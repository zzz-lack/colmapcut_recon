#!/usr/bin/env python3
"""Inspect a reconstructed plant from a square or circular mobile-base path."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import tomllib
from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/simulation/tomato_orbit.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--test-frames", type=int, default=0, help="Exit after N frames; 0 runs interactively")
    return parser.parse_args()


args = parse_args()
# Do not let Kit reinterpret this script's --headless/--test-frames options.
sys.argv = [sys.argv[0]]
simulation_app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.numpy.rotations import (
    euler_angles_to_quats,
    quats_to_euler_angles,
)
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)
from isaacsim.sensors.camera import Camera
from pxr import Sdf, UsdLux, UsdPhysics

sys.path.insert(0, str(ROOT / "src"))
from colmapcut_recon.simulation.orbit_control import (
    OrbitControllerConfig,
    OrbitState,
    SquareControllerConfig,
    camera_look_at_quaternion_wxyz,
    camera_roll_orientations_wxyz,
    compute_orbit_command,
    compute_square_command,
    end_effector_target,
    square_corners,
    wrap_angle,
)

LEFT_WHEELS = ("front_left_wheel_joint", "rear_left_wheel_joint")
RIGHT_WHEELS = ("front_right_wheel_joint", "rear_right_wheel_joint")
ARM_JOINTS = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
CAMERA_ROLL_CANDIDATES_RAD = tuple(index * math.pi / 4.0 for index in range(8))


def load_config(path: Path) -> dict:
    with path.resolve(strict=True).open("rb") as handle:
        return tomllib.load(handle)


def find_prim_path_by_name(name: str) -> str:
    stage = omni.usd.get_context().get_stage()
    matches = [str(prim.GetPath()) for prim in stage.Traverse() if prim.GetName() == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one prim named {name!r}, found {matches}")
    return matches[0]


def find_articulation_root_path() -> str:
    stage = omni.usd.get_context().get_stage()
    matches = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one articulation root, found {matches}")
    return matches[0]


def configure_wheel_velocity_drives() -> None:
    stage = omni.usd.get_context().get_stage()
    for name in LEFT_WHEELS + RIGHT_WHEELS:
        candidates = [prim for prim in stage.Traverse() if prim.GetName() == name and prim.IsA(UsdPhysics.Joint)]
        if len(candidates) != 1:
            raise RuntimeError(f"Could not uniquely resolve wheel joint {name}: {candidates}")
        drive = UsdPhysics.DriveAPI.Get(candidates[0], "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(candidates[0], "angular")
        drive.CreateStiffnessAttr(0.0).Set(0.0)
        drive.CreateDampingAttr(8.0).Set(8.0)
        drive.CreateMaxForceAttr(6.0).Set(6.0)


def wheel_action(robot: SingleArticulation, linear: float, angular: float, radius: float, separation: float):
    left = (linear - 0.5 * separation * angular) / radius
    right = (linear + 0.5 * separation * angular) / radius
    names = LEFT_WHEELS + RIGHT_WHEELS
    indices = np.array([robot.get_dof_index(name) for name in names], dtype=np.int32)
    velocities = np.array([left, left, right, right], dtype=np.float32)
    return ArticulationAction(joint_velocities=velocities, joint_indices=indices)


def main() -> int:
    print(
        f"simulation request: headless={args.headless} test_frames={args.test_frames}",
        flush=True,
    )
    config = load_config(args.config)
    assets = config["assets"]
    prims = config["prims"]
    scene_cfg = config["scene"]
    drive_cfg = config["drive"]
    path_cfg = config["path"]
    orbit_cfg = config["orbit"]
    square_cfg = config["square"]
    arm_cfg = config["arm"]
    camera_cfg = config["camera"]
    for key in (
        "robot_usd",
        "robot_urdf",
        "robot_descriptor",
        "environment_usdz",
        "tomato_entities_usd",
    ):
        Path(assets[key]).resolve(strict=True)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=float(scene_cfg["physics_dt_s"]),
        rendering_dt=float(scene_cfg["rendering_dt_s"]),
    )
    add_reference_to_stage(assets["environment_usdz"], prims["environment"])
    add_reference_to_stage(
        assets["tomato_entities_usd"], prims["tomato_entities"]
    )
    add_reference_to_stage(assets["robot_usd"], prims["robot"])
    stage = omni.usd.get_context().get_stage()
    articulation_path = find_articulation_root_path()

    light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/OrbitLight"))
    light.CreateIntensityAttr(1000.0)
    light.AddRotateXYZOp().Set((-35.0, 25.0, 20.0))

    plant = tuple(float(value) for value in scene_cfg["plant_center_xyz"])
    path_mode = str(path_cfg["mode"])
    if path_mode not in {"orbit", "square"}:
        raise ValueError(f"Unsupported path.mode: {path_mode!r}")
    drive_mode = str(drive_cfg["mode"])
    if drive_mode not in {"kinematic", "wheel"}:
        raise ValueError(f"Unsupported drive.mode: {drive_mode!r}")
    orbit_radius = float(orbit_cfg["radius_m"])
    orbit_direction = int(orbit_cfg["direction"])
    if path_mode == "square":
        corners = square_corners(
            (plant[0], plant[1]),
            float(square_cfg["half_extent_m"]),
            int(square_cfg["direction"]),
        )
        start_xy = corners[0]
        start_yaw = math.atan2(corners[1][1] - start_xy[1], corners[1][0] - start_xy[0])
    else:
        start_angle = float(scene_cfg["robot_start_angle_rad"])
        start_xy = (
            plant[0] + orbit_radius * math.cos(start_angle),
            plant[1] + orbit_radius * math.sin(start_angle),
        )
        start_yaw = start_angle + orbit_direction * math.pi / 2.0
    print(
        f"path setup: mode={path_mode} drive={drive_mode} "
        f"start_xy={np.round(start_xy, 3)} "
        f"start_yaw={start_yaw:.3f}",
        flush=True,
    )
    start_position = np.array([start_xy[0], start_xy[1], float(scene_cfg["robot_start_z_m"])])
    start_orientation = euler_angles_to_quats(np.array([0.0, 0.0, start_yaw]))
    robot = world.scene.add(
        SingleArticulation(
            prim_path=articulation_path,
            name="mobile_manipulator",
            position=start_position,
            orientation=start_orientation,
        )
    )
    configure_wheel_velocity_drives()

    camera_frame_path = find_prim_path_by_name(arm_cfg["end_effector_frame"])
    camera = Camera(
        prim_path=f"{camera_frame_path}/EefCamera",
        name=camera_cfg["name"],
        translation=np.zeros(3),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        frequency=int(camera_cfg["frequency_hz"]),
        resolution=tuple(int(value) for value in camera_cfg["resolution"]),
    )

    world.reset()
    arm_indices = np.array([robot.get_dof_index(name) for name in ARM_JOINTS], dtype=np.int32)
    robot.set_joint_positions(
        np.array([0.0, 0.35, 1.1, 0.0, 0.75, 0.0], dtype=np.float32),
        joint_indices=arm_indices,
    )
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()

    lula = LulaKinematicsSolver(
        robot_description_path=assets["robot_descriptor"],
        urdf_path=assets["robot_urdf"],
    )
    ik_solver = ArticulationKinematicsSolver(robot, lula, arm_cfg["end_effector_frame"])
    orbit_controller = OrbitControllerConfig(
        radius_m=orbit_radius,
        tangential_speed_mps=float(orbit_cfg["tangential_speed_mps"]),
        direction=orbit_direction,
        radial_gain=float(orbit_cfg["radial_gain"]),
        heading_gain=float(orbit_cfg["heading_gain"]),
        max_linear_speed_mps=float(orbit_cfg["max_linear_speed_mps"]),
        max_angular_speed_rps=float(orbit_cfg["max_angular_speed_rps"]),
    )
    square_controller = SquareControllerConfig(
        half_extent_m=float(square_cfg["half_extent_m"]),
        linear_speed_mps=float(square_cfg["linear_speed_mps"]),
        direction=int(square_cfg["direction"]),
        corner_tolerance_m=float(square_cfg["corner_tolerance_m"]),
        heading_gain=float(square_cfg["heading_gain"]),
        turn_in_place_threshold_rad=float(square_cfg["turn_in_place_threshold_rad"]),
        corner_linear_speed_mps=float(square_cfg["corner_linear_speed_mps"]),
        max_angular_speed_rps=float(square_cfg["max_angular_speed_rps"]),
    )

    warmup_frames = int(camera_cfg["warmup_frames"])
    for _ in range(warmup_frames):
        world.step(render=True)
    print(
        f"warmup complete: frames={warmup_frames} app_running={simulation_app.is_running()}",
        flush=True,
    )

    frame = 0
    # One control iteration advances one rendering interval; World internally
    # performs the required number of fixed physics substeps.
    dt = float(scene_cfg["rendering_dt_s"])
    last_ik_success = False
    ik_mode = "none"
    selected_camera_roll_rad = None
    desired_standoff = float(arm_cfg["initial_standoff_m"])
    square_target_corner_index = 1
    kinematic_state = OrbitState(float(start_xy[0]), float(start_xy[1]), start_yaw)
    while (args.test_frames > 0 and frame < args.test_frames) or (
        args.test_frames <= 0 and simulation_app.is_running()
    ):
        if drive_mode == "kinematic":
            robot.set_world_pose(
                position=np.array(
                    [
                        kinematic_state.x,
                        kinematic_state.y,
                        float(scene_cfg["robot_start_z_m"]),
                    ]
                ),
                orientation=euler_angles_to_quats(
                    np.array([0.0, 0.0, kinematic_state.yaw])
                ),
            )
        position, orientation = robot.get_world_pose()
        # Lula does not automatically follow a moving articulation root.  Keep
        # the URDF base_footprint pose aligned with the simulated mobile base.
        lula.set_robot_base_pose(np.asarray(position), np.asarray(orientation))
        yaw = float(quats_to_euler_angles(np.asarray(orientation))[2])
        state = OrbitState(float(position[0]), float(position[1]), yaw)
        if path_mode == "square":
            command = compute_square_command(
                state,
                (plant[0], plant[1]),
                square_controller,
                square_target_corner_index,
            )
            square_target_corner_index = command.target_corner_index
            path_status = (
                f"edge_error={command.edge_error_m:+.3f} "
                f"corner={command.target_corner_index}"
            )
        else:
            command = compute_orbit_command(
                state,
                (plant[0], plant[1]),
                orbit_controller,
            )
            path_status = f"radius_error={command.radius_error_m:+.3f}"
        if drive_mode == "wheel":
            robot.apply_action(
                wheel_action(
                    robot,
                    command.linear_mps,
                    command.angular_rps,
                    float(orbit_cfg["wheel_radius_m"]),
                    float(orbit_cfg["wheel_separation_m"]),
                )
            )

        if frame % int(arm_cfg["command_period_steps"]) == 0:
            elapsed = frame * dt
            blend = min(1.0, elapsed / float(arm_cfg["approach_duration_s"]))
            desired_standoff = (
                (1.0 - blend) * float(arm_cfg["initial_standoff_m"])
                + blend * float(arm_cfg["inspection_standoff_m"])
            )
            target_position = end_effector_target(
                command.polar_angle_rad,
                plant,
                desired_standoff,
                float(arm_cfg["inspection_height_m"]),
            )
            target_orientation = camera_look_at_quaternion_wxyz(target_position, plant)
            orientations = camera_roll_orientations_wxyz(
                target_orientation,
                CAMERA_ROLL_CANDIDATES_RAD,
            )
            last_ik_success = False
            ik_mode = "none"
            selected_camera_roll_rad = None
            for roll, orientation_candidate in zip(
                CAMERA_ROLL_CANDIDATES_RAD,
                orientations,
                strict=True,
            ):
                arm_action, last_ik_success = ik_solver.compute_inverse_kinematics(
                    target_position=np.asarray(target_position),
                    target_orientation=np.asarray(orientation_candidate),
                    orientation_tolerance=float(arm_cfg["orientation_tolerance_rad"]),
                )
                if last_ik_success:
                    selected_camera_roll_rad = roll
                    ik_mode = "pose"
                    break
            if not last_ik_success:
                arm_action, last_ik_success = ik_solver.compute_inverse_kinematics(
                    target_position=np.asarray(target_position),
                )
                if last_ik_success:
                    ik_mode = "position"
            if last_ik_success:
                robot.apply_action(arm_action)
            elif frame == 0:
                current_position, _ = ik_solver.compute_end_effector_pose()
                print(
                    f"IK diagnostic: current_eef={np.round(current_position, 3)} "
                    f"target={np.round(target_position, 3)} "
                    "position_only=False",
                    flush=True,
                )

        if frame % 120 == 0:
            current = camera.get_current_frame()
            depth = current.get("distance_to_image_plane")
            current_eef_position, _ = ik_solver.compute_end_effector_pose(position_only=True)
            current_standoff = math.hypot(
                float(current_eef_position[0]) - plant[0],
                float(current_eef_position[1]) - plant[1],
            )
            print(
                f"frame={frame} mode={path_mode}/{drive_mode} "
                f"pos=({state.x:+.3f},{state.y:+.3f}) "
                f"yaw={state.yaw:+.3f} cmd=({command.linear_mps:+.3f},"
                f"{command.angular_rps:+.3f}) {path_status} "
                f"ik={last_ik_success}/{ik_mode} roll={selected_camera_roll_rad} "
                f"standoff={current_standoff:.3f}/{desired_standoff:.3f} "
                f"rgb={camera.get_rgb().shape} "
                f"depth={None if depth is None else depth.shape}",
                flush=True,
            )
        world.step(render=True)
        if drive_mode == "kinematic":
            kinematic_state = OrbitState(
                state.x + command.linear_mps * math.cos(state.yaw) * dt,
                state.y + command.linear_mps * math.sin(state.yaw) * dt,
                wrap_angle(state.yaw + command.angular_rps * dt),
            )
        frame += 1
    return 0


try:
    raise SystemExit(main())
finally:
    simulation_app.close()
