# 番茄巡检仿真 / Tomato Inspection Simulation (Isaac Sim 5.1)

[中文概述](#中文概述) · [English details](#active-file-layout)

> 外部依赖与路径修改 / External dependencies and path configuration: [EXTERNAL_DEPENDENCIES.md](EXTERNAL_DEPENDENCIES.md)

## 中文概述

该流程在 Isaac Sim 5.1 中加载移动机械臂、NuRec 番茄场景、独立番茄实体和碰撞地面，执行绕植株巡检轨迹并验证关节、相机和碰撞状态。机器人、环境和实体资产路径在 `configs/simulation/tomato_orbit.toml` 修改。

外部依赖包括 Isaac Sim、其内置 Python、OpenUSD/`pxr`、PhysX 和 Lula motion generation。Isaac Sim 启动器不由普通 Python 环境替代；请在命令中把 `/home/linzz/isaacsim/.../python.sh` 改为本机启动器地址。

## English details

## Active file layout

```text
assets/
  robots/mobile_manipulator/
    urdf/combined_source.urdf       # preserved input
    urdf/combined_mobile.urdf       # repaired, active URDF
    lula/robot_descriptor.yaml      # six-axis arm kinematics
    usd/combined_mobile_merged/     # generated mobile articulation
  environments/roman_tomato_02/    # collision-proxy/target placeholders
configs/simulation/tomato_orbit.toml
scripts/isaac/
  01_audit_urdf.py
  02_import_robot_usd.py
  03_orbit_plant.py
src/colmapcut_recon/simulation/
  urdf_audit.py
  orbit_control.py
```

The active environment is
`data/scenes/roman_tomato_02/10_simulation_asset/roman_tomato_02_isaacsim51_tuned_collision.usdz`.

## Run

From the repository root:

```bash
env PYTHONPATH=src python scripts/isaac/01_audit_urdf.py

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  /home/linzz/isaacsim/_build/linux-x86_64/release/python.sh \
  scripts/isaac/02_import_robot_usd.py --headless

env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
  /home/linzz/isaacsim/_build/linux-x86_64/release/python.sh \
  scripts/isaac/03_orbit_plant.py
```

For a repeatable headless full-lap check, add `--headless --test-frames 1700`
to the last command. The interactive command runs until the Isaac Sim window is
closed.

## Path and drive modes

`configs/simulation/tomato_orbit.toml` selects two independent modes:

- `[path].mode = "square"` follows a 1.20 m by 1.20 m square centered on the
  configured plant. Change it to `"orbit"` for the earlier circular controller.
- `[drive].mode = "kinematic"` is the stable default for perception, arm-motion,
  and camera-data validation. It integrates the commanded base pose along each
  edge and turns at the vertices while the arm remains an Isaac articulation.
- `[drive].mode = "wheel"` sends the same controller output to the four PhysX
  wheel velocity drives. The imported TerraSentia wheel/contact model currently
  needs friction and skid-steer calibration before this mode is safe at corners.

The square centerline uses `half_extent_m = 0.60`. The collision ground spans
approximately `[-1, 1] m` in X and Y, so the centerline leaves room for the robot
footprint.

## Arm and wrist camera behavior

- Lula receives the moving `base_footprint` world pose every frame and solves the
  Lite6 joints for a world-space wrist-camera target.
- The target approaches from 0.50 m to 0.45 m, then follows the robot's polar
  angle while remaining 0.45 m horizontally from the plant center.
- Full look-at pose IK is used on the straight edges. At difficult corner poses,
  the solver can fall back to position-only IK so fixed standoff has priority;
  full camera orientation is reacquired after the turn.
- The wrist camera publishes RGB and `distance_to_image_plane` at 640x480.

## 3DGS and collision boundary

The NuRec/3DGS plant is visible to the RTX camera and produces RGB and depth. It
is not, by itself, PhysX collision geometry. The current asset contains an
invisible ground collider, but stems, leaves, and tomatoes still need mesh,
capsule, sphere, or other collision proxies before contact or grasping can be
simulated.

The controller currently uses the configured `plant_center_xyz`; it is a motion
baseline, not a visual detector. A later perception node can replace this center
with an estimate from RGB/depth while retaining the same path and standoff logic.

## Validation result

On Isaac Sim 5.1 with an RTX 5090, a 1700-frame headless run completed more than
one square lap. Checkpoints covered all four edges and four turns. Straight-edge
cross-track error was typically 0-5 mm and the largest logged corner error was
20 mm. After the initial approach, measured wrist-camera standoff was generally
0.443-0.460 m for a 0.450 m target. RGB and depth remained available at 640x480.

The machine-readable record is `outputs/simulation/square_validation.json`.
