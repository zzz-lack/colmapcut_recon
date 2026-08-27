import math

import pytest

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
)


def test_orbit_command_is_nominal_on_circle_with_tangent_heading() -> None:
    config = OrbitControllerConfig(radius_m=0.8, tangential_speed_mps=0.16)
    command = compute_orbit_command(
        OrbitState(x=0.8, y=0.0, yaw=math.pi / 2),
        (0.0, 0.0),
        config,
    )
    assert command.radius_error_m == pytest.approx(0.0)
    assert command.heading_error_rad == pytest.approx(0.0)
    assert command.linear_mps == pytest.approx(0.16)
    assert command.angular_rps == pytest.approx(0.2)


def test_orbit_command_steers_inward_when_outside_circle() -> None:
    config = OrbitControllerConfig(radius_m=0.8, tangential_speed_mps=0.16)
    command = compute_orbit_command(
        OrbitState(x=1.0, y=0.0, yaw=math.pi / 2),
        (0.0, 0.0),
        config,
    )
    assert command.radius_error_m == pytest.approx(0.2)
    assert command.heading_error_rad > 0.0


def test_end_effector_target_keeps_fixed_horizontal_standoff() -> None:
    target = end_effector_target(math.pi / 3, (0.1, -0.2, 0.4), 0.35, 0.55)
    assert math.hypot(target[0] - 0.1, target[1] + 0.2) == pytest.approx(0.35)
    assert target[2] == pytest.approx(0.55)


def test_camera_look_at_quaternion_is_normalized() -> None:
    quaternion = camera_look_at_quaternion_wxyz((0.4, 0.0, 0.5), (0.0, 0.0, 0.5))
    assert sum(value * value for value in quaternion) == pytest.approx(1.0)


def test_camera_roll_candidates_remain_normalized() -> None:
    look_at = camera_look_at_quaternion_wxyz((0.4, 0.0, 0.5), (0.0, 0.0, 0.5))
    candidates = camera_roll_orientations_wxyz(look_at, (0.0, math.pi / 2, math.pi))
    assert candidates[0] == pytest.approx(look_at)
    for quaternion in candidates:
        assert sum(value * value for value in quaternion) == pytest.approx(1.0)


def test_square_corners_follow_requested_direction() -> None:
    assert square_corners((0.0, 0.0), 0.6, 1) == (
        (-0.6, -0.6),
        (0.6, -0.6),
        (0.6, 0.6),
        (-0.6, 0.6),
    )
    assert square_corners((0.0, 0.0), 0.6, -1)[1] == (-0.6, 0.6)


def test_square_command_drives_straight_along_edge() -> None:
    config = SquareControllerConfig(half_extent_m=0.6, linear_speed_mps=0.12)
    command = compute_square_command(
        OrbitState(x=-0.4, y=-0.6, yaw=0.0),
        (0.0, 0.0),
        config,
        target_corner_index=1,
    )
    assert command.target_corner_index == 1
    assert command.linear_mps == pytest.approx(0.12)
    assert command.angular_rps == pytest.approx(0.0)
    assert command.edge_error_m == pytest.approx(0.0)


def test_square_command_switches_corner_and_uses_rolling_turn() -> None:
    config = SquareControllerConfig(
        half_extent_m=0.6,
        linear_speed_mps=0.12,
        corner_linear_speed_mps=0.0,
    )
    command = compute_square_command(
        OrbitState(x=0.57, y=-0.6, yaw=0.0),
        (0.0, 0.0),
        config,
        target_corner_index=1,
    )
    assert command.target_corner_index == 2
    assert command.linear_mps == pytest.approx(0.0)
    assert command.angular_rps > 0.0
