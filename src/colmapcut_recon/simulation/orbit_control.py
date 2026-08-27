"""Pure geometry and feedback laws for orbiting a plant at fixed standoff."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OrbitControllerConfig:
    radius_m: float
    tangential_speed_mps: float
    direction: int = 1
    radial_gain: float = 1.5
    heading_gain: float = 2.5
    max_linear_speed_mps: float = 0.25
    max_angular_speed_rps: float = 0.8

    def __post_init__(self) -> None:
        if self.radius_m <= 0 or self.tangential_speed_mps <= 0:
            raise ValueError("Orbit radius and speed must be positive")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")


@dataclass(frozen=True)
class OrbitState:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class OrbitCommand:
    linear_mps: float
    angular_rps: float
    radius_error_m: float
    heading_error_rad: float
    polar_angle_rad: float


@dataclass(frozen=True)
class SquareControllerConfig:
    half_extent_m: float
    linear_speed_mps: float
    direction: int = 1
    corner_tolerance_m: float = 0.05
    heading_gain: float = 2.5
    turn_in_place_threshold_rad: float = 0.35
    corner_linear_speed_mps: float = 0.0
    max_angular_speed_rps: float = 0.8

    def __post_init__(self) -> None:
        if self.half_extent_m <= 0 or self.linear_speed_mps <= 0:
            raise ValueError("Square half extent and speed must be positive")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if self.corner_tolerance_m <= 0:
            raise ValueError("corner tolerance must be positive")
        if not 0 <= self.corner_linear_speed_mps <= self.linear_speed_mps:
            raise ValueError("corner speed must be between zero and linear speed")


@dataclass(frozen=True)
class SquareCommand:
    linear_mps: float
    angular_rps: float
    edge_error_m: float
    heading_error_rad: float
    polar_angle_rad: float
    target_corner_index: int


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_orbit_command(
    state: OrbitState,
    plant_xy: tuple[float, float],
    config: OrbitControllerConfig,
) -> OrbitCommand:
    """Compute a differential-drive command that converges to a circular orbit."""

    dx = state.x - plant_xy[0]
    dy = state.y - plant_xy[1]
    radius = math.hypot(dx, dy)
    if radius < 1e-6:
        raise ValueError("Robot cannot be initialized at the plant center")
    radial_x, radial_y = dx / radius, dy / radius
    tangent_x = -config.direction * radial_y
    tangent_y = config.direction * radial_x
    radius_error = radius - config.radius_m
    desired_x = config.tangential_speed_mps * tangent_x - config.radial_gain * radius_error * radial_x
    desired_y = config.tangential_speed_mps * tangent_y - config.radial_gain * radius_error * radial_y
    desired_heading = math.atan2(desired_y, desired_x)
    heading_error = wrap_angle(desired_heading - state.yaw)
    nominal_angular = config.direction * config.tangential_speed_mps / config.radius_m
    linear = config.tangential_speed_mps * max(0.0, math.cos(heading_error))
    angular = nominal_angular + config.heading_gain * heading_error
    linear = max(-config.max_linear_speed_mps, min(config.max_linear_speed_mps, linear))
    angular = max(-config.max_angular_speed_rps, min(config.max_angular_speed_rps, angular))
    return OrbitCommand(linear, angular, radius_error, heading_error, math.atan2(dy, dx))


def square_corners(
    center_xy: tuple[float, float],
    half_extent_m: float,
    direction: int = 1,
) -> tuple[tuple[float, float], ...]:
    """Return a deterministic bottom-left-starting square waypoint sequence."""

    cx, cy = center_xy
    h = half_extent_m
    counterclockwise = (
        (cx - h, cy - h),
        (cx + h, cy - h),
        (cx + h, cy + h),
        (cx - h, cy + h),
    )
    if direction == 1:
        return counterclockwise
    if direction == -1:
        return (counterclockwise[0], *reversed(counterclockwise[1:]))
    raise ValueError("direction must be -1 or 1")


def compute_square_command(
    state: OrbitState,
    plant_xy: tuple[float, float],
    config: SquareControllerConfig,
    target_corner_index: int,
) -> SquareCommand:
    """Track square edges and use a configurable turn at each 90-degree corner."""

    corners = square_corners(plant_xy, config.half_extent_m, config.direction)
    target_index = target_corner_index % len(corners)
    previous_index = (target_index - 1) % len(corners)
    previous_x, previous_y = corners[previous_index]
    target_x, target_y = corners[target_index]
    edge_x = target_x - previous_x
    edge_y = target_y - previous_y
    edge_length = math.hypot(edge_x, edge_y)
    unit_x, unit_y = edge_x / edge_length, edge_y / edge_length
    from_previous_x = state.x - previous_x
    from_previous_y = state.y - previous_y
    progress_m = from_previous_x * unit_x + from_previous_y * unit_y
    distance_to_target = math.hypot(target_x - state.x, target_y - state.y)
    if (
        distance_to_target <= config.corner_tolerance_m
        or progress_m >= edge_length - config.corner_tolerance_m
    ):
        target_index = (target_index + 1) % len(corners)
        previous_index = (target_index - 1) % len(corners)
        previous_x, previous_y = corners[previous_index]
        target_x, target_y = corners[target_index]
        edge_x = target_x - previous_x
        edge_y = target_y - previous_y
        edge_length = math.hypot(edge_x, edge_y)
        unit_x, unit_y = edge_x / edge_length, edge_y / edge_length
        from_previous_x = state.x - previous_x
        from_previous_y = state.y - previous_y

    desired_heading = math.atan2(target_y - state.y, target_x - state.x)
    heading_error = wrap_angle(desired_heading - state.yaw)
    angular = max(
        -config.max_angular_speed_rps,
        min(config.max_angular_speed_rps, config.heading_gain * heading_error),
    )
    if abs(heading_error) >= config.turn_in_place_threshold_rad:
        # Zero produces an in-place kinematic turn. A small positive value can
        # be used for a calibrated skid-steer base that needs a rolling corner.
        linear = config.corner_linear_speed_mps
    else:
        linear = config.linear_speed_mps * max(0.0, math.cos(heading_error))

    # Signed perpendicular distance from the active directed edge.
    edge_error = unit_x * from_previous_y - unit_y * from_previous_x
    polar_angle = math.atan2(state.y - plant_xy[1], state.x - plant_xy[0])
    return SquareCommand(
        linear,
        angular,
        edge_error,
        heading_error,
        polar_angle,
        target_index,
    )


def end_effector_target(
    polar_angle_rad: float,
    plant_xyz: tuple[float, float, float],
    standoff_m: float,
    height_m: float,
) -> tuple[float, float, float]:
    """Place the wrist camera on the robot-facing side of a fixed-radius circle."""

    return (
        plant_xyz[0] + standoff_m * math.cos(polar_angle_rad),
        plant_xyz[1] + standoff_m * math.sin(polar_angle_rad),
        height_m,
    )


def camera_look_at_quaternion_wxyz(
    camera_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """Return a USD-camera orientation: local -Z looks at target and +Y is up."""

    fx = target_xyz[0] - camera_xyz[0]
    fy = target_xyz[1] - camera_xyz[1]
    fz = target_xyz[2] - camera_xyz[2]
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    if norm < 1e-9:
        raise ValueError("camera and target positions must differ")
    fx, fy, fz = fx / norm, fy / norm, fz / norm
    zx, zy, zz = -fx, -fy, -fz
    # local X = world-up cross local Z; fall back when looking vertically.
    xx, xy, xz = -zy, zx, 0.0
    xnorm = math.sqrt(xx * xx + xy * xy)
    if xnorm < 1e-9:
        xx, xy, xz, xnorm = 1.0, 0.0, 0.0, 1.0
    xx, xy, xz = xx / xnorm, xy / xnorm, xz / xnorm
    yx = zy * xz - zz * xy
    yy = zz * xx - zx * xz
    yz = zx * xy - zy * xx
    # Rotation matrix columns are local X/Y/Z in world coordinates.
    m00, m01, m02 = xx, yx, zx
    m10, m11, m12 = xy, yy, zy
    m20, m21, m22 = xz, yz, zz
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


def quaternion_multiply_wxyz(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Compose two scalar-first quaternions as ``left * right``."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def camera_roll_orientations_wxyz(
    look_at_orientation: tuple[float, float, float, float],
    rolls_rad: tuple[float, ...],
) -> tuple[tuple[float, float, float, float], ...]:
    """Vary camera roll without changing the local -Z viewing direction."""

    orientations = []
    for roll in rolls_rad:
        half = 0.5 * roll
        local_z_roll = (math.cos(half), 0.0, 0.0, math.sin(half))
        orientations.append(quaternion_multiply_wxyz(look_at_orientation, local_z_roll))
    return tuple(orientations)
