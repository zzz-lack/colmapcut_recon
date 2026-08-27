"""Build a clean, closed collision mesh from reconstructed ground data."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CollisionMeshConfig:
    """Parameters for fitting and closing a ground collision height field."""

    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    grid_size_m: float = 0.02
    initial_ground_height_m: float = 0.0
    max_slope_degrees: float = 40.0
    min_points_per_node: int = 2
    smoothing_iterations: int = 2
    mesh_weight: float = 0.75
    max_mesh_residual_m: float = 0.03
    min_triangle_vertical_normal: float = 0.3
    bottom_offset_m: float = 0.03

    def validate(self) -> None:
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("XY bounds must have positive area")
        if self.grid_size_m <= 0:
            raise ValueError("grid_size_m must be positive")
        if not 0 < self.max_slope_degrees < 89:
            raise ValueError("max_slope_degrees must be in (0, 89)")
        if self.min_points_per_node < 1:
            raise ValueError("min_points_per_node must be at least one")
        if self.smoothing_iterations < 0:
            raise ValueError("smoothing_iterations cannot be negative")
        if not 0 <= self.mesh_weight <= 1:
            raise ValueError("mesh_weight must be in [0, 1]")
        if self.max_mesh_residual_m <= 0:
            raise ValueError("max_mesh_residual_m must be positive")
        if not 0 <= self.min_triangle_vertical_normal <= 1:
            raise ValueError("min_triangle_vertical_normal must be in [0, 1]")
        if self.bottom_offset_m <= 0:
            raise ValueError("bottom_offset_m must be positive")


def make_xy_grid(config: CollisionMeshConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return inclusive, evenly spaced grid coordinates for the requested bounds."""

    config.validate()
    columns = max(2, int(round((config.x_max - config.x_min) / config.grid_size_m)) + 1)
    rows = max(2, int(round((config.y_max - config.y_min) / config.grid_size_m)) + 1)
    x = np.linspace(config.x_min, config.x_max, columns, dtype=np.float64)
    y = np.linspace(config.y_min, config.y_max, rows, dtype=np.float64)
    return x, y


def _median_smooth(height: np.ndarray, iterations: int) -> np.ndarray:
    from scipy.ndimage import median_filter

    filtered = np.asarray(height, dtype=np.float64).copy()
    for _ in range(iterations):
        filtered = median_filter(filtered, size=3, mode="nearest")
    return filtered


def limit_heightfield_slope(
    height: np.ndarray,
    x_spacing: float,
    y_spacing: float,
    max_slope_degrees: float,
    *,
    max_iterations: int = 100,
) -> np.ndarray:
    """Project a height grid onto an orthogonal maximum-slope constraint."""

    result = np.asarray(height, dtype=np.float64).copy()
    max_dx = math.tan(math.radians(max_slope_degrees)) * x_spacing
    max_dy = math.tan(math.radians(max_slope_degrees)) * y_spacing
    rows, columns = result.shape

    for _ in range(max_iterations):
        changed = False
        for row in range(rows):
            for column in range(1, columns):
                reference = result[row, column - 1]
                clipped = np.clip(result[row, column], reference - max_dx, reference + max_dx)
                changed |= abs(clipped - result[row, column]) > 1e-12
                result[row, column] = clipped
            for column in range(columns - 2, -1, -1):
                reference = result[row, column + 1]
                clipped = np.clip(result[row, column], reference - max_dx, reference + max_dx)
                changed |= abs(clipped - result[row, column]) > 1e-12
                result[row, column] = clipped
        for column in range(columns):
            for row in range(1, rows):
                reference = result[row - 1, column]
                clipped = np.clip(result[row, column], reference - max_dy, reference + max_dy)
                changed |= abs(clipped - result[row, column]) > 1e-12
                result[row, column] = clipped
            for row in range(rows - 2, -1, -1):
                reference = result[row + 1, column]
                clipped = np.clip(result[row, column], reference - max_dy, reference + max_dy)
                changed |= abs(clipped - result[row, column]) > 1e-12
                result[row, column] = clipped
        if not changed:
            break
    return result


def fit_reference_heightfield(
    points: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    config: CollisionMeshConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit a robust continuous height field from Gaussian center positions."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    finite = np.all(np.isfinite(points), axis=1)
    inside = (
        finite
        & (points[:, 0] >= x_grid[0])
        & (points[:, 0] <= x_grid[-1])
        & (points[:, 1] >= y_grid[0])
        & (points[:, 1] <= y_grid[-1])
    )
    cropped = points[inside]
    if len(cropped) == 0:
        raise ValueError("No finite ground points lie inside the requested 2D bounds")

    x_spacing = float(x_grid[1] - x_grid[0])
    y_spacing = float(y_grid[1] - y_grid[0])
    x_index = np.clip(
        np.rint((cropped[:, 0] - x_grid[0]) / x_spacing).astype(np.int64),
        0,
        len(x_grid) - 1,
    )
    y_index = np.clip(
        np.rint((cropped[:, 1] - y_grid[0]) / y_spacing).astype(np.int64),
        0,
        len(y_grid) - 1,
    )
    node_id = y_index * len(x_grid) + x_index
    order = np.argsort(node_id, kind="stable")
    ordered_nodes = node_id[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_nodes)) + 1]
    ends = np.r_[starts[1:], len(order)]

    raw = np.full((len(y_grid), len(x_grid)), np.nan, dtype=np.float64)
    populated = 0
    for start, end in zip(starts, ends):
        populated += 1
        if end - start < config.min_points_per_node:
            continue
        current = int(ordered_nodes[start])
        raw.flat[current] = float(np.median(cropped[order[start:end], 2]))

    finite_raw = np.isfinite(raw)
    if not np.any(finite_raw):
        raise ValueError("No grid node contains the requested minimum number of points")

    # Nearest-value filling makes gaps explicit and deterministic. The later
    # slope projection prevents an isolated plant/outlier node from producing
    # a vertical collision spike.
    from scipy.ndimage import distance_transform_edt

    nearest_indices = distance_transform_edt(
        ~finite_raw, return_distances=False, return_indices=True
    )
    filled = raw[tuple(nearest_indices)]
    filled = _median_smooth(filled, config.smoothing_iterations)
    limited = limit_heightfield_slope(
        filled,
        x_spacing,
        y_spacing,
        config.max_slope_degrees,
    )

    diagnostics = {
        "input_points": int(len(points)),
        "cropped_points": int(len(cropped)),
        "populated_nodes": int(populated),
        "measured_nodes": int(finite_raw.sum()),
        "filled_nodes": int(raw.size - finite_raw.sum()),
        "raw_z_range_m": [float(np.nanmin(raw)), float(np.nanmax(raw))],
        "reference_z_range_m": [float(limited.min()), float(limited.max())],
    }
    return limited, diagnostics


def rasterize_mesh_height(
    vertices: np.ndarray,
    triangles: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    reference_height: np.ndarray,
    *,
    max_residual_m: float,
    min_vertical_normal: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Rasterize trustworthy SuGaR triangles, selecting the Z nearest the reference."""

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    best_height = np.full(reference_height.shape, np.nan, dtype=np.float64)
    best_residual = np.full(reference_height.shape, np.inf, dtype=np.float64)
    x_spacing = float(x_grid[1] - x_grid[0])
    y_spacing = float(y_grid[1] - y_grid[0])
    considered = 0
    rasterized = 0

    for face in triangles:
        triangle = vertices[face]
        if not np.all(np.isfinite(triangle)):
            continue
        edge_a = triangle[1] - triangle[0]
        edge_b = triangle[2] - triangle[0]
        normal = np.cross(edge_a, edge_b)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-14 or abs(float(normal[2])) / normal_norm < min_vertical_normal:
            continue

        xy = triangle[:, :2]
        determinant = (
            (xy[1, 1] - xy[2, 1]) * (xy[0, 0] - xy[2, 0])
            + (xy[2, 0] - xy[1, 0]) * (xy[0, 1] - xy[2, 1])
        )
        if abs(float(determinant)) <= 1e-14:
            continue
        considered += 1

        column_min = max(0, int(math.ceil((xy[:, 0].min() - x_grid[0]) / x_spacing)))
        column_max = min(
            len(x_grid) - 1,
            int(math.floor((xy[:, 0].max() - x_grid[0]) / x_spacing)),
        )
        row_min = max(0, int(math.ceil((xy[:, 1].min() - y_grid[0]) / y_spacing)))
        row_max = min(
            len(y_grid) - 1,
            int(math.floor((xy[:, 1].max() - y_grid[0]) / y_spacing)),
        )
        if column_min > column_max or row_min > row_max:
            continue

        columns = np.arange(column_min, column_max + 1)
        rows = np.arange(row_min, row_max + 1)
        grid_x, grid_y = np.meshgrid(x_grid[columns], y_grid[rows])
        weight_0 = (
            (xy[1, 1] - xy[2, 1]) * (grid_x - xy[2, 0])
            + (xy[2, 0] - xy[1, 0]) * (grid_y - xy[2, 1])
        ) / determinant
        weight_1 = (
            (xy[2, 1] - xy[0, 1]) * (grid_x - xy[2, 0])
            + (xy[0, 0] - xy[2, 0]) * (grid_y - xy[2, 1])
        ) / determinant
        weight_2 = 1.0 - weight_0 - weight_1
        inside = (weight_0 >= -1e-9) & (weight_1 >= -1e-9) & (weight_2 >= -1e-9)
        if not np.any(inside):
            continue

        candidate = weight_0 * triangle[0, 2] + weight_1 * triangle[1, 2] + weight_2 * triangle[2, 2]
        reference = reference_height[row_min : row_max + 1, column_min : column_max + 1]
        residual = np.abs(candidate - reference)
        accepted = inside & (residual <= max_residual_m)
        current_best = best_residual[row_min : row_max + 1, column_min : column_max + 1]
        update = accepted & (residual < current_best)
        if np.any(update):
            current_height = best_height[row_min : row_max + 1, column_min : column_max + 1]
            current_height[update] = candidate[update]
            current_best[update] = residual[update]
            rasterized += int(update.sum())

    mask = np.isfinite(best_height)
    diagnostics = {
        "input_triangles": int(len(triangles)),
        "considered_triangles": int(considered),
        "covered_nodes": int(mask.sum()),
        "height_updates": int(rasterized),
    }
    return best_height, mask, diagnostics


def build_closed_heightfield_mesh(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    top_height: np.ndarray,
    bottom_offset_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an oriented watertight solid with a gridded top and flat bottom."""

    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    top_height = np.asarray(top_height, dtype=np.float64)
    rows, columns = top_height.shape
    if top_height.shape != (len(y_grid), len(x_grid)) or rows < 2 or columns < 2:
        raise ValueError("top_height shape must match a grid of at least 2 x 2")
    if not np.all(np.isfinite(top_height)):
        raise ValueError("top_height contains non-finite values")

    grid_x, grid_y = np.meshgrid(x_grid, y_grid)
    top_vertices = np.column_stack((grid_x.ravel(), grid_y.ravel(), top_height.ravel()))
    top_faces: list[list[int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = (row + 1) * columns + column
            upper_right = upper_left + 1
            top_faces.extend(
                ([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left])
            )

    boundary: list[int] = []
    boundary.extend(range(columns))
    boundary.extend(row * columns + columns - 1 for row in range(1, rows))
    boundary.extend((rows - 1) * columns + column for column in range(columns - 2, -1, -1))
    boundary.extend(row * columns for row in range(rows - 2, 0, -1))

    bottom_z = float(top_height.min() - bottom_offset_m)
    bottom_start = len(top_vertices)
    bottom_vertices = top_vertices[boundary].copy()
    bottom_vertices[:, 2] = bottom_z
    center_index = bottom_start + len(boundary)
    center = np.array(
        [[0.5 * (x_grid[0] + x_grid[-1]), 0.5 * (y_grid[0] + y_grid[-1]), bottom_z]],
        dtype=np.float64,
    )
    vertices = np.vstack((top_vertices, bottom_vertices, center))

    side_faces: list[list[int]] = []
    bottom_faces: list[list[int]] = []
    for index, top_a in enumerate(boundary):
        next_index = (index + 1) % len(boundary)
        top_b = boundary[next_index]
        bottom_a = bottom_start + index
        bottom_b = bottom_start + next_index
        side_faces.extend(([top_a, bottom_a, bottom_b], [top_a, bottom_b, top_b]))
        bottom_faces.append([center_index, bottom_b, bottom_a])

    faces = np.asarray(top_faces + side_faces + bottom_faces, dtype=np.int64)
    if signed_mesh_volume(vertices, faces) < 0:
        faces[:, [1, 2]] = faces[:, [2, 1]]
    return vertices, faces


def signed_mesh_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    triangle_vertices = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles)]
    return float(
        np.einsum(
            "ij,ij->i",
            triangle_vertices[:, 0],
            np.cross(triangle_vertices[:, 1], triangle_vertices[:, 2]),
        ).sum()
        / 6.0
    )


def topology_report(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    """Return dependency-free edge, degeneracy, and orientation diagnostics."""

    triangles = np.asarray(triangles, dtype=np.int64)
    triangle_vertices = np.asarray(vertices, dtype=np.float64)[triangles]
    double_area = np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    edges = Counter(
        tuple(sorted((int(a), int(b))))
        for face in triangles
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
    )
    counts = np.fromiter(edges.values(), dtype=np.int64)
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "unique_edges": int(len(edges)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
        "degenerate_triangles": int(np.count_nonzero(double_area <= 1e-14)),
        "signed_volume_m3": signed_mesh_volume(vertices, triangles),
    }


def _read_ply_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertex = PlyData.read(path)["vertex"].data
    return np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float64, copy=False
    )


def _read_triangle_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError(f"Mesh contains no triangles: {path}")
    return vertices, triangles


def _write_triangle_mesh(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
        raise OSError(f"Open3D failed to write {path}")


def _open3d_validation(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    validation = {
        "edge_manifold_closed": bool(mesh.is_edge_manifold(allow_boundary_edges=False)),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "self_intersecting": bool(mesh.is_self_intersecting()),
        "watertight": bool(mesh.is_watertight()),
        "orientable": bool(mesh.is_orientable()),
    }
    validation["passed"] = bool(
        validation["edge_manifold_closed"]
        and validation["vertex_manifold"]
        and not validation["self_intersecting"]
        and validation["watertight"]
        and validation["orientable"]
    )
    return validation


def build_ground_collision_asset(
    *,
    ground_points_path: Path,
    sugar_mesh_path: Path,
    ply_output: Path,
    obj_output: Path,
    report_output: Path,
    config: CollisionMeshConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit, reconstruct, validate, and write a simulation collision mesh."""

    config.validate()
    output_paths = (ply_output, obj_output, report_output)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite: " + ", ".join(existing))
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    x_grid, y_grid = make_xy_grid(config)
    ground_points = _read_ply_points(ground_points_path)
    reference, reference_diagnostics = fit_reference_heightfield(
        ground_points, x_grid, y_grid, config
    )
    sugar_vertices, sugar_triangles = _read_triangle_mesh(sugar_mesh_path)
    mesh_height, mesh_mask, raster_diagnostics = rasterize_mesh_height(
        sugar_vertices,
        sugar_triangles,
        x_grid,
        y_grid,
        reference,
        max_residual_m=config.max_mesh_residual_m,
        min_vertical_normal=config.min_triangle_vertical_normal,
    )
    surface = reference.copy()
    surface[mesh_mask] = (
        (1.0 - config.mesh_weight) * reference[mesh_mask]
        + config.mesh_weight * mesh_height[mesh_mask]
    )
    surface = _median_smooth(surface, config.smoothing_iterations)
    surface = limit_heightfield_slope(
        surface,
        float(x_grid[1] - x_grid[0]),
        float(y_grid[1] - y_grid[0]),
        config.max_slope_degrees,
    )
    vertices, triangles = build_closed_heightfield_mesh(
        x_grid, y_grid, surface, config.bottom_offset_m
    )
    topology = topology_report(vertices, triangles)
    validation = _open3d_validation(vertices, triangles)
    if (
        topology["boundary_edges"]
        or topology["nonmanifold_edges"]
        or topology["degenerate_triangles"]
        or topology["signed_volume_m3"] <= 0
        or not validation["passed"]
    ):
        raise RuntimeError(
            "Generated collision mesh failed validation: "
            + json.dumps({"topology": topology, "open3d": validation})
        )

    _write_triangle_mesh(ply_output, vertices, triangles)
    _write_triangle_mesh(obj_output, vertices, triangles)
    report: dict[str, Any] = {
        "method": "closed_heightfield_from_ground_gaussians_and_sugar_mesh",
        "config": asdict(config),
        "inputs": {
            "ground_gaussians": str(ground_points_path.resolve()),
            "sugar_mesh": str(sugar_mesh_path.resolve()),
        },
        "grid": {
            "rows": int(len(y_grid)),
            "columns": int(len(x_grid)),
            "nodes": int(len(x_grid) * len(y_grid)),
            "surface_z_range_m": [float(surface.min()), float(surface.max())],
        },
        "reference_fit": reference_diagnostics,
        "sugar_rasterization": raster_diagnostics,
        "topology": topology,
        "open3d_validation": validation,
        "outputs": {
            "ply": str(ply_output.resolve()),
            "obj": str(obj_output.resolve()),
            "report": str(report_output.resolve()),
        },
    }
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-points", type=Path, required=True)
    parser.add_argument("--sugar-mesh", type=Path, required=True)
    parser.add_argument("--ply-output", type=Path, required=True)
    parser.add_argument("--obj-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--y-min", type=float, default=-1.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--initial-ground-height", type=float, default=0.0)
    parser.add_argument("--max-slope", type=float, default=40.0)
    parser.add_argument("--min-points-per-node", type=int, default=2)
    parser.add_argument("--smoothing-iterations", type=int, default=2)
    parser.add_argument("--mesh-weight", type=float, default=0.75)
    parser.add_argument("--max-mesh-residual", type=float, default=0.03)
    parser.add_argument("--min-triangle-vertical-normal", type=float, default=0.3)
    parser.add_argument("--bottom-offset", type=float, default=0.03)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = CollisionMeshConfig(
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        grid_size_m=args.grid_size,
        initial_ground_height_m=args.initial_ground_height,
        max_slope_degrees=args.max_slope,
        min_points_per_node=args.min_points_per_node,
        smoothing_iterations=args.smoothing_iterations,
        mesh_weight=args.mesh_weight,
        max_mesh_residual_m=args.max_mesh_residual,
        min_triangle_vertical_normal=args.min_triangle_vertical_normal,
        bottom_offset_m=args.bottom_offset,
    )
    report = build_ground_collision_asset(
        ground_points_path=args.ground_points,
        sugar_mesh_path=args.sugar_mesh,
        ply_output=args.ply_output,
        obj_output=args.obj_output,
        report_output=args.report_output,
        config=config,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
