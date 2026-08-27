"""Build a plant-and-ground Gaussian asset without changing source files.

The default ground method estimates a continuous local height field from the
full Gaussian scene.  It follows sloped terrain from a known seed height and
uses every Gaussian's oriented ellipsoid plus locally measured reconstruction
spread to classify surface-supporting splats.  By default, selected plant
Gaussians have ownership priority: they are excluded before terrain estimation
and can never be written to the ground component.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from plyfile import PlyData, PlyElement


XYZ_PROPERTIES = ("x", "y", "z")
REQUIRED_GAUSSIAN_PROPERTIES = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
XYZ_KEY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])


@dataclass(frozen=True)
class GroundBounds:
    """Horizontal asset footprint plus legacy slab limits."""

    x_min: float = -0.5
    x_max: float = 0.5
    y_min: float = -0.5
    y_max: float = 0.5
    z_min: float = -0.05
    z_max: float = 0.05

    def validate(self, require_z: bool = True) -> None:
        """Reject empty or reversed intervals before processing large files."""

        if not (self.x_min < self.x_max):
            raise ValueError("x_min must be smaller than x_max")
        if not (self.y_min < self.y_max):
            raise ValueError("y_min must be smaller than y_max")
        if require_z and not (self.z_min < self.z_max):
            raise ValueError("z_min must be smaller than z_max")


@dataclass(frozen=True)
class AdaptiveHeightfieldConfig:
    """Parameters for slope-aware local terrain estimation.

    None of these values defines a global vertical crop.  The ground band is
    estimated per XY cell from the below-surface reconstruction distribution
    and each Gaussian's oriented vertical standard deviation.
    """

    grid_size_m: float = 0.02
    initial_ground_height_m: float = 0.0
    surface_quantile: float = 0.15
    max_slope_degrees: float = 40.0
    seed_tolerance_steps: float = 2.0
    smoothing_iterations: int = 3
    ground_band_quantile: float = 0.90
    ground_band_mad_multiplier: float = 3.0
    gaussian_sigma_multiplier: float = 5.0
    max_sigma_to_band_ratio: float = 2.0
    min_points_per_cell: int = 5

    def validate(self) -> None:
        """Validate terrain parameters before allocating a grid."""

        if self.grid_size_m <= 0:
            raise ValueError("grid_size_m must be positive")
        if not (0.0 < self.surface_quantile < 0.5):
            raise ValueError("surface_quantile must be between 0 and 0.5")
        if not (0.0 < self.max_slope_degrees < 89.0):
            raise ValueError("max_slope_degrees must be between 0 and 89")
        if self.seed_tolerance_steps <= 0:
            raise ValueError("seed_tolerance_steps must be positive")
        if self.smoothing_iterations < 0:
            raise ValueError("smoothing_iterations cannot be negative")
        if not (0.5 <= self.ground_band_quantile < 1.0):
            raise ValueError("ground_band_quantile must be in [0.5, 1.0)")
        if self.ground_band_mad_multiplier <= 0:
            raise ValueError("ground_band_mad_multiplier must be positive")
        if self.gaussian_sigma_multiplier <= 0:
            raise ValueError("gaussian_sigma_multiplier must be positive")
        if self.max_sigma_to_band_ratio < 1.0:
            raise ValueError("max_sigma_to_band_ratio must be at least one")
        if self.min_points_per_cell < 1:
            raise ValueError("min_points_per_cell must be at least one")


@dataclass(frozen=True)
class AssemblyOutputs:
    """Paths written by one asset-assembly run."""

    plant: Path
    ground: Path
    combined: Path
    report: Path


def _read_vertex_data(path: Path) -> tuple[PlyData, np.ndarray]:
    """Memory-map a PLY and return its single vertex table."""

    if not path.is_file():
        raise FileNotFoundError(f"PLY input does not exist: {path}")
    ply = PlyData.read(str(path), mmap="r")
    try:
        vertex = ply["vertex"].data
    except KeyError as exc:
        raise ValueError(f"PLY has no vertex element: {path}") from exc
    if vertex.dtype.names is None:
        raise ValueError(f"PLY vertex element is not a structured table: {path}")
    return ply, vertex


def _validate_gaussian_schema(vertex: np.ndarray, path: Path) -> None:
    """Ensure a PLY has the minimum fields needed by 3DGS/3DGRUT."""

    names = set(vertex.dtype.names or ())
    missing = [name for name in REQUIRED_GAUSSIAN_PROPERTIES if name not in names]
    if missing:
        raise ValueError(f"Gaussian PLY is missing {missing}: {path}")


def _xyz_keys(vertex: np.ndarray) -> np.ndarray:
    """Return exact float32 XYZ keys suitable for set membership checks."""

    keys = np.empty(len(vertex), dtype=XYZ_KEY_DTYPE)
    for name in XYZ_PROPERTIES:
        keys[name] = np.asarray(vertex[name], dtype=np.float32)
    return keys


def _normalize_schema(vertex: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Copy named fields into the target schema, dropping extras such as mask."""

    target_names = target_dtype.names or ()
    source_names = set(vertex.dtype.names or ())
    missing = [name for name in target_names if name not in source_names]
    if missing:
        raise ValueError(f"Source PLY cannot satisfy target fields: {missing}")
    normalized = np.empty(len(vertex), dtype=target_dtype)
    for name in target_names:
        normalized[name] = vertex[name]
    return normalized


def _finite_xyz(vertex: np.ndarray) -> np.ndarray:
    """Return a mask selecting records with finite XYZ positions."""

    finite = np.ones(len(vertex), dtype=bool)
    for name in XYZ_PROPERTIES:
        finite &= np.isfinite(vertex[name])
    return finite


def _median_filter(grid: np.ndarray, iterations: int) -> np.ndarray:
    """Apply a small NaN-aware median filter without an extra dependency."""

    filtered = np.asarray(grid, dtype=np.float64).copy()
    for _ in range(iterations):
        padded = np.pad(filtered, 1, mode="constant", constant_values=np.nan)
        updated = np.empty_like(filtered)
        for row in range(filtered.shape[0]):
            for column in range(filtered.shape[1]):
                window = padded[row : row + 3, column : column + 3]
                finite = window[np.isfinite(window)]
                updated[row, column] = np.median(finite) if len(finite) else np.nan
        filtered = updated
    return filtered


def _propagate_heightfield(
    raw_height: np.ndarray,
    initial_height: float,
    maximum_step_m: float,
    seed_tolerance_steps: float,
) -> tuple[np.ndarray, int]:
    """Grow a continuous terrain surface from height-consistent seed cells."""

    tolerance = maximum_step_m * seed_tolerance_steps
    seed_mask = np.isfinite(raw_height) & (np.abs(raw_height - initial_height) <= tolerance)
    if not np.any(seed_mask):
        finite_locations = np.argwhere(np.isfinite(raw_height))
        if not len(finite_locations):
            raise ValueError("No populated cells are available for height-field estimation")
        values = raw_height[np.isfinite(raw_height)]
        closest = finite_locations[int(np.argmin(np.abs(values - initial_height)))]
        seed_mask[tuple(closest)] = True

    surface = np.full_like(raw_height, np.nan, dtype=np.float64)
    surface[seed_mask] = raw_height[seed_mask]
    queue: deque[tuple[int, int]] = deque(
        (int(row), int(column)) for row, column in np.argwhere(seed_mask)
    )
    rows, columns = surface.shape
    offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while queue:
        row, column = queue.popleft()
        for row_offset, column_offset in offsets:
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if not (0 <= neighbor_row < rows and 0 <= neighbor_column < columns):
                continue
            if np.isfinite(surface[neighbor_row, neighbor_column]):
                continue
            neighborhood = surface[
                max(0, neighbor_row - 1) : min(rows, neighbor_row + 2),
                max(0, neighbor_column - 1) : min(columns, neighbor_column + 2),
            ]
            assigned = neighborhood[np.isfinite(neighborhood)]
            if not len(assigned):
                continue
            predicted = float(np.median(assigned))
            candidate = raw_height[neighbor_row, neighbor_column]
            if np.isfinite(candidate) and abs(candidate - predicted) <= maximum_step_m:
                surface[neighbor_row, neighbor_column] = candidate
            else:
                # An occluded or discontinuous cell inherits the continuous
                # terrain prediction instead of adopting canopy or outlier Z.
                surface[neighbor_row, neighbor_column] = predicted
            queue.append((neighbor_row, neighbor_column))
    return surface, int(seed_mask.sum())


def _fill_grid_from_neighbors(grid: np.ndarray) -> np.ndarray:
    """Fill missing grid values by nearest propagated neighborhood medians."""

    filled = np.asarray(grid, dtype=np.float64).copy()
    finite_locations = np.argwhere(np.isfinite(filled))
    if not len(finite_locations):
        raise ValueError("Cannot fill a grid without at least one finite value")
    queue: deque[tuple[int, int]] = deque(
        (int(row), int(column)) for row, column in finite_locations
    )
    rows, columns = filled.shape
    while queue:
        row, column = queue.popleft()
        for row_offset, column_offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor_row = row + row_offset
            neighbor_column = column + column_offset
            if not (0 <= neighbor_row < rows and 0 <= neighbor_column < columns):
                continue
            if np.isfinite(filled[neighbor_row, neighbor_column]):
                continue
            neighborhood = filled[
                max(0, neighbor_row - 1) : min(rows, neighbor_row + 2),
                max(0, neighbor_column - 1) : min(columns, neighbor_column + 2),
            ]
            assigned = neighborhood[np.isfinite(neighborhood)]
            if len(assigned):
                filled[neighbor_row, neighbor_column] = float(np.median(assigned))
                queue.append((neighbor_row, neighbor_column))
    return filled


def _vertical_gaussian_sigma(vertex: np.ndarray) -> np.ndarray:
    """Calculate each oriented Gaussian's standard deviation along world Z."""

    log_scales = np.column_stack([vertex[f"scale_{index}"] for index in range(3)]).astype(
        np.float64, copy=False
    )
    # Clipping affects classification only and prevents corrupted parameters
    # from overflowing. Output properties are never modified.
    scales = np.exp(np.clip(log_scales, -30.0, 10.0))
    quaternion = np.column_stack([vertex[f"rot_{index}"] for index in range(4)]).astype(
        np.float64, copy=False
    )
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = quaternion / np.maximum(norm, 1e-12)
    w, x, y, z = quaternion.T
    rotation_z_x = 2.0 * (x * z - w * y)
    rotation_z_y = 2.0 * (y * z + w * x)
    rotation_z_z = 1.0 - 2.0 * (x * x + y * y)
    sigma = np.sqrt(
        (rotation_z_x * scales[:, 0]) ** 2
        + (rotation_z_y * scales[:, 1]) ** 2
        + (rotation_z_z * scales[:, 2]) ** 2
    )
    return np.where(np.isfinite(sigma), sigma, 0.0)


def _adaptive_heightfield_ground_mask(
    vertex: np.ndarray,
    bounds: GroundBounds,
    config: AdaptiveHeightfieldConfig,
    excluded_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select ground splats using a continuous local terrain height field."""

    bounds.validate(require_z=False)
    config.validate()
    finite = _finite_xyz(vertex)
    footprint_before_exclusion = (
        finite
        & (vertex["x"] >= bounds.x_min)
        & (vertex["x"] <= bounds.x_max)
        & (vertex["y"] >= bounds.y_min)
        & (vertex["y"] <= bounds.y_max)
    )
    if excluded_mask is None:
        excluded_mask = np.zeros(len(vertex), dtype=bool)
    else:
        excluded_mask = np.asarray(excluded_mask, dtype=bool)
        if excluded_mask.shape != (len(vertex),):
            raise ValueError("excluded_mask must contain one value per Gaussian")
    excluded_in_footprint = footprint_before_exclusion & excluded_mask
    footprint = footprint_before_exclusion & ~excluded_mask
    source_indices = np.flatnonzero(footprint)
    if not len(source_indices):
        raise ValueError("No finite Gaussians lie inside the requested XY footprint")

    x = np.asarray(vertex["x"][source_indices], dtype=np.float64)
    y = np.asarray(vertex["y"][source_indices], dtype=np.float64)
    z = np.asarray(vertex["z"][source_indices], dtype=np.float64)
    columns = max(1, int(np.ceil((bounds.x_max - bounds.x_min) / config.grid_size_m)))
    rows = max(1, int(np.ceil((bounds.y_max - bounds.y_min) / config.grid_size_m)))
    if rows * columns > 2_000_000:
        raise ValueError(
            f"Height field would contain {rows * columns} cells; increase grid_size_m"
        )
    x_index = np.clip(
        ((x - bounds.x_min) / config.grid_size_m).astype(np.int64), 0, columns - 1
    )
    y_index = np.clip(
        ((y - bounds.y_min) / config.grid_size_m).astype(np.int64), 0, rows - 1
    )
    cell_id = y_index * columns + x_index
    order = np.argsort(cell_id, kind="stable")
    ordered_cells = cell_id[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(ordered_cells)) + 1]
    group_ends = np.r_[group_starts[1:], len(order)]

    raw_height_flat = np.full(rows * columns, np.nan, dtype=np.float64)
    groups: dict[int, np.ndarray] = {}
    for start, end in zip(group_starts, group_ends):
        group_indices = order[start:end]
        current_cell = int(ordered_cells[start])
        groups[current_cell] = group_indices
        if len(group_indices) >= config.min_points_per_cell:
            raw_height_flat[current_cell] = float(
                np.quantile(z[group_indices], config.surface_quantile)
            )
    raw_height = raw_height_flat.reshape(rows, columns)

    maximum_step = (
        np.tan(np.deg2rad(config.max_slope_degrees))
        * config.grid_size_m
        * np.sqrt(2.0)
    )
    surface, seed_cells = _propagate_heightfield(
        raw_height,
        config.initial_ground_height_m,
        maximum_step,
        config.seed_tolerance_steps,
    )
    surface = _median_filter(surface, config.smoothing_iterations)

    # Estimate the locally observed reconstruction band using only points on
    # or below the predicted surface. Plant/canopy points above the surface do
    # not inflate this value.
    band_flat = np.full(rows * columns, np.nan, dtype=np.float64)
    minimum_below_points = max(3, config.min_points_per_cell // 2)
    for current_cell, group_indices in groups.items():
        below_distance = surface.flat[current_cell] - z[group_indices]
        below_distance = below_distance[below_distance >= 0.0]
        if len(below_distance) >= minimum_below_points:
            median_distance = float(np.median(below_distance))
            mad = float(np.median(np.abs(below_distance - median_distance)))
            robust_limit = median_distance + (
                config.ground_band_mad_multiplier * 1.4826 * mad
            )
            band_flat[current_cell] = min(
                float(np.quantile(below_distance, config.ground_band_quantile)),
                robust_limit,
            )
    band = _fill_grid_from_neighbors(band_flat.reshape(rows, columns))
    band = _median_filter(band, config.smoothing_iterations)

    local_surface = surface[y_index, x_index]
    local_band = band[y_index, x_index]
    vertical_sigma = _vertical_gaussian_sigma(vertex[source_indices])
    sigma_allowance = np.minimum(
        config.gaussian_sigma_multiplier * vertical_sigma,
        config.max_sigma_to_band_ratio * local_band,
    )
    allowed_residual = np.maximum(local_band, sigma_allowance)
    selected_local = np.abs(z - local_surface) <= allowed_residual
    selected = np.zeros(len(vertex), dtype=bool)
    selected[source_indices[selected_local]] = True

    diagnostics = {
        "method": "adaptive_heightfield",
        "uses_global_z_slab": False,
        "xy_bounds_m": {
            "x": [bounds.x_min, bounds.x_max],
            "y": [bounds.y_min, bounds.y_max],
        },
        "config": asdict(config),
        "grid": {
            "rows": rows,
            "columns": columns,
            "cells": rows * columns,
            "populated_cells": len(groups),
            "raw_height_cells": int(np.isfinite(raw_height).sum()),
            "seed_cells": seed_cells,
        },
        "surface_height_m": {
            "min": float(np.min(surface)),
            "median": float(np.median(surface)),
            "max": float(np.max(surface)),
        },
        "adaptive_band_m": {
            "min": float(np.min(band)),
            "median": float(np.median(band)),
            "max": float(np.max(band)),
        },
        "footprint_gaussians": int(len(source_indices)),
        "excluded_plant_gaussians_in_footprint": int(excluded_in_footprint.sum()),
        "selected_gaussians": int(selected.sum()),
    }
    return selected, diagnostics


def _summarize_xyz(vertex: np.ndarray) -> dict[str, Any]:
    """Calculate compact spatial diagnostics for an output table."""

    if len(vertex) == 0:
        return {"count": 0, "min": None, "median": None, "max": None}
    xyz = np.column_stack([vertex[name] for name in XYZ_PROPERTIES]).astype(
        np.float64, copy=False
    )
    return {
        "count": int(len(vertex)),
        "min": np.min(xyz, axis=0).tolist(),
        "median": np.median(xyz, axis=0).tolist(),
        "max": np.max(xyz, axis=0).tolist(),
    }


def _sha256(path: Path) -> str:
    """Hash an input or output artifact without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_binary_ply(path: Path, vertex: np.ndarray, overwrite: bool) -> None:
    """Atomically write a binary little-endian vertex-only PLY."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        element = PlyElement.describe(vertex, "vertex")
        PlyData([element], text=False, byte_order="<").write(str(temporary_path))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    """Atomically write the reproducibility report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def assemble_ground_asset(
    full_scene_path: Path,
    segmented_plant_path: Path,
    outputs: AssemblyOutputs,
    bounds: GroundBounds = GroundBounds(),
    ground_method: str = "adaptive_heightfield",
    heightfield: AdaptiveHeightfieldConfig = AdaptiveHeightfieldConfig(),
    plant_selection: str = "positive_mask",
    minimum_membership_ratio: float = 0.999,
    exclude_plant_from_ground: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract visual ground Gaussians and combine them with plant Gaussians.

    Adaptive mode estimates ground after removing selected plant records from
    its candidates.  This plant-first ownership rule also applies to legacy
    slab mode.  Set ``exclude_plant_from_ground=False`` only to reproduce older
    overlapping outputs.
    """

    if ground_method not in {"adaptive_heightfield", "slab"}:
        raise ValueError("ground_method must be adaptive_heightfield or slab")
    bounds.validate(require_z=ground_method == "slab")
    if plant_selection not in {"positive_mask", "all_segmented"}:
        raise ValueError("plant_selection must be positive_mask or all_segmented")
    if not (0.0 <= minimum_membership_ratio <= 1.0):
        raise ValueError("minimum_membership_ratio must be between 0 and 1")

    full_scene_path = full_scene_path.resolve()
    segmented_plant_path = segmented_plant_path.resolve()
    output_paths = [outputs.plant, outputs.ground, outputs.combined, outputs.report]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("Every output path must be distinct")
    if not overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite outputs: {existing}")

    full_ply, full_vertex = _read_vertex_data(full_scene_path)
    plant_ply, segmented_vertex = _read_vertex_data(segmented_plant_path)
    _validate_gaussian_schema(full_vertex, full_scene_path)
    _validate_gaussian_schema(segmented_vertex, segmented_plant_path)

    full_keys = _xyz_keys(full_vertex)
    segmented_keys = _xyz_keys(segmented_vertex)
    segmented_in_full = np.isin(segmented_keys, full_keys)
    membership_ratio = float(np.mean(segmented_in_full)) if len(segmented_keys) else 0.0
    if membership_ratio < minimum_membership_ratio:
        raise ValueError(
            "Segmented plant is not a sufficiently complete exact subset of the full scene: "
            f"{membership_ratio:.6f} < {minimum_membership_ratio:.6f}"
        )

    if plant_selection == "positive_mask":
        if "mask" not in (segmented_vertex.dtype.names or ()):
            raise ValueError("positive_mask selection requires a mask property")
        plant_mask = _finite_xyz(segmented_vertex) & (segmented_vertex["mask"] > 0)
    else:
        plant_mask = _finite_xyz(segmented_vertex)

    plant_keys = segmented_keys[plant_mask]
    full_is_plant = np.isin(full_keys, plant_keys)
    ownership_exclusion = full_is_plant if exclude_plant_from_ground else None

    if ground_method == "adaptive_heightfield":
        ground_mask, ground_diagnostics = _adaptive_heightfield_ground_mask(
            full_vertex, bounds, heightfield, excluded_mask=ownership_exclusion
        )
    else:
        finite_full = _finite_xyz(full_vertex)
        ground_mask = (
            finite_full
            & (full_vertex["x"] >= bounds.x_min)
            & (full_vertex["x"] <= bounds.x_max)
            & (full_vertex["y"] >= bounds.y_min)
            & (full_vertex["y"] <= bounds.y_max)
            & (full_vertex["z"] >= bounds.z_min)
            & (full_vertex["z"] <= bounds.z_max)
        )
        if exclude_plant_from_ground:
            ground_mask &= ~full_is_plant
        ground_diagnostics = {
            "method": "slab",
            "uses_global_z_slab": True,
            "bounds_m": asdict(bounds),
            "excluded_plant_gaussians_in_footprint": int(
                (full_is_plant & finite_full).sum()
            )
            if exclude_plant_from_ground
            else 0,
            "selected_gaussians": int(ground_mask.sum()),
        }

    plant_vertex = _normalize_schema(segmented_vertex[plant_mask], full_vertex.dtype)
    ground_vertex = np.array(full_vertex[ground_mask], dtype=full_vertex.dtype, copy=True)
    if len(plant_vertex) == 0:
        raise ValueError("Plant selection is empty")
    if len(ground_vertex) == 0:
        raise ValueError("Ground selection is empty")

    ground_overlaps_plant = np.isin(_xyz_keys(ground_vertex), _xyz_keys(plant_vertex))
    cross_overlap = int(ground_overlaps_plant.sum())
    if exclude_plant_from_ground and cross_overlap:
        raise RuntimeError(
            "Plant-first ownership failed: ground output still contains plant Gaussians"
        )
    unique_ground_vertex = ground_vertex[~ground_overlaps_plant]

    combined_vertex = np.empty(
        len(plant_vertex) + len(unique_ground_vertex), dtype=full_vertex.dtype
    )
    combined_vertex[: len(plant_vertex)] = plant_vertex
    combined_vertex[len(plant_vertex) :] = unique_ground_vertex

    _write_binary_ply(outputs.plant, plant_vertex, overwrite)
    _write_binary_ply(outputs.ground, ground_vertex, overwrite)
    _write_binary_ply(outputs.combined, combined_vertex, overwrite)

    standard_properties = list(full_vertex.dtype.names or ())
    report: dict[str, Any] = {
        "operation": "extract_ground_and_combine_gaussians",
        "coordinate_system": {"unit": "meter", "up_axis": "Z", "origin": "plant_base"},
        "inputs": {
            "full_scene": {
                "path": str(full_scene_path),
                "bytes": full_scene_path.stat().st_size,
                "sha256": _sha256(full_scene_path),
                "vertices": int(len(full_vertex)),
            },
            "segmented_plant": {
                "path": str(segmented_plant_path),
                "bytes": segmented_plant_path.stat().st_size,
                "sha256": _sha256(segmented_plant_path),
                "vertices": int(len(segmented_vertex)),
                "exact_membership_count": int(segmented_in_full.sum()),
                "exact_membership_ratio": membership_ratio,
            },
        },
        "selection": {
            "ground": ground_diagnostics,
            "plant_selection": plant_selection,
            "plant_mask_threshold": 0.0 if plant_selection == "positive_mask" else None,
            "point_ownership": {
                "policy": "plant_priority_exact_xyz"
                if exclude_plant_from_ground
                else "allow_overlap_legacy",
                "selected_plant_records": int(plant_mask.sum()),
                "matched_full_scene_records": int(full_is_plant.sum()),
            },
            "merge_strategy": "disjoint_exact_xyz_union"
            if exclude_plant_from_ground
            else "plant_first_exact_xyz_union",
        },
        "schema": {
            "standard_properties": standard_properties,
            "dropped_plant_properties": sorted(
                set(segmented_vertex.dtype.names or ()) - set(standard_properties)
            ),
        },
        "outputs": {
            "plant": _summarize_xyz(plant_vertex),
            "ground": _summarize_xyz(ground_vertex),
            "combined": _summarize_xyz(combined_vertex),
            "cross_component_xyz_overlap": cross_overlap,
            "ground_records_removed_from_combined_as_duplicates": cross_overlap,
        },
    }
    for label, path in (
        ("plant", outputs.plant),
        ("ground", outputs.ground),
        ("combined", outputs.combined),
    ):
        report["outputs"][label].update(
            {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    _write_json(outputs.report, report, overwrite)

    # Keep mmap owners alive until every selection and write has completed.
    del full_ply, plant_ply
    return report


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser used by the thin scripts entry point."""

    parser = argparse.ArgumentParser(
        description="Extract a metric ground Gaussian crop and combine it with a segmented plant."
    )
    parser.add_argument("--full-scene", required=True, type=Path)
    parser.add_argument("--segmented-plant", required=True, type=Path)
    parser.add_argument("--plant-output", required=True, type=Path)
    parser.add_argument("--ground-output", required=True, type=Path)
    parser.add_argument("--combined-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--x-min", type=float, default=-0.5)
    parser.add_argument("--x-max", type=float, default=0.5)
    parser.add_argument("--y-min", type=float, default=-0.5)
    parser.add_argument("--y-max", type=float, default=0.5)
    parser.add_argument(
        "--ground-method",
        choices=("adaptive_heightfield", "slab"),
        default="adaptive_heightfield",
    )
    parser.add_argument("--z-min", type=float, default=-0.05, help="Legacy slab mode only")
    parser.add_argument("--z-max", type=float, default=0.05, help="Legacy slab mode only")
    parser.add_argument("--grid-size", type=float, default=0.02)
    parser.add_argument("--initial-ground-height", type=float, default=0.0)
    parser.add_argument("--surface-quantile", type=float, default=0.15)
    parser.add_argument("--max-slope-degrees", type=float, default=40.0)
    parser.add_argument("--seed-tolerance-steps", type=float, default=2.0)
    parser.add_argument("--smoothing-iterations", type=int, default=3)
    parser.add_argument("--ground-band-quantile", type=float, default=0.90)
    parser.add_argument("--ground-band-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--gaussian-sigma-multiplier", type=float, default=5.0)
    parser.add_argument("--max-sigma-to-band-ratio", type=float, default=2.0)
    parser.add_argument("--min-points-per-cell", type=int, default=5)
    parser.add_argument(
        "--plant-selection",
        choices=("positive_mask", "all_segmented"),
        default="positive_mask",
    )
    parser.add_argument("--minimum-membership-ratio", type=float, default=0.999)
    parser.add_argument(
        "--allow-plant-ground-overlap",
        action="store_true",
        help="Reproduce legacy behavior; selected plant Gaussians may remain in ground output",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ground extraction and combination from command-line arguments."""

    args = _build_parser().parse_args(argv)
    report = assemble_ground_asset(
        full_scene_path=args.full_scene,
        segmented_plant_path=args.segmented_plant,
        outputs=AssemblyOutputs(
            plant=args.plant_output,
            ground=args.ground_output,
            combined=args.combined_output,
            report=args.report_output,
        ),
        bounds=GroundBounds(
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            z_min=args.z_min,
            z_max=args.z_max,
        ),
        ground_method=args.ground_method,
        heightfield=AdaptiveHeightfieldConfig(
            grid_size_m=args.grid_size,
            initial_ground_height_m=args.initial_ground_height,
            surface_quantile=args.surface_quantile,
            max_slope_degrees=args.max_slope_degrees,
            seed_tolerance_steps=args.seed_tolerance_steps,
            smoothing_iterations=args.smoothing_iterations,
            ground_band_quantile=args.ground_band_quantile,
            ground_band_mad_multiplier=args.ground_band_mad_multiplier,
            gaussian_sigma_multiplier=args.gaussian_sigma_multiplier,
            max_sigma_to_band_ratio=args.max_sigma_to_band_ratio,
            min_points_per_cell=args.min_points_per_cell,
        ),
        plant_selection=args.plant_selection,
        minimum_membership_ratio=args.minimum_membership_ratio,
        exclude_plant_from_ground=not args.allow_plant_ground_overlap,
        overwrite=args.overwrite,
    )
    print(json.dumps(report["outputs"], indent=2, ensure_ascii=False))
    return 0
