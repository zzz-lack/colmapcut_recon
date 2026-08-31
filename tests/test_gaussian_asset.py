"""Tests for exact-subset ground extraction and Gaussian schema normalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from colmapcut_recon.postprocessing.clean_gaussians import (
    AdaptiveHeightfieldConfig,
    AssemblyOutputs,
    GroundBounds,
    assemble_ground_asset,
)
from colmapcut_recon.postprocessing.remove_background_gaussians import (
    SceneBounds,
    SeparationOutputs,
    separate_scene_gaussians,
)


STANDARD_DTYPE = np.dtype(
    [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
)


def _write_ply(path: Path, records: np.ndarray) -> None:
    PlyData([PlyElement.describe(records, "vertex")], text=False).write(path)


def test_extracts_disjoint_ground_and_strict_plant(tmp_path: Path) -> None:
    """Only positive plant-mask rows own points over the ground component."""

    full = np.zeros(5, dtype=STANDARD_DTYPE)
    full["rot_0"] = 1.0
    full_xyz = np.array(
        [
            [0.0, 0.0, 0.2],
            [0.1, 0.0, 0.01],
            [0.2, 0.2, 0.0],
            [-0.4, 0.4, 0.0],
            [0.7, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    for index, name in enumerate(("x", "y", "z")):
        full[name] = full_xyz[:, index]

    segmented_dtype = np.dtype(
        [("x", "f4"), ("y", "f4"), ("z", "f4"), ("mask", "f4")]
        + list(STANDARD_DTYPE.descr[3:])
    )
    segmented = np.zeros(2, dtype=segmented_dtype)
    for name in STANDARD_DTYPE.names or ():
        segmented[name] = full[name][:2]
    segmented["mask"] = [2.0, -1.0]

    full_path = tmp_path / "full.ply"
    segmented_path = tmp_path / "segmented.ply"
    _write_ply(full_path, full)
    _write_ply(segmented_path, segmented)
    outputs = AssemblyOutputs(
        plant=tmp_path / "plant.ply",
        ground=tmp_path / "ground.ply",
        combined=tmp_path / "combined.ply",
        report=tmp_path / "report.json",
    )

    report = assemble_ground_asset(
        full_path,
        segmented_path,
        outputs,
        GroundBounds(),
        ground_method="slab",
        plant_selection="positive_mask",
    )

    plant = PlyData.read(outputs.plant)["vertex"].data
    ground = PlyData.read(outputs.ground)["vertex"].data
    combined = PlyData.read(outputs.combined)["vertex"].data
    assert len(plant) == 1
    assert len(ground) == 3
    assert len(combined) == 4
    assert "mask" not in (plant.dtype.names or ())
    assert report["outputs"]["cross_component_xyz_overlap"] == 0
    assert report["selection"]["point_ownership"]["policy"] == "plant_priority_exact_xyz"
    assert json.loads(outputs.report.read_text())["outputs"]["combined"]["count"] == 4


def test_adaptive_heightfield_follows_slope_and_subtracts_plant(tmp_path: Path) -> None:
    """Adaptive mode follows local elevation and removes near-ground plant splats."""

    records: list[tuple[float, float, float]] = []
    for y in np.linspace(-0.4, 0.4, 5):
        for x in np.linspace(-0.4, 0.4, 5):
            ground_height = 0.2 * x + 0.1 * y
            for offset in (-0.02, -0.01, 0.0, 0.01, 0.02):
                records.append((float(x), float(y), float(ground_height + offset)))
            records.append((float(x), float(y), float(ground_height + 0.5)))

    full = np.zeros(len(records), dtype=STANDARD_DTYPE)
    full["rot_0"] = 1.0
    for scale_name in ("scale_0", "scale_1", "scale_2"):
        full[scale_name] = np.log(0.01)
    xyz = np.asarray(records, dtype=np.float32)
    for index, name in enumerate(("x", "y", "z")):
        full[name] = xyz[:, index]

    center_ground_index = 12 * 6 + 2
    center_canopy_index = 12 * 6 + 5
    segmented_dtype = np.dtype(
        [("x", "f4"), ("y", "f4"), ("z", "f4"), ("mask", "f4")]
        + list(STANDARD_DTYPE.descr[3:])
    )
    segmented = np.zeros(2, dtype=segmented_dtype)
    for name in STANDARD_DTYPE.names or ():
        segmented[name] = full[name][[center_ground_index, center_canopy_index]]
    segmented["mask"] = 1.0

    full_path = tmp_path / "slope_full.ply"
    segmented_path = tmp_path / "slope_segmented.ply"
    _write_ply(full_path, full)
    _write_ply(segmented_path, segmented)
    outputs = AssemblyOutputs(
        plant=tmp_path / "slope_plant.ply",
        ground=tmp_path / "slope_ground.ply",
        combined=tmp_path / "slope_combined.ply",
        report=tmp_path / "slope_report.json",
    )

    report = assemble_ground_asset(
        full_path,
        segmented_path,
        outputs,
        GroundBounds(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5),
        ground_method="adaptive_heightfield",
        heightfield=AdaptiveHeightfieldConfig(
            grid_size_m=0.2,
            initial_ground_height_m=0.0,
            surface_quantile=0.15,
            max_slope_degrees=30.0,
            seed_tolerance_steps=2.0,
            smoothing_iterations=1,
            ground_band_quantile=0.95,
            ground_band_mad_multiplier=3.0,
            gaussian_sigma_multiplier=3.0,
            max_sigma_to_band_ratio=2.0,
            min_points_per_cell=3,
        ),
    )

    plant = PlyData.read(outputs.plant)["vertex"].data
    ground = PlyData.read(outputs.ground)["vertex"].data
    combined = PlyData.read(outputs.combined)["vertex"].data
    assert len(plant) == 2
    assert len(ground) >= 5 * 5 * 3
    occupied_xy = set(zip(np.round(ground["x"], 3), np.round(ground["y"], 3)))
    assert len(occupied_xy) == 25
    assert float(np.max(ground["z"])) < 0.3
    assert report["selection"]["ground"]["uses_global_z_slab"] is False
    ground_keys = set(zip(ground["x"], ground["y"], ground["z"]))
    near_ground_plant_xyz = tuple(
        full[name][center_ground_index] for name in ("x", "y", "z")
    )
    assert near_ground_plant_xyz not in ground_keys
    assert report["selection"]["ground"]["excluded_plant_gaussians_in_footprint"] == 2
    assert report["outputs"]["cross_component_xyz_overlap"] == 0
    assert len(combined) == len(plant) + len(ground)


def test_geometric_separation_partitions_background_ground_and_plant(
    tmp_path: Path,
) -> None:
    records: list[tuple[float, float, float]] = []
    for y in np.linspace(-0.4, 0.4, 5):
        for x in np.linspace(-0.4, 0.4, 5):
            for offset in (-0.02, -0.01, 0.0, 0.01, 0.02):
                records.append((float(x), float(y), float(offset)))
            records.append((float(x), float(y), 0.5))
    records.extend(((2.0, 2.0, 0.0), (-2.0, -2.0, 0.0)))
    full = np.zeros(len(records), dtype=STANDARD_DTYPE)
    full["rot_0"] = 1.0
    full["opacity"] = 4.0
    for name in ("scale_0", "scale_1", "scale_2"):
        full[name] = np.log(0.005)
    xyz = np.asarray(records, dtype=np.float32)
    for index, name in enumerate(("x", "y", "z")):
        full[name] = xyz[:, index]
    source = tmp_path / "full_scene.ply"
    _write_ply(source, full)
    outputs = SeparationOutputs(
        plant=tmp_path / "plant.ply",
        ground=tmp_path / "ground.ply",
        background=tmp_path / "background.ply",
        combined=tmp_path / "combined.ply",
        report=tmp_path / "separation.json",
    )

    report = separate_scene_gaussians(
        source,
        outputs,
        scene_bounds=SceneBounds(
            x_min=-0.5,
            x_max=0.5,
            y_min=-0.5,
            y_max=0.5,
            z_min=-0.1,
            z_max=0.8,
        ),
        heightfield=AdaptiveHeightfieldConfig(
            grid_size_m=0.2,
            min_points_per_cell=3,
            smoothing_iterations=1,
            surface_quantile=0.4,
        ),
    )

    plant = PlyData.read(outputs.plant)["vertex"].data
    ground = PlyData.read(outputs.ground)["vertex"].data
    background = PlyData.read(outputs.background)["vertex"].data
    combined = PlyData.read(outputs.combined)["vertex"].data
    assert len(plant) == 25
    assert len(ground) == 125
    assert len(background) == 2
    assert len(combined) == 150
    assert report["partition"]["disjoint"] is True
