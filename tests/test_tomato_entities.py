from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from colmapcut_recon.segmentation.tomato_entities import (
    ExtractionConfig,
    extract_tomato_entities,
    load_gaussian_mask,
)


def _write_gaussians(path: Path) -> None:
    rng = np.random.default_rng(7)
    centers = (np.array([-0.06, 0.0, 0.12]), np.array([0.07, 0.0, 0.14]))
    clouds = [center + rng.normal(0.0, 0.009, size=(120, 3)) for center in centers]
    background = rng.uniform((-0.2, -0.1, 0.02), (0.2, 0.1, 0.25), size=(80, 3))
    xyz = np.vstack((*clouds, background)).astype(np.float32)
    dtype = np.dtype(
        [(name, "<f4") for name in (
            "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
            "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        )]
    )
    vertex = np.zeros(len(xyz), dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["f_dc_0"] = 1.6
    vertex["f_dc_1"] = -0.7
    vertex["f_dc_2"] = -0.7
    vertex["opacity"] = 3.0
    vertex["scale_0"] = vertex["scale_1"] = vertex["scale_2"] = -4.0
    vertex["rot_0"] = 1.0
    PlyData([PlyElement.describe(vertex, "vertex")]).write(path)


def test_mask_length_is_checked(tmp_path: Path) -> None:
    mask = tmp_path / "mask.npy"
    np.save(mask, np.ones(3, dtype=bool))
    try:
        load_gaussian_mask(mask, 4)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected a mask length error")


def test_extracts_entities_and_static_plant(tmp_path: Path) -> None:
    source = tmp_path / "scene.ply"
    output = tmp_path / "entities"
    _write_gaussians(source)
    stale_directory = output / "gaussians"
    stale_directory.mkdir(parents=True)
    (stale_directory / "tomato_999.ply").write_bytes(b"stale")
    (stale_directory / "keep.txt").write_text("user data")
    manifest = extract_tomato_entities(
        source,
        output,
        config=ExtractionConfig(
            voxel_size_m=0.012,
            minimum_seed_points=20,
            minimum_entity_points=20,
            maximum_component_span_m=0.09,
            minimum_radius_m=0.012,
            maximum_radius_m=0.05,
            ellipsoid_padding_m=0.005,
        ),
    )
    assert manifest["selection_source"] == "ripe_colour_bootstrap"
    assert manifest["statistics"]["entity_count"] == 2
    assert (output / "tomato_entities.usda").is_file()
    assert (output / "plant_without_tomatoes.ply").is_file()
    assert not (stale_directory / "tomato_999.ply").exists()
    assert (stale_directory / "keep.txt").is_file()
    assert len(list(stale_directory.glob("tomato_*.ply"))) == 2
    text = (output / "tomato_entities.usda").read_text()
    assert "PhysicsRigidBodyAPI" in text
    assert "tomato:stemAnchor" in text
