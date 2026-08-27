"""Tests for portable USD packaging of 3DGS data and static collision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement


pxr = pytest.importorskip("pxr")

from colmapcut_recon.export.export_usd import (  # noqa: E402
    UsdAssetConfig,
    package_simulation_usd,
)


def _write_gaussians(path: Path) -> None:
    dtype = [(name, "f4") for name in (
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
        "f_rest_0", "f_rest_1", "f_rest_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )]
    vertex = np.zeros(2, dtype=dtype)
    vertex["x"] = [-0.25, 0.25]
    vertex["z"] = [0.1, 0.2]
    vertex["opacity"] = [0.0, 2.0]
    vertex["scale_0"] = vertex["scale_1"] = vertex["scale_2"] = np.log(0.01)
    vertex["rot_0"] = 1.0
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def _write_collision(path: Path) -> None:
    vertex = np.array(
        [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    face = np.empty(1, dtype=[("vertex_indices", "O")])
    face["vertex_indices"][0] = np.array([0, 1, 2], dtype=np.int32)
    PlyData(
        [
            PlyElement.describe(vertex, "vertex"),
            PlyElement.describe(face, "face", val_types={"vertex_indices": "i4"}),
        ],
        text=False,
    ).write(path)


def test_packages_gaussians_and_collision_as_valid_usd(tmp_path: Path) -> None:
    gaussian_path = tmp_path / "combined.ply"
    collision_path = tmp_path / "collision.ply"
    _write_gaussians(gaussian_path)
    _write_collision(collision_path)

    report = package_simulation_usd(
        gaussian_ply=gaussian_path,
        collision_ply=collision_path,
        output_dir=tmp_path / "asset",
        config=UsdAssetConfig(asset_name="test_asset"),
    )

    assert report["gaussians"]["count"] == 2
    assert report["collision"]["triangles"] == 1
    assert report["validation"]["passed"] is True
    assert (tmp_path / "asset" / "test_asset.usdc").is_file()
    assert (tmp_path / "asset" / "data" / "combined.ply").is_file()
    assert json.loads((tmp_path / "asset" / "manifest.json").read_text())["stage"][
        "default_prim"
    ] == "/PlantGroundAsset"
