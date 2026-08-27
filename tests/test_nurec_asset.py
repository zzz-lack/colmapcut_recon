"""Tests for Isaac Sim 5.x NuRec package detection and validation."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402

from colmapcut_recon.export.nurec_asset import (  # noqa: E402
    isaac_sim_major_version,
    validate_nurec_collision_usdz,
)


def _make_mock_nurec_package(path: Path) -> None:
    source_dir = path.parent / "mock_source"
    source_dir.mkdir()

    mesh_stage = Usd.Stage.CreateNew(str(source_dir / "mesh.usd"))
    mesh = UsdGeom.Mesh.Define(mesh_stage, "/mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [(-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (0.0, 1.0, 0.0)]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
    mesh_stage.SetDefaultPrim(mesh.GetPrim())
    mesh_stage.GetRootLayer().Save()

    stage = Usd.Stage.CreateNew(str(source_dir / "default.usda"))
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    volume = stage.DefinePrim("/World/gauss", "Volume")
    volume.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    volume.CreateRelationship("proxy", custom=True).SetTargets([Sdf.Path("/World/mesh")])
    field = stage.DefinePrim("/World/gauss/density", "OmniNuRecFieldAsset")
    field.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("payload.nurec")
    )
    stage.OverridePrim("/World/mesh").GetReferences().AddReference("mesh.usd")
    stage.GetRootLayer().Save()

    (source_dir / "payload.nurec").write_bytes(b"mock-nurec")
    (source_dir / "mesh.ply").write_bytes(b"ply\n")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("default.usda", "payload.nurec", "mesh.usd", "mesh.ply"):
            archive.write(source_dir / name, arcname=name)


def test_parses_isaac_sim_major_version() -> None:
    assert isaac_sim_major_version("5.1.0-rc.19") == 5
    with pytest.raises(ValueError):
        isaac_sim_major_version("development")


def test_validates_nurec_volume_and_invisible_collision(tmp_path: Path) -> None:
    package = tmp_path / "asset.usdz"
    _make_mock_nurec_package(package)

    report = validate_nurec_collision_usdz(package)

    assert report["passed"] is True
    assert report["nurec_payloads"] == ["payload.nurec"]
    assert report["volume_prims"] == ["/World/gauss"]
    assert report["collision_mesh_prims"] == ["/World/mesh"]
    assert report["invisible_collision_mesh_prims"] == ["/World/mesh"]
