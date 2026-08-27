"""Package 3D Gaussian splats and a collision mesh as a portable USD asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SH_C0 = 0.28209479177387814
REQUIRED_GAUSSIAN_PROPERTIES = {
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
}


@dataclass(frozen=True)
class UsdAssetConfig:
    """Settings that control the USD representation and portable package."""

    asset_name: str = "plant_ground_asset"
    root_prim_name: str = "PlantGroundAsset"
    meters_per_unit: float = 1.0
    preview_min_width_m: float = 0.001
    preview_max_width_m: float = 0.10
    copy_source_files: bool = True

    def validate(self) -> None:
        if not self.asset_name:
            raise ValueError("asset_name cannot be empty")
        if not self.root_prim_name or "/" in self.root_prim_name:
            raise ValueError("root_prim_name must be one valid USD prim component")
        if self.meters_per_unit <= 0:
            raise ValueError("meters_per_unit must be positive")
        if self.preview_min_width_m <= 0:
            raise ValueError("preview_min_width_m must be positive")
        if self.preview_max_width_m < self.preview_min_width_m:
            raise ValueError("preview_max_width_m cannot be smaller than the minimum")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def gaussian_preview_arrays(vertex: np.ndarray, config: UsdAssetConfig) -> dict[str, np.ndarray]:
    """Convert raw 3DGS parameters to standard USD point-preview attributes."""

    names = set(vertex.dtype.names or ())
    missing = sorted(REQUIRED_GAUSSIAN_PROPERTIES - names)
    if missing:
        raise ValueError("Gaussian PLY is missing properties: " + ", ".join(missing))

    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32, copy=False
    )
    sh_dc = np.column_stack(
        (vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"])
    ).astype(np.float32, copy=False)
    colors = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0).astype(np.float32, copy=False)
    opacity = _sigmoid(vertex["opacity"]).astype(np.float32, copy=False)
    log_scales = np.column_stack(
        (vertex["scale_0"], vertex["scale_1"], vertex["scale_2"])
    ).astype(np.float32, copy=False)
    scales = np.exp(np.clip(log_scales, -30.0, 10.0))
    widths = np.clip(
        2.0 * np.max(scales, axis=1),
        config.preview_min_width_m,
        config.preview_max_width_m,
    ).astype(np.float32, copy=False)
    rotations = np.column_stack(
        (vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"])
    ).astype(np.float32, copy=False)
    rotation_norm = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations = rotations / np.maximum(rotation_norm, 1e-12)
    invalid_rotation = rotation_norm[:, 0] <= 1e-12
    rotations[invalid_rotation] = (1.0, 0.0, 0.0, 0.0)

    finite = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(colors), axis=1)
        & np.isfinite(opacity)
        & np.all(np.isfinite(log_scales), axis=1)
        & np.all(np.isfinite(rotations), axis=1)
    )
    if not np.all(finite):
        raise ValueError(f"Gaussian PLY contains {int((~finite).sum())} non-finite records")
    return {
        "points": np.ascontiguousarray(points),
        "colors": np.ascontiguousarray(colors),
        "opacity": np.ascontiguousarray(opacity),
        "widths": np.ascontiguousarray(widths),
        "sh_dc": np.ascontiguousarray(sh_dc),
        "log_scales": np.ascontiguousarray(log_scales),
        "rotations": np.ascontiguousarray(rotations),
        "opacity_logits": np.ascontiguousarray(vertex["opacity"], dtype=np.float32),
    }


def _read_gaussians(path: Path) -> tuple[Any, list[str]]:
    from plyfile import PlyData

    ply = PlyData.read(path, mmap=True)
    vertex = ply["vertex"].data
    names = list(vertex.dtype.names or ())
    missing = sorted(REQUIRED_GAUSSIAN_PROPERTIES - set(names))
    if missing:
        raise ValueError(f"{path} is not a supported 3DGS PLY; missing: {', '.join(missing)}")
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.removeprefix("f_rest_")),
    )
    return vertex, rest_names


def _read_collision_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from plyfile import PlyData

    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    vertices = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32, copy=False
    )
    face = ply["face"].data
    triangles = np.asarray([indices for indices in face["vertex_indices"]], dtype=np.int32)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Collision PLY must contain only triangular faces")
    if len(vertices) == 0 or len(triangles) == 0:
        raise ValueError("Collision PLY contains no mesh geometry")
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices):
        raise ValueError("Collision PLY contains an invalid vertex index")
    return np.ascontiguousarray(vertices), np.ascontiguousarray(triangles)


def _source_destination(source: Path, package_data_dir: Path, copy: bool) -> Path:
    destination = package_data_dir / source.name
    if source.resolve() == destination.resolve():
        return destination
    if copy:
        shutil.copy2(source, destination)
    return destination


def _set_asset_metadata(
    prim: Any,
    *,
    asset_name: str,
    source_asset_path: str,
    source_sha256: str,
) -> None:
    from pxr import Sdf

    prim.CreateAttribute("source:asset", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(source_asset_path)
    )
    prim.CreateAttribute("source:sha256", Sdf.ValueTypeNames.String).Set(source_sha256)
    prim.CreateAttribute("source:assetName", Sdf.ValueTypeNames.String).Set(asset_name)


def _author_gaussians(
    stage: Any,
    path: str,
    vertex: np.ndarray,
    rest_names: list[str],
    config: UsdAssetConfig,
    source_asset_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    from pxr import Sdf, UsdGeom, Vt

    preview = gaussian_preview_arrays(vertex, config)
    points = UsdGeom.Points.Define(stage, path)
    prim = points.GetPrim()
    points.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(preview["points"]))
    points.CreateWidthsAttr(Vt.FloatArray.FromNumpy(preview["widths"]))
    points.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(preview["colors"])
    )
    points.CreateDisplayOpacityPrimvar(UsdGeom.Tokens.vertex).Set(
        Vt.FloatArray.FromNumpy(preview["opacity"])
    )
    prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set(UsdGeom.Tokens.render)
    _set_asset_metadata(
        prim,
        asset_name=config.asset_name,
        source_asset_path=source_asset_path,
        source_sha256=source_sha256,
    )

    prim.CreateAttribute("gaussian:schema", Sdf.ValueTypeNames.String).Set("3DGS-Ply-v1")
    prim.CreateAttribute("gaussian:scaleActivation", Sdf.ValueTypeNames.Token).Set("exp")
    prim.CreateAttribute("gaussian:opacityActivation", Sdf.ValueTypeNames.Token).Set("sigmoid")
    prim.CreateAttribute("gaussian:rotationOrder", Sdf.ValueTypeNames.Token).Set("wxyz")
    sh_degree = int(round(math.sqrt(len(rest_names) / 3 + 1) - 1)) if rest_names else 0
    prim.CreateAttribute("gaussian:shDegree", Sdf.ValueTypeNames.Int).Set(sh_degree)

    primvars = UsdGeom.PrimvarsAPI(prim)
    primvars.CreatePrimvar(
        "gaussian:logScales", Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec3fArray.FromNumpy(preview["log_scales"]))
    primvars.CreatePrimvar(
        "gaussian:rotations", Sdf.ValueTypeNames.Float4Array, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec4fArray.FromNumpy(preview["rotations"]))
    primvars.CreatePrimvar(
        "gaussian:opacityLogits", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
    ).Set(Vt.FloatArray.FromNumpy(preview["opacity_logits"]))
    primvars.CreatePrimvar(
        "gaussian:shDC", Sdf.ValueTypeNames.Float3Array, UsdGeom.Tokens.vertex
    ).Set(Vt.Vec3fArray.FromNumpy(preview["sh_dc"]))

    if rest_names:
        sh_rest = np.column_stack([vertex[name] for name in rest_names]).astype(
            np.float32, copy=False
        )
        if not np.all(np.isfinite(sh_rest)):
            raise ValueError("Gaussian PLY contains non-finite higher-order SH coefficients")
        sh_primvar = primvars.CreatePrimvar(
            "gaussian:shRest", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
        )
        sh_primvar.SetElementSize(len(rest_names))
        sh_primvar.Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(sh_rest.ravel())))

    return {
        "count": int(len(vertex)),
        "sh_degree": sh_degree,
        "sh_rest_coefficients_per_gaussian": int(len(rest_names)),
        "bounds_min_m": preview["points"].min(axis=0).astype(float).tolist(),
        "bounds_max_m": preview["points"].max(axis=0).astype(float).tolist(),
        "preview_width_range_m": [
            float(preview["widths"].min()),
            float(preview["widths"].max()),
        ],
    }


def _author_collision_mesh(
    stage: Any,
    path: str,
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    asset_name: str,
    source_asset_path: str,
    source_sha256: str,
) -> dict[str, Any]:
    from pxr import Sdf, UsdGeom, UsdPhysics, Vt

    mesh = UsdGeom.Mesh.Define(stage, path)
    prim = mesh.GetPrim()
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray.FromNumpy(np.full(len(triangles), 3, dtype=np.int32))
    )
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray.FromNumpy(np.ascontiguousarray(triangles.ravel(), dtype=np.int32))
    )
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([(0.25, 0.55, 0.20)])
    prim.CreateAttribute("purpose", Sdf.ValueTypeNames.Token).Set(UsdGeom.Tokens.guide)
    _set_asset_metadata(
        prim,
        asset_name=asset_name,
        source_asset_path=source_asset_path,
        source_sha256=source_sha256,
    )

    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr(True)
    mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision_api.CreateApproximationAttr().Set("none")
    return {
        "vertices": int(len(vertices)),
        "triangles": int(len(triangles)),
        "bounds_min_m": vertices.min(axis=0).astype(float).tolist(),
        "bounds_max_m": vertices.max(axis=0).astype(float).tolist(),
        "collision_approximation": "none",
        "static_collision": True,
    }


def validate_usd_asset(
    stage_path: Path,
    *,
    root_prim_name: str,
    expected_gaussians: int,
    expected_triangles: int,
) -> dict[str, Any]:
    """Reopen the saved USD and verify geometry and physics schemas."""

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(stage_path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"USD could not be reopened: {stage_path}")
    root_path = f"/{root_prim_name}"
    root = stage.GetPrimAtPath(root_path)
    gaussian = UsdGeom.Points.Get(stage, f"{root_path}/Render/GaussianSplat")
    collision = UsdGeom.Mesh.Get(stage, f"{root_path}/Collision/Ground")
    points = gaussian.GetPointsAttr().Get() if gaussian else None
    face_counts = collision.GetFaceVertexCountsAttr().Get() if collision else None
    collision_prim = collision.GetPrim() if collision else None
    report = {
        "stage_opened": True,
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "z_up": UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z,
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "gaussian_count": len(points) if points is not None else 0,
        "collision_triangles": len(face_counts) if face_counts is not None else 0,
        "collision_api": bool(
            collision_prim and collision_prim.HasAPI(UsdPhysics.CollisionAPI)
        ),
        "mesh_collision_api": bool(
            collision_prim and collision_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        ),
    }
    report["passed"] = bool(
        root
        and report["default_prim"] == root_path
        and report["z_up"]
        and report["gaussian_count"] == expected_gaussians
        and report["collision_triangles"] == expected_triangles
        and report["collision_api"]
        and report["mesh_collision_api"]
    )
    if not report["passed"]:
        raise RuntimeError("USD validation failed: " + json.dumps(report))
    return report


def package_simulation_usd(
    *,
    gaussian_ply: Path,
    collision_ply: Path,
    output_dir: Path,
    config: UsdAssetConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create a self-contained USD folder with 3DGS data and static collision."""

    from pxr import Kind, Usd, UsdGeom

    config.validate()
    gaussian_ply = gaussian_ply.resolve(strict=True)
    collision_ply = collision_ply.resolve(strict=True)
    output_dir = output_dir.resolve()
    data_dir = output_dir / "data"
    stage_path = output_dir / f"{config.asset_name}.usdc"
    manifest_path = output_dir / "manifest.json"
    managed_outputs = [stage_path, manifest_path]
    if config.copy_source_files:
        managed_outputs.extend((data_dir / gaussian_ply.name, data_dir / collision_ply.name))
    existing = [str(path) for path in managed_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite: " + ", ".join(existing))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    if stage_path.exists():
        stage_path.unlink()

    gaussian_sha256 = _sha256(gaussian_ply)
    collision_sha256 = _sha256(collision_ply)
    packaged_gaussian = _source_destination(
        gaussian_ply, data_dir, config.copy_source_files
    )
    packaged_collision = _source_destination(
        collision_ply, data_dir, config.copy_source_files
    )
    if config.copy_source_files:
        gaussian_asset_path = f"./data/{packaged_gaussian.name}"
        collision_asset_path = f"./data/{packaged_collision.name}"
    else:
        gaussian_asset_path = str(gaussian_ply)
        collision_asset_path = str(collision_ply)

    vertex, rest_names = _read_gaussians(gaussian_ply)
    collision_vertices, collision_triangles = _read_collision_mesh(collision_ply)
    stage = Usd.Stage.CreateNew(str(stage_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, config.meters_per_unit)
    root_path = f"/{config.root_prim_name}"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    stage.SetDefaultPrim(root)
    Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
    root.SetAssetInfoByKey("name", config.asset_name)
    root.SetCustomDataByKey("asset:contains3DGS", True)
    root.SetCustomDataByKey("asset:containsStaticCollision", True)
    UsdGeom.Scope.Define(stage, f"{root_path}/Render")
    UsdGeom.Scope.Define(stage, f"{root_path}/Collision")

    gaussian_report = _author_gaussians(
        stage,
        f"{root_path}/Render/GaussianSplat",
        vertex,
        rest_names,
        config,
        gaussian_asset_path,
        gaussian_sha256,
    )
    collision_report = _author_collision_mesh(
        stage,
        f"{root_path}/Collision/Ground",
        collision_vertices,
        collision_triangles,
        asset_name=config.asset_name,
        source_asset_path=collision_asset_path,
        source_sha256=collision_sha256,
    )
    stage.GetRootLayer().Save()
    del stage

    validation = validate_usd_asset(
        stage_path,
        root_prim_name=config.root_prim_name,
        expected_gaussians=gaussian_report["count"],
        expected_triangles=collision_report["triangles"],
    )
    manifest: dict[str, Any] = {
        "format": "USD 3DGS rendering data + static triangle-mesh collision",
        "config": asdict(config),
        "coordinate_system": {"up_axis": "Z", "meters_per_unit": config.meters_per_unit},
        "stage": {
            "path": stage_path.name,
            "default_prim": root_path,
            "bytes": stage_path.stat().st_size,
            "sha256": _sha256(stage_path),
        },
        "gaussians": {
            **gaussian_report,
            "source": str(gaussian_ply),
            "packaged_source": (
                str(packaged_gaussian.relative_to(output_dir))
                if config.copy_source_files
                else None
            ),
            "source_sha256": gaussian_sha256,
        },
        "collision": {
            **collision_report,
            "source": str(collision_ply),
            "packaged_source": (
                str(packaged_collision.relative_to(output_dir))
                if config.copy_source_files
                else None
            ),
            "source_sha256": collision_sha256,
        },
        "validation": validation,
        "compatibility": {
            "collision": "Standard UsdPhysics collision; suitable for static ground.",
            "3dgs": (
                "Full 3DGS parameters are stored as gaussian:* primvars and the source PLY "
                "is packaged. Standard USD viewers show a point preview; true splat rendering "
                "requires a compatible Gaussian-splat renderer or simulator extension."
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--collision-mesh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--asset-name", default="plant_ground_asset")
    parser.add_argument("--root-prim-name", default="PlantGroundAsset")
    parser.add_argument("--meters-per-unit", type=float, default=1.0)
    parser.add_argument("--preview-min-width", type=float, default=0.001)
    parser.add_argument("--preview-max-width", type=float, default=0.10)
    parser.add_argument("--no-copy-sources", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = package_simulation_usd(
        gaussian_ply=args.gaussians,
        collision_ply=args.collision_mesh,
        output_dir=args.output_dir,
        config=UsdAssetConfig(
            asset_name=args.asset_name,
            root_prim_name=args.root_prim_name,
            meters_per_unit=args.meters_per_unit,
            preview_min_width_m=args.preview_min_width,
            preview_max_width_m=args.preview_max_width,
            copy_source_files=not args.no_copy_sources,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
