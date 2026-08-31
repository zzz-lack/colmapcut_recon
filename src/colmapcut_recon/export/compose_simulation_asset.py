"""Compose a NuRec environment and dynamic fruit meshes into one USDZ."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_composed_simulation_usdz(path: Path) -> dict[str, Any]:
    """Verify rendering, mesh, rigid-body, collision, scale, and axis schemas."""

    from pxr import Usd, UsdGeom, UsdPhysics

    path = path.resolve(strict=True)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Output is not a USDZ package: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.namelist()
    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open composed package: {path}")

    volumes: list[str] = []
    meshes: list[str] = []
    fruit_meshes: list[str] = []
    collisions: list[str] = []
    rigid_bodies: list[str] = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim.GetTypeName() == "Volume":
            volumes.append(prim_path)
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim_path)
            if "/Fruits/" in prim_path:
                fruit_meshes.append(prim_path)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collisions.append(prim_path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(prim_path)

    report = {
        "package_opened": True,
        "members": members,
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "z_up": UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z,
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "volume_prims": volumes,
        "mesh_prims": meshes,
        "fruit_mesh_prims": fruit_meshes,
        "collision_prims": collisions,
        "rigid_body_prims": rigid_bodies,
    }
    report["passed"] = bool(
        volumes
        and meshes
        and fruit_meshes
        and collisions
        and rigid_bodies
        and report["z_up"]
        and abs(report["meters_per_unit"] - 1.0) < 1e-9
    )
    if not report["passed"]:
        raise RuntimeError(
            "Composed simulation USDZ validation failed: " + json.dumps(report)
        )
    return report


def compose_simulation_usdz(
    *,
    environment_usdz: Path,
    fruit_entities_usd: Path,
    output_usdz: Path,
    manifest_output: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Package the static environment and dynamic fruit entity layer."""

    from pxr import Sdf, Usd, UsdGeom, UsdUtils

    environment_usdz = environment_usdz.expanduser().resolve(strict=True)
    fruit_entities_usd = fruit_entities_usd.expanduser().resolve(strict=True)
    output_usdz = output_usdz.expanduser().resolve()
    manifest_output = manifest_output.expanduser().resolve()
    if output_usdz.suffix.lower() != ".usdz":
        raise ValueError("Composed simulation output must use the .usdz extension")
    existing = [str(path) for path in (output_usdz, manifest_output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Simulation package outputs already exist; pass --overwrite: "
            + ", ".join(existing)
        )
    output_usdz.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="compose_simulation_", dir=str(output_usdz.parent)
    ) as temporary:
        temporary_path = Path(temporary)
        root_layer = temporary_path / "simulation_root.usda"
        stage = Usd.Stage.CreateNew(str(root_layer))
        world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        stage.SetDefaultPrim(world)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        environment = stage.OverridePrim("/World/Environment")
        environment.GetReferences().AddReference(str(environment_usdz))
        fruits = stage.OverridePrim("/World/Fruits")
        fruits.GetReferences().AddReference(str(fruit_entities_usd))
        stage.GetRootLayer().Save()

        previous_tmp = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = str(temporary_path)
        try:
            packaged = UsdUtils.CreateNewUsdzPackage(
                Sdf.AssetPath(str(root_layer)), str(output_usdz)
            )
        finally:
            if previous_tmp is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_tmp
        if not packaged:
            raise RuntimeError("OpenUSD failed to create the composed USDZ package")

    validation = validate_composed_simulation_usdz(output_usdz)
    manifest: dict[str, Any] = {
        "format": "composed simulation USDZ",
        "coordinate_system": {"unit": "meter", "up_axis": "Z"},
        "compatibility": {
            "isaac_sim": {
                "rendering": "NVIDIA NuRec volume",
                "physics": "UsdPhysics ground collision and dynamic fruit proxies",
            },
            "genesis": {
                "physics": "Standard ground and fruit mesh/primitive schemas",
                "rendering_limit": (
                    "NuRec Gaussian rendering is NVIDIA-specific; Genesis can use "
                    "the standard meshes but may ignore the Gaussian volume."
                ),
            },
        },
        "inputs": {
            "environment_usdz": {
                "path": str(environment_usdz),
                "bytes": environment_usdz.stat().st_size,
                "sha256": _sha256(environment_usdz),
            },
            "fruit_entities_usd": {
                "path": str(fruit_entities_usd),
                "bytes": fruit_entities_usd.stat().st_size,
                "sha256": _sha256(fruit_entities_usd),
            },
        },
        "output": {
            "path": str(output_usdz),
            "bytes": output_usdz.stat().st_size,
            "sha256": _sha256(output_usdz),
        },
        "validation": validation,
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-usdz", type=Path, required=True)
    parser.add_argument("--fruit-entities-usd", type=Path, required=True)
    parser.add_argument("--output-usdz", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_output = args.manifest_output or args.output_usdz.with_name(
        args.output_usdz.stem + "_manifest.json"
    )
    manifest = compose_simulation_usdz(
        environment_usdz=args.environment_usdz,
        fruit_entities_usd=args.fruit_entities_usd,
        output_usdz=args.output_usdz,
        manifest_output=manifest_output,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
