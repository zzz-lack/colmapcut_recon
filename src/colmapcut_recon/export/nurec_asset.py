"""Export an Isaac Sim 5.x-compatible NuRec USDZ with static collision."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NuRecExportConfig:
    """Local NVIDIA toolchain paths and compatibility settings."""

    threedgrut_root: Path = Path("/home/linzz/3dgrut")
    threedgrut_python: Path = Path("/home/linzz/3dgrut/.venv/bin/python")
    isaac_version_file: Path = Path("/home/linzz/isaacsim/VERSION")
    collision_visible: bool = False

    def validate(self) -> str:
        if not self.threedgrut_root.is_dir():
            raise FileNotFoundError(f"3DGRUT root does not exist: {self.threedgrut_root}")
        if not self.threedgrut_python.is_file():
            raise FileNotFoundError(f"3DGRUT Python does not exist: {self.threedgrut_python}")
        if not self.isaac_version_file.is_file():
            raise FileNotFoundError(
                f"Isaac Sim version file does not exist: {self.isaac_version_file}"
            )
        version = read_isaac_sim_version(self.isaac_version_file)
        if isaac_sim_major_version(version) != 5:
            raise ValueError(
                f"NuRec export is selected for Isaac Sim 5.x, but detected {version}"
            )
        return version


def read_isaac_sim_version(path: Path) -> str:
    version = path.read_text(encoding="utf-8").strip()
    if not version:
        raise ValueError(f"Isaac Sim version file is empty: {path}")
    return version


def isaac_sim_major_version(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, IndexError) as error:
        raise ValueError(f"Cannot parse Isaac Sim version: {version!r}") from error


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=str(cwd), check=True)


def validate_nurec_collision_usdz(path: Path) -> dict[str, Any]:
    """Verify NuRec payload, Volume schema, mesh composition, and collision APIs."""

    from pxr import Usd, UsdGeom, UsdPhysics

    path = path.resolve(strict=True)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Output is not a USDZ/ZIP package: {path}")
    with zipfile.ZipFile(path, "r") as archive:
        members = archive.namelist()
    nurec_members = sorted(name for name in members if name.lower().endswith(".nurec"))
    mesh_usd_members = sorted(
        name for name in members if Path(name).name.lower() == "mesh.usd"
    )
    mesh_ply_members = sorted(
        name for name in members if Path(name).name.lower() == "mesh.ply"
    )

    stage = Usd.Stage.Open(str(path), load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"OpenUSD could not open NuRec package: {path}")
    volumes = []
    meshes = []
    collision_meshes = []
    invisible_collision_meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Volume":
            marker = prim.GetAttribute("omni:nurec:isNuRecVolume")
            if marker and marker.Get():
                volumes.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh):
            prim_path = str(prim.GetPath())
            meshes.append(prim_path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_meshes.append(prim_path)
                visibility = UsdGeom.Imageable(prim).ComputeVisibility()
                if visibility == UsdGeom.Tokens.invisible:
                    invisible_collision_meshes.append(prim_path)

    report = {
        "package_opened": True,
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "members": members,
        "nurec_payloads": nurec_members,
        "volume_prims": volumes,
        "mesh_usd_members": mesh_usd_members,
        "mesh_ply_members": mesh_ply_members,
        "mesh_prims": meshes,
        "collision_mesh_prims": collision_meshes,
        "invisible_collision_mesh_prims": invisible_collision_meshes,
    }
    report["passed"] = bool(
        nurec_members
        and volumes
        and mesh_usd_members
        and mesh_ply_members
        and collision_meshes
    )
    if not report["passed"]:
        raise RuntimeError("NuRec USDZ validation failed: " + json.dumps(report))
    return report


def export_isaac_sim_5_asset(
    *,
    gaussian_ply: Path,
    collision_ply: Path,
    output_usdz: Path,
    manifest_output: Path,
    config: NuRecExportConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run NVIDIA's PLY-to-NuRec and mesh-injection tools, then validate."""

    isaac_version = config.validate()
    gaussian_ply = gaussian_ply.resolve(strict=True)
    collision_ply = collision_ply.resolve(strict=True)
    output_usdz = output_usdz.resolve()
    manifest_output = manifest_output.resolve()
    if output_usdz.suffix.lower() != ".usdz":
        raise ValueError("Isaac Sim 5.x NuRec output must use the .usdz extension")
    existing = [str(path) for path in (output_usdz, manifest_output) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Outputs already exist; pass --overwrite: " + ", ".join(existing))
    output_usdz.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="nurec_export_", dir=str(output_usdz.parent)
    ) as temporary_dir:
        intermediate = Path(temporary_dir) / "gaussians_only.usdz"
        convert_command = [
            str(config.threedgrut_python),
            "-m",
            "threedgrut.export.scripts.transcode",
            str(gaussian_ply),
            "--output",
            str(intermediate),
            "--format",
            "nurec",
        ]
        mesh_command = [
            str(config.threedgrut_python),
            "-m",
            "threedgrut.export.scripts.add_mesh_to_usdz",
            "--input_usdz",
            str(intermediate),
            "--output_usdz",
            str(output_usdz),
            "--mesh_ply",
            str(collision_ply),
            "--set_collision",
        ]
        if not config.collision_visible:
            mesh_command.append("--set_invisible")
        _run_checked(convert_command, cwd=config.threedgrut_root)
        _run_checked(mesh_command, cwd=config.threedgrut_root)

    validation = validate_nurec_collision_usdz(output_usdz)
    if not config.collision_visible and not validation["invisible_collision_mesh_prims"]:
        raise RuntimeError("Collision mesh was requested invisible but remains visible")
    manifest: dict[str, Any] = {
        "format": "NVIDIA NuRec USDZ with static triangle-mesh collision",
        "compatibility": {
            "target": "Isaac Sim 5.x",
            "detected_isaac_sim": isaac_version,
            "renderer": "Omniverse RTX NuRec",
        },
        "config": {
            **asdict(config),
            "threedgrut_root": str(config.threedgrut_root.resolve()),
            "threedgrut_python": str(config.threedgrut_python.resolve()),
            "isaac_version_file": str(config.isaac_version_file.resolve()),
        },
        "inputs": {
            "gaussians": {
                "path": str(gaussian_ply),
                "bytes": gaussian_ply.stat().st_size,
                "sha256": _sha256(gaussian_ply),
            },
            "collision_mesh": {
                "path": str(collision_ply),
                "bytes": collision_ply.stat().st_size,
                "sha256": _sha256(collision_ply),
            },
        },
        "output": {
            "path": output_usdz.name,
            "bytes": output_usdz.stat().st_size,
            "sha256": _sha256(output_usdz),
        },
        "validation": validation,
    }
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--collision-mesh", type=Path, required=True)
    parser.add_argument("--output-usdz", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--threedgrut-root", type=Path, default=Path("/home/linzz/3dgrut"))
    parser.add_argument(
        "--threedgrut-python",
        type=Path,
        default=Path("/home/linzz/3dgrut/.venv/bin/python"),
    )
    parser.add_argument(
        "--isaac-version-file",
        type=Path,
        default=Path("/home/linzz/isaacsim/VERSION"),
    )
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_output = args.manifest_output or args.output_usdz.with_name(
        args.output_usdz.stem + "_manifest.json"
    )
    manifest = export_isaac_sim_5_asset(
        gaussian_ply=args.gaussians,
        collision_ply=args.collision_mesh,
        output_usdz=args.output_usdz,
        manifest_output=manifest_output,
        config=NuRecExportConfig(
            threedgrut_root=args.threedgrut_root,
            threedgrut_python=args.threedgrut_python,
            isaac_version_file=args.isaac_version_file,
            collision_visible=args.show_collision,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
