"""Run a staged external COLMAP sparse reconstruction."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from colmapcut_recon.common.config import (
    PROJECT_ROOT,
    load_tool,
    load_yaml,
    resolve_project_path,
)
from colmapcut_recon.common.subprocess_utils import format_command, run_command

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MATCHER_COMMANDS = {
    "sequential": "sequential_matcher",
    "exhaustive": "exhaustive_matcher",
    "spatial": "spatial_matcher",
    "transitive": "transitive_matcher",
    "vocab_tree": "vocab_tree_matcher",
}


@dataclass(frozen=True)
class ColmapRunConfig:
    camera_model: str = "OPENCV"
    single_camera: bool = True
    matcher: str = "sequential"
    use_gpu: bool = True
    feature_extra_args: tuple[str, ...] = field(default_factory=tuple)
    matcher_extra_args: tuple[str, ...] = field(default_factory=tuple)
    mapper_extra_args: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ColmapRunConfig:
        matcher = str(value.get("matcher", "sequential"))
        if matcher not in MATCHER_COMMANDS:
            raise ValueError(
                f"Unsupported matcher '{matcher}'; choose from {sorted(MATCHER_COMMANDS)}"
            )
        return cls(
            camera_model=str(value.get("camera_model", "OPENCV")),
            single_camera=bool(value.get("single_camera", True)),
            matcher=matcher,
            use_gpu=bool(value.get("use_gpu", True)),
            feature_extra_args=tuple(
                str(x) for x in value.get("feature_extra_args", [])
            ),
            matcher_extra_args=tuple(
                str(x) for x in value.get("matcher_extra_args", [])
            ),
            mapper_extra_args=tuple(
                str(x)
                for x in value.get("mapper_extra_args", value.get("extra_args", []))
            ),
        )


def _flag(value: bool) -> str:
    return "1" if value else "0"


def build_colmap_commands(
    *,
    executable: Path,
    images_dir: Path,
    output_dir: Path,
    config: ColmapRunConfig,
) -> list[list[str]]:
    """Build COLMAP's feature, matching, and mapping commands."""

    database = output_dir / "database.db"
    sparse = output_dir / "sparse"
    common = [str(executable)]
    return [
        common
        + [
            "feature_extractor",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--ImageReader.camera_model",
            config.camera_model,
            "--ImageReader.single_camera",
            _flag(config.single_camera),
            "--FeatureExtraction.use_gpu",
            _flag(config.use_gpu),
            *config.feature_extra_args,
        ],
        common
        + [
            MATCHER_COMMANDS[config.matcher],
            "--database_path",
            str(database),
            "--FeatureMatching.use_gpu",
            _flag(config.use_gpu),
            *config.matcher_extra_args,
        ],
        common
        + [
            "mapper",
            "--database_path",
            str(database),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse),
            *config.mapper_extra_args,
        ],
    ]


def run_colmap_sparse(
    *,
    executable: Path,
    images_dir: Path,
    output_dir: Path,
    config: ColmapRunConfig,
    dry_run: bool = False,
) -> dict[str, object]:
    """Validate inputs and run (or preview) a sparse reconstruction."""

    executable = executable.expanduser().resolve()
    images_dir = images_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"COLMAP executable does not exist: {executable}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"COLMAP image directory does not exist: {images_dir}")
    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found under {images_dir}")

    model_zero = output_dir / "sparse" / "0"
    if not dry_run and (
        (output_dir / "database.db").exists()
        or any((model_zero / name).exists() for name in ("cameras.bin", "cameras.txt"))
    ):
        raise FileExistsError(
            f"COLMAP output already exists in {output_dir}; move it aside before rebuilding"
        )

    commands = build_colmap_commands(
        executable=executable,
        images_dir=images_dir,
        output_dir=output_dir,
        config=config,
    )
    manifest: dict[str, object] = {
        "adapter": "colmapcut_recon.colmap",
        "dry_run": dry_run,
        "input": {"images": str(images_dir), "image_count": len(images)},
        "output": {
            "root": str(output_dir),
            "database": str(output_dir / "database.db"),
            "sparse": str(output_dir / "sparse"),
            "primary_model": str(model_zero),
        },
        "commands": commands,
        "command_text": [format_command(command) for command in commands],
    }
    if dry_run:
        return manifest

    (output_dir / "sparse").mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    for index, command in enumerate(commands, start=1):
        run_command(
            command,
            record_path=logs_dir / f"{index:02d}_{command[1]}.json",
        )
    if not model_zero.is_dir():
        models = sorted(
            path for path in (output_dir / "sparse").iterdir() if path.is_dir()
        )
        raise RuntimeError(
            "COLMAP mapper did not produce sparse/0. "
            f"Models found: {[str(path) for path in models]}"
        )
    manifest_path = output_dir / "manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=PROJECT_ROOT / "configs/scenes/plant_001.yaml",
    )
    parser.add_argument(
        "--tools-config", type=Path, default=PROJECT_ROOT / "configs/tools.local.yaml"
    )
    parser.add_argument(
        "--colmap-config",
        type=Path,
        default=PROJECT_ROOT / "configs/colmap/default.yaml",
    )
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_yaml(args.scene_config)
    raw_config = load_yaml(args.colmap_config)
    config = ColmapRunConfig.from_mapping(raw_config)
    data_root = resolve_project_path(scene["data_root"])
    images = (
        args.images
        or data_root / str(raw_config.get("image_root_stage", "01_frames")) / "images"
    )
    output = args.output or data_root / str(
        raw_config.get("output_stage", "02_colmap_full")
    )
    executable = args.executable or Path(
        load_tool("colmap", args.tools_config)["executable"]
    )
    manifest = run_colmap_sparse(
        executable=executable,
        images_dir=images,
        output_dir=output,
        config=config,
        dry_run=args.dry_run,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
