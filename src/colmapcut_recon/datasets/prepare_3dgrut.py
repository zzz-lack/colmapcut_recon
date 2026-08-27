"""Stage a lightweight COLMAP-style dataset for external 3DGRUT training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from colmapcut_recon.colmap.run_colmap import IMAGE_SUFFIXES
from colmapcut_recon.common.config import PROJECT_ROOT, load_yaml, resolve_project_path

MODEL_STEMS = ("cameras", "images", "points3D")
MODEL_SUFFIXES = (".bin", ".txt")


def _safe_link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    if destination.is_symlink():
        if destination.resolve(strict=False) == source:
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(
            f"Refusing to replace non-symlink dataset file: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=False)


def _find_model_files(model_dir: Path) -> list[Path]:
    selected: list[Path] = []
    for stem in MODEL_STEMS:
        candidates = [model_dir / f"{stem}{suffix}" for suffix in MODEL_SUFFIXES]
        existing = [path for path in candidates if path.is_file()]
        if not existing:
            raise FileNotFoundError(
                f"COLMAP model is missing {stem}.bin or {stem}.txt in {model_dir}"
            )
        selected.append(existing[0])
    return selected


def _remove_stale_links(root: Path, expected: set[Path]) -> list[str]:
    """Remove only adapter-owned symlinks that are absent from the new input set."""

    removed: list[str] = []
    if not root.is_dir():
        return removed
    for path in root.rglob("*"):
        if path.is_symlink() and path not in expected:
            path.unlink()
            removed.append(str(path))
    return removed


def prepare_3dgrut_dataset(
    *,
    dataset_root: Path,
    images_dir: Path,
    sparse_model_dir: Path,
    load_loss_mask: bool = False,
) -> dict[str, object]:
    """Link composite images and an aligned COLMAP model into 3DGRUT layout."""

    dataset_root = dataset_root.expanduser().resolve()
    images_dir = images_dir.expanduser().resolve()
    sparse_model_dir = sparse_model_dir.expanduser().resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Composite image directory does not exist: {images_dir}"
        )
    if not sparse_model_dir.is_dir():
        raise FileNotFoundError(
            f"Aligned COLMAP model does not exist: {sparse_model_dir}"
        )

    images = sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and (load_loss_mask or not path.name.lower().endswith("_mask.png"))
    )
    if not images:
        raise ValueError(f"No training images found under {images_dir}")
    model_files = _find_model_files(sparse_model_dir)

    expected_image_links = {
        dataset_root / "images" / source.relative_to(images_dir) for source in images
    }
    removed_links = _remove_stale_links(dataset_root / "images", expected_image_links)
    image_links: list[str] = []
    for source in images:
        relative = source.relative_to(images_dir)
        destination = dataset_root / "images" / relative
        _safe_link(source, destination)
        image_links.append(str(destination))

    expected_model_links = {
        dataset_root / "sparse" / "0" / source.name for source in model_files
    }
    removed_links.extend(
        _remove_stale_links(dataset_root / "sparse" / "0", expected_model_links)
    )
    model_links: list[str] = []
    for source in model_files:
        destination = dataset_root / "sparse" / "0" / source.name
        _safe_link(source, destination)
        model_links.append(str(destination))

    manifest: dict[str, object] = {
        "adapter": "3DGRUT COLMAP dataset",
        "layout": {
            "root": str(dataset_root),
            "images": str(dataset_root / "images"),
            "sparse_model": str(dataset_root / "sparse" / "0"),
        },
        "sources": {
            "images": str(images_dir),
            "aligned_sparse_model": str(sparse_model_dir),
        },
        "load_loss_mask": load_loss_mask,
        "image_count": len(images),
        "linked_images": image_links,
        "linked_model_files": model_links,
        "removed_stale_links": removed_links,
        "notes": [
            "Inputs are symlinked so deterministic scene data is not duplicated.",
            "3DGRUT training outputs belong under runs/, never under this dataset directory.",
        ],
    }
    manifest_path = dataset_root / "dataset_manifest.json"
    manifest["manifest"] = str(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
        "--training-config",
        type=Path,
        default=PROJECT_ROOT / "configs/training/3dgrut.yaml",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--sparse-model", type=Path)
    parser.add_argument("--load-loss-mask", action=argparse.BooleanOptionalAction)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_yaml(args.scene_config)
    training = load_yaml(args.training_config)
    data_root = resolve_project_path(scene["data_root"])
    dataset_root = args.dataset_root or data_root / str(
        training.get("dataset_stage", "07_datasets/3dgrut")
    )
    images = args.images or data_root / str(
        training.get("images_stage", "06_composite/images")
    )
    sparse_model = args.sparse_model or data_root / str(
        training.get("sparse_model_stage", "05_alignment/sparse/0")
    )
    load_loss_mask = (
        args.load_loss_mask
        if args.load_loss_mask is not None
        else bool(training.get("load_loss_mask", False))
    )
    manifest = prepare_3dgrut_dataset(
        dataset_root=dataset_root,
        images_dir=images,
        sparse_model_dir=sparse_model,
        load_loss_mask=load_loss_mask,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
