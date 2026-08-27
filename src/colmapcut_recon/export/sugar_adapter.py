"""Prepare a vanilla-3DGS-style input adapter for SuGaR mesh extraction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


SUPPORTED_CAMERA_MODELS = {
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "FULL_OPENCV",
    "OPENCV_FISHEYE",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
}

DUAL_FOCAL_CAMERA_MODELS = {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"}


def _read_colmap_cameras(path: Path) -> dict[int, dict[str, float | int | str]]:
    cameras: dict[int, dict[str, float | int | str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(f"Invalid COLMAP camera line in {path}: {line}")

            camera_id = int(fields[0])
            model = fields[1]
            if model not in SUPPORTED_CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model for SuGaR adapter: {model}")
            width, height = int(fields[2]), int(fields[3])
            params = [float(value) for value in fields[4:]]

            if model in DUAL_FOCAL_CAMERA_MODELS:
                fx, fy = params[0], params[1]
            else:
                fx = fy = params[0]

            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
            }
    if not cameras:
        raise ValueError(f"No cameras found in {path}")
    return cameras


def _quaternion_to_rotation(
    qw: float, qx: float, qy: float, qz: float
) -> list[list[float]]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0:
        raise ValueError("COLMAP image contains a zero-length quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return [
        [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ],
        [
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ],
        [
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ],
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[row][column] for row in range(3)] for column in range(3)]


def _camera_center(
    world_to_camera_rotation: list[list[float]], translation: Iterable[float]
) -> list[float]:
    camera_to_world_rotation = _transpose(world_to_camera_rotation)
    tx, ty, tz = translation
    return [
        -sum(camera_to_world_rotation[row][column] * (tx, ty, tz)[column] for column in range(3))
        for row in range(3)
    ]


def build_sugar_cameras_json(sparse_text_dir: Path) -> list[dict[str, object]]:
    """Convert COLMAP text cameras/images to vanilla 3DGS cameras.json records."""

    cameras = _read_colmap_cameras(sparse_text_dir / "cameras.txt")
    records: list[dict[str, object]] = []
    images_path = sparse_text_dir / "images.txt"

    with images_path.open("r", encoding="utf-8") as handle:
        expecting_image = True
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("#"):
                continue
            if not expecting_image:
                expecting_image = True
                continue
            if not line:
                continue

            fields = line.split()
            if len(fields) < 10:
                raise ValueError(f"Invalid COLMAP image line in {images_path}: {line}")
            image_id = int(fields[0])
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            translation = tuple(float(value) for value in fields[5:8])
            camera_id = int(fields[8])
            image_name = " ".join(fields[9:])
            camera = cameras[camera_id]
            world_to_camera_rotation = _quaternion_to_rotation(qw, qx, qy, qz)

            records.append(
                {
                    "id": image_id,
                    "img_name": Path(image_name).stem,
                    "width": camera["width"],
                    "height": camera["height"],
                    "position": _camera_center(world_to_camera_rotation, translation),
                    "rotation": _transpose(world_to_camera_rotation),
                    "fy": camera["fy"],
                    "fx": camera["fx"],
                }
            )
            expecting_image = False

    if not records:
        raise ValueError(f"No images found in {images_path}")
    return records


def _replace_symlink(link: Path, target: Path, overwrite: bool) -> None:
    target = target.resolve(strict=True)
    if link.is_symlink():
        if link.resolve(strict=False) == target:
            return
        if not overwrite:
            raise FileExistsError(f"Symlink already points elsewhere: {link}")
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace a non-symlink path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_sugar_ground_adapter(
    *,
    adapter_root: Path,
    images_dir: Path,
    sparse_dir: Path,
    ground_ply: Path,
    iteration: int = 7000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Create lightweight, reproducible SuGaR scene and checkpoint directories."""

    adapter_root = adapter_root.resolve()
    scene_dir = adapter_root / "scene"
    checkpoint_dir = adapter_root / "checkpoint"
    mesh_output_dir = adapter_root / "mesh_output"
    point_cloud_link = (
        checkpoint_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    )

    _replace_symlink(scene_dir / "images", images_dir, overwrite)
    _replace_symlink(scene_dir / "sparse", sparse_dir, overwrite)
    _replace_symlink(point_cloud_link, ground_ply, overwrite)
    mesh_output_dir.mkdir(parents=True, exist_ok=True)

    cameras = build_sugar_cameras_json(sparse_dir / "0")
    cameras_path = checkpoint_dir / "cameras.json"
    cameras_path.parent.mkdir(parents=True, exist_ok=True)
    cameras_path.write_text(json.dumps(cameras, indent=2) + "\n", encoding="utf-8")

    manifest: dict[str, object] = {
        "adapter": "SuGaR vanilla 3DGS checkpoint adapter",
        "iteration": iteration,
        "camera_count": len(cameras),
        "paths": {
            "scene": str(scene_dir),
            "images_source": str(images_dir.resolve(strict=True)),
            "sparse_source": str(sparse_dir.resolve(strict=True)),
            "checkpoint": str(checkpoint_dir),
            "ground_gaussians_source": str(ground_ply.resolve(strict=True)),
            "point_cloud": str(point_cloud_link),
            "cameras_json": str(cameras_path),
            "mesh_output": str(mesh_output_dir),
        },
        "notes": [
            "Large immutable inputs are linked read-only instead of duplicated.",
            "Pass checkpoint path with a trailing slash because SuGaR concatenates cameras.json.",
            "The source COLMAP model and ground PLY must use the same metric coordinates.",
        ],
    }
    manifest_path = adapter_root / "adapter_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--sparse-dir", type=Path, required=True)
    parser.add_argument("--ground-ply", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_sugar_ground_adapter(
        adapter_root=args.adapter_root,
        images_dir=args.images_dir,
        sparse_dir=args.sparse_dir,
        ground_ply=args.ground_ply,
        iteration=args.iteration,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))
    return 0
