#!/usr/bin/env python3
"""Create a local SAGA scene adapter without duplicating the source dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = PROJECT_ROOT / "data/scenes/roman_tomato_02/07_saga"
DEFAULT_IMAGES = Path("/home/linzz/Desktop/realplantrecon_romantomato2/images")
DEFAULT_SPARSE = Path("/home/linzz/Desktop/realplantrecon_romantomato2/metric_scene/sparse")
DEFAULT_GAUSSIANS = Path(
    "/home/linzz/Desktop/realplantrecon_romantomato2/metric_scene/3dgrut_out/"
    "plant_metric_original_coords.ply"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--sparse", type=Path, default=DEFAULT_SPARSE)
    parser.add_argument("--gaussians", type=Path, default=DEFAULT_GAUSSIANS)
    parser.add_argument("--scene-iteration", type=int, default=30000)
    parser.add_argument("--feature-iterations", type=int, default=10000)
    parser.add_argument("--resolution", type=int, default=4)
    return parser.parse_args()


def _replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace a non-symlink path: {link}")
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def _namespace_text(args: argparse.Namespace, runtime: Path) -> str:
    values = {
        "sh_degree": 3,
        "feature_dim": 32,
        "init_from_3dgs_pcd": False,
        "source_path": str(runtime),
        "model_path": str(runtime),
        "feature_model_path": "",
        "images": "images",
        "resolution": args.resolution,
        "white_background": False,
        "data_device": "cuda",
        "eval": False,
        "need_features": False,
        "need_masks": True,
        "allow_principle_point_shift": False,
        "mask_camera_filter": False,
        "iterations": args.feature_iterations,
        "position_lr_init": 0.00016,
        "position_lr_final": 0.0000016,
        "position_lr_delay_mult": 0.01,
        "position_lr_max_steps": 30000,
        "feature_lr": 0.0025,
        "opacity_lr": 0.05,
        "scaling_lr": 0.005,
        "rotation_lr": 0.001,
        "percent_dense": 0.01,
        "lambda_dssim": 0.2,
        "densification_interval": 100,
        "opacity_reset_interval": 3000,
        "densify_from_iter": 500,
        "densify_until_iter": 15000,
        "densify_grad_threshold": 0.0002,
        "mask_lr": 1.0,
        "optimization_times": 2,
        "IoU_thresh": 0.5,
        "IoA_thresh": 0.8,
        "lamb": 0.3,
        "ray_sample_rate": 0.0,
        "num_sampled_rays": 512,
        "smooth_K": 1,
        "scale_aware_dim": -1,
        "rfn": 1.0,
        "convert_SHs_python": False,
        "compute_cov3D_python": False,
        "debug": False,
        "target": "contrastive_feature",
        "iteration": args.scene_iteration,
    }
    entries = ",\n    ".join(f"{key}={value!r}" for key, value in values.items())
    return f"Namespace(\n    {entries}\n)\n"


def main() -> int:
    args = parse_args()
    for label, path in (
        ("images", args.images),
        ("sparse model", args.sparse),
        ("metric Gaussian PLY", args.gaussians),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    _replace_symlink(runtime / "images", args.images)
    _replace_symlink(runtime / "sparse", args.sparse)
    point_cloud = runtime / "point_cloud" / f"iteration_{args.scene_iteration}"
    point_cloud.mkdir(parents=True, exist_ok=True)
    _replace_symlink(point_cloud / "scene_point_cloud.ply", args.gaussians)
    config = _namespace_text(args, runtime)
    (runtime / "cfg_args").write_text(config, encoding="utf-8")
    (runtime / "feature_cfg_args").write_text(config, encoding="utf-8")
    (runtime / "sam_masks").mkdir(exist_ok=True)
    (runtime / "mask_scales").mkdir(exist_ok=True)
    print(f"SAGA runtime prepared: {runtime}")
    print(f"Scene Gaussians: {args.gaussians.resolve()}")
    print(f"Images: {args.images.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
