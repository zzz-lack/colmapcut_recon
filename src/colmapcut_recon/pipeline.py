"""Resumable orchestration for the implemented mask-free reconstruction path."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from colmapcut_recon.common.config import PROJECT_ROOT, load_yaml, resolve_project_path
from colmapcut_recon.common.subprocess_utils import format_command, run_command

STAGES = (
    "extract_frames",
    "run_colmap",
    "align_scale_axes",
    "prepare_3dgrut_dataset",
    "train_3dgrut",
    "separate_gaussians",
    "build_ground_collision_mesh",
    "extract_fruit_meshes",
    "export_static_usdz",
    "compose_simulation_usdz",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _model_exists(path: Path) -> bool:
    return path.is_dir() and any(
        (path / name).is_file() for name in ("images.bin", "images.txt")
    )


def _primary_colmap_model(data_root: Path) -> Path | None:
    manifest = _read_json(data_root / "02_colmap_full" / "manifest.json")
    output = manifest.get("output", {})
    if not isinstance(output, dict) or not output.get("primary_model"):
        return None
    model = Path(str(output["primary_model"]))
    return model if _model_exists(model) else None


def _latest_gaussian_ply(run_root: Path) -> Path | None:
    candidates = [path for path in run_root.rglob("export_last.ply") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def stage_complete(
    stage: str,
    data_root: Path,
    run_root: Path,
    asset_config: dict[str, Any] | None = None,
) -> bool:
    """Check that a stage's durable success outputs are present."""

    asset_config = asset_config or {}
    separation_config = dict(asset_config.get("separation", {}))
    collision_config = dict(asset_config.get("ground_collision", {}))
    fruit_config = dict(asset_config.get("fruits", {}))
    export_config = dict(asset_config.get("export", {}))
    separation_root = data_root / str(
        separation_config.get("output_stage", "08_asset_separation")
    )
    collision_root = data_root / str(
        collision_config.get("output_stage", "09_ground_collision")
    )
    fruit_root = data_root / str(
        fruit_config.get("output_stage", "10_fruit_entities")
    )
    export_root = data_root / str(
        export_config.get("output_stage", "11_simulation_asset")
    )
    environment_name = str(
        export_config.get("environment_usdz", "fruit_tomato_environment.usdz")
    )
    combined_name = str(
        export_config.get("combined_usdz", "fruit_tomato_simulation.usdz")
    )

    if stage == "extract_frames":
        images = data_root / "01_frames" / "images"
        return (data_root / "01_frames" / "frame_manifest.json").is_file() and any(
            images.glob("frame_*.*")
        )
    if stage == "run_colmap":
        return _primary_colmap_model(data_root) is not None
    if stage == "align_scale_axes":
        report = _read_json(data_root / "05_alignment" / "alignment_report.json")
        output_model = report.get("output_model")
        return bool(output_model and _model_exists(Path(str(output_model))))
    if stage == "prepare_3dgrut_dataset":
        dataset = data_root / "07_datasets" / "3dgrut"
        return (
            (dataset / "dataset_manifest.json").is_file()
            and (dataset / "images").is_dir()
            and _model_exists(dataset / "sparse" / "0")
        )
    if stage == "train_3dgrut":
        process = _read_json(run_root / "last_process.json")
        invocation = _read_json(run_root / "last_invocation.json")
        directories = invocation.get("new_run_directories", [])
        return process.get("returncode") == 0 and bool(
            isinstance(directories, list)
            and any(Path(str(path)).is_dir() for path in directories)
        )
    if stage == "separate_gaussians":
        return (
            (separation_root / "separation_report.json").is_file()
            and (separation_root / "plant_gaussians.ply").is_file()
            and (separation_root / "ground_gaussians.ply").is_file()
        )
    if stage == "build_ground_collision_mesh":
        return (
            (collision_root / "ground_collision_report.json").is_file()
            and (collision_root / "ground_collision.ply").is_file()
        )
    if stage == "extract_fruit_meshes":
        manifest = _read_json(fruit_root / "tomato_instances.json")
        statistics = manifest.get("statistics", {})
        return bool(
            isinstance(statistics, dict)
            and statistics.get("entity_count", 0) > 0
            and (fruit_root / "tomato_entities.usda").is_file()
            and any((fruit_root / "meshes").glob("tomato_*.ply"))
        )
    if stage == "export_static_usdz":
        manifest = _read_json(
            export_root / f"{Path(environment_name).stem}_manifest.json"
        )
        validation = manifest.get("validation", {})
        return bool(
            isinstance(validation, dict)
            and validation.get("passed")
            and (export_root / environment_name).is_file()
        )
    if stage == "compose_simulation_usdz":
        manifest = _read_json(
            export_root / f"{Path(combined_name).stem}_manifest.json"
        )
        validation = manifest.get("validation", {})
        return bool(
            isinstance(validation, dict)
            and validation.get("passed")
            and (export_root / combined_name).is_file()
        )
    raise ValueError(f"Unknown pipeline stage: {stage}")


def _config_path(scene: dict[str, Any], key: str, fallback: str) -> Path:
    return resolve_project_path(str(scene.get(key, fallback)))


def build_stage_command(
    stage: str,
    *,
    scene_config: Path,
    scene: dict[str, Any],
    tools_config: Path,
    video: Path | None,
    data_root: Path,
) -> list[str]:
    """Build one stage command, resolving the COLMAP model at stage runtime."""

    python = str(Path(sys.executable).absolute())
    scripts = PROJECT_ROOT / "scripts"
    if stage == "extract_frames":
        if video is None:
            raise ValueError("--video is required when frame extraction is incomplete")
        return [
            python,
            str(scripts / "01_extract_frames.py"),
            "--scene-config",
            str(scene_config),
            "--sampling-config",
            str(
                _config_path(
                    scene,
                    "frame_sampling_config",
                    "configs/preprocessing/video_sampling.yaml",
                )
            ),
            "--video",
            str(video),
        ]
    if stage == "run_colmap":
        return [
            python,
            str(scripts / "02_run_colmap.py"),
            "--scene-config",
            str(scene_config),
            "--tools-config",
            str(tools_config),
            "--colmap-config",
            str(_config_path(scene, "colmap_config", "configs/colmap/default.yaml")),
        ]
    if stage == "align_scale_axes":
        model = _primary_colmap_model(data_root)
        if model is None:
            raise RuntimeError("COLMAP manifest has no readable primary model")
        return [
            python,
            str(scripts / "05_align_scale_axes.py"),
            "--scene-config",
            str(scene_config),
            "--alignment-config",
            str(
                _config_path(
                    scene, "alignment_config", "configs/alignment/default.yaml"
                )
            ),
            "--model",
            str(model),
        ]
    training_config = _config_path(
        scene,
        "training_config",
        str(
            dict(scene.get("training_configs", {})).get(
                "threedgrut", "configs/training/3dgrut.yaml"
            )
        ),
    )
    if stage == "prepare_3dgrut_dataset":
        return [
            python,
            str(scripts / "07_prepare_datasets.py"),
            "--scene-config",
            str(scene_config),
            "--training-config",
            str(training_config),
        ]
    if stage == "train_3dgrut":
        return [
            python,
            str(scripts / "08_train_3dgrut.py"),
            "--scene-config",
            str(scene_config),
            "--tools-config",
            str(tools_config),
            "--training-config",
            str(training_config),
        ]
    asset_config = load_yaml(
        _config_path(
            scene,
            "simulation_asset_config",
            "configs/simulation/fruit_tomato_asset.yaml",
        )
    )
    tools = load_yaml(tools_config)
    threedgrut = dict(tools.get("threedgrut", {}))
    sugar = dict(tools.get("sugar", {}))
    threedgrut_python = Path(str(threedgrut.get("python", ""))).expanduser()
    sugar_python = Path(str(sugar.get("python", ""))).expanduser()
    separation = dict(asset_config.get("separation", {}))
    collision = dict(asset_config.get("ground_collision", {}))
    fruits = dict(asset_config.get("fruits", {}))
    export = dict(asset_config.get("export", {}))
    separation_root = data_root / str(
        separation.get("output_stage", "08_asset_separation")
    )
    collision_root = data_root / str(
        collision.get("output_stage", "09_ground_collision")
    )
    fruit_root = data_root / str(fruits.get("output_stage", "10_fruit_entities"))
    export_root = data_root / str(export.get("output_stage", "11_simulation_asset"))
    if stage == "separate_gaussians":
        gaussians = _latest_gaussian_ply(PROJECT_ROOT / "runs" / str(scene["scene_id"]) / "3dgrut")
        if gaussians is None:
            raise RuntimeError("No 3DGRUT export_last.ply was found for asset separation")
        x_bounds = list(separation.get("x_bounds_m", [-1.0, 1.0]))
        y_bounds = list(separation.get("y_bounds_m", [-1.0, 1.0]))
        z_bounds = list(separation.get("z_bounds_m", [-0.15, 2.0]))
        heightfield = dict(separation.get("heightfield", {}))
        return [
            str(threedgrut_python),
            str(scripts / "10_clean_gaussians.py"),
            "--full-scene",
            str(gaussians),
            "--output-directory",
            str(separation_root),
            "--x-min",
            str(x_bounds[0]),
            "--x-max",
            str(x_bounds[1]),
            "--y-min",
            str(y_bounds[0]),
            "--y-max",
            str(y_bounds[1]),
            "--z-min",
            str(z_bounds[0]),
            "--z-max",
            str(z_bounds[1]),
            "--opacity-minimum",
            str(separation.get("opacity_minimum", 0.02)),
            "--grid-size",
            str(heightfield.get("grid_size_m", 0.05)),
            "--initial-ground-height",
            str(heightfield.get("initial_ground_height_m", 0.0)),
            "--surface-quantile",
            str(heightfield.get("surface_quantile", 0.15)),
            "--max-slope-degrees",
            str(heightfield.get("max_slope_degrees", 35.0)),
            "--seed-tolerance-steps",
            str(heightfield.get("seed_tolerance_steps", 2.0)),
            "--smoothing-iterations",
            str(heightfield.get("smoothing_iterations", 2)),
            "--ground-band-quantile",
            str(heightfield.get("ground_band_quantile", 0.90)),
            "--ground-band-mad-multiplier",
            str(heightfield.get("ground_band_mad_multiplier", 3.0)),
            "--gaussian-sigma-multiplier",
            str(heightfield.get("gaussian_sigma_multiplier", 4.0)),
            "--max-sigma-to-band-ratio",
            str(heightfield.get("max_sigma_to_band_ratio", 2.0)),
            "--min-points-per-cell",
            str(heightfield.get("min_points_per_cell", 2)),
        ]
    if stage == "build_ground_collision_mesh":
        x_bounds = list(collision.get("x_bounds_m", [-1.0, 1.0]))
        y_bounds = list(collision.get("y_bounds_m", [-1.0, 1.0]))
        return [
            str(sugar_python),
            str(scripts / "16_build_ground_collision_mesh.py"),
            "--ground-points",
            str(separation_root / "ground_gaussians.ply"),
            "--ply-output",
            str(collision_root / "ground_collision.ply"),
            "--obj-output",
            str(collision_root / "ground_collision.obj"),
            "--report-output",
            str(collision_root / "ground_collision_report.json"),
            "--x-min",
            str(x_bounds[0]),
            "--x-max",
            str(x_bounds[1]),
            "--y-min",
            str(y_bounds[0]),
            "--y-max",
            str(y_bounds[1]),
            "--grid-size",
            str(collision.get("grid_size_m", 0.05)),
            "--initial-ground-height",
            str(collision.get("initial_ground_height_m", 0.0)),
            "--max-slope",
            str(collision.get("max_slope_degrees", 35.0)),
            "--min-points-per-node",
            str(collision.get("min_points_per_node", 1)),
            "--smoothing-iterations",
            str(collision.get("smoothing_iterations", 2)),
            "--bottom-offset",
            str(collision.get("bottom_offset_m", 0.04)),
        ]
    if stage == "extract_fruit_meshes":
        return [
            str(threedgrut_python),
            str(scripts / "17_extract_tomato_entities.py"),
            "--config",
            str(resolve_project_path(str(fruits["config"]))),
            "--gaussians",
            str(separation_root / "plant_gaussians.ply"),
            "--ground-gaussians",
            str(separation_root / "ground_gaussians.ply"),
            "--output-directory",
            str(fruit_root),
            "--bootstrap-colour",
        ]
    environment_name = str(export.get("environment_usdz", "environment.usdz"))
    combined_name = str(export.get("combined_usdz", "simulation.usdz"))
    if stage == "export_static_usdz":
        return [
            str(threedgrut_python),
            str(scripts / "11_export_simulation_asset.py"),
            "--gaussians",
            str(fruit_root / "static_scene_without_tomatoes.ply"),
            "--collision-mesh",
            str(collision_root / "ground_collision.ply"),
            "--output-usdz",
            str(export_root / environment_name),
            "--manifest-output",
            str(export_root / f"{Path(environment_name).stem}_manifest.json"),
            "--threedgrut-root",
            str(threedgrut.get("repository", "")),
            "--threedgrut-python",
            str(threedgrut_python),
            "--isaac-version-file",
            str(export.get("isaac_version_file", "/home/linzz/isaacsim/VERSION")),
        ]
    if stage == "compose_simulation_usdz":
        return [
            str(threedgrut_python),
            str(scripts / "21_compose_simulation_usdz.py"),
            "--environment-usdz",
            str(export_root / environment_name),
            "--fruit-entities-usd",
            str(fruit_root / "tomato_entities.usda"),
            "--output-usdz",
            str(export_root / combined_name),
            "--manifest-output",
            str(export_root / f"{Path(combined_name).stem}_manifest.json"),
        ]
    raise ValueError(f"Unknown pipeline stage: {stage}")


def run_pipeline(
    *,
    pipeline_config: Path,
    scene_config: Path,
    tools_config: Path,
    video: Path | None,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Execute or preview the configured mask-free stage sequence."""

    pipeline = load_yaml(pipeline_config)
    scene = load_yaml(scene_config)
    policies = dict(pipeline.get("policies", {}))
    if not (
        policies.get("skip_masks")
        and policies.get("skip_foreground_sparse_filtering")
        and policies.get("skip_compositing")
    ):
        raise ValueError("This runner currently supports only the mask-free pipeline")
    stages = tuple(str(stage) for stage in pipeline.get("stages", STAGES))
    unsupported = [stage for stage in stages if stage not in STAGES]
    if unsupported:
        raise ValueError(f"Unsupported pipeline stages: {unsupported}")

    data_root = resolve_project_path(scene["data_root"])
    scene_id = str(scene["scene_id"])
    run_root = PROJECT_ROOT / "runs" / scene_id / "3dgrut"
    if video is not None:
        video = video.expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {video}")

    report: dict[str, Any] = {
        "adapter": "colmapcut_recon.mask_free_pipeline",
        "pipeline_config": str(pipeline_config.resolve()),
        "scene_config": str(scene_config.resolve()),
        "scene_id": scene_id,
        "source_video": str(video) if video else None,
        "resume": resume,
        "dry_run": dry_run,
        "stages": [],
    }
    manifest_path = data_root / "pipeline_manifest.json"
    asset_config = (
        load_yaml(resolve_project_path(str(scene["simulation_asset_config"])))
        if scene.get("simulation_asset_config")
        else {}
    )
    for stage in stages:
        complete = stage_complete(stage, data_root, run_root, asset_config)
        if resume and complete:
            report["stages"].append({"name": stage, "status": "skipped_complete"})
            continue
        deferred_dry_run = dry_run and (
            (
                stage == "align_scale_axes"
                and _primary_colmap_model(data_root) is None
            )
            or (
                stage == "separate_gaussians"
                and _latest_gaussian_ply(run_root) is None
            )
        )
        if deferred_dry_run:
            command = ["<resolved-after-prerequisites>", stage]
        else:
            command = build_stage_command(
                stage,
                scene_config=scene_config.resolve(),
                scene=scene,
                tools_config=tools_config.resolve(),
                video=video,
                data_root=data_root,
            )
        entry: dict[str, Any] = {
            "name": stage,
            "status": "planned" if dry_run else "running",
            "command": command,
            "command_text": format_command(command),
        }
        report["stages"].append(entry)
        if dry_run:
            continue
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        run_command(
            command,
            cwd=PROJECT_ROOT,
            record_path=data_root / "pipeline_logs" / f"{stage}.json",
        )
        entry["status"] = "completed"
        if not stage_complete(stage, data_root, run_root, asset_config):
            raise RuntimeError(
                f"Stage returned successfully but outputs are incomplete: {stage}"
            )

    report["status"] = "planned" if dry_run else "completed"
    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report["manifest"] = str(manifest_path)
        manifest_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=PROJECT_ROOT / "configs/pipeline_no_masks.yaml",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=PROJECT_ROOT / "configs/scenes/fruit_tomato.yaml",
    )
    parser.add_argument(
        "--tools-config", type=Path, default=PROJECT_ROOT / "configs/tools.local.yaml"
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_pipeline(
        pipeline_config=args.pipeline_config,
        scene_config=args.scene_config,
        tools_config=args.tools_config,
        video=args.video,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
