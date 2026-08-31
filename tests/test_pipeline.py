from __future__ import annotations

import json
import struct
from pathlib import Path

from colmapcut_recon import pipeline
from colmapcut_recon.common.config import PROJECT_ROOT
from colmapcut_recon.pipeline import stage_complete


def test_pipeline_stage_completion_checks_durable_outputs(tmp_path: Path) -> None:
    data = tmp_path / "data"
    runs = tmp_path / "runs"

    frames = data / "01_frames"
    (frames / "images").mkdir(parents=True)
    (frames / "images" / "frame_000000.jpg").touch()
    (frames / "frame_manifest.json").write_text("{}", encoding="utf-8")

    model = data / "02_colmap_full" / "sparse" / "1"
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(struct.pack("<Q", 10))
    (data / "02_colmap_full" / "manifest.json").write_text(
        json.dumps({"output": {"primary_model": str(model)}}), encoding="utf-8"
    )

    assert stage_complete("extract_frames", data, runs)
    assert stage_complete("run_colmap", data, runs)
    assert not stage_complete("align_scale_axes", data, runs)


def test_dry_run_defers_outputs_created_by_earlier_stages(tmp_path: Path) -> None:
    scene_config = tmp_path / "scene.yaml"
    scene_config.write_text(
        "scene_id: dry_scene\n"
        f"data_root: {tmp_path / 'scene_data'}\n"
        "simulation_asset_config: configs/simulation/fruit_tomato_asset.yaml\n",
        encoding="utf-8",
    )
    pipeline_config = tmp_path / "pipeline.yaml"
    pipeline_config.write_text(
        "policies:\n"
        "  skip_masks: true\n"
        "  skip_foreground_sparse_filtering: true\n"
        "  skip_compositing: true\n"
        "stages: [align_scale_axes, separate_gaussians]\n",
        encoding="utf-8",
    )

    report = pipeline.run_pipeline(
        pipeline_config=pipeline_config,
        scene_config=scene_config,
        tools_config=PROJECT_ROOT / "configs/tools.example.yaml",
        video=None,
        resume=True,
        dry_run=True,
    )

    assert [entry["command"] for entry in report["stages"]] == [
        ["<resolved-after-prerequisites>", "align_scale_axes"],
        ["<resolved-after-prerequisites>", "separate_gaussians"],
    ]
