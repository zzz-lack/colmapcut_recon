from __future__ import annotations

import json
from pathlib import Path

from colmapcut_recon.colmap.run_colmap import (
    ColmapRunConfig,
    build_colmap_commands,
    run_colmap_sparse,
)
from colmapcut_recon.datasets.prepare_3dgrut import prepare_3dgrut_dataset
from colmapcut_recon.reconstruction.train_3dgrut import build_3dgrut_command


def _write_text_model(model: Path) -> None:
    model.mkdir(parents=True)
    (model / "cameras.txt").write_text("# cameras\n", encoding="utf-8")
    (model / "images.txt").write_text("# images\n", encoding="utf-8")
    (model / "points3D.txt").write_text("# points\n", encoding="utf-8")


def test_colmap_commands_match_external_cli_layout(tmp_path: Path) -> None:
    executable = tmp_path / "colmap"
    executable.touch()
    images = tmp_path / "input frames"
    images.mkdir()
    (images / "frame 01.jpg").touch()
    output = tmp_path / "colmap_output"
    config = ColmapRunConfig(matcher="sequential", use_gpu=False)

    commands = build_colmap_commands(
        executable=executable,
        images_dir=images,
        output_dir=output,
        config=config,
    )

    assert [command[1] for command in commands] == [
        "feature_extractor",
        "sequential_matcher",
        "mapper",
    ]
    assert str(output / "database.db") in commands[0]
    assert str(output / "sparse") in commands[2]
    assert "--FeatureExtraction.use_gpu" in commands[0]
    assert commands[0][commands[0].index("--FeatureExtraction.use_gpu") + 1] == "0"

    manifest = run_colmap_sparse(
        executable=executable,
        images_dir=images,
        output_dir=output,
        config=config,
        dry_run=True,
    )
    assert manifest["input"]["image_count"] == 1
    assert "input frames'" in manifest["command_text"][0]


def test_prepare_3dgrut_dataset_links_inputs_and_excludes_masks(tmp_path: Path) -> None:
    images = tmp_path / "composite"
    images.mkdir()
    image = images / "frame_0001.png"
    image.touch()
    mask = images / "frame_0001_mask.png"
    mask.touch()
    sparse = tmp_path / "aligned" / "0"
    _write_text_model(sparse)
    dataset = tmp_path / "dataset"
    stale_mask = dataset / "images" / mask.name
    stale_mask.parent.mkdir(parents=True)
    stale_mask.symlink_to(mask)

    manifest = prepare_3dgrut_dataset(
        dataset_root=dataset,
        images_dir=images,
        sparse_model_dir=sparse,
    )

    assert (dataset / "images" / image.name).resolve() == image.resolve()
    assert not (dataset / "images" / "frame_0001_mask.png").exists()
    assert (dataset / "sparse" / "0" / "cameras.txt").resolve() == (
        sparse / "cameras.txt"
    ).resolve()
    assert manifest["image_count"] == 1
    assert str(stale_mask) in manifest["removed_stale_links"]
    saved = json.loads((dataset / "dataset_manifest.json").read_text())
    assert saved["load_loss_mask"] is False


def test_3dgrut_command_uses_project_io_boundaries(tmp_path: Path) -> None:
    repository = tmp_path / "3dgrut"
    python = repository / ".venv" / "bin" / "python"
    dataset = tmp_path / "data" / "scene" / "07_datasets" / "3dgrut"
    runs = tmp_path / "runs" / "scene" / "3dgrut"

    command = build_3dgrut_command(
        python=python,
        repository=repository,
        dataset_root=dataset,
        run_dir=runs,
        config={
            "app_config": "apps/colmap_3dgut.yaml",
            "background_color": [1.0, 1.0, 1.0],
            "downsample_factor": 1,
            "export_ply": True,
        },
    )

    assert command[:4] == [
        str(python),
        str(repository / "train.py"),
        "--config-name",
        "apps/colmap_3dgut.yaml",
    ]
    assert f"path={dataset}" in command
    assert f"out_dir={runs}" in command
    assert "model.background.color=white" in command
    assert "export_ply.enabled=true" in command
