from __future__ import annotations

import json
from pathlib import Path

from colmapcut_recon.export.sugar_adapter import (
    build_sugar_cameras_json,
    prepare_sugar_ground_adapter,
)


def _write_colmap_text_model(sparse_zero: Path) -> None:
    sparse_zero.mkdir(parents=True)
    (sparse_zero / "cameras.txt").write_text(
        "# cameras\n1 PINHOLE 640 480 500 510 320 240\n",
        encoding="utf-8",
    )
    (sparse_zero / "images.txt").write_text(
        "# images\n7 1 0 0 0 1 2 3 1 frame_000007.jpg\n\n",
        encoding="utf-8",
    )


def test_build_sugar_cameras_json(tmp_path: Path) -> None:
    sparse_zero = tmp_path / "sparse" / "0"
    _write_colmap_text_model(sparse_zero)

    records = build_sugar_cameras_json(sparse_zero)

    assert len(records) == 1
    assert records[0]["img_name"] == "frame_000007"
    assert records[0]["position"] == [-1.0, -2.0, -3.0]
    assert records[0]["rotation"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert records[0]["fx"] == 500.0
    assert records[0]["fy"] == 510.0


def test_prepare_sugar_ground_adapter(tmp_path: Path) -> None:
    images = tmp_path / "source_images"
    images.mkdir()
    (images / "frame_000007.jpg").touch()
    sparse = tmp_path / "source_sparse"
    _write_colmap_text_model(sparse / "0")
    ground = tmp_path / "ground.ply"
    ground.write_bytes(b"ply\n")
    adapter = tmp_path / "adapter"

    manifest = prepare_sugar_ground_adapter(
        adapter_root=adapter,
        images_dir=images,
        sparse_dir=sparse,
        ground_ply=ground,
    )

    assert (adapter / "scene" / "images").resolve() == images.resolve()
    assert (adapter / "scene" / "sparse").resolve() == sparse.resolve()
    assert (
        adapter / "checkpoint" / "point_cloud" / "iteration_7000" / "point_cloud.ply"
    ).resolve() == ground.resolve()
    cameras = json.loads((adapter / "checkpoint" / "cameras.json").read_text())
    assert len(cameras) == 1
    assert manifest["camera_count"] == 1
