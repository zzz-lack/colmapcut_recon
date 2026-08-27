from __future__ import annotations

from pathlib import Path

import numpy as np

from colmapcut_recon.geometry.apriltag_alignment import (
    AprilTagAlignmentConfig,
    _prepare_output,
    calculate_alignment,
)
from colmapcut_recon.geometry.similarity_transform import SimilarityTransform


def _square(center: tuple[float, float], edge: float = 2.0) -> np.ndarray:
    half = edge / 2
    x, y = center
    return np.asarray(
        [
            [x - half, y - half, 0.0],
            [x + half, y - half, 0.0],
            [x + half, y + half, 0.0],
            [x - half, y + half, 0.0],
        ]
    )


def test_similarity_transform_applies_row_vector_points() -> None:
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transform = SimilarityTransform(2.0, rotation, np.asarray([1.0, 2.0, 3.0]))

    result = transform.apply(np.asarray([[1.0, 0.0, 0.0]]))

    np.testing.assert_allclose(result, [[1.0, 4.0, 3.0]])
    np.testing.assert_allclose(transform.matrix[:3, :3], 2.0 * rotation)


def test_calculate_alignment_recovers_metric_tag_frame() -> None:
    corners = {
        3: _square((-3.0, -2.0)),
        0: _square((3.0, -2.0)),
        1: _square((-3.0, 2.0)),
        2: _square((3.0, 2.0)),
    }
    sizes = {tag_id: 0.1 for tag_id in corners}
    camera_centers = np.asarray([[0.0, 0.0, 5.0], [2.0, 1.0, 4.0]])

    transform, diagnostics = calculate_alignment(
        corners, sizes, camera_centers, x_axis_tags=(3, 0)
    )

    assert np.isclose(transform.scale, 0.05)
    np.testing.assert_allclose(transform.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(transform.translation, np.zeros(3), atol=1e-12)
    centers = np.stack(
        [transform.apply(value).mean(axis=0) for value in corners.values()]
    )
    np.testing.assert_allclose(centers.mean(axis=0), np.zeros(3), atol=1e-12)
    assert diagnostics["ground_plane_rms_m"] == 0.0


def test_alignment_config_supports_common_and_per_tag_sizes() -> None:
    config = AprilTagAlignmentConfig.from_mapping(
        {
            "family": "36h11",
            "tag_ids": [0, 1],
            "tag_size_m": 0.1,
            "tag_sizes_m": {"1": 0.2},
            "x_axis_tags": [0, 1],
        }
    )

    assert config.tag_sizes_m == {0: 0.1, 1: 0.2}


def test_empty_stage_skeleton_is_not_treated_as_alignment_output(
    tmp_path: Path,
) -> None:
    model = tmp_path / "05_alignment" / "sparse" / "0"
    model.mkdir(parents=True)
    (model / ".gitkeep").touch()

    assert _prepare_output(tmp_path / "05_alignment", overwrite=False) == model
