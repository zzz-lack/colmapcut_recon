from __future__ import annotations

from pathlib import Path

import pytest

from colmapcut_recon.preprocessing.extract_frames import (
    FrameSamplingConfig,
    VideoFrame,
    build_ffmpeg_command,
    select_video_frames,
)


def _frames(count: int, fps: float = 10.0) -> list[VideoFrame]:
    return [
        VideoFrame(index, index / fps, key_frame=index % round(fps) == 0)
        for index in range(count)
    ]


def test_interval_sampling_uses_timestamps() -> None:
    config = FrameSamplingConfig(
        method="interval_seconds",
        interval_seconds=0.5,
        start_seconds=0.2,
        end_seconds=1.3,
    )

    selected = select_video_frames(_frames(20), config)

    assert [frame.source_frame_index for frame in selected] == [2, 7, 12]
    assert [frame.timestamp_seconds for frame in selected] == pytest.approx(
        [0.2, 0.7, 1.2]
    )


def test_every_n_frames_is_relative_to_selected_time_range() -> None:
    config = FrameSamplingConfig(
        method="every_n_frames",
        every_n_frames=3,
        start_seconds=0.2,
        end_seconds=1.1,
    )

    selected = select_video_frames(_frames(20), config)

    assert [frame.source_frame_index for frame in selected] == [2, 5, 8, 11]


def test_target_count_includes_both_ends() -> None:
    config = FrameSamplingConfig(method="target_count", target_count=4)

    selected = select_video_frames(_frames(10), config)

    assert [frame.source_frame_index for frame in selected] == [0, 3, 6, 9]


def test_ffmpeg_command_selects_exact_source_indices(tmp_path: Path) -> None:
    config = FrameSamplingConfig(method="every_n_frames", every_n_frames=10)
    selected = [_frames(30)[index] for index in (0, 10, 20)]

    command = build_ffmpeg_command(
        ffmpeg=Path("/usr/bin/ffmpeg"),
        video=tmp_path / "source video.mp4",
        selected=selected,
        output_pattern=tmp_path / "frame_%06d.jpg",
        config=config,
    )

    assert command[command.index("-vf") + 1] == "select=eq(n\\,0)+eq(n\\,10)+eq(n\\,20)"
    assert command[-1].endswith("frame_%06d.jpg")
