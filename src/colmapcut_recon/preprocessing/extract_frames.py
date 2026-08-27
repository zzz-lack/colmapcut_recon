"""Deterministically sample reconstruction images from a captured video."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from colmapcut_recon.common.config import (
    PROJECT_ROOT,
    load_tool,
    load_yaml,
    resolve_project_path,
)
from colmapcut_recon.common.subprocess_utils import format_command, run_command

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


@dataclass(frozen=True)
class VideoFrame:
    source_frame_index: int
    timestamp_seconds: float
    key_frame: bool


@dataclass(frozen=True)
class FrameSamplingConfig:
    method: str = "interval_seconds"
    interval_seconds: float = 1.0
    every_n_frames: int = 1
    target_count: int | None = None
    start_seconds: float = 0.0
    end_seconds: float | None = None
    output_format: str = "jpg"
    jpeg_qscale: int = 2
    filename_prefix: str = "frame_"
    filename_digits: int = 6

    def __post_init__(self) -> None:
        if self.method not in {"interval_seconds", "every_n_frames", "target_count"}:
            raise ValueError(f"Unsupported frame sampling method: {self.method}")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be finite and positive")
        if self.every_n_frames < 1:
            raise ValueError("every_n_frames must be at least 1")
        if self.target_count is not None and self.target_count < 1:
            raise ValueError("target_count must be positive")
        if self.method == "target_count" and self.target_count is None:
            raise ValueError("target_count is required for target_count sampling")
        if not math.isfinite(self.start_seconds) or self.start_seconds < 0:
            raise ValueError("start_seconds must be finite and non-negative")
        if self.end_seconds is not None and (
            not math.isfinite(self.end_seconds)
            or self.end_seconds <= self.start_seconds
        ):
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.output_format not in {"jpg", "png"}:
            raise ValueError("output_format must be jpg or png")
        if not 1 <= self.jpeg_qscale <= 31:
            raise ValueError(
                "jpeg_qscale must be in [1, 31], where 1 is highest quality"
            )
        if not self.filename_prefix or "/" in self.filename_prefix:
            raise ValueError("filename_prefix must be a non-empty filename prefix")
        if self.filename_digits < 1:
            raise ValueError("filename_digits must be positive")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> FrameSamplingConfig:
        return cls(
            method=str(raw.get("method", "interval_seconds")),
            interval_seconds=float(raw.get("interval_seconds", 1.0)),
            every_n_frames=int(raw.get("every_n_frames", 1)),
            target_count=(
                int(raw["target_count"])
                if raw.get("target_count") is not None
                else None
            ),
            start_seconds=float(raw.get("start_seconds", 0.0)),
            end_seconds=(
                float(raw["end_seconds"])
                if raw.get("end_seconds") is not None
                else None
            ),
            output_format=str(raw.get("output_format", "jpg")).lower(),
            jpeg_qscale=int(raw.get("jpeg_qscale", 2)),
            filename_prefix=str(raw.get("filename_prefix", "frame_")),
            filename_digits=int(raw.get("filename_digits", 6)),
        )


def _parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    numerator, separator, denominator = value.partition("/")
    try:
        result = float(numerator) / float(denominator) if separator else float(value)
    except (ValueError, ZeroDivisionError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def probe_video(video: Path, ffprobe: Path) -> dict[str, Any]:
    """Read stream metadata and per-frame presentation timestamps with ffprobe."""

    video = video.expanduser().resolve()
    ffprobe = ffprobe.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video}")
    if not ffprobe.is_file():
        raise FileNotFoundError(f"ffprobe executable does not exist: {ffprobe}")
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=index,codec_name,width,height,avg_frame_rate,r_frame_rate,"
            "time_base,duration,nb_frames:format=duration:"
            "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,key_frame"
        ),
        "-of",
        "json",
        str(video),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"Video has no readable video stream: {video}")
    stream = streams[0]
    average_fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(
        stream.get("r_frame_rate")
    )
    frames: list[VideoFrame] = []
    for index, raw_frame in enumerate(payload.get("frames", [])):
        timestamp_text = raw_frame.get("best_effort_timestamp_time") or raw_frame.get(
            "pts_time"
        )
        if timestamp_text is None:
            if average_fps is None:
                raise ValueError(
                    f"Frame {index} has no timestamp and stream FPS is unavailable"
                )
            timestamp = index / average_fps
        else:
            timestamp = float(timestamp_text)
        frames.append(
            VideoFrame(
                source_frame_index=index,
                timestamp_seconds=timestamp,
                key_frame=bool(int(raw_frame.get("key_frame", 0))),
            )
        )
    if not frames:
        raise ValueError(f"ffprobe returned no decoded frame records for {video}")
    duration_raw = stream.get("duration") or payload.get("format", {}).get("duration")
    duration = (
        float(duration_raw)
        if duration_raw not in {None, "N/A"}
        else frames[-1].timestamp_seconds
    )
    return {
        "video": str(video),
        "probe_command": command,
        "stream": {
            "codec_name": stream.get("codec_name"),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "average_fps": average_fps,
            "time_base": stream.get("time_base"),
            "duration_seconds": duration,
            "reported_frame_count": (
                int(stream["nb_frames"])
                if stream.get("nb_frames") not in {None, "N/A"}
                else None
            ),
            "probed_frame_count": len(frames),
        },
        "frames": frames,
    }


def select_video_frames(
    frames: list[VideoFrame], config: FrameSamplingConfig
) -> list[VideoFrame]:
    """Select frame records without decoding image pixels."""

    candidates = [
        frame
        for frame in frames
        if frame.timestamp_seconds >= config.start_seconds - 1e-9
        and (
            config.end_seconds is None
            or frame.timestamp_seconds <= config.end_seconds + 1e-9
        )
    ]
    if not candidates:
        raise ValueError("No video frames fall inside the configured time range")
    if config.method == "every_n_frames":
        return candidates[:: config.every_n_frames]
    if config.method == "target_count":
        count = min(config.target_count or 1, len(candidates))
        if count == 1:
            return [candidates[0]]
        indices = [
            round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)
        ]
        return [candidates[index] for index in dict.fromkeys(indices)]

    selected: list[VideoFrame] = []
    next_timestamp = config.start_seconds
    for frame in candidates:
        if frame.timestamp_seconds + 1e-9 >= next_timestamp:
            selected.append(frame)
            skipped_intervals = max(
                1,
                math.floor(
                    (frame.timestamp_seconds - next_timestamp) / config.interval_seconds
                )
                + 1,
            )
            next_timestamp += skipped_intervals * config.interval_seconds
    return selected


def build_ffmpeg_command(
    *,
    ffmpeg: Path,
    video: Path,
    selected: list[VideoFrame],
    output_pattern: Path,
    config: FrameSamplingConfig,
) -> list[str]:
    """Build a frame-index select filter using an argument vector, not a shell."""

    if not selected:
        raise ValueError("At least one frame must be selected")
    select_expression = "+".join(
        f"eq(n\\,{frame.source_frame_index})" for frame in selected
    )
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-i",
        str(video),
        "-map",
        "0:v:0",
        "-vf",
        f"select={select_expression}",
        "-fps_mode",
        "vfr",
        "-start_number",
        "0",
    ]
    if config.output_format == "jpg":
        command.extend(["-q:v", str(config.jpeg_qscale)])
    command.append(str(output_pattern))
    return command


def _is_managed_frame(path: Path, config: FrameSamplingConfig) -> bool:
    if path.suffix.lower() != f".{config.output_format}":
        return False
    stem = path.stem
    if not stem.startswith(config.filename_prefix):
        return False
    index = stem[len(config.filename_prefix) :]
    return len(index) == config.filename_digits and index.isdigit()


def extract_video_frames(
    *,
    video: Path,
    output_dir: Path,
    manifest_path: Path,
    ffmpeg: Path,
    ffprobe: Path,
    config: FrameSamplingConfig,
    overwrite: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Probe, sample, and atomically publish reconstruction frames."""

    video = video.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    ffmpeg = ffmpeg.expanduser().resolve()
    ffprobe = ffprobe.expanduser().resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg executable does not exist: {ffmpeg}")
    if output_dir == video.parent or output_dir.is_relative_to(video):
        raise ValueError("Frame output must not overlap the source video")

    probe = probe_video(video, ffprobe)
    selected = select_video_frames(probe["frames"], config)
    output_pattern_name = (
        f"{config.filename_prefix}%0{config.filename_digits}d.{config.output_format}"
    )
    preview_pattern = output_dir / output_pattern_name
    preview_command = build_ffmpeg_command(
        ffmpeg=ffmpeg,
        video=video,
        selected=selected,
        output_pattern=preview_pattern,
        config=config,
    )
    manifest: dict[str, Any] = {
        "adapter": "colmapcut_recon.video_frame_sampling",
        "dry_run": dry_run,
        "source_video": str(video),
        "source_video_bytes": video.stat().st_size,
        "output_directory": str(output_dir),
        "output_pattern": output_pattern_name,
        "sampling": asdict(config),
        "stream": probe["stream"],
        "selected_frame_count": len(selected),
        "selected_frames": [
            {
                "output_file": (
                    f"{config.filename_prefix}{output_index:0{config.filename_digits}d}."
                    f"{config.output_format}"
                ),
                **asdict(frame),
            }
            for output_index, frame in enumerate(selected)
        ],
        "ffprobe_command": probe["probe_command"],
        "ffmpeg_command": preview_command,
        "ffmpeg_command_text": format_command(preview_command),
    }
    if dry_run:
        return manifest

    existing_frames = (
        [
            path
            for path in output_dir.iterdir()
            if path.is_file() and _is_managed_frame(path, config)
        ]
        if output_dir.is_dir()
        else []
    )
    if (existing_frames or manifest_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Extracted frames or manifest already exist under {output_dir.parent}; "
            "move them aside or pass --overwrite"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".frame-extract-", dir=str(output_dir.parent))
    )
    try:
        temporary_pattern = temporary_root / output_pattern_name
        command = build_ffmpeg_command(
            ffmpeg=ffmpeg,
            video=video,
            selected=selected,
            output_pattern=temporary_pattern,
            config=config,
        )
        run_command(command, record_path=temporary_root / "ffmpeg_process.json")
        generated = sorted(
            path for path in temporary_root.iterdir() if _is_managed_frame(path, config)
        )
        if len(generated) != len(selected):
            raise RuntimeError(
                f"ffmpeg generated {len(generated)} images for {len(selected)} selected frames"
            )
        if overwrite:
            for path in existing_frames:
                path.unlink()
            manifest_path.unlink(missing_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in generated:
            shutil.move(str(path), output_dir / path.name)
        manifest["dry_run"] = False
        manifest["process_record"] = str(manifest_path.with_name("ffmpeg_process.json"))
        process_source = temporary_root / "ffmpeg_process.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(process_source), manifest["process_record"])
        manifest["manifest"] = str(manifest_path)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return manifest


def _find_single_video(video_dir: Path) -> Path:
    videos = (
        sorted(
            path
            for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
        if video_dir.is_dir()
        else []
    )
    if not videos:
        raise FileNotFoundError(f"No supported video found in {video_dir}")
    if len(videos) > 1:
        raise ValueError(
            f"Multiple videos found; select one with --video: {[str(path) for path in videos]}"
        )
    return videos[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=PROJECT_ROOT / "configs/scenes/plant_001.yaml",
    )
    parser.add_argument(
        "--sampling-config",
        type=Path,
        default=PROJECT_ROOT / "configs/preprocessing/video_sampling.yaml",
    )
    parser.add_argument(
        "--tools-config", type=Path, default=PROJECT_ROOT / "configs/tools.local.yaml"
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--method", choices=["interval_seconds", "every_n_frames", "target_count"]
    )
    parser.add_argument("--interval-seconds", type=float)
    parser.add_argument("--every-n-frames", type=int)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--start-seconds", type=float)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_yaml(args.scene_config)
    raw = load_yaml(args.sampling_config)
    overrides = {
        "method": args.method,
        "interval_seconds": args.interval_seconds,
        "every_n_frames": args.every_n_frames,
        "target_count": args.target_count,
        "start_seconds": args.start_seconds,
        "end_seconds": args.end_seconds,
    }
    raw.update({key: value for key, value in overrides.items() if value is not None})
    config = FrameSamplingConfig.from_mapping(raw)
    data_root = resolve_project_path(scene["data_root"])
    video = args.video or _find_single_video(
        data_root / str(raw.get("input_stage", "00_capture/videos"))
    )
    output_dir = args.output_dir or data_root / str(
        raw.get("output_stage", "01_frames/images")
    )
    manifest = args.manifest or output_dir.parent / "frame_manifest.json"
    tool = load_tool("ffmpeg", args.tools_config)
    result = extract_video_frames(
        video=video,
        output_dir=output_dir,
        manifest_path=manifest,
        ffmpeg=Path(tool["executable"]),
        ffprobe=Path(tool["ffprobe"]),
        config=config,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
