"""Geometrically separate an aligned Gaussian scene for simulation export."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from colmapcut_recon.postprocessing.clean_gaussians import (
    AdaptiveHeightfieldConfig,
    GroundBounds,
    _adaptive_heightfield_ground_mask,
    _finite_xyz,
    _read_vertex_data,
    _sha256,
    _summarize_xyz,
    _validate_gaussian_schema,
    _write_binary_ply,
    _write_json,
)


@dataclass(frozen=True)
class SceneBounds:
    """Metric crop retained as the simulation environment."""

    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    z_min: float = -0.15
    z_max: float = 2.0
    opacity_minimum: float = 0.02

    def validate(self) -> None:
        if not self.x_min < self.x_max or not self.y_min < self.y_max:
            raise ValueError("Scene XY bounds must have positive area")
        if not self.z_min < self.z_max:
            raise ValueError("Scene Z bounds must have positive height")
        if not 0.0 <= self.opacity_minimum <= 1.0:
            raise ValueError("opacity_minimum must be in [0, 1]")


@dataclass(frozen=True)
class SeparationOutputs:
    plant: Path
    ground: Path
    background: Path
    combined: Path
    report: Path


def _activated_opacity(vertex: np.ndarray) -> np.ndarray:
    logits = np.asarray(vertex["opacity"], dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def separate_scene_gaussians(
    full_scene_path: Path,
    outputs: SeparationOutputs,
    *,
    scene_bounds: SceneBounds = SceneBounds(),
    heightfield: AdaptiveHeightfieldConfig = AdaptiveHeightfieldConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Partition one aligned scene into plant, ground, and discarded background."""

    scene_bounds.validate()
    heightfield.validate()
    full_scene_path = full_scene_path.expanduser().resolve()
    output_paths = [
        outputs.plant,
        outputs.ground,
        outputs.background,
        outputs.combined,
        outputs.report,
    ]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise ValueError("Every separation output path must be distinct")
    if not overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(
                "Separation outputs already exist; pass --overwrite: "
                + ", ".join(existing)
            )

    ply, vertex = _read_vertex_data(full_scene_path)
    _validate_gaussian_schema(vertex, full_scene_path)
    finite = _finite_xyz(vertex)
    if not np.all(finite):
        raise ValueError(
            f"Full Gaussian scene contains {int((~finite).sum())} non-finite records"
        )
    opacity = _activated_opacity(vertex)
    retained = (
        (vertex["x"] >= scene_bounds.x_min)
        & (vertex["x"] <= scene_bounds.x_max)
        & (vertex["y"] >= scene_bounds.y_min)
        & (vertex["y"] <= scene_bounds.y_max)
        & (vertex["z"] >= scene_bounds.z_min)
        & (vertex["z"] <= scene_bounds.z_max)
        & (opacity >= scene_bounds.opacity_minimum)
    )

    ground_bounds = GroundBounds(
        x_min=scene_bounds.x_min,
        x_max=scene_bounds.x_max,
        y_min=scene_bounds.y_min,
        y_max=scene_bounds.y_max,
        z_min=scene_bounds.z_min,
        z_max=scene_bounds.z_max,
    )
    ground_mask, ground_diagnostics = _adaptive_heightfield_ground_mask(
        vertex, ground_bounds, heightfield
    )
    ground_mask &= retained
    plant_mask = retained & ~ground_mask
    background_mask = ~retained
    if not np.any(plant_mask):
        raise ValueError("Geometric separation selected no plant Gaussians")
    if not np.any(ground_mask):
        raise ValueError("Geometric separation selected no ground Gaussians")

    plant = np.array(vertex[plant_mask], dtype=vertex.dtype, copy=True)
    ground = np.array(vertex[ground_mask], dtype=vertex.dtype, copy=True)
    background = np.array(vertex[background_mask], dtype=vertex.dtype, copy=True)
    combined = np.empty(len(plant) + len(ground), dtype=vertex.dtype)
    combined[: len(plant)] = plant
    combined[len(plant) :] = ground

    _write_binary_ply(outputs.plant, plant, overwrite)
    _write_binary_ply(outputs.ground, ground, overwrite)
    _write_binary_ply(outputs.background, background, overwrite)
    _write_binary_ply(outputs.combined, combined, overwrite)
    report: dict[str, Any] = {
        "operation": "geometric_gaussian_scene_separation",
        "coordinate_system": {"unit": "meter", "up_axis": "Z"},
        "input": {
            "path": str(full_scene_path),
            "bytes": full_scene_path.stat().st_size,
            "sha256": _sha256(full_scene_path),
            "gaussians": int(len(vertex)),
        },
        "configuration": {
            "scene_bounds": asdict(scene_bounds),
            "heightfield": asdict(heightfield),
        },
        "ground_estimation": ground_diagnostics,
        "outputs": {
            "plant": _summarize_xyz(plant),
            "ground": _summarize_xyz(ground),
            "background": _summarize_xyz(background),
            "combined": _summarize_xyz(combined),
        },
        "partition": {
            "retained_gaussians": int(retained.sum()),
            "discarded_background_gaussians": int(background_mask.sum()),
            "retained_fraction": float(retained.mean()),
            "disjoint": not bool(np.any(plant_mask & ground_mask)),
        },
        "limitations": [
            "Geometric separation is a post-training fallback, not semantic segmentation.",
            "Objects inside the crop but above the fitted ground remain in the plant component.",
        ],
    }
    for label, path in (
        ("plant", outputs.plant),
        ("ground", outputs.ground),
        ("background", outputs.background),
        ("combined", outputs.combined),
    ):
        report["outputs"][label].update(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(outputs.report, report, overwrite)
    del ply
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-scene", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--y-min", type=float, default=-1.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=-0.15)
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--opacity-minimum", type=float, default=0.02)
    parser.add_argument("--grid-size", type=float, default=0.04)
    parser.add_argument("--initial-ground-height", type=float, default=0.0)
    parser.add_argument("--surface-quantile", type=float, default=0.15)
    parser.add_argument("--max-slope-degrees", type=float, default=40.0)
    parser.add_argument("--seed-tolerance-steps", type=float, default=2.0)
    parser.add_argument("--smoothing-iterations", type=int, default=3)
    parser.add_argument("--ground-band-quantile", type=float, default=0.90)
    parser.add_argument("--ground-band-mad-multiplier", type=float, default=3.0)
    parser.add_argument("--gaussian-sigma-multiplier", type=float, default=5.0)
    parser.add_argument("--max-sigma-to-band-ratio", type=float, default=2.0)
    parser.add_argument("--min-points-per-cell", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_directory.expanduser().resolve()
    report = separate_scene_gaussians(
        args.full_scene,
        SeparationOutputs(
            plant=output / "plant_gaussians.ply",
            ground=output / "ground_gaussians.ply",
            background=output / "discarded_background_gaussians.ply",
            combined=output / "plant_and_ground_gaussians.ply",
            report=output / "separation_report.json",
        ),
        scene_bounds=SceneBounds(
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
            z_min=args.z_min,
            z_max=args.z_max,
            opacity_minimum=args.opacity_minimum,
        ),
        heightfield=AdaptiveHeightfieldConfig(
            grid_size_m=args.grid_size,
            initial_ground_height_m=args.initial_ground_height,
            surface_quantile=args.surface_quantile,
            max_slope_degrees=args.max_slope_degrees,
            seed_tolerance_steps=args.seed_tolerance_steps,
            smoothing_iterations=args.smoothing_iterations,
            ground_band_quantile=args.ground_band_quantile,
            ground_band_mad_multiplier=args.ground_band_mad_multiplier,
            gaussian_sigma_multiplier=args.gaussian_sigma_multiplier,
            max_sigma_to_band_ratio=args.max_sigma_to_band_ratio,
            min_points_per_cell=args.min_points_per_cell,
        ),
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
