#!/usr/bin/env python3
"""Extract tomato instances from a SAGA mask and author Isaac Sim entities."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 environments used by SAGA/SuGaR.
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from colmapcut_recon.segmentation.tomato_entities import (
    ExtractionConfig,
    extract_tomato_entities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/segmentation/saga_tomato.toml",
    )
    parser.add_argument("--mask", type=Path, help="SAGA .pt/.npy per-Gaussian mask")
    parser.add_argument("--gaussians", type=Path, help="Override paths.gaussian_ply")
    parser.add_argument(
        "--ground-gaussians", type=Path, help="Override paths.ground_gaussian_ply"
    )
    parser.add_argument(
        "--output-directory", type=Path, help="Override paths.output_directory"
    )
    parser.add_argument(
        "--bootstrap-colour",
        action="store_true",
        help="Use ripe-red Gaussian seeds for commissioning instead of a SAGA mask",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open("rb") as handle:
        raw = tomllib.load(handle)
    paths = raw["paths"]
    extraction = ExtractionConfig(**raw.get("extraction", {}))
    configured_mask = paths.get("saga_mask", "").strip()
    mask = args.mask or (Path(configured_mask) if configured_mask else None)
    if mask is None and not args.bootstrap_colour:
        raise SystemExit(
            "No SAGA mask supplied. Pass --mask MASK.pt, configure paths.saga_mask, "
            "or explicitly use --bootstrap-colour for commissioning."
        )
    manifest = extract_tomato_entities(
        args.gaussians or Path(paths["gaussian_ply"]),
        args.output_directory or Path(paths["output_directory"]),
        saga_mask=mask,
        saga_mask_source_ply=(
            Path(paths["saga_mask_source_ply"])
            if paths.get("saga_mask_source_ply", "").strip()
            else None
        ),
        ground_gaussian_ply=(
            args.ground_gaussians
            or Path(paths["ground_gaussian_ply"])
            if args.ground_gaussians
            or paths.get("ground_gaussian_ply", "").strip()
            else None
        ),
        config=extraction,
        overwrite=args.overwrite,
    )
    stats = manifest["statistics"]
    print(
        f"Generated {stats['entity_count']} tomato entities from "
        f"{stats['seed_gaussians']} selected Gaussians."
    )
    print(manifest["outputs"]["usd"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
