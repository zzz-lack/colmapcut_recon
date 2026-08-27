#!/usr/bin/env python3
"""Create an automatic tomato mask from trained SAGA features and colour seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

SH_C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    root = Path("/home/linzz/Desktop/colmapcut_recon")
    runtime = root / "data/scenes/roman_tomato_02/07_saga"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-ply",
        type=Path,
        default=runtime / "point_cloud/iteration_30000/scene_point_cloud.ply",
    )
    parser.add_argument(
        "--feature-ply",
        type=Path,
        default=runtime / "point_cloud/iteration_1000/contrastive_feature_point_cloud.ply",
    )
    parser.add_argument(
        "--scale-gate",
        type=Path,
        default=runtime / "point_cloud/iteration_1000/scale_gate.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=runtime / "tomato_saga_mask.pt",
    )
    parser.add_argument(
        "--seed-region-ply",
        type=Path,
        default=root / "data/scenes/roman_tomato_02/08_asset_assembly/plant_strict.ply",
        help="Restrict automatic colour prompts to an exact XYZ subset of the scene.",
    )
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.70)
    parser.add_argument("--prototype-count", type=int, default=64)
    parser.add_argument("--support-radius-m", type=float, default=0.065)
    parser.add_argument("--red-minimum", type=float, default=0.45)
    parser.add_argument("--red-margin", type=float, default=0.12)
    parser.add_argument("--opacity-minimum", type=float, default=0.10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def _read(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return PlyData.read(str(path), mmap="r")["vertex"].data


def _normalize(values: torch.Tensor) -> torch.Tensor:
    return values / torch.clamp(torch.linalg.vector_norm(values, dim=1, keepdim=True), min=1e-8)


def _xyz_membership(scene_xyz: np.ndarray, subset_vertex: np.ndarray) -> np.ndarray:
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

    def pack(values: np.ndarray) -> np.ndarray:
        result = np.empty(len(values), dtype=dtype)
        result["x"], result["y"], result["z"] = values.T
        return result

    subset_xyz = np.column_stack(
        (subset_vertex["x"], subset_vertex["y"], subset_vertex["z"])
    ).astype(np.float32)
    return np.isin(pack(scene_xyz), pack(subset_xyz))


def _select_prototypes(features: torch.Tensor, count: int) -> torch.Tensor:
    if len(features) <= count:
        return features
    mean = _normalize(features.mean(dim=0, keepdim=True))[0]
    selected = [int(torch.argmax(features @ mean).item())]
    closest = 1.0 - features @ features[selected[0]]
    while len(selected) < count:
        index = int(torch.argmax(closest).item())
        selected.append(index)
        distance = 1.0 - features @ features[index]
        closest = torch.minimum(closest, distance)
    return features[selected]


def _spatial_support(xyz: np.ndarray, seed_mask: np.ndarray, radius: float) -> np.ndarray:
    voxel_size = max(radius / 3.0, 0.005)
    keys = np.floor(xyz / voxel_size).astype(np.int32)
    seed_keys = np.unique(keys[seed_mask], axis=0)
    reach = int(np.ceil(radius / voxel_size))
    offsets = np.array(
        [
            (x, y, z)
            for x in range(-reach, reach + 1)
            for y in range(-reach, reach + 1)
            for z in range(-reach, reach + 1)
            if x * x + y * y + z * z <= reach * reach
        ],
        dtype=np.int32,
    )
    dilated = (seed_keys[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    key_dtype = np.dtype([("x", "<i4"), ("y", "<i4"), ("z", "<i4")])

    def packed(values: np.ndarray) -> np.ndarray:
        result = np.empty(len(values), dtype=key_dtype)
        result["x"], result["y"], result["z"] = values.T
        return result

    return np.isin(packed(keys), np.unique(packed(dilated)))


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.scale <= 1.0:
        raise ValueError("--scale must be in [0, 1]")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be in [0, 1]")
    scene = _read(args.scene_ply)
    feature = _read(args.feature_ply)
    if len(scene) != len(feature):
        raise ValueError(f"Scene has {len(scene)} points but features have {len(feature)}")
    xyz = np.column_stack((scene["x"], scene["y"], scene["z"])).astype(np.float32)
    feature_xyz = np.column_stack((feature["x"], feature["y"], feature["z"])).astype(
        np.float32
    )
    if not np.array_equal(xyz, feature_xyz):
        raise ValueError("Scene and SAGA feature PLY XYZ/order do not match")
    rgb = np.clip(
        0.5
        + SH_C0
        * np.column_stack((scene["f_dc_0"], scene["f_dc_1"], scene["f_dc_2"])),
        0.0,
        1.0,
    )
    opacity = 1.0 / (1.0 + np.exp(-np.clip(scene["opacity"], -30.0, 30.0)))
    seed_mask = (
        (rgb[:, 0] >= args.red_minimum)
        & ((rgb[:, 0] - rgb[:, 1]) >= args.red_margin)
        & ((rgb[:, 0] - rgb[:, 2]) >= args.red_margin)
        & (opacity >= args.opacity_minimum)
    )
    if args.seed_region_ply is not None:
        seed_region = _read(args.seed_region_ply)
        seed_mask &= _xyz_membership(xyz, seed_region)
    if int(seed_mask.sum()) < args.prototype_count:
        raise ValueError("Too few ripe-colour seeds to construct SAGA prototypes")
    feature_names = sorted(
        (name for name in feature.dtype.names or () if name.startswith("f_")),
        key=lambda name: int(name.split("_")[-1]),
    )
    if not feature_names:
        raise ValueError("Feature PLY has no f_N properties")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    features = torch.from_numpy(
        np.column_stack([feature[name] for name in feature_names]).astype(np.float32)
    ).to(device)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, len(feature_names)), torch.nn.Sigmoid()
    ).to(device)
    state = torch.load(args.scale_gate, map_location="cpu", weights_only=True)
    scale_gate.load_state_dict(state)
    scale_gate.eval()
    with torch.no_grad():
        gate = scale_gate(
            torch.tensor([[args.scale]], dtype=torch.float32, device=device)
        )[0]
        gated = _normalize(features * gate[None, :])
        prototypes = _select_prototypes(
            gated[torch.from_numpy(seed_mask).to(device)], args.prototype_count
        )
        score = torch.empty(len(gated), dtype=torch.float32)
        for start in range(0, len(gated), 100_000):
            similarity = gated[start : start + 100_000] @ prototypes.T
            score[start : start + 100_000] = (
                (similarity.max(dim=1).values + 1.0) * 0.5
            ).cpu()
    support = _spatial_support(xyz, seed_mask, args.support_radius_m)
    mask = (score.numpy() >= args.score_threshold) & support & (opacity >= 0.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(mask), args.output)
    np.save(args.output.with_suffix(".npy"), mask)
    report = {
        "scene_ply": str(args.scene_ply.resolve()),
        "feature_ply": str(args.feature_ply.resolve()),
        "scale_gate": str(args.scale_gate.resolve()),
        "output_mask": str(args.output.resolve()),
        "seed_region_ply": (
            str(args.seed_region_ply.resolve()) if args.seed_region_ply else None
        ),
        "gaussians": len(scene),
        "colour_seeds": int(seed_mask.sum()),
        "selected_gaussians": int(mask.sum()),
        "scale": args.scale,
        "score_threshold": args.score_threshold,
        "prototype_count": args.prototype_count,
        "support_radius_m": args.support_radius_m,
        "device": str(device),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved SAGA tomato mask with {int(mask.sum())} / {len(mask)} Gaussians")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
