"""Metric-align a COLMAP reconstruction from coplanar AprilTags.

Tag corners are detected in registered images and triangulated using the
existing COLMAP camera poses. Known black-square edge lengths determine scale,
the tag plane determines +Z, and two configured tag centers determine +X.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from colmapcut_recon.common.config import PROJECT_ROOT, load_yaml, resolve_project_path
from colmapcut_recon.geometry.similarity_transform import SimilarityTransform

FAMILY_ATTRIBUTES = {
    "16h5": "DICT_APRILTAG_16h5",
    "25h9": "DICT_APRILTAG_25h9",
    "36h10": "DICT_APRILTAG_36h10",
    "36h11": "DICT_APRILTAG_36h11",
}


@dataclass(frozen=True)
class Observation:
    image_id: int
    image_name: str
    camera_id: int
    uv: np.ndarray
    normalized: np.ndarray
    projection_normalized: np.ndarray


@dataclass(frozen=True)
class AprilTagAlignmentConfig:
    family: str
    tag_ids: tuple[int, ...]
    tag_sizes_m: dict[int, float]
    x_axis_tags: tuple[int, int]
    reprojection_threshold_px: float = 3.0
    ransac_trials: int = 1200
    min_observations: int = 5
    diagnostic_images: int = 6

    def __post_init__(self) -> None:
        if self.family not in FAMILY_ATTRIBUTES:
            raise ValueError(f"Unsupported AprilTag family: {self.family}")
        if not self.tag_ids or len(set(self.tag_ids)) != len(self.tag_ids):
            raise ValueError("tag_ids must be non-empty and contain no duplicates")
        missing_sizes = sorted(set(self.tag_ids) - set(self.tag_sizes_m))
        if missing_sizes:
            raise ValueError(
                f"No physical black-square size for tag IDs: {missing_sizes}"
            )
        if any(
            not np.isfinite(size) or size <= 0 for size in self.tag_sizes_m.values()
        ):
            raise ValueError("All tag sizes must be finite positive metres")
        if len(self.x_axis_tags) != 2 or self.x_axis_tags[0] == self.x_axis_tags[1]:
            raise ValueError("x_axis_tags must contain two different IDs")
        if not set(self.x_axis_tags).issubset(self.tag_ids):
            raise ValueError("x_axis_tags must both occur in tag_ids")
        if self.reprojection_threshold_px <= 0:
            raise ValueError("reprojection_threshold_px must be positive")
        if self.ransac_trials < 1:
            raise ValueError("ransac_trials must be positive")
        if self.min_observations < 2:
            raise ValueError("min_observations must be at least 2")
        if self.diagnostic_images < 0:
            raise ValueError("diagnostic_images must be non-negative")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> AprilTagAlignmentConfig:
        tag_ids = tuple(int(value) for value in raw.get("tag_ids", [0, 1, 2, 3]))
        common_size = raw.get("tag_size_m")
        sizes = {
            int(key): float(value) for key, value in raw.get("tag_sizes_m", {}).items()
        }
        if common_size is not None:
            sizes = {tag_id: float(common_size) for tag_id in tag_ids} | sizes
        return cls(
            family=str(raw.get("family", "36h11")),
            tag_ids=tag_ids,
            tag_sizes_m=sizes,
            x_axis_tags=tuple(int(value) for value in raw.get("x_axis_tags", [3, 0])),
            reprojection_threshold_px=float(raw.get("reprojection_threshold_px", 3.0)),
            ransac_trials=int(raw.get("ransac_trials", 1200)),
            min_observations=int(raw.get("min_observations", 5)),
            diagnostic_images=int(raw.get("diagnostic_images", 6)),
        )


def _runtime() -> tuple[Any, Any]:
    try:
        import cv2
        import pycolmap
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AprilTag alignment requires OpenCV contrib and pycolmap. "
            "Run `uv run scripts/05_align_scale_axes.py ...` or install the alignment extra."
        ) from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV was installed without the contrib aruco module")
    return cv2, pycolmap


def projection_and_center(image: Any) -> tuple[np.ndarray, np.ndarray]:
    cam_from_world = image.cam_from_world()
    if cam_from_world is None:
        raise ValueError(f"COLMAP image {image.name} has no pose")
    matrix = np.asarray(cam_from_world.matrix(), dtype=np.float64)
    center = np.asarray(cam_from_world.inverse().translation, dtype=np.float64)
    return matrix, center


def detect_tags(
    reconstruction: Any,
    image_dir: Path,
    family: str,
    tag_ids: set[int],
    cv2: Any,
) -> tuple[
    dict[tuple[int, int], list[Observation]], dict[int, int], list[dict[str, Any]]
]:
    dictionary_id = getattr(cv2.aruco, FAMILY_ATTRIBUTES[family])
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    observations: dict[tuple[int, int], list[Observation]] = defaultdict(list)
    counts: dict[int, int] = defaultdict(int)
    detections: list[dict[str, Any]] = []
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)

    for index, image in enumerate(images, start=1):
        path = image_dir / image.name
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Could not read registered image: {path}")
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            if index % 100 == 0:
                print(f"detect: {index}/{len(images)} images", flush=True)
            continue
        projection, _ = projection_and_center(image)
        camera = reconstruction.cameras[image.camera_id]
        image_detections: list[dict[str, Any]] = []
        for marker_corners, marker_id_raw in zip(corners, ids.ravel(), strict=True):
            marker_id = int(marker_id_raw)
            if marker_id not in tag_ids:
                continue
            marker_corners = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            counts[marker_id] += 1
            image_detections.append(
                {"tag_id": marker_id, "corners_px": marker_corners.tolist()}
            )
            for corner_index, uv in enumerate(marker_corners):
                observation = Observation(
                    image_id=image.image_id,
                    image_name=image.name,
                    camera_id=image.camera_id,
                    uv=uv,
                    normalized=np.asarray(camera.cam_from_img(uv), dtype=np.float64),
                    projection_normalized=projection.copy(),
                )
                observation.uv.setflags(write=False)
                observations[(marker_id, corner_index)].append(observation)
        if image_detections:
            detections.append({"image": image.name, "markers": image_detections})
        if index % 100 == 0:
            print(f"detect: {index}/{len(images)} images", flush=True)
    return observations, dict(counts), detections


def dlt_triangulate(
    observations: list[Observation], indices: np.ndarray
) -> np.ndarray | None:
    rows = []
    for observation_index in indices:
        observation = observations[int(observation_index)]
        x, y = observation.normalized
        projection = observation.projection_normalized
        rows.extend(
            (x * projection[2] - projection[0], y * projection[2] - projection[1])
        )
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[0] < 4:
        return None
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    homogeneous = vh[-1]
    if abs(homogeneous[3]) < 1e-12:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.all(np.isfinite(point)) else None


def reprojection_errors(
    point: np.ndarray, observations: list[Observation], reconstruction: Any
) -> np.ndarray:
    errors = np.full(len(observations), np.inf, dtype=np.float64)
    point_h = np.r_[point, 1.0]
    for index, observation in enumerate(observations):
        point_camera = observation.projection_normalized @ point_h
        if point_camera[2] <= 1e-9:
            continue
        prediction = reconstruction.cameras[observation.camera_id].img_from_cam(
            point_camera
        )
        errors[index] = np.linalg.norm(np.asarray(prediction) - observation.uv)
    return errors


def ray_angle_degrees(
    first: Observation, second: Observation, reconstruction: Any
) -> float:
    first_pose = reconstruction.images[first.image_id].cam_from_world()
    second_pose = reconstruction.images[second.image_id].cam_from_world()
    first_ray = first_pose.rotation.inverse() * np.r_[first.normalized, 1.0]
    second_ray = second_pose.rotation.inverse() * np.r_[second.normalized, 1.0]
    first_ray = np.asarray(first_ray) / np.linalg.norm(first_ray)
    second_ray = np.asarray(second_ray) / np.linalg.norm(second_ray)
    cosine = float(np.clip(abs(first_ray @ second_ray), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def robust_triangulate(
    observations: list[Observation],
    reconstruction: Any,
    threshold_px: float,
    trials: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(observations) < 2:
        raise ValueError("A corner needs observations in at least two images")
    rng = np.random.default_rng(seed)
    best_inliers: np.ndarray | None = None
    best_key = (-1, -np.inf)
    candidates = np.arange(len(observations))
    for _ in range(trials):
        pair = rng.choice(candidates, size=2, replace=False)
        if observations[int(pair[0])].image_id == observations[int(pair[1])].image_id:
            continue
        if (
            ray_angle_degrees(
                observations[int(pair[0])], observations[int(pair[1])], reconstruction
            )
            < 1.0
        ):
            continue
        point = dlt_triangulate(observations, pair)
        if point is None:
            continue
        errors = reprojection_errors(point, observations, reconstruction)
        inliers = np.flatnonzero(errors <= threshold_px)
        median_error = float(np.median(errors[inliers])) if len(inliers) else np.inf
        key = (len(inliers), -median_error)
        if key > best_key:
            best_key, best_inliers = key, inliers
    if best_inliers is None or len(best_inliers) < 2:
        raise ValueError("RANSAC could not find a valid triangulation")
    for _ in range(5):
        point = dlt_triangulate(observations, best_inliers)
        if point is None:
            raise ValueError("Degenerate triangulation")
        errors = reprojection_errors(point, observations, reconstruction)
        updated = np.flatnonzero(errors <= threshold_px)
        if np.array_equal(updated, best_inliers) or len(updated) < 2:
            break
        best_inliers = updated
    point = dlt_triangulate(observations, best_inliers)
    if point is None:
        raise ValueError("Degenerate final triangulation")
    return point, best_inliers, reprojection_errors(point, observations, reconstruction)


def calculate_alignment(
    corners: dict[int, np.ndarray],
    sizes_m: dict[int, float],
    camera_centers: np.ndarray,
    x_axis_tags: tuple[int, int],
) -> tuple[SimilarityTransform, dict[str, Any]]:
    """Estimate the metric right-handed target frame from triangulated tag corners."""

    centers = {tag_id: value.mean(axis=0) for tag_id, value in corners.items()}
    all_corners = np.concatenate(list(corners.values()), axis=0)
    origin = np.mean(np.stack(list(centers.values())), axis=0)
    centered = all_corners - origin
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    z_axis = vh[-1]
    if np.median((camera_centers - origin) @ z_axis) < 0:
        z_axis = -z_axis
    z_axis /= np.linalg.norm(z_axis)
    x_from, x_to = x_axis_tags
    if x_from not in centers or x_to not in centers:
        raise ValueError(f"x-axis tag IDs must be triangulated: {sorted(centers)}")
    x_axis = centers[x_to] - centers[x_from]
    x_axis -= z_axis * (x_axis @ z_axis)
    if np.linalg.norm(x_axis) < 1e-9:
        raise ValueError("The selected x-axis tag centers are coincident")
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.stack((x_axis, y_axis, z_axis), axis=0)

    samples: list[float] = []
    edge_records: list[dict[str, Any]] = []
    for tag_id, tag_corners in corners.items():
        for corner_index in range(4):
            model_length = float(
                np.linalg.norm(
                    tag_corners[(corner_index + 1) % 4] - tag_corners[corner_index]
                )
            )
            sample = sizes_m[tag_id] / model_length
            samples.append(sample)
            edge_records.append(
                {
                    "tag_id": tag_id,
                    "edge": corner_index,
                    "model_length": model_length,
                    "scale_m_per_model_unit": sample,
                }
            )
    sample_array = np.asarray(samples)
    scale = float(np.median(sample_array))
    transform = SimilarityTransform(scale, rotation, -scale * rotation @ origin)
    diagnostics = {
        "edge_scale_samples": edge_records,
        "scale_sample_median": scale,
        "scale_sample_min": float(sample_array.min()),
        "scale_sample_max": float(sample_array.max()),
        "scale_sample_relative_mad": float(
            np.median(np.abs(sample_array - scale)) / scale
        ),
        "ground_plane_rms_m": float(np.sqrt(np.mean((centered @ z_axis) ** 2)) * scale),
        "ground_plane_singular_values_model": singular_values.tolist(),
        "tag_centers_model": {
            str(key): value.tolist() for key, value in centers.items()
        },
    }
    return transform, diagnostics


def annotate_detections(
    image_dir: Path,
    detections: list[dict[str, Any]],
    output_dir: Path,
    limit: int,
    cv2: Any,
) -> None:
    if limit <= 0:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in detections[:limit]:
        image = cv2.imread(str(image_dir / record["image"]))
        if image is None:
            continue
        for marker in record["markers"]:
            corners = np.rint(np.asarray(marker["corners_px"])).astype(np.int32)
            cv2.polylines(image, [corners], True, (0, 255, 0), 4, cv2.LINE_AA)
            center = tuple(np.rint(corners.mean(axis=0)).astype(int))
            cv2.putText(
                image,
                f"id={marker['tag_id']}",
                center,
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
        destination = output_dir / record["image"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(destination), image)


def _model_exists(model_dir: Path) -> bool:
    return any(
        (model_dir / f"cameras{suffix}").is_file() for suffix in (".bin", ".txt")
    )


def _prepare_output(output_root: Path, overwrite: bool) -> Path:
    managed = [
        output_root / "sparse" / "0",
        output_root / "alignment_report.json",
        output_root / "tag_detections.json",
        output_root / "tag_diagnostics",
    ]
    existing = [
        path
        for path in managed
        if path.exists()
        and (
            not path.is_dir()
            or any(child.name != ".gitkeep" for child in path.iterdir())
        )
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Alignment outputs already exist: {[str(path) for path in existing]}. "
            "Move them aside or pass --overwrite."
        )
    if overwrite:
        for path in managed:
            if path.is_symlink() or path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path)
    model_output = output_root / "sparse" / "0"
    model_output.mkdir(parents=True, exist_ok=True)
    return model_output


def align_colmap_apriltags(
    *,
    model_dir: Path,
    image_dir: Path,
    output_root: Path,
    config: AprilTagAlignmentConfig,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Detect, estimate, transform, and write a metric COLMAP reconstruction."""

    cv2, pycolmap = _runtime()
    model_dir = model_dir.expanduser().resolve()
    image_dir = image_dir.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not _model_exists(model_dir):
        raise FileNotFoundError(f"No COLMAP model found in {model_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Registered image directory does not exist: {image_dir}"
        )
    if (
        output_root == model_dir
        or output_root.is_relative_to(model_dir)
        or model_dir.is_relative_to(output_root)
        or output_root == image_dir
        or output_root.is_relative_to(image_dir)
        or image_dir.is_relative_to(output_root)
    ):
        raise ValueError("Alignment output must not overlap the input model or images")

    reconstruction = pycolmap.Reconstruction(model_dir)
    print(reconstruction.summary(), flush=True)
    observations, counts, detections = detect_tags(
        reconstruction, image_dir, config.family, set(config.tag_ids), cv2
    )
    missing = sorted(set(config.tag_ids) - set(counts))
    if missing:
        raise RuntimeError(f"No detections for required tag IDs: {missing}")
    corners: dict[int, np.ndarray] = {}
    triangulation_report: dict[str, dict[str, Any]] = {}
    for tag_id in config.tag_ids:
        triangulated = []
        for corner_index in range(4):
            corner_observations = observations[(tag_id, corner_index)]
            if len(corner_observations) < config.min_observations:
                raise RuntimeError(
                    f"Tag {tag_id} corner {corner_index} has only "
                    f"{len(corner_observations)} observations"
                )
            point, inliers, errors = robust_triangulate(
                corner_observations,
                reconstruction,
                config.reprojection_threshold_px,
                config.ransac_trials,
                seed=tag_id * 4 + corner_index,
            )
            triangulated.append(point)
            inlier_errors = errors[inliers]
            triangulation_report[f"{tag_id}:{corner_index}"] = {
                "point_model": point.tolist(),
                "observations": len(corner_observations),
                "inliers": len(inliers),
                "reprojection_median_px": float(np.median(inlier_errors)),
                "reprojection_max_px": float(np.max(inlier_errors)),
            }
        corners[tag_id] = np.asarray(triangulated)
    camera_centers = np.stack(
        [projection_and_center(image)[1] for image in reconstruction.images.values()]
    )
    transform, alignment_diagnostics = calculate_alignment(
        corners, config.tag_sizes_m, camera_centers, config.x_axis_tags
    )

    model_output = _prepare_output(output_root, overwrite)
    reconstruction.transform(
        pycolmap.Sim3d(
            transform.scale,
            pycolmap.Rotation3d(transform.rotation),
            transform.translation,
        )
    )
    reconstruction.write_binary(model_output)
    reconstruction.write_text(model_output)
    if (model_dir / "project.ini").is_file():
        shutil.copy2(model_dir / "project.ini", model_output / "project.ini")

    transformed_corners = {
        str(tag_id): transform.apply(value).tolist()
        for tag_id, value in corners.items()
    }
    centers_metric = [
        np.mean(np.asarray(value), axis=0) for value in transformed_corners.values()
    ]
    metric_edges = []
    for value in transformed_corners.values():
        value_array = np.asarray(value)
        metric_edges.extend(
            float(np.linalg.norm(value_array[(index + 1) % 4] - value_array[index]))
            for index in range(4)
        )
    report: dict[str, Any] = {
        "adapter": "colmapcut_recon.apriltag_alignment",
        "input_model": str(model_dir),
        "input_images": str(image_dir),
        "output_model": str(model_output),
        "coordinate_units": "metres",
        "tag_family": config.family,
        "tag_sizes_m": {str(key): value for key, value in config.tag_sizes_m.items()},
        "detections_per_tag": {str(key): value for key, value in counts.items()},
        "images_with_detections": len(detections),
        "x_axis_tags": list(config.x_axis_tags),
        "scale_m_per_model_unit": transform.scale,
        "rotation_new_from_old": transform.rotation.tolist(),
        "translation_new_from_old_m": transform.translation.tolist(),
        "transform_new_from_old": transform.matrix.tolist(),
        "triangulation": triangulation_report,
        "alignment_diagnostics": alignment_diagnostics,
        "tag_corners_metric": transformed_corners,
        "validation": {
            "mean_of_tag_centers_m": np.mean(np.stack(centers_metric), axis=0).tolist(),
            "metric_edge_length_median": float(np.median(metric_edges)),
            "metric_edge_length_min": float(np.min(metric_edges)),
            "metric_edge_length_max": float(np.max(metric_edges)),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "tag_detections.json").write_text(
        json.dumps(detections, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    transform_record = transform.to_dict() | {
        "source_frame": "colmap_world",
        "target_frame": "plant_world",
        "unit": "meter",
        "up_axis": "Z",
        "method": "coplanar_apriltags",
        "alignment_report": str(output_root / "alignment_report.json"),
    }
    (output_root / "transform.json").write_text(
        json.dumps(transform_record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    annotate_detections(
        image_dir,
        detections,
        output_root / "tag_diagnostics",
        config.diagnostic_images,
        cv2,
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=PROJECT_ROOT / "configs/scenes/plant_001.yaml",
    )
    parser.add_argument(
        "--alignment-config",
        type=Path,
        default=PROJECT_ROOT / "configs/alignment/default.yaml",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--family", choices=sorted(FAMILY_ATTRIBUTES))
    parser.add_argument("--tag-ids", type=int, nargs="+")
    parser.add_argument("--tag-size-m", type=float)
    parser.add_argument("--tag-size", action="append", default=[], metavar="ID=METERS")
    parser.add_argument("--x-axis-tags", type=int, nargs=2, metavar=("FROM", "TO"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _parse_tag_sizes(values: list[str]) -> dict[int, float]:
    result: dict[int, float] = {}
    for value in values:
        try:
            tag_id_text, size_text = value.split("=", 1)
            result[int(tag_id_text)] = float(size_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --tag-size '{value}'; expected ID=METERS"
            ) from exc
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = load_yaml(args.scene_config)
    raw = load_yaml(args.alignment_config)
    if raw.get("method") not in {None, "apriltag"}:
        raise ValueError(f"Unsupported alignment method: {raw.get('method')}")
    april = dict(raw.get("apriltag", {}))
    if args.family:
        april["family"] = args.family
    if args.tag_ids:
        april["tag_ids"] = args.tag_ids
    if args.tag_size_m is not None:
        april["tag_size_m"] = args.tag_size_m
    if args.tag_size:
        april["tag_sizes_m"] = {
            **april.get("tag_sizes_m", {}),
            **_parse_tag_sizes(args.tag_size),
        }
    if args.x_axis_tags:
        april["x_axis_tags"] = args.x_axis_tags
    config = AprilTagAlignmentConfig.from_mapping(april)
    data_root = resolve_project_path(scene["data_root"])
    model = args.model or data_root / str(
        raw.get("input_model_stage", "04_colmap_foreground/sparse/0")
    )
    images = args.images or data_root / str(raw.get("image_stage", "01_frames/images"))
    output = args.output_root or data_root / str(
        raw.get("output_stage", "05_alignment")
    )
    plan = {
        "method": "coplanar_apriltags",
        "input_model": str(model.resolve()),
        "input_images": str(images.resolve()),
        "output_root": str(output.resolve()),
        "tag_family": config.family,
        "tag_sizes_m": config.tag_sizes_m,
        "x_axis_tags": config.x_axis_tags,
    }
    if args.dry_run:
        if not _model_exists(model):
            raise FileNotFoundError(f"No COLMAP model found in {model}")
        if not images.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {images}")
        _runtime()
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    report = align_colmap_apriltags(
        model_dir=model,
        image_dir=images,
        output_root=output,
        config=config,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {"output_model": report["output_model"], **plan},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
