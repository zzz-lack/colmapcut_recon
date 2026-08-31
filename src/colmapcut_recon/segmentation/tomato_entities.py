"""Turn a SAGA Gaussian mask into independent tomato simulation entities.

SAGA produces a per-Gaussian selection, while a simulator needs individual,
closed collision proxies.  This module separates the selection spatially,
fits robust oriented ellipsoids, removes the selected volumes from the static
Gaussian plant, and writes an Isaac Sim compatible USDA layer.

For commissioning before a trained SAGA feature field exists, a conservative
ripe-red colour seed can be used.  The output manifest records which source was
used so a colour bootstrap can never be mistaken for a SAGA result.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class ExtractionConfig:
    """Geometry and quality thresholds expressed in metres."""

    voxel_size_m: float = 0.008
    minimum_seed_points: int = 18
    minimum_entity_points: int = 40
    maximum_component_span_m: float = 0.115
    minimum_radius_m: float = 0.018
    maximum_radius_m: float = 0.060
    ellipsoid_padding_m: float = 0.008
    removal_scale: float = 1.12
    opacity_minimum: float = 0.10
    red_minimum: float = 0.45
    red_margin: float = 0.12
    minimum_height_m: float = 0.015
    tomato_density_kg_m3: float = 850.0
    initially_kinematic: bool = True
    saga_require_ripe_colour_seed: bool = True
    mesh_latitude_segments: int = 12
    mesh_longitude_segments: int = 24

    def validate(self) -> None:
        positive = {
            "voxel_size_m": self.voxel_size_m,
            "minimum_seed_points": self.minimum_seed_points,
            "minimum_entity_points": self.minimum_entity_points,
            "maximum_component_span_m": self.maximum_component_span_m,
            "minimum_radius_m": self.minimum_radius_m,
            "maximum_radius_m": self.maximum_radius_m,
            "ellipsoid_padding_m": self.ellipsoid_padding_m,
            "removal_scale": self.removal_scale,
            "tomato_density_kg_m3": self.tomato_density_kg_m3,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Values must be positive: {', '.join(invalid)}")
        if self.maximum_radius_m <= self.minimum_radius_m:
            raise ValueError("maximum_radius_m must exceed minimum_radius_m")
        if not 0.0 <= self.opacity_minimum <= 1.0:
            raise ValueError("opacity_minimum must be in [0, 1]")
        if self.mesh_latitude_segments < 4:
            raise ValueError("mesh_latitude_segments must be at least 4")
        if self.mesh_longitude_segments < 8:
            raise ValueError("mesh_longitude_segments must be at least 8")


@dataclass
class TomatoEntity:
    identifier: str
    center_m: list[float]
    radii_m: list[float]
    orientation_wxyz: list[float]
    stem_anchor_m: list[float]
    mass_kg: float
    seed_gaussians: int
    entity_gaussians: int
    mean_rgb: list[float]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def _load_vertex(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian PLY does not exist: {path}")
    ply = PlyData.read(str(path), mmap="r")
    try:
        vertex = ply["vertex"].data
    except KeyError as exc:
        raise ValueError(f"PLY has no vertex element: {path}") from exc
    required = {
        "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - set(vertex.dtype.names or ()))
    if missing:
        raise ValueError(f"Gaussian PLY is missing properties: {', '.join(missing)}")
    return vertex


def _xyz_rgb_opacity(vertex: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32, copy=False
    )
    sh_dc = np.column_stack(
        (vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"])
    ).astype(np.float32, copy=False)
    rgb = np.clip(0.5 + SH_C0 * sh_dc, 0.0, 1.0)
    opacity = _sigmoid(vertex["opacity"])
    finite = np.all(np.isfinite(xyz), axis=1) & np.all(np.isfinite(rgb), axis=1)
    if not np.all(finite):
        raise ValueError(f"Input contains {int((~finite).sum())} non-finite Gaussians")
    return xyz, rgb, opacity


def _unwrap_mask(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("mask", "masks", "scores", "selection"):
            if key in value:
                return value[key]
        raise ValueError("Mask dictionary has none of: mask, masks, scores, selection")
    return value


def load_gaussian_mask(path: Path, count: int, threshold: float = 0.5) -> np.ndarray:
    """Load a SAGA GUI mask from PT or a portable NPY/NPZ file."""

    if not path.is_file():
        raise FileNotFoundError(f"SAGA mask does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        key = "mask" if "mask" in archive.files else archive.files[0]
        value = archive[key]
    elif suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to read a SAGA .pt mask") from exc
        value = torch.load(path, map_location="cpu", weights_only=False)
        value = _unwrap_mask(value)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    else:
        raise ValueError(f"Unsupported mask extension: {path.suffix}")
    value = np.asarray(_unwrap_mask(value))
    if value.ndim == 2 and value.shape[-1] == count:
        value = np.any(value > threshold, axis=0)
    value = np.squeeze(value)
    if value.ndim != 1 or value.shape[0] != count:
        raise ValueError(
            f"Mask has shape {value.shape}, but the Gaussian PLY has {count} vertices"
        )
    return value if value.dtype == bool else value > threshold


def _xyz_keys(vertex: np.ndarray) -> np.ndarray:
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    keys = np.empty(len(vertex), dtype=dtype)
    keys["x"] = np.asarray(vertex["x"], dtype=np.float32)
    keys["y"] = np.asarray(vertex["y"], dtype=np.float32)
    keys["z"] = np.asarray(vertex["z"], dtype=np.float32)
    return keys


def _map_mask_by_exact_xyz(
    mask: np.ndarray,
    source_vertex: np.ndarray,
    target_vertex: np.ndarray,
) -> np.ndarray:
    """Map a full-scene SAGA mask onto an exact-subset plant PLY."""

    selected_keys = _xyz_keys(source_vertex)[mask]
    if not len(selected_keys):
        return np.zeros(len(target_vertex), dtype=bool)
    return np.isin(_xyz_keys(target_vertex), selected_keys)


def _colour_seed(
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    config: ExtractionConfig,
) -> np.ndarray:
    return (
        (rgb[:, 0] >= config.red_minimum)
        & ((rgb[:, 0] - rgb[:, 1]) >= config.red_margin)
        & ((rgb[:, 0] - rgb[:, 2]) >= config.red_margin)
        & (opacity >= config.opacity_minimum)
        & (xyz[:, 2] >= config.minimum_height_m)
    )


def _voxel_components(points: np.ndarray, voxel_size: float) -> list[np.ndarray]:
    if len(points) == 0:
        return []
    keys = np.floor(points / voxel_size).astype(np.int64)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    key_to_voxel = {tuple(key): index for index, key in enumerate(unique)}
    visited = np.zeros(len(unique), dtype=bool)
    voxel_components: list[list[int]] = []
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    for start in range(len(unique)):
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[int] = deque([start])
        component: list[int] = []
        while queue:
            voxel_index = queue.popleft()
            component.append(voxel_index)
            key = unique[voxel_index]
            for offset in offsets:
                neighbor = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
                neighbor_index = key_to_voxel.get(neighbor)
                if neighbor_index is not None and not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    queue.append(neighbor_index)
        voxel_components.append(component)
    return [
        np.flatnonzero(np.isin(inverse, component))
        for component in voxel_components
    ]


def _robust_span(points: np.ndarray) -> np.ndarray:
    low, high = np.quantile(points, (0.03, 0.97), axis=0)
    return high - low


def _split_oversized(
    points: np.ndarray,
    indices: np.ndarray,
    maximum_span: float,
    minimum_points: int,
) -> list[np.ndarray]:
    pending = [indices]
    output: list[np.ndarray] = []
    while pending:
        current = pending.pop()
        cloud = points[current]
        span = _robust_span(cloud)
        if float(span.max()) <= maximum_span or len(current) < 2 * minimum_points:
            output.append(current)
            continue
        axis = int(np.argmax(span))
        median = float(np.median(cloud[:, axis]))
        left = current[cloud[:, axis] <= median]
        right = current[cloud[:, axis] > median]
        if len(left) < minimum_points or len(right) < minimum_points:
            output.append(current)
        else:
            pending.extend((left, right))
    return output


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to wxyz."""

    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s,
                      (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s,
                          (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif axis == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
                          0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
                          (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q.astype(np.float32)


def _fit_ellipsoid(
    points: np.ndarray, config: ExtractionConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, axes = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    axes = axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = centered @ axes
    low, high = np.quantile(local, (0.03, 0.97), axis=0)
    local_center = (low + high) * 0.5
    center = center + axes @ local_center
    radii = (high - low) * 0.5 + config.ellipsoid_padding_m
    radii = np.clip(radii, config.minimum_radius_m, config.maximum_radius_m)
    return center.astype(np.float32), radii.astype(np.float32), axes.astype(np.float32)


def _inside_ellipsoid(
    xyz: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    axes: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    local = (xyz - center) @ axes
    normalized = local / np.maximum(radii * scale, 1e-6)
    return np.einsum("ij,ij->i", normalized, normalized) <= 1.0


def _write_vertex_ply(path: Path, records: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    element = PlyElement.describe(np.asarray(records), "vertex")
    PlyData([element], text=False).write(str(path))


def _ellipsoid_mesh(
    radii: np.ndarray | list[float], latitude_segments: int, longitude_segments: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed local-space triangle mesh for one ellipsoid."""

    radii_array = np.asarray(radii, dtype=np.float32)
    points: list[tuple[float, float, float]] = [
        (0.0, 0.0, float(radii_array[2]))
    ]
    for latitude in range(1, latitude_segments):
        theta = math.pi * latitude / latitude_segments
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        for longitude in range(longitude_segments):
            phi = 2.0 * math.pi * longitude / longitude_segments
            points.append(
                (
                    float(radii_array[0] * sin_theta * math.cos(phi)),
                    float(radii_array[1] * sin_theta * math.sin(phi)),
                    float(radii_array[2] * cos_theta),
                )
            )
    south = len(points)
    points.append((0.0, 0.0, -float(radii_array[2])))

    faces: list[tuple[int, int, int]] = []
    first_ring = 1
    for longitude in range(longitude_segments):
        current = first_ring + longitude
        following = first_ring + (longitude + 1) % longitude_segments
        faces.append((0, following, current))
    for latitude in range(latitude_segments - 2):
        current_ring = 1 + latitude * longitude_segments
        next_ring = current_ring + longitude_segments
        for longitude in range(longitude_segments):
            current = current_ring + longitude
            following = current_ring + (longitude + 1) % longitude_segments
            below = next_ring + longitude
            below_following = next_ring + (longitude + 1) % longitude_segments
            faces.extend(
                (
                    (current, following, below_following),
                    (current, below_following, below),
                )
            )
    last_ring = 1 + (latitude_segments - 2) * longitude_segments
    for longitude in range(longitude_segments):
        current = last_ring + longitude
        following = last_ring + (longitude + 1) % longitude_segments
        faces.append((south, current, following))
    return np.asarray(points, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _write_triangle_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(len(vertices), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
    vertex["x"], vertex["y"], vertex["z"] = vertices.T
    face = np.empty(len(triangles), dtype=[("vertex_indices", "O")])
    face["vertex_indices"] = [np.asarray(values, dtype=np.int32) for values in triangles]
    PlyData(
        [PlyElement.describe(vertex, "vertex"), PlyElement.describe(face, "face")],
        text=False,
    ).write(str(path))


def _combine_static_plant_and_ground(
    plant_vertex: np.ndarray,
    ground_ply: Path,
    output_path: Path,
) -> tuple[int, int]:
    """Write a plant-first exact XYZ union and return ground/overlap counts."""

    ground_vertex = _load_vertex(ground_ply)
    if ground_vertex.dtype != plant_vertex.dtype:
        names = plant_vertex.dtype.names or ()
        missing = [name for name in names if name not in (ground_vertex.dtype.names or ())]
        if missing:
            raise ValueError(f"Ground PLY cannot satisfy plant schema: {missing}")
        normalized = np.empty(len(ground_vertex), dtype=plant_vertex.dtype)
        for name in names:
            normalized[name] = ground_vertex[name]
        ground_vertex = normalized
    overlap = np.isin(_xyz_keys(ground_vertex), _xyz_keys(plant_vertex))
    combined = np.concatenate((plant_vertex, ground_vertex[~overlap]))
    _write_vertex_ply(output_path, combined)
    return len(ground_vertex), int(overlap.sum())


def _format_tuple(values: list[float] | np.ndarray) -> str:
    return ", ".join(f"{float(value):.9g}" for value in values)


def _write_usda(
    path: Path,
    entities: list[TomatoEntity],
    kinematic: bool,
    latitude_segments: int,
    longitude_segments: int,
) -> None:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "TomatoEntities"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "TomatoEntities"',
        "{",
        '    def Scope "Looks"',
        "    {",
        '        def Material "TomatoMaterial"',
        "        {",
        "            token outputs:surface.connect = "
        "</TomatoEntities/Looks/TomatoMaterial/PreviewSurface.outputs:surface>",
        '            def Shader "PreviewSurface"',
        "            {",
        '                uniform token info:id = "UsdPreviewSurface"',
        "                color3f inputs:diffuseColor = (0.72, 0.045, 0.025)",
        "                float inputs:roughness = 0.42",
        '                token outputs:surface',
        "            }",
        "        }",
        "    }",
    ]
    for entity in entities:
        center = _format_tuple(entity.center_m)
        quat = entity.orientation_wxyz
        stem = _format_tuple(entity.stem_anchor_m)
        collider_radius = max(entity.radii_m)
        mesh_points, mesh_faces = _ellipsoid_mesh(
            entity.radii_m, latitude_segments, longitude_segments
        )
        point_text = ", ".join(
            f"({x:.9g}, {y:.9g}, {z:.9g})" for x, y, z in mesh_points
        )
        count_text = ", ".join("3" for _ in mesh_faces)
        index_text = ", ".join(str(int(value)) for value in mesh_faces.ravel())
        lines.extend(
            [
                "",
                f'    def Xform "{entity.identifier}" (',
                '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
                "    )",
                "    {",
                f"        double3 xformOp:translate = ({center})",
                "        quatf xformOp:orient = "
                f"({quat[0]:.9g}, {quat[1]:.9g}, {quat[2]:.9g}, {quat[3]:.9g})",
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
                f"        bool physics:kinematicEnabled = {str(kinematic).lower()}",
                f"        float physics:mass = {entity.mass_kg:.9g}",
                "        bool tomato:attached = true",
                f"        double3 tomato:stemAnchor = ({stem})",
                f'        string tomato:instanceId = "{entity.identifier}"',
                "",
                '        def Mesh "Visual" (',
                '            prepend apiSchemas = ["MaterialBindingAPI"]',
                "        )",
                "        {",
                f"            point3f[] points = [{point_text}]",
                f"            int[] faceVertexCounts = [{count_text}]",
                f"            int[] faceVertexIndices = [{index_text}]",
                '            uniform token subdivisionScheme = "none"',
                '            rel material:binding = </TomatoEntities/Looks/TomatoMaterial>',
                "        }",
                "",
                '        def Sphere "Collision" (',
                '            prepend apiSchemas = ["PhysicsCollisionAPI"]',
                "        )",
                "        {",
                f"            double radius = {collider_radius:.9g}",
                '            token visibility = "invisible"',
                "        }",
                "    }",
            ]
        )
    lines.extend(("}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def extract_tomato_entities(
    gaussian_ply: Path,
    output_directory: Path,
    *,
    saga_mask: Path | None = None,
    saga_mask_source_ply: Path | None = None,
    ground_gaussian_ply: Path | None = None,
    config: ExtractionConfig | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract tomato instances and write Gaussian plus USD deliverables."""

    config = config or ExtractionConfig()
    config.validate()
    gaussian_ply = Path(gaussian_ply).resolve()
    output_directory = Path(output_directory).resolve()
    expected_outputs = [
        output_directory / "tomato_instances.json",
        output_directory / "tomato_entities.usda",
        output_directory / "tomatoes_combined.ply",
        output_directory / "plant_without_tomatoes.ply",
    ]
    if ground_gaussian_ply is not None:
        expected_outputs.append(output_directory / "static_scene_without_tomatoes.ply")
    existing = [str(path) for path in expected_outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Tomato entity outputs already exist; pass overwrite=True: "
            + ", ".join(existing)
        )
    output_directory.mkdir(parents=True, exist_ok=True)

    vertex = _load_vertex(gaussian_ply)
    xyz, rgb, opacity = _xyz_rgb_opacity(vertex)
    if saga_mask is None:
        source = "ripe_colour_bootstrap"
        seed_mask = _colour_seed(xyz, rgb, opacity, config)
        mask_path: str | None = None
    else:
        source = "saga_gaussian_mask"
        saga_mask = Path(saga_mask).resolve()
        if saga_mask_source_ply is None:
            seed_mask = load_gaussian_mask(saga_mask, len(vertex))
            resolved_mask_source = gaussian_ply
        else:
            resolved_mask_source = Path(saga_mask_source_ply).resolve()
            source_vertex = _load_vertex(resolved_mask_source)
            source_mask = load_gaussian_mask(saga_mask, len(source_vertex))
            seed_mask = _map_mask_by_exact_xyz(source_mask, source_vertex, vertex)
        seed_mask &= opacity >= config.opacity_minimum
        if config.saga_require_ripe_colour_seed:
            seed_mask &= _colour_seed(xyz, rgb, opacity, config)
            source = "saga_gaussian_mask_with_ripe_colour_seed"
        mask_path = str(saga_mask)

    seed_indices = np.flatnonzero(seed_mask)
    if len(seed_indices) < config.minimum_seed_points:
        raise ValueError(
            f"Only {len(seed_indices)} candidate Gaussians were selected; "
            f"at least {config.minimum_seed_points} are required"
        )
    seed_xyz = xyz[seed_indices]
    components = _voxel_components(seed_xyz, config.voxel_size_m)
    split_components: list[np.ndarray] = []
    for component in components:
        if len(component) < config.minimum_seed_points:
            continue
        split_components.extend(
            _split_oversized(
                seed_xyz,
                component,
                config.maximum_component_span_m,
                config.minimum_seed_points,
            )
        )

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for component in split_components:
        if len(component) < config.minimum_seed_points:
            continue
        center, radii, axes = _fit_ellipsoid(seed_xyz[component], config)
        if np.any(radii >= config.maximum_radius_m - 1e-7):
            continue
        entity_mask = _inside_ellipsoid(xyz, center, radii, axes)
        entity_mask &= opacity >= max(config.opacity_minimum * 0.25, 0.01)
        if int(entity_mask.sum()) < config.minimum_entity_points:
            continue
        candidates.append((component, center, radii, axes))

    # Prefer strongly supported candidates when two fitted volumes overlap at
    # almost the same centre.  This suppresses fragments without merging two
    # adjacent fruits in a tight truss.
    candidates.sort(key=lambda item: len(item[0]), reverse=True)
    accepted: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for candidate in candidates:
        _, center, radii, _ = candidate
        duplicate = False
        for _, other_center, other_radii, _ in accepted:
            scale = max(float(np.mean(radii + other_radii)), 1e-6)
            if float(np.linalg.norm(center - other_center)) / scale < 0.42:
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)
    accepted.sort(key=lambda item: tuple(float(value) for value in item[1]))

    removal_mask = np.zeros(len(vertex), dtype=bool)
    entities: list[TomatoEntity] = []
    combined_entity_mask = np.zeros(len(vertex), dtype=bool)
    gaussian_directory = output_directory / "gaussians"
    gaussian_directory.mkdir(parents=True, exist_ok=True)
    mesh_directory = output_directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    # Parameter sweeps can produce fewer instances than a previous run.  Only
    # remove this exporter\'s numbered outputs so the directory remains an
    # exact representation of the current manifest without touching user data.
    for stale_path in gaussian_directory.glob("tomato_[0-9][0-9][0-9].ply"):
        stale_path.unlink()
    for stale_path in mesh_directory.glob("tomato_[0-9][0-9][0-9].ply"):
        stale_path.unlink()
    for index, (component, center, radii, axes) in enumerate(accepted, start=1):
        identifier = f"tomato_{index:03d}"
        entity_mask = _inside_ellipsoid(xyz, center, radii, axes)
        entity_mask &= opacity >= max(config.opacity_minimum * 0.25, 0.01)
        remove = _inside_ellipsoid(
            xyz, center, radii, axes, scale=config.removal_scale
        )
        combined_entity_mask |= entity_mask
        removal_mask |= remove
        entity_xyz = xyz[entity_mask]
        entity_rgb = rgb[entity_mask]
        _write_vertex_ply(gaussian_directory / f"{identifier}.ply", vertex[entity_mask])
        mesh_vertices, mesh_triangles = _ellipsoid_mesh(
            radii, config.mesh_latitude_segments, config.mesh_longitude_segments
        )
        _write_triangle_ply(
            mesh_directory / f"{identifier}.ply", mesh_vertices, mesh_triangles
        )
        quaternion = _rotation_matrix_to_quaternion(axes)
        stem_anchor = center.copy()
        stem_anchor[2] += float(radii[np.argmax(np.abs(axes[2, :]))])
        volume = 4.0 / 3.0 * math.pi * float(np.prod(radii))
        entities.append(
            TomatoEntity(
                identifier=identifier,
                center_m=center.astype(float).tolist(),
                radii_m=radii.astype(float).tolist(),
                orientation_wxyz=quaternion.astype(float).tolist(),
                stem_anchor_m=stem_anchor.astype(float).tolist(),
                mass_kg=volume * config.tomato_density_kg_m3,
                seed_gaussians=len(component),
                entity_gaussians=len(entity_xyz),
                mean_rgb=entity_rgb.mean(axis=0).astype(float).tolist(),
            )
        )

    if not entities:
        raise ValueError(
            "Candidate Gaussians were found, but none passed the tomato geometry filters"
        )

    _write_vertex_ply(
        output_directory / "tomatoes_combined.ply", vertex[combined_entity_mask]
    )
    _write_vertex_ply(
        output_directory / "plant_without_tomatoes.ply", vertex[~removal_mask]
    )
    static_scene_path: Path | None = None
    ground_statistics: dict[str, Any] | None = None
    if ground_gaussian_ply is not None:
        ground_gaussian_ply = Path(ground_gaussian_ply).resolve()
        static_scene_path = output_directory / "static_scene_without_tomatoes.ply"
        ground_count, overlap_count = _combine_static_plant_and_ground(
            vertex[~removal_mask], ground_gaussian_ply, static_scene_path
        )
        ground_statistics = {
            "source": str(ground_gaussian_ply),
            "source_gaussians": ground_count,
            "plant_ground_xyz_overlap_removed": overlap_count,
            "combined_gaussians": (
                len(vertex) - int(removal_mask.sum()) + ground_count - overlap_count
            ),
        }
    _write_usda(
        output_directory / "tomato_entities.usda",
        entities,
        config.initially_kinematic,
        config.mesh_latitude_segments,
        config.mesh_longitude_segments,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source_gaussian_ply": str(gaussian_ply),
        "selection_source": source,
        "saga_mask": mask_path,
        "saga_mask_source_ply": (
            str(resolved_mask_source) if saga_mask is not None else None
        ),
        "coordinate_system": {"unit": "meter", "up_axis": "Z"},
        "configuration": asdict(config),
        "statistics": {
            "source_gaussians": len(vertex),
            "seed_gaussians": int(seed_mask.sum()),
            "entity_gaussians": int(combined_entity_mask.sum()),
            "removed_from_static_gaussians": int(removal_mask.sum()),
            "entity_count": len(entities),
            "static_scene": ground_statistics,
        },
        "entities": [asdict(entity) for entity in entities],
        "outputs": {
            "usd": str(output_directory / "tomato_entities.usda"),
            "combined_tomato_gaussians": str(output_directory / "tomatoes_combined.ply"),
            "static_plant_gaussians": str(output_directory / "plant_without_tomatoes.ply"),
            "per_instance_directory": str(gaussian_directory),
            "per_instance_mesh_directory": str(mesh_directory),
            "static_scene_gaussians": (
                str(static_scene_path) if static_scene_path is not None else None
            ),
        },
    }
    (output_directory / "tomato_instances.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
