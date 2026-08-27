"""Tests for deterministic closed height-field collision mesh construction."""

from __future__ import annotations

import numpy as np

from colmapcut_recon.export.collision_mesh import (
    CollisionMeshConfig,
    build_closed_heightfield_mesh,
    fit_reference_heightfield,
    rasterize_mesh_height,
    topology_report,
)


def test_closed_heightfield_has_manifold_oriented_topology() -> None:
    x = np.linspace(-1.0, 1.0, 3)
    y = np.linspace(-1.0, 1.0, 3)
    grid_x, grid_y = np.meshgrid(x, y)
    height = 0.05 * grid_x + 0.02 * grid_y

    vertices, triangles = build_closed_heightfield_mesh(x, y, height, 0.03)
    report = topology_report(vertices, triangles)

    assert vertices.shape == (18, 3)
    assert triangles.shape == (32, 3)
    assert report["boundary_edges"] == 0
    assert report["nonmanifold_edges"] == 0
    assert report["degenerate_triangles"] == 0
    assert report["signed_volume_m3"] > 0


def test_reference_fit_rejects_isolated_vertical_outlier() -> None:
    config = CollisionMeshConfig(
        x_min=-0.1,
        x_max=0.1,
        y_min=-0.1,
        y_max=0.1,
        grid_size_m=0.1,
        min_points_per_node=2,
        smoothing_iterations=1,
        max_slope_degrees=20.0,
    )
    x = np.linspace(-0.1, 0.1, 3)
    y = np.linspace(-0.1, 0.1, 3)
    points = []
    for y_value in y:
        for x_value in x:
            points.extend(
                ([x_value, y_value, 0.01 * x_value], [x_value, y_value, 0.01 * x_value])
            )
    points.extend(([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]))

    height, diagnostics = fit_reference_heightfield(np.asarray(points), x, y, config)

    assert diagnostics["measured_nodes"] == 9
    assert float(height.max()) < 0.01


def test_rasterizer_uses_mesh_only_within_reference_residual() -> None:
    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 0.5, 1.0])
    reference = np.zeros((3, 3), dtype=np.float64)
    vertices = np.array(
        [
            [0.0, 0.0, 0.02],
            [1.0, 0.0, 0.02],
            [1.0, 1.0, 0.02],
            [0.0, 1.0, 0.02],
            [0.0, 0.0, 0.2],
            [1.0, 0.0, 0.2],
            [1.0, 1.0, 0.2],
        ]
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])

    mesh_height, mask, diagnostics = rasterize_mesh_height(
        vertices,
        triangles,
        x,
        y,
        reference,
        max_residual_m=0.03,
        min_vertical_normal=0.3,
    )

    assert diagnostics["covered_nodes"] == 9
    assert np.allclose(mesh_height[mask], 0.02)
