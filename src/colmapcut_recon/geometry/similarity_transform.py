"""Represent the shared scale, rotation, and translation transform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    """Map COLMAP world coordinates into a target world coordinate frame."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Similarity-transform scale must be finite and positive")
        if rotation.shape != (3, 3):
            raise ValueError("Similarity-transform rotation must have shape (3, 3)")
        if translation.shape != (3,):
            raise ValueError("Similarity-transform translation must have shape (3,)")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("Similarity-transform values must be finite")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7):
            raise ValueError("Similarity-transform rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("Similarity-transform rotation must be right-handed")
        rotation.setflags(write=False)
        translation.setflags(write=False)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("Points must have final dimension 3")
        return self.scale * (points @ self.rotation.T) + self.translation

    def to_dict(self) -> dict[str, object]:
        return {
            "scale": self.scale,
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "matrix": self.matrix.tolist(),
            "convention": "target = scale * rotation * source + translation",
        }
