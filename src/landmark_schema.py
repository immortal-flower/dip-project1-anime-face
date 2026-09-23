"""Canonical 28-point order used by the landmark teacher and course model."""
from __future__ import annotations

import numpy as np


LANDMARK_ORDER = "hysts28-v1"
LANDMARK_COUNT = 28

# Pairs copied from the teacher model's official flip configuration.  They are
# also a compact, machine-checkable definition of left/right correspondence.
FLIP_PAIRS = (
    (0, 4), (1, 3),
    (5, 10), (6, 9), (7, 8),
    (11, 19), (12, 18), (13, 17),
    (14, 22), (15, 21), (16, 20),
    (24, 26),
)

FACE_OUTLINE = tuple(range(0, 5))
LEFT_BROW = tuple(range(5, 8))
RIGHT_BROW = tuple(range(8, 11))
LEFT_EYE = tuple(range(11, 17))
RIGHT_EYE = tuple(range(17, 23))
NOSE = (23,)
MOUTH = tuple(range(24, 28))


def validate_landmark_order(value: str) -> str:
    if value != LANDMARK_ORDER:
        raise ValueError(f"Expected landmark_order={LANDMARK_ORDER!r}, got {value!r}")
    return value


def validate_landmarks(points, visibility=None):
    points = np.asarray(points, dtype=float)
    if points.shape != (LANDMARK_COUNT, 2) or not np.isfinite(points).all():
        raise ValueError("Expected 28 finite xy landmarks")
    if visibility is None:
        return points
    visibility = np.asarray(visibility)
    if visibility.shape != (LANDMARK_COUNT,) or not np.isin(visibility, [0, 1]).all():
        raise ValueError("Expected 28 binary visibility flags")
    return points, visibility.astype(bool)
