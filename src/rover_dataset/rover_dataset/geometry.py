"""Rotation helpers shared by the projection code.

Deliberately self-contained (module 2 has its own copy) so ``rover_dataset``
depends on the *generated artefacts* of modules 1 and 2 -- ``ground_truth.yaml``
and ``calibration.yaml`` -- and never on their Python packages.

Convention: ZYX Euler ``(roll, pitch, yaw)`` in radians, REP-103,
``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["rpy_matrix", "rolling_yaw_matrix", "matrix_to_rpy"]


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles to a 3x3 rotation matrix (same convention as module 2)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def rolling_yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation about +z only."""
    return rpy_matrix(0.0, 0.0, yaw)


def matrix_to_rpy(matrix: np.ndarray) -> tuple:
    """Inverse of :func:`rpy_matrix` for a proper rotation matrix."""
    m = np.asarray(matrix, dtype=float)
    sy = math.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    if sy > 1e-9:
        roll = math.atan2(m[2, 1], m[2, 2])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock
        roll = math.atan2(-m[1, 2], m[1, 1])
        pitch = math.atan2(-m[2, 0], sy)
        yaw = 0.0
    return (roll, pitch, yaw)
