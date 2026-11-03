"""Rotation and rigid-transform helpers shared by the whole perception stack.

Conventions are inherited from modules 1 and 2 and are not re-invented here:

* Euler angles are **ZYX intrinsic** (``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``),
  which is what SDF/Gazebo and ``cat_arena``/``rover_description`` use,
* the map (arena) frame has its origin at the west-south arena corner, ``+x``
  east, ``+y`` north, ``+z`` up,
* a camera *optical* frame follows REP-103/105 (``+z`` forward, ``+x`` right,
  ``+y`` down) and is related to its camera *link* frame by
  ``rpy = (-pi/2, 0, -pi/2)``.

Only numpy is used; nothing here imports ROS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "rpy_matrix",
    "matrix_to_rpy",
    "rpy_to_quaternion",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "rotation_matrix_to_rpy",
    "normalize_angle",
    "wrap_pi",
    "yaw_of",
    "Pose3",
]

_EPS = 1e-12


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler angles (SDF convention) to a 3x3 rotation matrix."""
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


def rotation_matrix_to_rpy(matrix: np.ndarray) -> Tuple[float, float, float]:
    """3x3 rotation matrix to ZYX Euler angles (inverse of :func:`rpy_matrix`)."""
    r = np.asarray(matrix, dtype=float).reshape(3, 3)
    sy = -float(r[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    if abs(sy) < 1.0 - 1e-9:
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:  # gimbal lock: pitch = +-pi/2, roll and yaw are not separable
        roll = math.atan2(-float(r[1, 2]), float(r[1, 1]))
        yaw = 0.0
    return roll, pitch, yaw


#: Alias kept so the name used by module 1's sdf.py works here too.
matrix_to_rpy = rotation_matrix_to_rpy


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """ZYX Euler angles to ``(x, y, z, w)``."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """``(x, y, z, w)`` (any norm) to a 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= _EPS:
        raise ValueError("quaternion_to_matrix: zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    """3x3 rotation matrix to ``(x, y, z, w)`` (Shepperd's method)."""
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (float(m[2, 1]) - float(m[1, 2])) / s
        y = (float(m[0, 2]) - float(m[2, 0])) / s
        z = (float(m[1, 0]) - float(m[0, 1])) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[0, 0]) - float(m[1, 1]) - float(m[2, 2])) * 2.0
        w = (float(m[2, 1]) - float(m[1, 2])) / s
        x = 0.25 * s
        y = (float(m[0, 1]) + float(m[1, 0])) / s
        z = (float(m[0, 2]) + float(m[2, 0])) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[1, 1]) - float(m[0, 0]) - float(m[2, 2])) * 2.0
        w = (float(m[0, 2]) - float(m[2, 0])) / s
        x = (float(m[0, 1]) + float(m[1, 0])) / s
        y = 0.25 * s
        z = (float(m[1, 2]) + float(m[2, 1])) / s
    else:
        s = math.sqrt(1.0 + float(m[2, 2]) - float(m[0, 0]) - float(m[1, 1])) * 2.0
        w = (float(m[1, 0]) - float(m[0, 1])) / s
        x = (float(m[0, 2]) + float(m[2, 0])) / s
        y = (float(m[1, 2]) + float(m[2, 1])) / s
        z = 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= _EPS:  # pragma: no cover - only for a degenerate matrix
        raise ValueError("matrix_to_quaternion: degenerate rotation matrix")
    return (x / norm, y / norm, z / norm, w / norm)


def normalize_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    wrapped = math.fmod(float(angle) + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


#: Alias used where "wrap to pi" reads better.
wrap_pi = normalize_angle


def yaw_of(matrix: np.ndarray) -> float:
    """Yaw of a rotation matrix (ZYX)."""
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    return math.atan2(float(m[1, 0]), float(m[0, 0]))


@dataclass(frozen=True)
class Pose3:
    """A rigid transform: ``p_parent = R_world_child @ p_child + t_world_child``.

    ``rotation`` maps *child* coordinates into the *parent* frame and
    ``translation`` is the child origin in the parent frame, i.e. the same
    convention as a TF parent -> child transform (``T_parent_child``).
    """

    rotation: np.ndarray
    translation: np.ndarray

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def identity(cls) -> "Pose3":
        return cls(np.eye(3), np.zeros(3))

    @classmethod
    def from_xy_yaw(
        cls, x: float, y: float, yaw: float, z: float = 0.0
    ) -> "Pose3":
        return cls(rpy_matrix(0.0, 0.0, yaw), np.array([x, y, z], dtype=float))

    @classmethod
    def from_rpy(
        cls,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> "Pose3":
        return cls(rpy_matrix(roll, pitch, yaw), np.array([x, y, z], dtype=float))

    @classmethod
    def from_transform(
        cls,
        translation: Sequence[float],
        rotation_matrix: np.ndarray,
    ) -> "Pose3":
        return cls(np.asarray(rotation_matrix, dtype=float).reshape(3, 3), np.array(translation, dtype=float))

    # -- algebra ----------------------------------------------------------- #
    def compose(self, other: "Pose3") -> "Pose3":
        """``self * other``: apply ``other`` first, then ``self``."""
        return Pose3(self.rotation @ other.rotation, self.rotation @ other.translation + self.translation)

    def __matmul__(self, other: "Pose3") -> "Pose3":
        return self.compose(other)

    def inverse(self) -> "Pose3":
        rotation_t = self.rotation.T
        return Pose3(rotation_t, -rotation_t @ self.translation)

    # -- application ------------------------------------------------------- #
    def transform_point(self, point: Sequence[float]) -> np.ndarray:
        return self.rotation @ np.asarray(point, dtype=float).reshape(3) + self.translation

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return pts @ self.rotation.T + self.translation

    def transform_rotation(self, rotation: np.ndarray) -> np.ndarray:
        return self.rotation @ np.asarray(rotation, dtype=float).reshape(3, 3)

    # -- inspection -------------------------------------------------------- #
    @property
    def x(self) -> float:
        return float(self.translation[0])

    @property
    def y(self) -> float:
        return float(self.translation[1])

    @property
    def z(self) -> float:
        return float(self.translation[2])

    @property
    def yaw(self) -> float:
        return yaw_of(self.rotation)

    @property
    def quaternion_xyzw(self) -> Tuple[float, float, float, float]:
        """The rotation as ``(x, y, z, w)`` (the ROS convention)."""
        return matrix_to_quaternion(self.rotation)

    def as_rpy(self) -> Tuple[float, float, float, float, float, float]:
        roll, pitch, yaw = rotation_matrix_to_rpy(self.rotation)
        return (self.x, self.y, self.z, roll, pitch, yaw)

    def as_homogeneous(self) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = self.rotation
        out[:3, 3] = self.translation
        return out

    @classmethod
    def from_homogeneous(cls, matrix: np.ndarray) -> "Pose3":
        m = np.asarray(matrix, dtype=float).reshape(4, 4)
        return cls(m[:3, :3].copy(), m[:3, 3].copy())

    def is_finite(self) -> bool:
        return bool(np.all(np.isfinite(self.rotation)) and np.all(np.isfinite(self.translation)))

    def almost_equal(self, other: "Pose3", tol: float = 1e-9) -> bool:
        return bool(
            np.allclose(self.rotation, other.rotation, atol=tol)
            and np.allclose(self.translation, other.translation, atol=tol)
        )


def look_at_rotation(
    from_point: Sequence[float],
    to_point: Sequence[float],
    up: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Rotation whose ``+z`` axis points from ``from_point`` to ``to_point``.

    Used to orient tag debug images and by the tests; SDF planes face ``+z``.
    """
    frm = np.asarray(from_point, dtype=float).reshape(3)
    to = np.asarray(to_point, dtype=float).reshape(3)
    forward = to - frm
    norm = float(np.linalg.norm(forward))
    if norm <= _EPS:
        raise ValueError("look_at_rotation: from_point == to_point")
    z_axis = forward / norm
    up_vec = np.asarray(up if up is not None else (0.0, 0.0, 1.0), dtype=float).reshape(3)
    if abs(float(np.dot(up_vec, z_axis))) > 1.0 - 1e-9:
        up_vec = np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(up_vec, z_axis)
    x_axis /= float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])
