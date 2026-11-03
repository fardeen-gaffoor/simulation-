"""3-D poses and small angles, numpy-only.

Module 5 talks to module 4 and to the drive firmware entirely in ROS messages,
so it may not import module 4's Python (that would couple two packages that must
be deployable independently).  It therefore carries the handful of pose helpers
it needs itself; they are the same conventions, restated:

* rotation matrices are right-handed, applied as ``R @ v``;
* ``rpy`` is the ZYX intrinsic composition (yaw about ``+z``, pitch about
  ``+y``, roll about ``+x``), the convention of ``geometry_msgs`` and REP-103;
* quaternions are ``(x, y, z, w)``;
* the rover drives on flat sand, so every *applied* correction is planar:
  :func:`planarize` discards z/roll/pitch rather than letting them tilt a frame.

Nothing here imports ROS, so the planner, controller and sequencer tests run
in-process on any machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

__all__ = [
    "wrap_angle",
    "angle_difference",
    "Pose3",
    "rpy_matrix",
    "rotation_matrix_to_rpy",
    "quaternion_to_matrix",
    "matrix_to_quaternion",
    "planarize",
    "yaw_to_quaternion",
    "unit",
]


def wrap_angle(angle: float) -> float:
    """Wrap to ``(-pi, pi]``."""
    value = math.fmod(float(angle) + math.pi, 2.0 * math.pi)
    if value <= 0.0:
        value += 2.0 * math.pi
    return value - math.pi


def angle_difference(a: float, b: float) -> float:
    """``a - b`` wrapped to ``(-pi, pi]`` - the shortest way round."""
    return wrap_angle(float(a) - float(b))


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    """``(x, y, z, w)`` for a pure yaw rotation."""
    half = 0.5 * float(yaw)
    return (0.0, 0.0, math.sin(half), math.cos(half))


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX rotation matrix from roll/pitch/yaw."""
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


def rotation_matrix_to_rpy(rotation: np.ndarray) -> Tuple[float, float, float]:
    """``(roll, pitch, yaw)`` from a rotation matrix (the inverse of :func:`rpy_matrix`)."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation_matrix_to_rpy: expected 3x3, got {matrix.shape}")
    # The pitch branch is the gimbal-lock edge; clamp keeps sqrt of a tiny
    # negative from a rounding error from producing NaN.
    sin_pitch = float(np.clip(-matrix[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) > 1.0 - 1e-9:  # pragma: no cover - degenerate only
        roll = 0.0
        yaw = math.atan2(matrix[0, 1], matrix[1, 1])
        return (roll, math.copysign(math.pi / 2.0, pitch), yaw)
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return (roll, pitch, yaw)


def quaternion_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """``(x, y, z, w)`` -> rotation matrix (normalised, so a noisy message is safe)."""
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:  # pragma: no cover - only for a zeroed message
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    """Rotation matrix -> ``(x, y, z, w)`` (Shepperd's method, always finite)."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(matrix[0, 0] + matrix[1, 1] + matrix[2, 2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    return (x, y, z, w)


def unit(x: float, y: float) -> Tuple[float, float]:
    """A 2-D unit vector; ``(1, 0)`` for the degenerate input."""
    norm = math.hypot(x, y)
    if norm <= 1e-12:
        return (1.0, 0.0)
    return (x / norm, y / norm)


@dataclass(frozen=True, eq=False)
class Pose3:
    """A rigid transform: rotation matrix + translation.

    Equality is by value with a small absolute tolerance rather than numpy's
    elementwise ``==`` (which would make ``a == b`` raise instead of returning a
    bool, because comparing two arrays yields an array).  That matters here: the
    controller and the tests both ask "is this the same pose".
    """

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=float).reshape(3, 3))
        object.__setattr__(self, "translation", np.asarray(self.translation, dtype=float).reshape(3))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pose3):
            return NotImplemented
        return bool(
            np.allclose(self.rotation, other.rotation, atol=1e-9)
            and np.allclose(self.translation, other.translation, atol=1e-9)
        )

    def __hash__(self) -> int:
        # Rounded so that two poses equal under `__eq__` hash the same.
        return hash(
            tuple(round(float(value), 9) for value in self.translation)
            + tuple(round(float(value), 9) for value in self.rotation.ravel())
        )

    # -- constructors ------------------------------------------------------- #
    @classmethod
    def identity(cls) -> "Pose3":
        return cls(np.eye(3), (0.0, 0.0, 0.0))

    @classmethod
    def from_xy_yaw(cls, x: float, y: float, yaw: float, z: float = 0.0) -> "Pose3":
        return cls(rpy_matrix(0.0, 0.0, yaw), (float(x), float(y), float(z)))

    @classmethod
    def from_pose(cls, pose: object) -> "Pose3":
        """From a ``geometry_msgs/Pose``-like object (anything with position/orientation)."""
        position = pose.position  # type: ignore[attr-defined]
        orientation = pose.orientation  # type: ignore[attr-defined]
        return cls(
            quaternion_to_matrix((orientation.x, orientation.y, orientation.z, orientation.w)),
            (float(position.x), float(position.y), float(position.z)),
        )

    # -- accessors ---------------------------------------------------------- #
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
        return rotation_matrix_to_rpy(self.rotation)[2]

    @property
    def roll(self) -> float:
        return rotation_matrix_to_rpy(self.rotation)[0]

    @property
    def pitch(self) -> float:
        return rotation_matrix_to_rpy(self.rotation)[1]

    def as_rpy(self) -> Tuple[float, float, float, float, float, float]:
        """``(x, y, z, roll, pitch, yaw)``."""
        roll, pitch, yaw = rotation_matrix_to_rpy(self.rotation)
        return (self.x, self.y, self.z, roll, pitch, yaw)

    @property
    def quaternion_xyzw(self) -> Tuple[float, float, float, float]:
        return matrix_to_quaternion(self.rotation)

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    # -- algebra ------------------------------------------------------------ #
    def compose(self, other: "Pose3") -> "Pose3":
        """``self @ other``: ``other`` expressed in ``self``'s frame."""
        return Pose3(self.rotation @ other.rotation, self.rotation @ other.translation + self.translation)

    def inverse(self) -> "Pose3":
        rotation = self.rotation.T
        return Pose3(rotation, -(rotation @ self.translation))

    def transform_point(self, point: Sequence[float]) -> np.ndarray:
        return self.rotation @ np.asarray(point, dtype=float).reshape(3) + self.translation

    def planar_distance(self, other: "Pose3") -> float:
        """Ground-plane distance between two poses."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def planar_angle_to(self, other: "Pose3") -> float:
        """Heading error: how far ``other``'s yaw is from this pose's yaw."""
        return angle_difference(other.yaw, self.yaw)

    def bearing_to(self, x: float, y: float) -> float:
        """The yaw that points from this pose at ``(x, y)``."""
        return math.atan2(float(y) - self.y, float(x) - self.x)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        x, y, z, roll, pitch, yaw = self.as_rpy()
        return f"Pose3(x={x:.4f}, y={y:.4f}, z={z:.4f}, yaw={yaw:.4f})"


def planarize(pose: Pose3) -> Pose3:
    """Drop z, roll and pitch: the rover lives on the sand plane."""
    return Pose3.from_xy_yaw(pose.x, pose.y, pose.yaw)
