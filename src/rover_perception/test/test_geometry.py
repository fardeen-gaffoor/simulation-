"""Geometry helpers: the conventions everything else is built on.

The two conventions worth pinning down here are the ones that cost the most when
they are wrong: the ZYX Euler order (shared with SDF/Gazebo and module 1) and the
``Pose3`` direction (``rotation`` maps child -> parent, i.e. a TF parent->child
transform).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rover_perception.geometry import (
    Pose3,
    look_at_rotation,
    matrix_to_quaternion,
    normalize_angle,
    quaternion_to_matrix,
    rotation_matrix_to_rpy,
    rpy_matrix,
    rpy_to_quaternion,
    yaw_of,
)


def test_rpy_matrix_is_zyx():
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll): apply roll, then pitch, then yaw.
    roll, pitch, yaw = 0.3, -0.7, 2.1
    expected = (
        rpy_matrix(0.0, 0.0, yaw) @ rpy_matrix(0.0, pitch, 0.0) @ rpy_matrix(roll, 0.0, 0.0)
    )
    assert np.allclose(rpy_matrix(roll, pitch, yaw), expected, atol=1e-12)


def test_rpy_matrix_is_orthonormal():
    for angles in [(0, 0, 0), (0.1, 0.2, 0.3), (-1.0, 0.5, 3.0), (math.pi / 2, 0.0, 0.0)]:
        matrix = rpy_matrix(*angles)
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-12)


def test_yaw_only_matrix_matches_the_definition():
    yaw = math.radians(37.0)
    matrix = rpy_matrix(0.0, 0.0, yaw)
    assert matrix[0, 0] == pytest.approx(math.cos(yaw))
    assert matrix[1, 0] == pytest.approx(math.sin(yaw))
    assert yaw_of(matrix) == pytest.approx(yaw)


@pytest.mark.parametrize(
    "angles",
    [
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 0.4),
        (0.31, -0.62, 1.23),
        (-0.9, 0.15, -2.7),
    ],
)
def test_rpy_round_trip(angles):
    assert rotation_matrix_to_rpy(rpy_matrix(*angles)) == pytest.approx(angles, abs=1e-9)


def test_quaternion_matches_matrix():
    for angles in [(0.0, 0.0, 0.0), (0.2, 0.4, 0.6), (-1.1, 0.3, 2.9)]:
        quaternion = rpy_to_quaternion(*angles)
        assert np.allclose(quaternion_to_matrix(quaternion), rpy_matrix(*angles), atol=1e-12)
        assert matrix_to_quaternion(rpy_matrix(*angles)) == pytest.approx(quaternion, abs=1e-9)


def test_quaternion_is_normalised_and_requires_non_zero():
    assert sum(v * v for v in rpy_to_quaternion(0.3, 0.4, 0.5)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        quaternion_to_matrix((0.0, 0.0, 0.0, 0.0))


def test_north_wall_board_rotation():
    """Module 1's north-wall board: rpy (0, pi/2, -pi/2) -> normal -y.

    The board's front face must look *into* the arena (south, i.e. -y), which is
    the assumption ``tags.py`` relies on when it stands the tag plane off the
    board along that normal.
    """
    rotation = rpy_matrix(0.0, math.pi / 2, -math.pi / 2)
    assert rotation[:, 2] == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)
    assert matrix_to_quaternion(rotation) == pytest.approx((0.5, 0.5, -0.5, 0.5), abs=1e-9)


def test_west_wall_board_rotation():
    rotation = rpy_matrix(0.0, math.pi / 2, 0.0)
    assert rotation[:, 2] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert matrix_to_quaternion(rotation) == pytest.approx((0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)), abs=1e-9)


# --------------------------------------------------------------------------- #
# Pose3
# --------------------------------------------------------------------------- #
def test_identity_is_identity():
    identity = Pose3.identity()
    assert np.allclose(identity.rotation, np.eye(3))
    assert np.allclose(identity.translation, np.zeros(3))
    assert identity.is_finite()


def test_from_xy_yaw_maps_child_into_parent():
    pose = Pose3.from_xy_yaw(1.0, 2.0, math.pi / 2)
    point = pose.rotation @ np.array([1.0, 0.0, 0.0]) + pose.translation
    # +x in the child becomes +y in the parent
    assert point == pytest.approx((1.0, 3.0, 0.0), abs=1e-12)
    assert pose.yaw == pytest.approx(math.pi / 2)


def test_compose_matches_homogeneous_multiplication():
    a = Pose3.from_rpy(1.0, 2.0, 0.5, 0.1, -0.2, 0.3)
    b = Pose3.from_rpy(-0.4, 0.7, 0.2, 0.0, 0.5, -0.9)
    composed = a.compose(b)
    assert np.allclose(
        composed.as_homogeneous(), a.as_homogeneous() @ b.as_homogeneous(), atol=1e-12
    )


def test_inverse_round_trips():
    pose = Pose3.from_rpy(0.3, -1.2, 0.7, 0.2, 0.1, -0.4)
    assert pose.compose(pose.inverse()).almost_equal(Pose3.identity(), tol=1e-12)
    assert pose.inverse().compose(pose).almost_equal(Pose3.identity(), tol=1e-12)


def test_homogeneous_round_trip():
    pose = Pose3.from_rpy(1.0, 2.0, 3.0, 0.4, 0.5, 0.6)
    assert Pose3.from_homogeneous(pose.as_homogeneous()).almost_equal(pose, tol=1e-12)


def test_as_rpy_round_trip():
    pose = Pose3.from_rpy(1.0, 2.0, 3.0, 0.4, -0.5, 0.6)
    assert pose.as_rpy() == pytest.approx((1.0, 2.0, 3.0, 0.4, -0.5, 0.6), abs=1e-9)


def test_quaternion_xyzw_property():
    pose = Pose3.from_xy_yaw(0.0, 0.0, math.pi / 2)
    qx, qy, qz, qw = pose.quaternion_xyzw
    assert (qx, qy, qz, qw) == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)), abs=1e-12)


def test_is_finite_detects_nan():
    good = Pose3.from_xy_yaw(0.0, 0.0, 0.0)
    assert good.is_finite()
    bad = Pose3(np.full((3, 3), np.nan), np.zeros(3))
    assert not bad.is_finite()


# --------------------------------------------------------------------------- #
# angles and look-at
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "angle,expected",
    [
        (0.0, 0.0),
        (math.pi, math.pi),
        (-math.pi, math.pi),
        (3 * math.pi, math.pi),
        (2 * math.pi + 0.5, 0.5),
        (-2 * math.pi - 0.5, -0.5),
    ],
)
def test_normalize_angle(angle, expected):
    assert normalize_angle(angle) == pytest.approx(expected, abs=1e-12)


def test_look_at_rotation_points_z_along_the_target():
    rotation = look_at_rotation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert rotation[:, 2] == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    with pytest.raises(ValueError):
        look_at_rotation((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
