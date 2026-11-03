"""Poses and angles: the conventions every other module in the stack assumes.

Module 5 may not import module 4 (they must be independently deployable), so it
restates the pose conventions itself.  That makes this file the place the
restatement is checked: ZYX rpy, ``(x, y, z, w)`` quaternions, right-handed
rotations applied as ``R @ v``, and ``wrap_angle`` behaving on the exactly-pi
edge that the controller's heading tests sit on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rover_navigation.geometry import (
    Pose3,
    angle_difference,
    matrix_to_quaternion,
    planarize,
    quaternion_to_matrix,
    rotation_matrix_to_rpy,
    rpy_matrix,
    unit,
    wrap_angle,
    yaw_to_quaternion,
)


# --------------------------------------------------------------------------- #
# angles
# --------------------------------------------------------------------------- #
def test_wrap_angle_lands_in_the_half_open_interval():
    for angle in (0.0, 1.0, math.pi, -math.pi, 3 * math.pi, -3 * math.pi, 100.0):
        wrapped = wrap_angle(angle)
        assert -math.pi < wrapped <= math.pi


def test_wrap_angle_is_idempotent():
    for angle in (0.3, -2.9, 12.5, -0.0):
        assert wrap_angle(wrap_angle(angle)) == pytest.approx(wrap_angle(angle))


def test_wrap_angle_maps_pi_to_pi_and_minus_pi_to_pi():
    # The convention is (-pi, pi]: exactly-pi is positive, so a 180 deg turn has
    # one representation and `abs(error) <= tol` cannot flip sign at the edge.
    assert wrap_angle(math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi) == pytest.approx(math.pi)


def test_angle_difference_takes_the_short_way_round():
    assert angle_difference(0.1, 0.0) == pytest.approx(0.1)
    # +170 deg to -170 deg: the short way is -20 deg, not +340.
    assert angle_difference(math.radians(170), math.radians(-170)) == pytest.approx(math.radians(-20))
    assert angle_difference(math.radians(-170), math.radians(170)) == pytest.approx(math.radians(20))


def test_angle_difference_is_antisymmetric():
    for a, b in ((0.4, 1.9), (-2.5, 2.5), (3.0, -3.0)):
        assert angle_difference(a, b) == pytest.approx(-angle_difference(b, a))


def test_unit_handles_the_degenerate_input():
    assert unit(0.0, 0.0) == (1.0, 0.0)
    assert unit(3.0, 4.0) == pytest.approx((0.6, 0.8))


# --------------------------------------------------------------------------- #
# rotations
# --------------------------------------------------------------------------- #
def test_rpy_matrix_matches_hand_written_yaw():
    yaw = 0.7
    expected = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert rpy_matrix(0.0, 0.0, yaw) == pytest.approx(expected)


def test_rpy_round_trips():
    for roll, pitch, yaw in (
        (0.0, 0.0, 0.0),
        (0.2, -0.3, 1.1),
        (-0.6, 0.4, -2.2),
    ):
        recovered = rotation_matrix_to_rpy(rpy_matrix(roll, pitch, yaw))
        assert recovered == pytest.approx((roll, pitch, yaw))


def test_positive_pitch_tilts_the_forward_axis_down():
    """The convention modules 2 and 4 share: +pitch about +y points +x downwards."""
    down = rpy_matrix(0.0, 0.3, 0.0) @ np.array([1.0, 0.0, 0.0])
    assert down[2] < 0.0
    assert down[0] == pytest.approx(math.cos(0.3))


def test_rotation_matrix_to_rpy_rejects_a_bad_shape():
    with pytest.raises(ValueError):
        rotation_matrix_to_rpy(np.eye(4))


def test_quaternion_of_a_pure_yaw():
    x, y, z, w = yaw_to_quaternion(math.pi / 2)
    assert (x, y, w) == pytest.approx((0.0, 0.0, math.cos(math.pi / 4)))
    assert z == pytest.approx(math.sin(math.pi / 4))


def test_quaternion_round_trip():
    for roll, pitch, yaw in ((0.0, 0.0, 0.0), (0.3, -0.2, 2.4), (-0.1, 0.9, -1.3)):
        matrix = rpy_matrix(roll, pitch, yaw)
        recovered = quaternion_to_matrix(matrix_to_quaternion(matrix))
        assert recovered == pytest.approx(matrix, abs=1e-9)


def test_quaternion_to_matrix_normalises_a_noisy_message():
    """A message is not guaranteed normalised; the conversion must still be a rotation."""
    matrix = quaternion_to_matrix((0.0, 0.0, 0.0, 4.0))
    assert matrix == pytest.approx(np.eye(3))
    assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_quaternion_to_matrix_survives_a_zeroed_message():
    assert quaternion_to_matrix((0.0, 0.0, 0.0, 0.0)) == pytest.approx(np.eye(3))


# --------------------------------------------------------------------------- #
# poses
# --------------------------------------------------------------------------- #
def test_identity_is_the_origin():
    pose = Pose3.identity()
    assert pose.as_rpy() == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert pose.xy == (0.0, 0.0)


def test_from_xy_yaw_round_trips_through_the_accessors():
    pose = Pose3.from_xy_yaw(1.5, -2.25, 0.75, z=0.3)
    assert (pose.x, pose.y, pose.z) == pytest.approx((1.5, -2.25, 0.3))
    assert pose.yaw == pytest.approx(0.75)
    x, y, z, roll, pitch, yaw = pose.as_rpy()
    assert (x, y, z) == pytest.approx((1.5, -2.25, 0.3))
    assert (roll, pitch) == pytest.approx((0.0, 0.0))
    assert yaw == pytest.approx(0.75)


def test_compose_is_matrix_multiplication():
    a = Pose3.from_xy_yaw(1.0, 2.0, math.pi / 2)
    b = Pose3.from_xy_yaw(0.5, 0.0, 0.0)
    composed = a.compose(b)
    # b is half a metre along a's own +x, which is arena +y.
    assert composed.x == pytest.approx(1.0, abs=1e-9)
    assert composed.y == pytest.approx(2.5, abs=1e-9)
    assert composed.yaw == pytest.approx(math.pi / 2)


def test_compose_identity_is_a_no_op():
    pose = Pose3.from_xy_yaw(3.0, 1.0, -0.4)
    composed = pose.compose(Pose3.identity())
    assert composed.as_rpy() == pytest.approx(pose.as_rpy())


def test_inverse_undoes_compose():
    a = Pose3.from_xy_yaw(1.0, 2.0, 0.5)
    b = Pose3.from_xy_yaw(-3.0, 0.25, -1.2)
    assert a.compose(b).compose(b.inverse()).as_rpy() == pytest.approx(a.as_rpy())
    assert a.inverse().compose(a).as_rpy() == pytest.approx(Pose3.identity().as_rpy())


def test_compose_order_matters():
    a = Pose3.from_xy_yaw(0.0, 0.0, math.pi / 2)
    b = Pose3.from_xy_yaw(1.0, 0.0, 0.0)
    assert a.compose(b).y == pytest.approx(1.0)
    assert b.compose(a).y == pytest.approx(0.0)


def test_transform_point_matches_compose():
    pose = Pose3.from_xy_yaw(1.0, 2.0, 0.3)
    point = pose.transform_point((0.5, -0.25, 0.0))
    assert point[:2] == pytest.approx(pose.compose(Pose3.from_xy_yaw(0.5, -0.25, 0.0)).xy)


def test_planar_distance_and_angle_to():
    a = Pose3.from_xy_yaw(0.0, 0.0, 0.0)
    b = Pose3.from_xy_yaw(3.0, 4.0, math.pi / 2)
    assert a.planar_distance(b) == pytest.approx(5.0)
    assert a.planar_angle_to(b) == pytest.approx(math.pi / 2)
    assert b.planar_angle_to(a) == pytest.approx(-math.pi / 2)


def test_planar_distance_ignores_height():
    a = Pose3.from_xy_yaw(0.0, 0.0, 0.0, z=0.0)
    b = Pose3.from_xy_yaw(3.0, 4.0, 0.0, z=9.0)
    assert a.planar_distance(b) == pytest.approx(5.0)


def test_bearing_to_points_at_the_target():
    pose = Pose3.from_xy_yaw(1.0, 1.0, 0.0)
    assert pose.bearing_to(1.0, 5.0) == pytest.approx(math.pi / 2)
    assert pose.bearing_to(5.0, 1.0) == pytest.approx(0.0)
    assert pose.bearing_to(0.0, 1.0) == pytest.approx(math.pi)


def test_planarize_drops_height_and_tilt():
    tilted = Pose3(rpy_matrix(0.3, 0.2, 1.1), (1.0, 2.0, 0.4))
    flat = planarize(tilted)
    assert (flat.x, flat.y, flat.z) == pytest.approx((1.0, 2.0, 0.0))
    assert flat.yaw == pytest.approx(1.1)
    assert (flat.roll, flat.pitch) == pytest.approx((0.0, 0.0))


def test_from_pose_reads_a_ros_style_message():
    class _Vector:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Orientation:
        def __init__(self, x, y, z, w):
            self.x, self.y, self.z, self.w = x, y, z, w

    class _Pose:
        position = _Vector(2.0, 3.0, 0.1)
        orientation = _Orientation(0.0, 0.0, math.sin(0.2), math.cos(0.2))

    pose = Pose3.from_pose(_Pose())
    assert (pose.x, pose.y) == pytest.approx((2.0, 3.0))
    assert pose.yaw == pytest.approx(0.4)


def test_quaternion_xyzw_round_trips_through_the_message_conversion():
    pose = Pose3.from_xy_yaw(0.5, 0.6, 2.0)
    recovered = quaternion_to_matrix(pose.quaternion_xyzw)
    assert recovered == pytest.approx(pose.rotation, abs=1e-9)


def test_pose_is_hashable_and_value_equal():
    a = Pose3.from_xy_yaw(1.0, 2.0, 0.5)
    b = Pose3.from_xy_yaw(1.0, 2.0, 0.5)
    assert a == b
    assert len({a, b}) == 1
