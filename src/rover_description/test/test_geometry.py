"""Tests for the analytic geometry: rpy, inertias and the envelope AABB."""

from __future__ import annotations

import math

import pytest

from rover_description.geometry import (
    box_inertia,
    compute_envelope,
    cylinder_inertia_z,
    ladder_box_center,
    link_inertias,
    rotate,
    rpy_matrix,
    wheel_centers,
)


def test_rpy_identity():
    m = rpy_matrix(0.0, 0.0, 0.0)
    assert m == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_rpy_orthonormal_and_proper(cfg):
    m = rpy_matrix(0.3, -0.4, 1.1)
    # rows orthonormal
    for i in range(3):
        assert sum(m[i][k] ** 2 for k in range(3)) == pytest.approx(1.0)
    for i in range(3):
        for j in range(i + 1, 3):
            assert sum(m[i][k] * m[j][k] for k in range(3)) == pytest.approx(0.0)
    # determinant +1
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    assert det == pytest.approx(1.0)


def test_positive_pitch_points_down():
    # R_y(pitch) @ ex = (cos, 0, -sin): positive pitch -> downward
    m = rpy_matrix(0.0, math.radians(20.0), 0.0)
    v = rotate(m, (1.0, 0.0, 0.0))
    assert v[2] < 0.0


def test_box_inertia_cube():
    ixx, ixy, ixz, iyy, iyz, izz = box_inertia(12.0, (2.0, 2.0, 2.0))
    assert ixx == pytest.approx(8.0)
    assert iyy == pytest.approx(8.0)
    assert izz == pytest.approx(8.0)
    assert (ixy, ixz, iyz) == (0.0, 0.0, 0.0)


def test_box_inertia_rectangular():
    # 3 x 2 x 1 box, mass 6
    ixx, _, _, iyy, _, izz = box_inertia(6.0, (3.0, 2.0, 1.0))
    assert ixx == pytest.approx(6.0 * (4.0 + 1.0) / 12.0)
    assert iyy == pytest.approx(6.0 * (9.0 + 1.0) / 12.0)
    assert izz == pytest.approx(6.0 * (9.0 + 4.0) / 12.0)


def test_cylinder_inertia_z():
    ixx, _, _, iyy, _, izz = cylinder_inertia_z(6.0, 1.0, 2.0)
    assert ixx == pytest.approx(6.0 * (3.0 + 4.0) / 12.0)
    assert iyy == pytest.approx(ixx)
    assert izz == pytest.approx(6.0 * 1.0 / 2.0)


def test_inertias_physical(cfg):
    for name, iner in link_inertias(cfg).items():
        assert iner.mass > 0.0
        for moment in (iner.ixx, iner.iyy, iner.izz):
            assert moment > 0.0
        # triangle inequality holds for any rigid body
        assert iner.ixx + iner.iyy >= iner.izz - 1e-9
        assert iner.iyy + iner.izz >= iner.ixx - 1e-9
        assert iner.izz + iner.ixx >= iner.iyy - 1e-9


def test_link_inertias_keys(cfg):
    keys = set(link_inertias(cfg))
    assert {
        "base_link",
        "wheel_front_left",
        "wheel_front_right",
        "wheel_rear_left",
        "wheel_rear_right",
        "ladder_link",
        "belt_link",
        "tub_link",
        "funnel_link",
    } <= keys


def test_wheel_centers_symmetric(cfg):
    c = wheel_centers(cfg)
    assert c["wheel_front_left"][0] == pytest.approx(-c["wheel_rear_left"][0])
    assert c["wheel_front_left"][1] == pytest.approx(-c["wheel_front_right"][1])
    assert c["wheel_front_left"][2] == pytest.approx(cfg.drive.wheel_radius)


def test_envelope_within_limits(cfg):
    env = compute_envelope(cfg)
    limit = cfg.envelope_limit
    assert env.length_x <= limit.length_x + 1e-9
    assert env.width_y <= limit.width_y + 1e-9
    assert env.height_z <= limit.height_z + 1e-9


def test_envelope_ground_contact(cfg):
    env = compute_envelope(cfg)
    # wheels define the bottom of the robot at z = 0
    assert env.z_min == pytest.approx(0.0, abs=1e-6)
    assert env.z_max > 0.0


def test_ladder_stowed_is_raised(cfg):
    # stowed_pitch is negative -> the ladder tip is above the pivot
    pitch = math.radians(cfg.excavation.ladder.stowed_pitch_deg)
    center = ladder_box_center(cfg, pitch)
    assert center[2] > cfg.excavation.ladder.pivot[2]
