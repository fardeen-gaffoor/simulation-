"""Fiducial marker board geometry tests.

The boards are the only absolute reference the rover gets for free, so their
ground-truth poses must be exact.  These tests pin the two properties that are
easy to get wrong and silently produce an invisible board:

* the board must be offset from the wall face *into* the arena (the original
  implementation offset it the wrong way and buried it in the wall), and
* the board's local +z (its viewing direction) must point into the arena, away
  from the wall it is mounted on.
"""

from __future__ import annotations

import math

import pytest

from cat_arena.markers import (
    marker_poses,
    markers_yaml_document,
    quaternion_to_rpy,
    rpy_to_quaternion,
)
from cat_arena.sdf import rpy_matrix

#: The direction that points from the wall face INTO the arena.
_INTO_ARENA = {
    "north_wall": (0.0, -1.0),
    "south_wall": (0.0, 1.0),
    "west_wall": (1.0, 0.0),
    "east_wall": (-1.0, 0.0),
}


def _local_z(rpy):
    """World direction of a board's local +z axis for the given RPY."""
    matrix = rpy_matrix(*rpy)
    return (matrix[0][2], matrix[1][2], matrix[2][2])


def test_one_pose_per_placement(cfg):
    assert len(marker_poses(cfg)) == len(cfg.markers.placements)


@pytest.mark.parametrize("placement_index", [0, 1, 2])
def test_board_is_offset_into_the_arena(cfg, placement_index):
    placement = cfg.markers.placements[placement_index]
    pose = marker_poses(cfg)[placement_index]
    markers = cfg.markers

    nx, ny = _INTO_ARENA[placement.mount]
    offset = markers.stand_off + 0.5 * markers.board_thickness
    expected = (
        placement.position[0] + nx * offset,
        placement.position[1] + ny * offset,
    )

    assert pose.position[0] == pytest.approx(expected[0], abs=1e-9)
    assert pose.position[1] == pytest.approx(expected[1], abs=1e-9)

    # the whole board (up to its thickness) must lie on the arena side of the wall
    half = 0.5 * markers.board_thickness
    if placement.mount == "north_wall":
        assert pose.position[1] + half <= cfg.width_y + 1e-9
        assert pose.position[1] > 0.0
    elif placement.mount == "south_wall":
        assert pose.position[1] - half >= -1e-9
    elif placement.mount == "west_wall":
        assert pose.position[0] + half <= cfg.length_x + 1e-9
        assert pose.position[0] > 0.0
    elif placement.mount == "east_wall":
        assert pose.position[0] - half >= -1e-9


@pytest.mark.parametrize("placement_index", [0, 1, 2])
def test_board_looks_into_the_arena(cfg, placement_index):
    placement = cfg.markers.placements[placement_index]
    pose = marker_poses(cfg)[placement_index]
    lx, ly, lz = _local_z(pose.rpy)
    nx, ny = _INTO_ARENA[placement.mount]
    assert (lx, ly, lz) == pytest.approx((nx, ny, 0.0), abs=1e-9)


def test_boards_are_upright(cfg):
    """A board's plate spans local +x (vertical) and +y (wall tangent)."""
    for pose in marker_poses(cfg):
        matrix = rpy_matrix(*pose.rpy)
        # local +x in world coordinates must be vertical: the plate stands up
        ux = (matrix[0][0], matrix[1][0], matrix[2][0])
        assert abs(ux[2]) == pytest.approx(1.0, abs=1e-9)
        # and local +z (viewing direction) must be horizontal (no pitch)
        assert _local_z(pose.rpy)[2] == pytest.approx(0.0, abs=1e-9)


def test_boards_respect_the_height_limit(cfg):
    for pose in marker_poses(cfg):
        assert pose.top_z <= 0.60 + 1e-9
        assert pose.top_z == pytest.approx(cfg.markers.center_height + 0.5 * cfg.markers.board_size)


def test_board_normal_matches_the_reported_normal(cfg):
    for pose in marker_poses(cfg):
        lx, ly, lz = _local_z(pose.rpy)
        assert (lx, ly, lz) == pytest.approx(pose.normal, abs=1e-9)


def test_quaternion_round_trips_through_rpy():
    for rpy in [(0.0, math.pi / 2, -math.pi / 2), (0.0, math.pi / 2, 0.0), (0.1, -0.2, 0.3)]:
        q = rpy_to_quaternion(*rpy)
        norm = math.sqrt(sum(c * c for c in q))
        assert norm == pytest.approx(1.0)
        back = quaternion_to_rpy(q)
        # compare via rotation matrices to avoid Euler ambiguity
        a = rpy_matrix(*rpy)
        b = rpy_matrix(*back)
        for i in range(3):
            for j in range(3):
                assert a[i][j] == pytest.approx(b[i][j], abs=1e-9)


def test_markers_yaml_document_shape(cfg):
    document = markers_yaml_document(cfg, seed=11)
    assert document["count"] == len(cfg.markers.placements)
    assert document["family"] == cfg.markers.family
    assert document["limits"]["max_count"] == cfg.markers.max_count
    assert len(document["apriltag_ros"]["standalone_tags"]) == document["count"]
    for tag in document["apriltag_ros"]["standalone_tags"]:
        assert tag["size"] == pytest.approx(cfg.markers.tag_size)
        q = tag["pose"]["orientation"]
        norm = math.sqrt(q["x"] ** 2 + q["y"] ** 2 + q["z"] ** 2 + q["w"] ** 2)
        assert norm == pytest.approx(1.0)
