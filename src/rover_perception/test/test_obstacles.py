"""The wall mask and the obstacle payload: perception's public promise.

The competition forbids using the walls as sensors, so a wall that looks like a
rock must be discarded by *known geometry* rather than fed to the planner.  And
because the payload is what module 5 consumes, its field names and shapes are
checked here against the message definition (``rover_perception_msgs/Obstacle``)
and the PointCloud2 layout nav2 reads.
"""

from __future__ import annotations

import numpy as np
import pytest

from rover_perception.config import load_config
from rover_perception.obstacles import (
    KIND_UNKNOWN,
    ORIENTATION_VARIANCE_FLOOR,
    POINT_FIELDS,
    build_obstacle_payload,
    covariance_from_sigma,
    mask_tracks,
    obstacle_fields,
    pointcloud_dtype,
    pointcloud_payload,
    wall_mask_reason,
)
from support import track

import os


@pytest.fixture(scope="module")
def arena():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_config(os.path.join(here, "..", "config", "perception.yaml")).arena


# --------------------------------------------------------------------------- #
# covariance
# --------------------------------------------------------------------------- #
def test_covariance_is_diagonal_with_an_orientation_floor():
    covariance = covariance_from_sigma(0.2)
    assert len(covariance) == 36
    assert covariance[0] == pytest.approx(0.04)
    assert covariance[7] == pytest.approx(0.04)
    assert covariance[14] == pytest.approx(0.04), "z uses xy when nothing else is known"
    for index in (21, 28, 35):
        assert covariance[index] == pytest.approx(ORIENTATION_VARIANCE_FLOOR)
    off_diagonal = [
        covariance[row * 6 + column]
        for row in range(6)
        for column in range(6)
        if row != column and row * 6 + column not in (21, 28, 35)
    ]
    assert max(abs(value) for value in off_diagonal) == 0.0


def test_covariance_uses_a_given_z_sigma_and_floors_zero():
    assert covariance_from_sigma(0.2, 0.05)[14] == pytest.approx(0.0025)
    assert covariance_from_sigma(0.0)[0] == pytest.approx(1e-12), "never publish a zero sigma"


# --------------------------------------------------------------------------- #
# the wall mask
# --------------------------------------------------------------------------- #
def test_a_mid_arena_track_is_a_legal_obstacle(arena):
    assert wall_mask_reason(track(4.0, 2.5), arena) is None


def test_a_track_outside_the_x_bounds_is_a_wall(arena):
    assert wall_mask_reason(track(9.5, 2.5), arena) == "outside_arena_x"
    assert wall_mask_reason(track(-0.5, 2.5), arena) == "outside_arena_x"


def test_a_track_outside_the_y_bounds_is_a_wall(arena):
    assert wall_mask_reason(track(4.0, 5.5), arena) == "outside_arena_y"
    assert wall_mask_reason(track(4.0, -0.2), arena) == "outside_arena_y"


def test_a_track_inside_the_margin_is_a_wall(arena):
    """Inside the rectangle but within wall_margin_m of a wall."""
    assert arena.wall_margin_m > 0.0
    assert wall_mask_reason(track(4.0, 0.05), arena) == "inside_wall_margin"
    assert wall_mask_reason(track(8.95, 2.5), arena) == "inside_wall_margin"


def test_mask_tracks_splits_and_keeps_every_rejection(arena):
    legal = track(4.0, 2.5, identifier="legal")
    wall = track(4.0, 5.5, identifier="wall")
    kept, dropped = mask_tracks([legal, wall], arena)
    assert [one.id for one in kept] == ["obstacle_legal"]
    assert len(dropped) == 1
    assert dropped[0][0].id == "obstacle_wall"
    assert dropped[0][1] == "outside_arena_y"


# --------------------------------------------------------------------------- #
# the payload
# --------------------------------------------------------------------------- #
def test_obstacle_fields_match_the_message(arena):
    fields = obstacle_fields(track(3.0, 2.0, kind="boulder", height=0.21, now=10.0), 10.0, "map")
    assert fields["kind"] == 1, "Boulder"
    assert fields["class_id"] == 0
    assert fields["z"] == pytest.approx(0.21)
    assert fields["height"] == pytest.approx(0.21)
    assert fields["depth"] == 0.0, "a boulder has no depth"
    assert fields["frame_id"] == "map"
    assert len(fields["covariance"]) == 36
    assert fields["age_s"] >= 0.0


def test_crater_fields_zero_the_height(arena):
    fields = obstacle_fields(track(3.0, 2.0, kind="crater", now=10.0), 10.0, "map")
    assert fields["kind"] == 2, "Crater"
    assert fields["class_id"] == 1
    assert fields["z"] == 0.0
    assert fields["height"] == 0.0
    assert fields["depth"] == 0.0, "monocular perception cannot measure crater depth"


def test_unknown_kind_maps_to_zero(arena):
    fields = obstacle_fields(track(3.0, 2.0, kind="sinkhole", now=0.0), 0.0, "map")
    assert fields["kind"] == KIND_UNKNOWN
    assert fields["class_id"] == 255


def test_payload_has_counters_and_every_obstacle():
    tracks = [track(3.0, 2.0), track(4.0, 2.0)]
    payload = build_obstacle_payload(tracks, 0.0, "map", {"frames_processed": 42, "dropped_wall": 3})
    assert payload["frame_id"] == "map"
    assert len(payload["obstacles"]) == 2
    assert payload["frames_processed"] == 42
    assert payload["dropped_wall"] == 3
    assert payload["dropped_range"] == 0
    assert payload["frames_without_fix"] == 0


def test_pointcloud_layout_is_xyzi_float32():
    assert pointcloud_dtype().itemsize == 16
    assert [name for name, _offset, _type in POINT_FIELDS] == ["x", "y", "z", "intensity"]
    assert all(offset == 4 * index for index, (_n, offset, _t) in enumerate(POINT_FIELDS))
    assert all(datatype == 7 for _n, _o, datatype in POINT_FIELDS)


def test_pointcloud_payload_carries_the_footprint_and_the_confidence():
    obstacles = [
        {"x": 3.0, "y": 2.0, "z": 0.2, "confidence": 0.8},
        {"x": 4.0, "y": 1.0, "z": 0.0, "confidence": 0.4},
    ]
    points, step = pointcloud_payload(obstacles, "map")
    assert step == 16
    assert points.shape == (2,)
    assert points["x"][0] == pytest.approx(3.0)
    assert points["z"][0] == pytest.approx(0.2)
    assert points["intensity"][1] == pytest.approx(0.4)
    assert points.dtype.names == ("x", "y", "z", "intensity")


def test_empty_pointcloud_is_a_valid_array():
    points, step = pointcloud_payload([], "map")
    assert points.shape == (0,)
    assert step == 16
    assert np.asarray(points).size == 0
