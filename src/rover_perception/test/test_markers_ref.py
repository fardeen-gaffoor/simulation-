"""Module 1's ``markers.yaml``: the wall fiducial catalogue.

Module 4 never imports ``cat_arena``; it reads this file.  The tests therefore
check both the reader's validation and the invariants module 4 *depends* on
(the board normal points into the arena, the quaternion agrees with the rpy, the
tag fits on the board) because a violation there is a wrong tag pose here.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import yaml

from rover_perception.markers_ref import MarkersError, load_markers, resolve_markers_path
from support import write_markers

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORLDS = os.path.expanduser("~/cat/worlds")


def test_reference_markers_load(markers_path):
    doc = load_markers(markers_path)
    assert doc.family == "tag36h11"
    assert doc.tag_size == pytest.approx(0.18)
    assert doc.board_size == pytest.approx(0.25)
    assert doc.ids == [0, 1, 2]


def test_board_poses_and_normals(markers_path):
    doc = load_markers(markers_path)
    north = doc.by_id(0)
    assert north.position == pytest.approx((1.0, 4.983, 0.4))
    assert north.normal == pytest.approx((0.0, -1.0, 0.0)), "the north board must face south"
    assert north.mount == "north_wall"
    west = doc.by_id(1)
    assert west.normal == pytest.approx((1.0, 0.0, 0.0)), "the west board must face east"
    mid = doc.by_id(2)
    assert mid.position[0] == pytest.approx(6.0)


def test_normal_agrees_with_rpy(markers_path):
    """``normal`` and ``rpy`` describe the same rotation - module 4 relies on it."""
    doc = load_markers(markers_path)
    for board in doc.markers:
        from rover_perception.geometry import rpy_matrix

        assert board.normal == pytest.approx(rpy_matrix(*board.rpy)[:, 2], abs=1e-9)


def test_quaternion_agrees_with_rpy(markers_path):
    doc = load_markers(markers_path)
    for board in doc.markers:
        assert board.standoff_pose.rotation == pytest.approx(board.pose.rotation, abs=1e-9)
        assert board.standoff_pose.translation == pytest.approx(board.pose.translation, abs=1e-9)


def test_boards_are_inside_the_arena_and_low_enough(markers_path, cfg):
    doc = load_markers(markers_path)
    for board in doc.markers:
        assert cfg.arena.inside(board.position[0], board.position[1], margin=0.0)
        assert board.top_z <= 0.60 + 1e-9, "competition rule: boards are <= 60 cm high"


def test_at_most_five_boards_and_no_duplicate_ids(markers_path):
    doc = load_markers(markers_path)
    assert len(doc.ids) <= 5
    assert len(set(doc.ids)) == len(doc.ids)


def test_tag_fits_on_the_board(markers_path):
    doc = load_markers(markers_path)
    for board in doc.markers:
        assert board.tag_size <= board.board_size + 1e-9


def test_by_id_missing_raises(markers_path):
    doc = load_markers(markers_path)
    with pytest.raises(MarkersError):
        doc.by_id(99)


# --------------------------------------------------------------------------- #
# reader validation
# --------------------------------------------------------------------------- #
def test_explicit_path_wins(tmp_path, markers_path):
    assert resolve_markers_path(markers_path) == os.path.abspath(markers_path)


def test_explicit_missing_path_is_an_error(tmp_path):
    """A typo'd --markers must not silently resolve to another arena's file."""
    with pytest.raises(MarkersError) as excinfo:
        resolve_markers_path(str(tmp_path / "typo.yaml"))
    assert "requested path" in str(excinfo.value)


def test_missing_file_is_reported():
    with pytest.raises(MarkersError) as excinfo:
        load_markers("/nonexistent/markers.yaml")
    assert "requested path" in str(excinfo.value)


def test_wrong_schema_version_is_rejected(tmp_path):
    path = str(tmp_path / "markers.yaml")
    write_markers(path, [{"id": 0, "position": (1.0, 4.0, 0.4)}])
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document["schema_version"] = 3
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle)
    with pytest.raises(MarkersError):
        load_markers(path)


def test_no_markers_is_rejected(tmp_path):
    path = str(tmp_path / "markers.yaml")
    write_markers(path, [])
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document["markers"] = []
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle)
    with pytest.raises(MarkersError):
        load_markers(path)


def test_duplicate_ids_are_rejected(tmp_path):
    path = str(tmp_path / "markers.yaml")
    write_markers(
        path,
        [
            {"id": 0, "position": (1.0, 4.0, 0.4)},
            {"id": 0, "position": (2.0, 4.0, 0.4)},
        ],
    )
    with pytest.raises(MarkersError) as excinfo:
        load_markers(path)
    assert "duplicate" in str(excinfo.value)


def test_tag_larger_than_the_board_is_rejected(tmp_path):
    path = str(tmp_path / "markers.yaml")
    write_markers(path, [{"id": 0, "position": (1.0, 4.0, 0.4)}], board_size=0.10, tag_size=0.18)
    with pytest.raises(MarkersError):
        load_markers(path)


def test_zero_size_is_rejected(tmp_path):
    path = str(tmp_path / "markers.yaml")
    write_markers(path, [{"id": 0, "position": (1.0, 4.0, 0.4)}], tag_size=0.0)
    with pytest.raises(MarkersError):
        load_markers(path)


def test_short_position_vector_is_rejected(tmp_path):
    path = str(tmp_path / "markers.yaml")
    write_markers(path, [{"id": 0, "position": (1.0, 4.0, 0.4)}])
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document["markers"][0]["position"] = [1.0, 4.0]
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle)
    with pytest.raises(MarkersError):
        load_markers(path)


def test_normal_falls_back_to_the_rpy_when_absent(tmp_path):
    path = str(tmp_path / "markers.yaml")
    write_markers(
        path,
        [{"id": 0, "position": (1.0, 4.0, 0.4), "rpy": (0.0, math.pi / 2, 0.0)}],
        include_normal=False,
    )
    doc = load_markers(path)
    assert doc.by_id(0).normal == pytest.approx((0.0, 0.0, 0.0))  # absent -> zero vector
    # tags.py is the layer that falls back to the rpy-derived normal
    from rover_perception import tags

    spec = tags.build_document(doc, _board_cfg(), frame_id="map")["tags"][0]
    assert spec["normal"] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def _board_cfg():
    from rover_perception.config import load_config

    return load_config(os.path.join(_HERE, "..", "config", "perception.yaml")).tags.board


# --------------------------------------------------------------------------- #
# the real generated world, when this machine has one
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.isdir(_WORLDS), reason="no generated worlds on this machine")
def test_generated_world_markers_load_if_present():
    import glob

    found = sorted(glob.glob(os.path.join(_WORLDS, "seed_*", "markers.yaml")))
    if not found:
        pytest.skip("no markers.yaml in any generated world")
    for path in found:
        doc = load_markers(path)
        assert doc.ids, f"{path} has no boards"
        for board in doc.markers:
            assert np.linalg.norm(board.normal) == pytest.approx(1.0, abs=1e-6)
