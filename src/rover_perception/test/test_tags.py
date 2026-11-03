"""``tags.yaml``: module 1's board placement turned into tag poses.

The standoff is the part to get right.  Module 1 centres a board *on* the wall
plane; the tag's front face is half a board thickness in front of that.  If
module 4 forgets the standoff, every fix is biased by a centimetre - small
enough to look plausible, large enough to matter.  So the tests check the
arithmetic against ``perception.yaml``'s configured standoff and check the
catalogue survives a write/read cycle byte for byte.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest
import yaml

from rover_perception import tags as tags_module
from rover_perception.markers_ref import load_markers
from rover_perception.tags import (
    SCHEMA_VERSION,
    TagCatalogue,
    TagsError,
    build_document,
    default_tags_path,
    load_tags,
    write_catalogue,
)


def test_build_document_lists_every_board(markers_path, cfg):
    document = build_document(load_markers(markers_path), cfg.tags.board, frame_id=cfg.arena.frame_id)
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["count"] == 3
    assert [entry["id"] for entry in document["tags"]] == [0, 1, 2]
    assert document["family"] == "tag36h11"
    assert document["frame"] == cfg.arena.frame_id


def test_standoff_is_applied_along_the_board_normal(markers_path, cfg):
    markers = load_markers(markers_path)
    document = build_document(markers, cfg.tags.board, frame_id="map")
    for board, entry in zip(markers.markers, document["tags"]):
        normal = np.asarray(entry["normal"], dtype=float)
        board_position = np.asarray(entry["board_position"], dtype=float)
        position = np.asarray(entry["position"], dtype=float)
        assert board_position == pytest.approx(np.asarray(board.position), abs=1e-9)
        assert position == pytest.approx(board_position + normal * cfg.tags.board.standoff_m, abs=1e-9)
        assert entry["standoff_m"] == pytest.approx(cfg.tags.board.standoff_m)


def test_source_record_is_auditable(markers_path, cfg):
    document = build_document(load_markers(markers_path), cfg.tags.board, frame_id="map")
    source = document["source"]
    assert source["markers"] == os.path.abspath(markers_path)
    with open(markers_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    assert source["markers_sha256"] == digest
    assert source["protrusion_m"] == pytest.approx(cfg.tags.board.protrusion_m)
    assert source["lift_m"] == pytest.approx(cfg.tags.board.lift_m)


def test_tag_sizes_and_drawing(markers_path, cfg):
    document = build_document(load_markers(markers_path), cfg.tags.board, frame_id="map")
    for entry in document["tags"]:
        assert entry["size_m"] == pytest.approx(0.18)
        assert entry["drawing_size_m"] == pytest.approx(0.225)
        assert entry["texture"] == f"tag_{entry['id']}_tag36h11.png"
        assert np.linalg.norm(np.asarray(entry["normal"], dtype=float)) == pytest.approx(1.0)


def test_normals_face_into_the_arena(markers_path, cfg):
    """A tag whose front face pointed at the wall could never be seen."""
    document = build_document(load_markers(markers_path), cfg.tags.board, frame_id="map")
    for entry in document["tags"]:
        position = np.asarray(entry["position"], dtype=float)
        normal = np.asarray(entry["normal"], dtype=float)
        # stepping along the normal from a wall-mounted tag must move inward
        inward = position + normal * 0.5
        assert cfg.arena.inside(inward[0], inward[1], margin=0.0), entry["id"]


def test_quaternion_matches_rpy(markers_path, cfg):
    from rover_perception.geometry import quaternion_to_matrix, rpy_matrix

    document = build_document(load_markers(markers_path), cfg.tags.board, frame_id="map")
    for entry in document["tags"]:
        assert quaternion_to_matrix(entry["quaternion_xyzw"]) == pytest.approx(
            rpy_matrix(*entry["rpy"]), abs=1e-9
        )


def test_tag_frame_is_the_viewer_frame(markers_path, cfg):
    """The catalogue rotation must be the one a detector reports.

    Measured fact (``test_tagdetect.py``): AprilTag's frame is the viewer's frame
    - ``+x`` right, ``+y`` down, ``+z`` *into* the board.  Getting this wrong
    flips a fix by ~180 degrees, so the reference boards are pinned by hand here.
    """
    document = build_document(load_markers(markers_path), cfg.tags.board, frame_id="map")
    by_id = {entry["id"]: entry for entry in document["tags"]}

    north = by_id[0]
    # north wall, face points south into the arena: the viewer stands south of it
    # looking north, so their right is east (+x) and down is -z, and the frame's
    # +z points north (into the wall)
    from rover_perception.geometry import rpy_matrix

    north_rotation = rpy_matrix(*north["rpy"])
    assert north_rotation[:, 0] == pytest.approx((1.0, 0.0, 0.0), abs=1e-9), "x = east"
    assert north_rotation[:, 1] == pytest.approx((0.0, 0.0, -1.0), abs=1e-9), "y = down"
    assert north_rotation[:, 2] == pytest.approx((0.0, 1.0, 0.0), abs=1e-9), "z = into the wall"
    assert north["normal"] == pytest.approx((0.0, -1.0, 0.0), abs=1e-9), "the face still points south"
    assert np.asarray(north["frame_z"]) == pytest.approx(-np.asarray(north["normal"]), abs=1e-9)

    west = by_id[1]
    west_rotation = rpy_matrix(*west["rpy"])
    # west wall, face points east: the viewer faces west, so their right is north
    assert west_rotation[:, 0] == pytest.approx((0.0, 1.0, 0.0), abs=1e-9), "x = north"
    assert west_rotation[:, 1] == pytest.approx((0.0, 0.0, -1.0), abs=1e-9), "y = down"
    assert west_rotation[:, 2] == pytest.approx((-1.0, 0.0, 0.0), abs=1e-9), "z = into the wall"


def test_tag_frame_rotation_is_orthonormal_and_right_handed(markers_path, cfg):
    from rover_perception.tags import tag_frame_rotation

    document = build_document(load_markers(markers_path), cfg.tags.board, frame_id="map")
    for entry in document["tags"]:
        rotation = tag_frame_rotation(np.asarray(entry["normal"], dtype=float))
        assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
        assert float(np.linalg.det(rotation)) == pytest.approx(1.0)
        assert rotation[:, 2] == pytest.approx(-np.asarray(entry["normal"], dtype=float), abs=1e-9)


def test_tag_frame_rotation_rejects_a_zero_normal():
    from rover_perception.tags import tag_frame_rotation

    with pytest.raises(TagsError):
        tag_frame_rotation(np.zeros(3))


def test_face_normal_and_pose_are_opposite(catalogue):
    for tag_id in catalogue.ids:
        tag = catalogue.tag(tag_id)
        assert tag.face_normal == pytest.approx(-tag.pose.rotation[:, 2], abs=1e-9)
        assert tag.face_normal == pytest.approx(np.asarray(tag.normal), abs=1e-9)


# --------------------------------------------------------------------------- #
# the round trip
# --------------------------------------------------------------------------- #
def test_write_and_reload(catalogue, markers_path):
    assert isinstance(catalogue, TagCatalogue)
    assert catalogue.ids == (0, 1, 2)
    assert catalogue.count == 3
    assert catalogue.family == "tag36h11"
    assert catalogue.source_markers == os.path.abspath(markers_path)
    assert catalogue.size_m(0) == pytest.approx(0.18)
    assert catalogue.has(0) and not catalogue.has(99)


def test_catalogue_summary(catalogue):
    summary = catalogue.summary()
    assert summary["count"] == 3
    assert summary["ids"] == [0, 1, 2]
    assert summary["family"] == "tag36h11"


def test_catalogue_lookup_error(catalogue):
    with pytest.raises(TagsError) as excinfo:
        catalogue.tag(99)
    assert "99" in str(excinfo.value)


def test_written_file_is_deterministic(markers_path, cfg, tmp_path):
    document = build_document(load_markers(markers_path), cfg.tags.board, frame_id="map")
    first = write_catalogue(str(tmp_path / "a.yaml"), document)
    second = write_catalogue(str(tmp_path / "b.yaml"), document)
    with open(first, "rb") as handle_a, open(second, "rb") as handle_b:
        assert handle_a.read() == handle_b.read()
    # the header records what generated it, so a stale file is obvious
    with open(first, "r", encoding="utf-8") as handle:
        assert handle.readline().startswith("# rover_perception tag catalogue")


def test_load_rejects_a_missing_file():
    with pytest.raises(TagsError) as excinfo:
        load_tags("/nonexistent/tags.yaml")
    assert "generate_tags" in str(excinfo.value)


def test_load_rejects_a_bad_schema(catalogue, tmp_path):
    document = _reload(catalogue.path)
    document["schema_version"] = 99
    with pytest.raises(TagsError):
        load_tags(_write(tmp_path, document))


def test_load_rejects_duplicate_ids(catalogue, tmp_path):
    document = _reload(catalogue.path)
    document["tags"].append(dict(document["tags"][0]))
    with pytest.raises(TagsError) as excinfo:
        load_tags(_write(tmp_path, document))
    assert "duplicate" in str(excinfo.value)


def test_load_rejects_no_tags(tmp_path):
    document = {"schema_version": SCHEMA_VERSION, "family": "tag36h11", "tags": []}
    with pytest.raises(TagsError):
        load_tags(_write(tmp_path, document))


def test_load_rejects_a_short_position(catalogue, tmp_path):
    document = _reload(catalogue.path)
    document["tags"][0]["position"] = [1.0, 2.0]
    with pytest.raises(TagsError):
        load_tags(_write(tmp_path, document))


def test_load_rejects_a_non_positive_size(catalogue, tmp_path):
    document = _reload(catalogue.path)
    document["tags"][0]["size_m"] = 0.0
    with pytest.raises(TagsError):
        load_tags(_write(tmp_path, document))


def test_default_tags_path():
    assert default_tags_path("") == os.path.join(os.getcwd(), "tags.yaml")
    assert default_tags_path("/tmp/x") == os.path.join("/tmp/x", "tags.yaml")
    assert default_tags_path("/tmp/x", "other.yaml") == os.path.join("/tmp/x", "other.yaml")


def _reload(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _write(tmp_path, document):
    target = tmp_path / "tags.yaml"
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return str(target)


def test_module_exports_the_expected_names():
    for name in ("TagsError", "TagSpec", "TagCatalogue", "build_document", "write_catalogue", "load_tags"):
        assert hasattr(tags_module, name)
