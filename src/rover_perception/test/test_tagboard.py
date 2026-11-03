"""The spawnable tag board: a textured slab whose pose comes from the CLI.

Spawn pose is deliberately *not* in the SDF, because a model that carries a pose
and is also spawned with ``-x -y -z`` has two authorities for the tag's position
and no way to tell which one won.  These tests pin that down, plus the texture
size the board is drawn at and the visual orientation that makes the bitmap
decode.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rover_perception import tagboard
from rover_perception.geometry import rpy_matrix
from rover_perception.tagboard import (
    TagBoardError,
    board_sdf,
    build_plan,
    file_uri,
    model_name,
    plan_document,
    spawn_arguments,
    write_plan,
)
from rover_perception.tagimages import render


# --------------------------------------------------------------------------- #
# the SDF
# --------------------------------------------------------------------------- #
def test_sdf_has_no_model_pose(catalogue, cfg):
    tag = catalogue.tag(0)
    rendered = render(0, tag.family, tag.size_m, cfg.tags.board.texture_size_px)
    text = board_sdf(tag, "file:///tmp/tag_0_tag36h11.png", rendered.drawing_size_m)
    root = ET.fromstring(text)
    assert root.tag == "sdf"
    model = root.find("model")
    assert model is not None
    assert model.get("name") == model_name(0)
    assert model.find("static") is not None and model.find("static").text == "true"
    assert model.find("pose") is None, "the spawn command owns the pose, not the SDF"


def test_sdf_flips_the_visual_to_face_out_of_the_wall(catalogue):
    """The model frame is AprilTag's (+z into the board), so the slab must turn.

    The slab's own normal is its local ``+z``; the visual's ``_TAG_FACE_RPY``
    points that normal *out* of the wall (so the tag is visible from the arena)
    and lands the *visible face* on the tag plane rather than half a thickness in
    front of it.
    """
    tag = catalogue.tag(0)
    root = ET.fromstring(board_sdf(tag, "file:///x.png", 0.225))
    visual_pose = root.find("model/link/visual/pose")
    assert visual_pose is not None, "the visual must carry the face pose"
    values = [float(v) for v in visual_pose.text.split()]
    assert values[:3] == pytest.approx(
        [0.0, 0.0, tagboard._BOARD_THICKNESS_M / 2.0], abs=1e-9
    )
    assert values[3:] == pytest.approx(list(tagboard._TAG_FACE_RPY), abs=1e-9)
    # composed with the tag frame the visible normal is the tag's face normal
    flip = rpy_matrix(*values[3:])
    visible_normal = tag.pose.rotation @ flip @ np.array([0.0, 0.0, 1.0])
    assert visible_normal == pytest.approx(tag.face_normal, abs=1e-9)
    assert np.asarray(tag.normal) == pytest.approx(tag.face_normal, abs=1e-9)


def test_sdf_box_uv_lands_on_the_viewer_frame(catalogue):
    """A gz box ``+z`` face maps texture ``u`` to local ``-y`` and ``v`` to local ``-x``.

    That was measured on Harmonic 8.15 with a four-quadrant texture (a plane has
    no UVs at all, which is why a box is used).  Composed with
    ``_TAG_FACE_RPY``, ``u`` must run along the model's ``+x`` (the viewer's
    right) and ``v`` along the model's ``+y`` (the viewer's down), so a head-on
    viewer sees the bitmap exactly as the PNG and the detector returns ``R = I``.
    """
    rotation = rpy_matrix(*tagboard._TAG_FACE_RPY)
    u_local = np.array([0.0, -1.0, 0.0])
    v_local = np.array([-1.0, 0.0, 0.0])
    assert rotation @ u_local == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)  # viewer right
    assert rotation @ v_local == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)  # viewer down
    # and the textured face is the one that ends up facing the viewer
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_sdf_box_geometry_matches_the_bitmap(catalogue, cfg):
    tag = catalogue.tag(0)
    rendered = render(0, tag.family, tag.size_m, cfg.tags.board.texture_size_px)
    root = ET.fromstring(board_sdf(tag, "file:///x.png", rendered.drawing_size_m))
    box = root.find("model/link/visual/geometry/box")
    size = [float(v) for v in box.find("size").text.split()]
    assert size == pytest.approx(
        [rendered.drawing_size_m, rendered.drawing_size_m, tagboard._BOARD_THICKNESS_M]
    )
    assert size[0] == pytest.approx(0.225)
    assert root.find("model/link/visual/geometry/plane") is None, "a plane does not texture in gz"


def test_sdf_carries_the_texture(catalogue, cfg):
    tag = catalogue.tag(1)
    text = board_sdf(tag, "file:///tmp/tag_1_tag36h11.png", 0.225)
    root = ET.fromstring(text)
    albedo = root.find("model/link/visual/material/pbr/metal/albedo_map")
    assert albedo is not None
    assert albedo.text == "file:///tmp/tag_1_tag36h11.png"


def test_sdf_rejects_a_non_positive_size(catalogue):
    with pytest.raises(TagBoardError):
        board_sdf(catalogue.tag(0), "file:///x.png", 0.0)


def test_model_name_and_uri():
    assert model_name(3) == "tag_board_3"
    assert file_uri("/tmp/a b.png") == "file:///tmp/a b.png"
    assert file_uri("relative.png").startswith("file:///")


# --------------------------------------------------------------------------- #
# the plan
# --------------------------------------------------------------------------- #
def test_build_plan_without_writing_touches_nothing(catalogue, cfg, tmp_path):
    out = str(tmp_path / "boards")
    boards = build_plan(catalogue, cfg.tags.board, out, write=False)
    assert [board.tag.id for board in boards] == [0, 1, 2]
    assert not os.path.exists(out), "write=False must not create the directory"


def test_build_plan_writes_texture_and_sdf(catalogue, cfg, tmp_path):
    out = str(tmp_path / "boards")
    boards = build_plan(catalogue, cfg.tags.board, out, write=True)
    for board in boards:
        assert os.path.isfile(board.texture_path)
        assert os.path.isfile(board.sdf_path)
        assert board.size_m == pytest.approx(0.225)
        assert os.path.basename(board.sdf_path) == f"{board.name}.sdf"
        with open(board.sdf_path, "r", encoding="utf-8") as handle:
            ET.fromstring(handle.read())


def test_build_plan_is_disabled_by_config(catalogue, cfg, tmp_path):
    import dataclasses

    disabled = dataclasses.replace(cfg.tags.board, enabled=False)
    assert build_plan(catalogue, disabled, str(tmp_path), write=True) == []


def test_spawn_arguments_pass_the_pose_explicitly(catalogue):
    board = build_plan(catalogue, _board_cfg(), "/tmp/unused", write=False)[0]
    argv = spawn_arguments(board, "cat_arena")
    assert argv[0] == "ros2"
    assert argv[1:3] == ["run", "ros_gz_sim"]
    assert "-world" in argv and argv[argv.index("-world") + 1] == "cat_arena"
    assert argv[argv.index("-name") + 1] == "tag_board_0"
    for flag in ("-x", "-y", "-z", "-R", "-P", "-Y"):
        assert flag in argv, f"pose flag {flag} missing"
    assert float(argv[argv.index("-x") + 1]) == pytest.approx(board.position[0])
    assert float(argv[argv.index("-Y") + 1]) == pytest.approx(board.rpy[2])
    assert argv[argv.index("-file") + 1] == board.sdf_path


def test_plan_document_and_write(catalogue, cfg, tmp_path):
    out = str(tmp_path / "boards")
    boards = build_plan(catalogue, cfg.tags.board, out, write=True)
    document = plan_document(boards, out, world="cat_arena")
    assert document["count"] == 3
    assert len(document["spawn_commands"]) == 3
    assert all("ros_gz_sim create" in command for command in document["spawn_commands"])
    path = write_plan(str(tmp_path / "plan.yaml"), document)
    assert os.path.isfile(path)
    with open(path, "r", encoding="utf-8") as handle:
        assert handle.readline().startswith("# rover_perception tag board spawn plan")


def test_board_to_dict(catalogue, cfg):
    board = build_plan(catalogue, cfg.tags.board, "/tmp/unused", write=False)[0]
    entry = board.to_dict()
    assert entry["id"] == 0
    assert entry["name"] == "tag_board_0"
    assert entry["size_m"] == pytest.approx(0.225)
    assert len(entry["position"]) == 3 and len(entry["rpy"]) == 3 and len(entry["normal"]) == 3
    assert board.rpy == board.tag.rpy
    assert board.position == board.tag.position


def _board_cfg():
    from rover_perception.config import load_config

    here = os.path.dirname(os.path.abspath(__file__))
    return load_config(os.path.join(here, "..", "config", "perception.yaml")).tags.board
