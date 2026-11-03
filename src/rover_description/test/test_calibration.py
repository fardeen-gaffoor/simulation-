"""Tests for the camera calibration document (intrinsics + extrinsics)."""

from __future__ import annotations

import math

import pytest

from rover_description.calibration import (
    OPTICAL_FRAME_RPY,
    calibration_document,
    camera_calibration,
)


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def test_intrinsics_from_hfov(cfg):
    for cam in cfg.sensors.cameras:
        rec = camera_calibration(cam)
        intr = rec["intrinsics"]
        expected_fx = (cam.width / 2.0) / math.tan(math.radians(cam.hfov_deg) / 2.0)
        assert intr["fx"] == pytest.approx(expected_fx)
        assert intr["fy"] == pytest.approx(expected_fx)
        assert intr["cx"] == pytest.approx(cam.width / 2.0)
        assert intr["cy"] == pytest.approx(cam.height / 2.0)


def test_view_direction_unit_and_directions(cfg):
    nav = camera_calibration(cfg.camera("cam_nav"))["extrinsics"]["view_direction_base"]
    assert _norm(nav) == pytest.approx(1.0)
    assert nav[0] > 0.9 and nav[2] < 0.0  # forward and tilted down

    tag = camera_calibration(cfg.camera("cam_tag"))["extrinsics"]["view_direction_base"]
    assert tag[0] == pytest.approx(1.0) and abs(tag[2]) < 1e-9  # level, forward

    dump = camera_calibration(cfg.camera("cam_dump"))["extrinsics"]["view_direction_base"]
    assert dump[0] < 0.0  # points rearward
    assert dump[2] < 0.0  # and down


def test_optical_matrix_columns(cfg):
    rec = camera_calibration(cfg.camera("cam_nav"))
    r = rec["extrinsics"]["rotation_matrix_base_optical"]
    # column 2 = optical +z (forward) in base == the view direction
    col2 = [r[i][2] for i in range(3)]
    view = rec["extrinsics"]["view_direction_base"]
    assert col2 == pytest.approx(view, abs=1e-6)
    # column 0 = optical +x (right) should have a negative base-y component
    col0 = [r[i][0] for i in range(3)]
    assert col0[1] == pytest.approx(-1.0, abs=1e-6)


def test_optical_frame_rpy_constant():
    assert OPTICAL_FRAME_RPY == pytest.approx((-math.pi / 2, 0.0, -math.pi / 2))
    rec = camera_calibration.__wrapped__ if hasattr(camera_calibration, "__wrapped__") else None
    assert rec is None  # sanity: camera_calibration is a plain function


def test_one_file_has_all_four(cfg):
    doc = calibration_document(cfg)
    assert doc["frame"] == "base_link"
    assert set(doc["cameras"]) == {"cam_nav", "cam_tag", "cam_excavation", "cam_dump"}
    assert doc["config_sha256"] == cfg.config_sha256()
    for name, rec in doc["cameras"].items():
        assert rec["topic"] == f"/{name}"
        assert rec["camera_info_topic"] == f"/{name}/camera_info"
        assert rec["frame_id"] == f"{name}_optical_frame"


def test_extrinsics_are_orthonormal(cfg):
    for cam in cfg.sensors.cameras:
        r = camera_calibration(cam)["extrinsics"]["rotation_matrix_base_camera"]
        for i in range(3):
            assert _norm([r[i][j] for j in range(3)]) == pytest.approx(1.0)
        for i in range(3):
            for j in range(i + 1, 3):
                dot = sum(r[i][k] * r[j][k] for k in range(3))
                assert dot == pytest.approx(0.0, abs=1e-6)
