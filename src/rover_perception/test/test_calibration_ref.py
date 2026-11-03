"""Module 2's ``calibration.yaml``: the intrinsics/extrinsics module 4 trusts.

Every metric number module 4 produces - a tag's range, a boulder's ground
position - is only as good as this file, so the tests check the reader against
the *real* generated artefact (produced in ``conftest`` by module 2's own
generator) and then pin down the geometry it implies.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import yaml

from rover_perception.calibration_ref import (
    CalibrationError,
    CameraModel,
    load_calibration,
    resolve_calibration_path,
)
from support import synthetic_calibration


def test_real_calibration_loads(calibration_path):
    doc = load_calibration(calibration_path)
    assert doc.schema_version == 1
    assert doc.frame == "base_link"
    assert doc.declared_forward == "+x"
    assert set(doc.cameras) == {"cam_nav", "cam_tag", "cam_excavation", "cam_dump"}


def test_intrinsics_match_module_two(cfg, calibration_path):
    doc = load_calibration(calibration_path)
    nav = doc.camera("cam_nav")
    assert (nav.width, nav.height) == (640, 480)
    assert nav.fx == pytest.approx(268.511881977, rel=1e-9)
    assert nav.cx == 320.0 and nav.cy == 240.0
    tag = doc.camera("cam_tag")
    assert (tag.width, tag.height) == (1280, 720)
    assert tag.fx == pytest.approx(1004.598769355, rel=1e-9)


def test_image_metadata_matches_perception_yaml(cfg, calibration_path):
    doc = load_calibration(calibration_path)
    for camera in cfg.cameras:
        model = doc.camera(camera.name)
        assert (model.width, model.height) == (camera.width, camera.height)
        assert model.topic == camera.topic
        assert model.camera_info_topic == camera.camera_info_topic
        image_size = model.width * model.height
        assert image_size > 0


def test_optical_frame_convention(calibration_path):
    doc = load_calibration(calibration_path)
    nav = doc.camera("cam_nav")
    # R_base_optical = R_base_link @ rpy(-pi/2, 0, -pi/2): the optical +z (forward)
    # must be the direction the camera looks, tilted 18 deg down for cam_nav.
    pitch = math.radians(18.0)
    forward = nav.r_base_optical_matrix[:, 2]
    assert forward == pytest.approx((math.cos(pitch), 0.0, -math.sin(pitch)), abs=1e-9)
    # optical +x (image right) is base -y (REP-103: +y is the robot's left);
    # optical +y (image down) is the tilted -z.
    assert nav.r_base_optical_matrix[:, 0] == pytest.approx((0.0, -1.0, 0.0), abs=1e-9)
    assert nav.r_base_optical_matrix[:, 1] == pytest.approx((-math.sin(pitch), 0.0, -math.cos(pitch)), abs=1e-9)


def test_mount_height_is_positive_and_sane(calibration_path):
    doc = load_calibration(calibration_path)
    for name, model in doc.cameras.items():
        height = float(model.t_base_cam[2])
        assert height > 0.0, f"{name} mount height must be above the ground"
        assert height < 1.5


def test_pixel_to_ray_is_principal_point_centred(calibration_path):
    doc = load_calibration(calibration_path)
    nav = doc.camera("cam_nav")
    ray = nav.pixel_to_ray(nav.cx, nav.cy)
    assert ray == pytest.approx((0.0, 0.0, 1.0), abs=1e-12), "the principal point looks along +z"
    # a pixel to the right of centre has +x in the optical frame; below centre +y
    right = nav.pixel_to_ray(nav.cx + 100.0, nav.cy)
    assert right[0] > 0.0 and right[1] == pytest.approx(0.0)
    below = nav.pixel_to_ray(nav.cx, nav.cy + 100.0)
    assert below[1] > 0.0 and below[0] == pytest.approx(0.0)
    assert right[0] == pytest.approx(100.0 / nav.fx)


def test_k_matrix_and_fovs(calibration_path):
    doc = load_calibration(calibration_path)
    tag = doc.camera("cam_tag")
    k = tag.k
    assert k.shape == (3, 3)
    assert k[0, 0] == pytest.approx(tag.fx)
    assert k[1, 1] == pytest.approx(tag.fy)
    assert k[0, 2] == pytest.approx(tag.cx)
    assert k[1, 2] == pytest.approx(tag.cy)
    # hfov from fx: 2 atan((w/2)/fx)
    assert tag.hfov_rad == pytest.approx(2.0 * math.atan(0.5 * tag.width / tag.fx), abs=1e-12)
    assert tag.vfov_rad == pytest.approx(2.0 * math.atan(0.5 * tag.height / tag.fy), abs=1e-12)


def test_camera_lookup_error(calibration_path):
    doc = load_calibration(calibration_path)
    assert doc.has("cam_nav")
    assert not doc.has("cam_sideways")
    with pytest.raises(CalibrationError):
        doc.camera("cam_sideways")


# --------------------------------------------------------------------------- #
# the synthetic fixture and the error paths
# --------------------------------------------------------------------------- #
def test_synthetic_calibration_is_valid(tmp_path):
    path = synthetic_calibration(str(tmp_path / "calibration.yaml"))
    doc = load_calibration(path)
    assert len(doc.cameras) == 4
    assert doc.camera("cam_nav").fx == pytest.approx(268.511881977)


def test_missing_required_camera_is_rejected(tmp_path):
    """Module 4's geometry needs cam_nav and cam_tag; the others are optional."""
    path = synthetic_calibration(str(tmp_path / "calibration.yaml"))
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    del document["cameras"]["cam_tag"]
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle)
    with pytest.raises(CalibrationError) as excinfo:
        load_calibration(path)
    assert "cam_tag" in str(excinfo.value)


def test_optional_camera_may_be_absent(tmp_path):
    path = synthetic_calibration(str(tmp_path / "calibration.yaml"))
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    del document["cameras"]["cam_dump"]
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle)
    doc = load_calibration(path)
    assert set(doc.cameras) == {"cam_nav", "cam_tag", "cam_excavation"}


def test_missing_intrinsics_is_rejected(tmp_path):
    path = synthetic_calibration(str(tmp_path / "calibration.yaml"))
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    del document["cameras"]["cam_nav"]["intrinsics"]["fx"]
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle)
    with pytest.raises(CalibrationError) as excinfo:
        load_calibration(path)
    assert "fx" in str(excinfo.value)


def test_non_positive_focal_length_is_rejected(tmp_path):
    path = synthetic_calibration(str(tmp_path / "calibration.yaml"))
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document["cameras"]["cam_nav"]["intrinsics"]["fx"] = 0.0
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle)
    with pytest.raises(CalibrationError):
        load_calibration(path)


def test_wrong_schema_version_is_rejected(tmp_path):
    path = synthetic_calibration(str(tmp_path / "calibration.yaml"))
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document["schema_version"] = 2
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle)
    with pytest.raises(CalibrationError):
        load_calibration(path)


def test_missing_file_is_reported():
    with pytest.raises(CalibrationError):
        load_calibration("/nonexistent/calibration.yaml")


def test_explicit_path_wins(tmp_path):
    path = synthetic_calibration(str(tmp_path / "calibration.yaml"))
    assert resolve_calibration_path(path) == os.path.abspath(path)


def test_explicit_missing_path_is_an_error(tmp_path):
    """A typo'd calibration path must not silently load a different machine's file."""
    with pytest.raises(CalibrationError) as excinfo:
        resolve_calibration_path(str(tmp_path / "typo.yaml"))
    assert "requested path" in str(excinfo.value)


def test_camera_model_is_hashable_and_comparable():
    model = CameraModel(
        name="cam",
        topic="/cam",
        camera_info_topic="/cam/camera_info",
        frame_id="cam_optical",
        width=640,
        height=480,
        fx=300.0,
        fy=300.0,
        cx=320.0,
        cy=240.0,
        t_base_cam=(0.0, 0.0, 0.5),
        r_base_optical=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    )
    assert model == model
    assert model.image_size == (640, 480)
    assert model.r_base_optical_matrix.shape == (3, 3)
    assert model.t_base_cam_vector.shape == (3,)
    assert model.k.shape == (3, 3)
