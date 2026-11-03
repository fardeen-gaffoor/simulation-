"""The AprilTag detector: a real 36h11 decode, verified against rendered tags.

``dt-apriltags`` is a genuine AprilTag 3 implementation and it is the one
non-ROS dependency module 4 truly needs, so it is exercised for real: a tag
bitmap of a *known* physical size is placed at a *known* distance from a
*pinhole* camera and the recovered pose is compared with the closed-form answer.
A synthetic-only test would happily pass with a sign error in the camera frame;
this one cannot.

The pure helpers (corner flattening, degenerate-pose rejection, mixed-size
rescaling) are tested without the library so they keep working on a machine
where ``dt-apriltags`` has not been installed yet.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from rover_perception.calibration_ref import load_calibration
from rover_perception.config import load_config
from rover_perception.geometry import Pose3, rpy_matrix
from rover_perception.tagdetect import (
    TagDetectionResult,
    TagDetector,
    TagDetectorError,
    available,
    corner_array,
    install_hint,
)
from rover_perception.tagdetect import _pose_from_result, _rescale_pose
from rover_perception.tagimages import module_grid

dt_apriltags = pytest.importorskip("dt_apriltags", reason="dt-apriltags is not installed")

_HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# synthetic camera images
# --------------------------------------------------------------------------- #
def tag_image(camera, tag_id: int, distance_m: float, size_m: float, background: int = 150):
    """A crisp, exactly-sized tag at ``distance_m`` straight ahead of ``camera``.

    The black square spans ``fx * size_m / distance`` pixels, so the image is a
    real pinhole view of a board of that physical size and the recovered range
    has a closed-form answer to compare against.
    """
    canvas = np.full((camera.height, camera.width), int(background), dtype=np.uint8)
    black_px = camera.fx * size_m / distance_m
    total_modules, black_modules = 10, 8
    total_px = int(round(black_px * total_modules / black_modules))
    if total_px > min(camera.height, camera.width):
        raise ValueError(
            f"a {size_m} m tag at {distance_m} m is {total_px} px across, "
            f"which does not fit in {camera.width}x{camera.height}"
        )
    grid = module_grid(tag_id)
    # nearest-neighbour module lookup: no interpolation, no blurring
    index = (np.arange(total_px) * total_modules) // max(1, total_px)
    index = np.clip(index, 0, total_modules - 1)
    block = grid[np.ix_(index, index)]  # already 0/255 (uint8: never multiply it)
    top = int(round(0.5 * (camera.height - total_px)))
    left = int(round(0.5 * (camera.width - total_px)))
    canvas[top : top + total_px, left : left + total_px] = block
    return canvas


def make_detector(calibration_path, cfg, sizes=None):
    camera = load_calibration(calibration_path).camera("cam_tag")
    return TagDetector(
        cfg.tags.detection,
        camera,
        sizes if sizes is not None else {0: 0.18, 1: 0.18, 2: 0.18},
        family="tag36h11",
        default_size_m=0.18,
    )


# --------------------------------------------------------------------------- #
# the real detector
# --------------------------------------------------------------------------- #
def test_dt_apriltags_is_available(cfg):
    assert available(), install_hint()


def test_detector_describes_itself(calibration_path, cfg):
    detector = make_detector(calibration_path, cfg)
    description = detector.describe()
    assert description["family"] == "tag36h11"
    assert description["camera"] == "cam_tag"
    assert description["max_detections"] >= 1
    assert set(description["sizes_m"]) == {"0", "1", "2"}


def test_detector_requires_the_library(monkeypatch, calibration_path, cfg):
    monkeypatch.setattr("rover_perception.tagdetect.available", lambda: False)
    camera = load_calibration(calibration_path).camera("cam_tag")
    with pytest.raises(TagDetectorError) as excinfo:
        TagDetector(cfg.tags.detection, camera, {0: 0.18})
    assert "dt-apriltags" in str(excinfo.value)


@pytest.mark.parametrize("tag_id", [0, 1, 2, 3, 4, 7, 100, 586])
def test_every_reference_id_decodes(calibration_path, cfg, tag_id):
    detector = make_detector(calibration_path, cfg)
    camera = detector.camera
    image = tag_image(camera, tag_id, 0.5, 0.18)
    detections = detector.detect(image)
    assert len(detections) == 1, f"tag {tag_id} was not decoded"
    found = detections[0]
    assert found.tag_id == tag_id
    assert found.hamming == 0
    assert found.decision_margin > cfg.tags.detection.decision_margin_min
    assert found.pose_valid


@pytest.mark.parametrize("distance_m", [0.35, 0.50, 1.00, 2.00])
def test_range_estimate_matches_the_pinhole_prediction(calibration_path, cfg, distance_m):
    detector = make_detector(calibration_path, cfg)
    image = tag_image(detector.camera, 0, distance_m, 0.18)
    detections = detector.detect(image)
    assert detections, f"no detection at {distance_m} m"
    pose = detections[0].pose_camera
    assert pose is not None
    # the optical frame is +z forward, so a head-on tag sits at (0, 0, range)
    assert pose.translation[0] == pytest.approx(0.0, abs=0.01)
    assert pose.translation[1] == pytest.approx(0.0, abs=0.01)
    assert pose.translation[2] == pytest.approx(distance_m, rel=0.05)
    assert detections[0].range_m == pytest.approx(distance_m, rel=0.05)


def test_head_on_tag_reports_the_viewer_frame(calibration_path, cfg):
    """The measured convention: AprilTag's tag frame is the *viewer's* frame.

    A head-on, upright tag gives ``R == I``: the tag's ``+x`` is the viewer's
    right, ``+y`` is down and ``+z`` points *into* the board (away from the
    viewer).  ``tags.tag_frame_rotation`` is built from exactly this, so a
    regression here invalidates every ``map -> odom`` fix.
    """
    detector = make_detector(calibration_path, cfg)
    detections = detector.detect(tag_image(detector.camera, 0, 0.5, 0.18))
    assert detections
    pose = detections[0].pose_camera
    assert pose is not None
    # the residual is pixel/module quantisation in the synthetic image (~0.3 deg)
    assert pose.rotation == pytest.approx(np.eye(3), abs=2e-2)
    assert float(np.linalg.det(pose.rotation)) == pytest.approx(1.0, abs=1e-6)
    assert pose.translation[2] > 0.0, "the tag is in front of the camera (+z forward)"
    # and an in-plane rotation is reported as an in-plane rotation, not a flip
    rotated = np.ascontiguousarray(np.rot90(tag_image(detector.camera, 0, 0.5, 0.18), 1))
    found = detector.detect(rotated)
    assert found
    spin = found[0].pose_camera.rotation
    assert abs(float(spin[2, 2]) - 1.0) < 1e-3, "z stays into the board under in-plane rotation"
    assert float(spin[0, 0]) == pytest.approx(0.0, abs=1e-2)


def test_empty_image_yields_nothing(calibration_path, cfg):
    detector = make_detector(calibration_path, cfg)
    blank = np.full((detector.camera.height, detector.camera.width), 128, dtype=np.uint8)
    assert detector.detect(blank) == []


def test_detect_requires_a_greyscale_image(calibration_path, cfg):
    detector = make_detector(calibration_path, cfg)
    with pytest.raises(TagDetectorError):
        detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))


def test_sizes_come_from_the_catalogue(calibration_path, cfg):
    detector = make_detector(calibration_path, cfg, sizes={0: 0.18, 9: 0.30})
    assert detector.size_for(0) == pytest.approx(0.18)
    assert detector.size_for(9) == pytest.approx(0.30)
    assert detector.size_for(1234) == pytest.approx(0.18), "unknown ids use the default size"
    assert detector.distinct_sizes() == (0.18, 0.30)


# --------------------------------------------------------------------------- #
# helpers (no library needed)
# --------------------------------------------------------------------------- #
def test_corner_array_flattens_and_tolerates_junk():
    assert corner_array([[1, 2], [3, 4], [5, 6], [7, 8]]) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
    assert corner_array(None) == ()
    assert corner_array([[1, 2]]) == ()
    assert corner_array("nonsense") == ()


class StubResult:
    def __init__(self, rotation=None, translation=None, **extra):
        self.pose_R = rotation
        self.pose_t = translation
        for key, value in extra.items():
            setattr(self, key, value)


def test_pose_from_result_accepts_a_valid_rotation():
    rotation = rpy_matrix(0.0, 0.0, 0.3)
    pose = _pose_from_result(StubResult(rotation, np.array([[0.0], [0.0], [1.5]])))
    assert pose is not None
    assert pose.translation == pytest.approx((0.0, 0.0, 1.5))
    assert pose.rotation == pytest.approx(rotation)


def test_pose_from_result_rejects_a_degenerate_rotation():
    noise = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert _pose_from_result(StubResult(noise, np.zeros(3))) is None
    assert _pose_from_result(StubResult(None, None)) is None
    assert _pose_from_result(StubResult(np.eye(3), np.array([np.nan, 0.0, 0.0]))) is None


def test_rescale_pose_is_linear_in_the_size():
    result = StubResult(np.eye(3), np.array([[0.0], [0.0], [1.0]]))
    pose = _rescale_pose(result, nominal=0.30, used=0.20)
    assert pose is not None
    assert pose.translation[2] == pytest.approx(1.5)
    assert _rescale_pose(result, 0.3, 0.0) is None


def test_detection_result_dict_and_properties():
    pose = Pose3(rpy_matrix(0.0, 0.0, 0.0), (0.0, 0.0, 2.0))
    found = TagDetectionResult(
        tag_id=3, family="tag36h11", size_m=0.18, hamming=0.0, decision_margin=80.0,
        pose_camera=pose, corners=(0.0,) * 8, center=(640.0, 360.0),
    )
    assert found.pose_valid
    assert found.range_m == pytest.approx(2.0)
    entry = found.to_dict()
    assert entry["id"] == 3 and entry["pose_valid"] is True and entry["range_m"] == 2.0
    without = TagDetectionResult(
        tag_id=4, family="tag36h11", size_m=0.18, hamming=0.0, decision_margin=10.0, pose_camera=None
    )
    assert not without.pose_valid
    assert math.isnan(without.range_m)
    assert without.to_dict()["range_m"] is None
