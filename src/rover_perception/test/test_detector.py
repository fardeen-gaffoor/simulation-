"""The detector interface: image-space boxes, and the backend factory.

Everything downstream - ground projection, tracking, the wall mask - is written
against :class:`Detection`, so a change here changes every node at once.  The
factory is tested for its *failures* as much as its success: pointing the config
at a backend that is not installed must stop the node with a readable message,
not fall back to something that would quietly drive the rover with an untested
detector.
"""

from __future__ import annotations

import numpy as np
import pytest

from rover_perception.backends.classical import ClassicalDetector
from rover_perception.calibration_ref import load_calibration
from rover_perception.config import load_config
from rover_perception.detector import (
    KIND_COLOURS,
    Detection,
    DetectorError,
    annotate,
    create_backend,
)
from support import detection

import os


@pytest.fixture(scope="module")
def camera(calibration_path):
    return load_calibration(calibration_path).camera("cam_nav")


# --------------------------------------------------------------------------- #
# the value object
# --------------------------------------------------------------------------- #
def test_detection_geometry():
    box = detection("boulder", x0=100.0, y0=200.0, x1=140.0, y1=260.0)
    assert box.width == pytest.approx(40.0)
    assert box.height == pytest.approx(60.0)
    assert box.area == pytest.approx(2400.0)
    assert box.center == pytest.approx((120.0, 230.0))
    assert box.contact_point == pytest.approx((120.0, 260.0)), "the ground contact is the box foot"
    assert box.class_id == 0
    assert not box.is_degenerate()


def test_crater_class_id_matches_module_three():
    assert detection("crater").class_id == 1
    assert detection("boulder").class_id == 0


def test_clamped_stays_inside_the_image():
    box = detection("boulder", x0=-20.0, y0=-5.0, x1=1000.0, y1=2000.0)
    clamped = box.clamped(640, 480)
    assert (clamped.x0, clamped.y0) == pytest.approx((0.0, 0.0))
    assert (clamped.x1, clamped.y1) == pytest.approx((639.0, 479.0))
    assert clamped.features == box.features


def test_degenerate_boxes_are_flagged():
    assert detection("boulder", x0=10.0, y0=10.0, x1=10.5, y1=40.0).is_degenerate()
    assert detection("boulder", x0=10.0, y0=10.0, x1=40.0, y1=10.5).is_degenerate()
    assert not detection("boulder", x0=10.0, y0=10.0, x1=40.0, y1=40.0).is_degenerate()


def test_features_survive_a_round_trip():
    box = detection("boulder", shading_delta=12.5, fill=0.7)
    assert box.features == {"shading_delta": 12.5, "fill": 0.7}


# --------------------------------------------------------------------------- #
# the debug drawing
# --------------------------------------------------------------------------- #
def test_annotate_draws_the_box_and_the_contact_cross():
    canvas = np.zeros((60, 80, 3), dtype=np.uint8)
    box = detection("boulder", x0=20.0, y0=20.0, x1=60.0, y1=50.0)
    drawn = annotate(canvas, [box], thickness=1)
    assert np.all(drawn[20, 20:61] == KIND_COLOURS["boulder"])
    assert np.all(drawn[50, 20:61] == KIND_COLOURS["boulder"])
    assert np.all(drawn[20:51, 20] == KIND_COLOURS["boulder"])
    assert np.all(drawn[20:51, 60] == KIND_COLOURS["boulder"])
    assert np.all(drawn[50, 38:43] == KIND_COLOURS["boulder"]), "the contact cross is on the foot"
    assert canvas.max() == 0, "the input image is not modified"


def test_annotate_promotes_a_grey_image(camera):
    grey = np.zeros((40, 40), dtype=np.uint8)
    drawn = annotate(grey, [detection("crater", x0=5.0, y0=5.0, x1=20.0, y1=20.0)])
    assert drawn.shape == (40, 40, 3)
    assert np.all(drawn[5, 5:21] == KIND_COLOURS["crater"])


def test_annotate_uses_the_unknown_colour():
    canvas = np.zeros((20, 20, 3), dtype=np.uint8)
    drawn = annotate(canvas, [detection("sinkhole", x0=2.0, y0=2.0, x1=10.0, y1=10.0)])
    assert np.all(drawn[2, 2:11] == KIND_COLOURS["unknown"])


def test_empty_detection_list_is_a_copy():
    canvas = np.zeros((5, 5, 3), dtype=np.uint8)
    drawn = annotate(canvas, [])
    assert drawn is not canvas
    assert np.array_equal(drawn, canvas)


# --------------------------------------------------------------------------- #
# the factory
# --------------------------------------------------------------------------- #
def _config(tmp_path, **overrides):
    import dataclasses

    cfg = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "perception.yaml"))
    detector = dataclasses.replace(cfg.detector, **overrides)
    return dataclasses.replace(cfg, detector=detector)


def test_classical_backend_is_created(camera, cfg):
    backend = create_backend(cfg.detector, camera)
    assert isinstance(backend, ClassicalDetector)
    assert backend.describe()["backend"] == "classical"


def test_disabled_detector_is_an_error(camera, cfg):
    with pytest.raises(DetectorError) as excinfo:
        create_backend(_config(None, enabled=False).detector, camera)
    assert "enabled" in str(excinfo.value)


def test_unknown_backend_is_an_error(camera, cfg):
    with pytest.raises(DetectorError) as excinfo:
        create_backend(_config(None, backend="magic").detector, camera)
    assert "magic" in str(excinfo.value)


def test_onnx_backend_without_a_model_is_an_error(camera, cfg):
    import dataclasses

    onnx = dataclasses.replace(cfg.detector.onnx, model="")
    detector = dataclasses.replace(cfg.detector, backend="onnx", onnx=onnx)
    with pytest.raises(DetectorError) as excinfo:
        create_backend(detector, camera)
    assert "model" in str(excinfo.value)
