"""The learned detector's numpy decode path, exercised without onnxruntime.

``onnxruntime`` is not installed here (and the fine-tuned model does not exist
yet), so every piece of the ONNX backend that *matters for correctness* is
tested against a **stub session**: the letterbox, the head decoding, the NMS,
and the mapping from letterbox pixels back to image pixels.  A real session only
supplies one tensor; everything after it is the code under test here.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from rover_perception.backends.onnx_yolo import (
    OnnxYoloDetector,
    decode_predictions,
    install_hint,
    is_available,
    letterbox,
    nms,
    resize_bilinear,
)
from rover_perception.calibration_ref import load_calibration
from rover_perception.config import load_config
from rover_perception.detector import DetectorError


@pytest.fixture(scope="module")
def cfg():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_config(os.path.join(here, "..", "config", "perception.yaml")).detector.onnx


@pytest.fixture(scope="module")
def camera(calibration_path):
    return load_calibration(calibration_path).camera("cam_nav")


# --------------------------------------------------------------------------- #
# the stub "graph"
# --------------------------------------------------------------------------- #
class StubSession:
    """The smallest thing :class:`OnnxYoloDetector` will accept as a session."""

    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)
        self.runs = []
        self._inputs = [SimpleNamespace(name="images")]

    def get_inputs(self):
        return list(self._inputs)

    def run(self, output_names, feed):
        self.runs.append((output_names, feed))
        return [self.output]


def head(rows, num_channels=6):
    """A YOLOv8 head laid out ``(1, 4 + classes, N)`` from ``(cx, cy, w, h, ...)`` rows."""
    if not len(rows):
        return np.zeros((1, num_channels, 0), dtype=np.float32)
    return np.array(rows, dtype=np.float32).T[None, :, :]


# --------------------------------------------------------------------------- #
# resize
# --------------------------------------------------------------------------- #
def test_resize_returns_a_copy_at_the_same_size():
    image = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = resize_bilinear(image, 3, 4)
    assert np.array_equal(out, image)
    assert out is not image


def test_resize_rejects_a_bad_rank():
    with pytest.raises(ValueError):
        resize_bilinear(np.zeros((2, 2, 2, 2), dtype=np.float32), 4, 4)


def test_resize_of_a_constant_stays_constant():
    image = np.full((10, 10), 42.0, dtype=np.float32)
    out = resize_bilinear(image, 25, 25)
    assert out.shape == (25, 25)
    assert out == pytest.approx(42.0)


def test_resize_upsamples_and_preserves_the_corners():
    image = np.array([[0.0, 100.0], [100.0, 0.0]], dtype=np.float32)
    out = resize_bilinear(image, 4, 4)
    assert out.shape == (4, 4)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[0, 3] == pytest.approx(100.0)
    assert out[3, 0] == pytest.approx(100.0)
    assert out[3, 3] == pytest.approx(0.0)
    assert 0.0 <= out.min() and out.max() <= 100.0


def test_resize_keeps_three_channels():
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[:, :, 2] = 255.0
    out = resize_bilinear(image, 8, 8)
    assert out.shape == (8, 8, 3)
    assert out[..., 2] == pytest.approx(255.0)
    assert out[..., 0] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# letterbox
# --------------------------------------------------------------------------- #
def test_letterbox_pads_a_wide_image_top_and_bottom():
    image = np.full((480, 640, 3), 10, dtype=np.uint8)
    padded, scale, dx, dy = letterbox(image, 640, pad=114)
    assert scale == pytest.approx(1.0)
    assert (dx, dy) == (0, 80)
    assert padded.shape == (640, 640, 3)
    assert padded.dtype == np.uint8
    assert np.all(padded[:80] == 114) and np.all(padded[560:] == 114)
    assert np.all(padded[80:560] == 10)


def test_letterbox_pads_a_tall_image_left_and_right():
    image = np.full((100, 50, 3), 7, dtype=np.uint8)
    _padded, scale, dx, dy = letterbox(image, 640, pad=114)
    assert scale == pytest.approx(min(640 / 100.0, 640 / 50.0))
    assert dx > 0 and dy == 0


def test_letterbox_promotes_grey_to_three_channels():
    padded, _scale, _dx, _dy = letterbox(np.full((640, 640), 5, dtype=np.uint8), 640)
    assert padded.shape == (640, 640, 3)
    assert np.all(padded == 5)


def test_letterbox_drops_a_fourth_channel():
    rgba = np.zeros((640, 640, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    padded, _s, _dx, _dy = letterbox(rgba, 640)
    assert padded.shape == (640, 640, 3)


def test_letterbox_rejects_an_empty_image():
    with pytest.raises(ValueError):
        letterbox(np.zeros((0, 10, 3), dtype=np.uint8), 640)


# --------------------------------------------------------------------------- #
# non-maximum suppression
# --------------------------------------------------------------------------- #
def test_nms_on_nothing_is_nothing():
    assert nms(np.zeros((0, 4)), np.zeros((0,)), 0.5).tolist() == []


def test_nms_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=float)
    kept = nms(boxes, np.array([0.9, 0.8]), 0.5)
    assert sorted(kept.tolist()) == [0, 1]


def test_nms_keeps_the_highest_score_of_identical_boxes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
    kept = nms(boxes, np.array([0.5, 0.9, 0.4]), 0.5)
    assert kept.tolist() == [1]


def test_nms_is_greedy_over_a_chain():
    a = [0, 0, 10, 10]
    b = [5, 0, 15, 10]  # IoU(a, b) = 0.33
    c = [10, 0, 20, 10]  # IoU(a, c) = 0, IoU(b, c) = 0.33
    boxes = np.array([a, b, c], dtype=float)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.4).tolist()) == [0, 1, 2]
    # at 0.3 the middle box is suppressed by the best box, but the far one
    # survives because it does not overlap the survivor at all
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.3).tolist()) == [0, 2]


# --------------------------------------------------------------------------- #
# decoding the head
# --------------------------------------------------------------------------- #
def test_decode_reads_channels_first_and_last():
    row = [100.0, 200.0, 40.0, 60.0, 0.9, 0.1]
    channels_first = np.array(row, dtype=np.float32).reshape(1, 6, 1)
    channels_last = np.array(row, dtype=np.float32).reshape(1, 1, 6)
    for output in (channels_first, channels_last):
        boxes, scores, class_ids = decode_predictions(output, 0.25, 2)
        assert boxes.shape == (1, 4)
        assert boxes[0] == pytest.approx([80.0, 170.0, 120.0, 230.0]), "cxcywh -> xyxy"
        assert scores[0] == pytest.approx(0.9)
        assert class_ids[0] == 0


def test_decode_picks_the_argmax_class():
    rows = [
        [10.0, 10.0, 4.0, 4.0, 0.1, 0.8],  # crater wins
        [50.0, 50.0, 4.0, 4.0, 0.7, 0.2],  # boulder wins
    ]
    _boxes, _scores, class_ids = decode_predictions(head(rows), 0.25, 2)
    assert class_ids.tolist() == [1, 0]


def test_decode_thresholds_on_the_winning_score():
    rows = [
        [10.0, 10.0, 4.0, 4.0, 0.9, 0.0],
        [50.0, 50.0, 4.0, 4.0, 0.1, 0.0],
    ]
    boxes, scores, class_ids = decode_predictions(head(rows), 0.5, 2)
    assert boxes.shape[0] == 1
    assert scores.tolist() == pytest.approx([0.9])
    assert class_ids.tolist() == [0]


def test_decode_of_an_empty_head():
    boxes, scores, class_ids = decode_predictions(np.zeros((1, 6, 0)), 0.25, 2)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)
    assert class_ids.shape == (0,)


def test_decode_accepts_a_bare_two_dimensional_head():
    boxes, _scores, _ids = decode_predictions(np.array([[100.0, 200.0, 40.0, 60.0, 0.9, 0.1]]), 0.25, 2)
    assert boxes[0] == pytest.approx([80.0, 170.0, 120.0, 230.0])


def test_decode_rejects_a_batch_and_a_wrong_channel_count():
    with pytest.raises(ValueError):
        decode_predictions(np.zeros((2, 6, 3)), 0.25, 2)
    with pytest.raises(ValueError):
        decode_predictions(np.zeros((1, 5, 3)), 0.25, 2)


# --------------------------------------------------------------------------- #
# the detector with a stub graph
# --------------------------------------------------------------------------- #
def test_availability_is_reported_not_assumed():
    # onnxruntime is deliberately absent on the dev laptop; the hint must say how
    assert isinstance(is_available(), bool)
    if not is_available():
        assert "onnxruntime" in install_hint()


def test_preprocessing_matches_the_yolo_convention(cfg, camera):
    session = StubSession(head([]))
    detector = OnnxYoloDetector(cfg, camera, session=session)
    detector.detect(np.full((480, 640, 3), 255, dtype=np.uint8))
    _names, feed = session.runs[0]
    blob = feed["images"]
    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert blob.max() <= 1.0
    assert blob[0, :, 0, 0] == pytest.approx(114.0 / 255.0, abs=1e-6), "the letterbox pad"


def test_boxes_map_back_out_of_letterbox_pixels(cfg, camera):
    # cam_nav is 640x480, so scale 1.0 and dy 80: image_y = letterbox_y - 80
    rows = [[320.0, 300.0, 40.0, 60.0, 0.9, 0.1]]
    detector = OnnxYoloDetector(cfg, camera, session=StubSession(head(rows)))
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 1
    found = detections[0]
    assert found.kind == "boulder"
    assert found.class_id == 0
    assert found.score == pytest.approx(0.9)
    assert (found.x0, found.y0, found.x1, found.y1) == pytest.approx((300.0, 190.0, 340.0, 250.0))
    assert found.features["letterbox_scale"] == pytest.approx(1.0)
    assert found.features["pad_y"] == pytest.approx(80.0)


def test_class_names_come_from_module_threes_order(cfg, camera):
    rows = [[100.0, 100.0, 20.0, 20.0, 0.1, 0.9]]
    detector = OnnxYoloDetector(cfg, camera, session=StubSession(head(rows)))
    found = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))[0]
    assert found.kind == "crater"
    assert found.class_id == 1, "1 = crater, matching module 3's dataset"


def test_nms_is_per_class_so_overlaps_survive(cfg, camera):
    """A boulder sitting in front of a crater is two obstacles, not one."""
    rows = [
        [320.0, 300.0, 60.0, 60.0, 0.9, 0.1],
        [320.0, 300.0, 60.0, 60.0, 0.1, 0.9],
    ]
    detector = OnnxYoloDetector(cfg, camera, session=StubSession(head(rows)))
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert sorted(found.kind for found in detections) == ["boulder", "crater"]


def test_two_overlapping_boulders_become_one(cfg, camera):
    rows = [
        [320.0, 300.0, 60.0, 60.0, 0.9, 0.0],
        [325.0, 302.0, 60.0, 60.0, 0.8, 0.0],
    ]
    detector = OnnxYoloDetector(cfg, camera, session=StubSession(head(rows)))
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.9)


def test_a_sub_pixel_box_is_dropped(cfg, camera):
    rows = [
        [320.0, 300.0, 0.4, 0.4, 0.9, 0.0],
        [200.0, 300.0, 40.0, 40.0, 0.7, 0.0],
    ]
    detector = OnnxYoloDetector(cfg, camera, session=StubSession(head(rows)))
    detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].x0 == pytest.approx(180.0)


def test_boxes_are_clamped_to_the_frame(cfg, camera):
    rows = [[10.0, 10.0, 400.0, 400.0, 0.9, 0.0]]
    detector = OnnxYoloDetector(cfg, camera, session=StubSession(head(rows)))
    found = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))[0]
    assert found.x0 >= 0.0 and found.y0 >= 0.0
    assert found.x1 <= 640.0 and found.y1 <= 480.0


def test_max_detections_is_honoured(cfg, camera):
    import dataclasses

    small = dataclasses.replace(cfg, max_detections=2)
    rows = [
        [100.0 + 80 * index, 100.0, 40.0, 40.0, 0.9 - 0.1 * index, 0.0] for index in range(4)
    ]
    detector = OnnxYoloDetector(small, camera, session=StubSession(head(rows)))
    assert len(detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))) == 2


def test_describe_and_close(cfg, camera):
    detector = OnnxYoloDetector(cfg, camera, session=StubSession(head([])))
    description = detector.describe()
    assert description["backend"] == "onnx"
    assert description["classes"] == ["boulder", "crater"]
    assert description["input_size"] == 640
    detector.close()
    assert detector._session is None


def test_a_missing_model_names_the_problem(cfg, camera):
    import dataclasses

    empty = dataclasses.replace(cfg, model="")
    with pytest.raises(DetectorError) as excinfo:
        OnnxYoloDetector(empty, camera)
    assert "model" in str(excinfo.value)

    absent = dataclasses.replace(cfg, model="/nonexistent/boulder.onnx")
    with pytest.raises(DetectorError) as excinfo:
        OnnxYoloDetector(absent, camera)
    assert "not found" in str(excinfo.value)
