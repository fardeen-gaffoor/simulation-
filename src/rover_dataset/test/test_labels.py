"""YOLO label I/O tests."""

from __future__ import annotations

import os

import pytest

from rover_dataset.labels import (
    LabelError,
    YoloBox,
    boxes_to_sidecar,
    clamp_normalised,
    read_classes,
    read_labels,
    validate_box,
    write_classes,
    write_data_yaml,
    write_labels,
)


def test_from_pixels_round_trip():
    box = YoloBox.from_pixels(1, 10.0, 20.0, 74.0, 68.0, 64, 48)
    assert box.class_id == 1
    assert box.cx == pytest.approx((10.0 + 74.0) / 2.0 / 64)
    assert box.w == pytest.approx(1.0)  # spans the full width
    x0, y0, x1, y1 = box.pixels(64, 48)
    assert (x0, y0, x1, y1) == pytest.approx((10.0, 20.0, 74.0, 68.0))


def test_from_pixels_normalises_inverted_corners():
    box = YoloBox.from_pixels(0, 74.0, 68.0, 10.0, 20.0, 64, 48)
    assert box.w == pytest.approx(1.0)
    assert box.h == pytest.approx(1.0)


def test_validate_rejects_out_of_range():
    with pytest.raises(LabelError, match="outside"):
        validate_box(YoloBox(0, 0.5, 0.5, 1.2, 0.2), 64, 48)


def test_validate_rejects_negative_class():
    with pytest.raises(LabelError, match="class_id"):
        validate_box(YoloBox(-1, 0.5, 0.5, 0.2, 0.2), 64, 48)


def test_validate_rejects_zero_size():
    with pytest.raises(LabelError, match="non-positive"):
        validate_box(YoloBox(0, 0.5, 0.5, 0.0, 0.2), 64, 48)


def test_validate_rejects_a_box_beyond_the_image():
    with pytest.raises(LabelError):
        validate_box(YoloBox(0, 0.9, 0.5, 0.4, 0.2), 64, 48)


def test_validate_accepts_a_legal_box():
    validate_box(YoloBox(0, 0.5, 0.5, 0.25, 0.25), 64, 48)


def test_validate_accepts_a_box_touching_the_left_edge():
    """Regression: a boulder clipped to x=0 was written by a real capture run as

        cx=0.023965 w=0.047931   ->   x0 = 0.023965 - 0.0239655 = -3.2e-4 px

    ``w/2`` needs one more decimal than ``cx``, so an exactly-clipped box can
    round to a hair outside the image.  That is not a labelling error.
    """
    validate_box(YoloBox(0, 0.023965, 0.465163, 0.047931, 0.113623), 640, 480)


def test_validate_accepts_a_box_touching_the_right_edge():
    validate_box(YoloBox(1, 0.976035, 0.465163, 0.047931, 0.113623), 640, 480)


def test_validate_still_rejects_a_box_really_off_the_image():
    # x0 = 0.02 - 0.05 = -0.03 -> -19 px, far beyond any rounding noise
    with pytest.raises(LabelError, match="outside the image"):
        validate_box(YoloBox(0, 0.02, 0.5, 0.10, 0.20), 640, 480)


def test_write_read_round_trip(tmp_path):
    path = str(tmp_path / "x.txt")
    boxes = [
        YoloBox.from_pixels(0, 1.0, 2.0, 11.0, 12.0, 64, 48, object_id="boulder_0"),
        YoloBox.from_pixels(1, 30.0, 20.0, 50.0, 40.0, 64, 48, object_id="crater_0"),
    ]
    write_labels(path, boxes)
    back = read_labels(path)
    assert [b.class_id for b in back] == [0, 1]
    assert back[0].cx == pytest.approx(boxes[0].cx)
    assert back[1].h == pytest.approx(boxes[1].h)


def test_empty_label_file_is_a_valid_negative(tmp_path):
    path = str(tmp_path / "empty.txt")
    write_labels(path, [])
    assert os.path.getsize(path) == 0
    assert read_labels(path) == []


def test_malformed_label_file_is_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")
    with pytest.raises(LabelError, match="5 fields"):
        read_labels(str(path))


def test_non_numeric_label_is_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("0 x 0.5 0.2 0.2\n", encoding="utf-8")
    with pytest.raises(LabelError, match="non-numeric"):
        read_labels(str(path))


def test_classes_round_trip(tmp_path):
    path = str(tmp_path / "classes.txt")
    write_classes(path, ["boulder", "crater"])
    assert read_classes(path) == ["boulder", "crater"]


def test_data_yaml_shape(tmp_path):
    path = str(tmp_path / "data.yaml")
    write_data_yaml(path, str(tmp_path), ["boulder", "crater"])
    text = open(path, encoding="utf-8").read()
    assert "nc: 2" in text
    assert "0: boulder" in text
    assert "1: crater" in text
    assert "train: images/train" in text
    assert "val: images/val" in text


def test_sidecar_records_dropped_objects(tmp_path):
    path = str(tmp_path / "side.json")
    boxes_to_sidecar(path, [], [{"kind": "boulder", "id": "b0", "reason": "occluded"}], {"camera": "cam_nav"})
    import json

    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["camera"] == "cam_nav"
    assert payload["dropped"][0]["reason"] == "occluded"


def test_clamp_normalised_repairs_numerical_overshoot():
    box = clamp_normalised(YoloBox(0, 1.0000001, 0.5, 1.0, 1.0))
    assert box.cx == 1.0
