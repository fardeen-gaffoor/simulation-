"""Contact-sheet tests: the one tool whose whole job is to be looked at."""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from rover_dataset.dataset import DatasetLayout, write_sample
from rover_dataset.labels import YoloBox
from rover_dataset.preview import (
    BOX_COLOURS,
    build_contact_sheet,
    draw_boxes,
    pick_stems,
)

CLASSES = ["boulder", "crater"]


def _image(value=90, width=64, height=48) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _box(class_id=0, value=8.0) -> YoloBox:
    return YoloBox.from_pixels(class_id, value, value, value + 20.0, value + 16.0, 64, 48)


def _dataset(tmp_path, *, train=3, val=2, empty_val=False):
    layout = DatasetLayout(root=str(tmp_path), image_format="jpg", image_quality=90)
    for index in range(train):
        write_sample(
            layout, split="train", scene_id=f"sim_{index:04d}", camera="cam_nav",
            pose_index=0, frame_index=0, image=_image(90 + index),
            boxes=[_box(0, 4.0), _box(1, 30.0)],
        )
    for index in range(val):
        boxes = [] if (empty_val and index == 0) else [_box(0, 4.0)]
        write_sample(
            layout, split="val", scene_id=f"sim_{100 + index:04d}", camera="cam_tag",
            pose_index=0, frame_index=0, image=_image(120 + index), boxes=boxes,
        )
    return layout


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_pick_stems_is_deterministic_and_evenly_spaced():
    entries = list(range(100))
    first = pick_stems(entries, 10)
    assert first == pick_stems(entries, 10)
    assert len(first) == 10
    assert first[0] == 0
    # spread across the range, not a contiguous slice
    assert max(first) - min(first) > 80
    assert first == sorted(first)


def test_pick_stems_returns_everything_when_asked_for_more():
    assert pick_stems([1, 2, 3], 10) == [1, 2, 3]


def test_pick_stems_handles_zero_and_empty():
    assert pick_stems([], 5) == []
    assert pick_stems([1, 2, 3], 0) == []


def test_pick_stems_never_repeats_an_entry():
    picked = pick_stems(list(range(5)), 5)
    assert len(set(picked)) == len(picked)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def test_draw_boxes_colours_by_class():
    image = Image.fromarray(_image())
    # two boxes at different places: identical rects would let the second
    # (magenta) paint over the first (green)
    drawn = draw_boxes(image, [_box(0, 4.0), _box(1, 30.0)], width=1)
    pixels = np.asarray(drawn)
    assert (pixels == np.array(BOX_COLOURS[0])).all(axis=2).any()
    assert (pixels == np.array(BOX_COLOURS[1])).all(axis=2).any()


def test_draw_boxes_does_not_mutate_the_input():
    image = Image.fromarray(_image())
    before = np.asarray(image).copy()
    draw_boxes(image, [_box(0)])
    assert np.array_equal(np.asarray(image), before)


def test_draw_boxes_survives_a_box_on_the_edge():
    image = Image.fromarray(_image())
    edge = YoloBox(0, 0.5, 0.5, 1.0, 1.0)  # exactly the whole frame
    drawn = draw_boxes(image, [edge], width=2)
    assert drawn.size == image.size


def test_draw_boxes_with_no_boxes_is_a_copy():
    image = Image.fromarray(_image())
    assert np.array_equal(np.asarray(draw_boxes(image, [])), np.asarray(image))


# --------------------------------------------------------------------------- #
# the sheet
# --------------------------------------------------------------------------- #
def test_build_writes_a_grid_of_the_requested_shape(tmp_path):
    _dataset(tmp_path)
    result = build_contact_sheet(str(tmp_path), count=4, columns=2, cell_px=64, classes=CLASSES)
    assert os.path.isfile(result.output)
    with Image.open(result.output) as sheet:
        assert sheet.size == (128, 128)  # 2 columns x 2 rows of 64 px
    assert result.cells == 4
    assert result.per_class == {"boulder": 4, "crater": 3}
    assert result.per_split == {"train": 3, "val": 1}


def test_build_counts_background_only_frames(tmp_path):
    _dataset(tmp_path, train=1, val=1, empty_val=True)
    result = build_contact_sheet(str(tmp_path), count=9, columns=3, cell_px=64, classes=CLASSES)
    assert result.empty_images == 1
    assert result.images == 2


def test_build_is_reproducible(tmp_path):
    _dataset(tmp_path, train=6, val=4)
    first = build_contact_sheet(
        str(tmp_path), count=5, columns=5, cell_px=48, classes=CLASSES, output=str(tmp_path / "a.png")
    )
    second = build_contact_sheet(
        str(tmp_path), count=5, columns=5, cell_px=48, classes=CLASSES, output=str(tmp_path / "b.png")
    )
    assert first.as_dict()["per_split"] == second.as_dict()["per_split"]
    with Image.open(first.output) as a, Image.open(second.output) as b:
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_build_can_restrict_to_one_split(tmp_path):
    _dataset(tmp_path)
    result = build_contact_sheet(
        str(tmp_path), count=9, cell_px=64, classes=CLASSES, splits=["val"]
    )
    assert result.per_split == {"val": 2}


def test_build_reports_missing_labels(tmp_path):
    layout = _dataset(tmp_path, train=1, val=0)
    stem = os.path.splitext(os.listdir(os.path.join(layout.images_dir, "train"))[0])[0]
    os.remove(layout.label_path("train", stem))
    result = build_contact_sheet(str(tmp_path), count=4, cell_px=64, classes=CLASSES)
    assert result.boxes == 0
    assert result.missing_labels == [f"train/{stem}"]


def test_build_raises_on_an_empty_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="no images"):
        build_contact_sheet(str(tmp_path), count=4)


def test_build_with_zero_count_writes_a_blank_sheet(tmp_path):
    # count/columns/cell of 0 must not divide by zero
    _dataset(tmp_path)
    result = build_contact_sheet(str(tmp_path), count=0, columns=0, cell_px=0, classes=CLASSES)
    assert result.cells == 0
    assert os.path.isfile(result.output)
