"""Dataset layout, writing and verification tests."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rover_dataset.dataset import (
    DatasetLayout,
    RawLayout,
    dataset_stats,
    image_stem,
    load_image,
    raw_stems,
    save_image,
    scene_of_stem,
    verify_dataset,
    write_sample,
)
from rover_dataset.labels import YoloBox

CLASSES = ["boulder", "crater"]


def _layout(tmp_path) -> DatasetLayout:
    return DatasetLayout(root=str(tmp_path), image_format="jpg", image_quality=90)


def _raw(tmp_path) -> RawLayout:
    return RawLayout(root=str(tmp_path), image_format="jpg", image_quality=90)


def _image(width=64, height=48) -> np.ndarray:
    return np.full((height, width, 3), 127, dtype=np.uint8)


def test_stem_encodes_scene_camera_pose_and_frame():
    stem = image_stem("sim_0003", "cam_nav", 12, 1)
    assert stem == "sim_0003__cam_nav__p0012__f01"
    assert scene_of_stem(stem) == "sim_0003"


def test_image_round_trip(tmp_path):
    path = str(tmp_path / "a.jpg")
    image = _image()
    save_image(path, image, "jpg", 95)
    back = load_image(path)
    assert back.shape == image.shape
    assert back.dtype == np.uint8


def test_png_round_trip_is_lossless(tmp_path):
    path = str(tmp_path / "a.png")
    image = _image()
    save_image(path, image, "png")
    assert np.array_equal(load_image(path), image)


def test_write_sample_creates_image_label_and_sidecar(tmp_path):
    raw = _raw(tmp_path)
    boxes = [YoloBox.from_pixels(0, 4.0, 8.0, 20.0, 24.0, 64, 48, object_id="b0")]
    record = write_sample(
        raw,
        split="train",
        scene_id="sim_0000",
        camera="cam_nav",
        pose_index=3,
        frame_index=0,
        image=_image(),
        boxes=boxes,
        dropped=[{"kind": "crater", "id": "c0", "reason": "too_small"}],
        meta={"source": "sim"},
    )
    assert record["boxes"] == 1
    stem = record["stem"]
    assert os.path.isfile(raw.image_path("train", stem))
    assert os.path.isfile(raw.label_path("train", stem))
    payload = json.loads(open(raw.meta_path("train", stem), encoding="utf-8").read())
    assert payload["source"] == "sim"
    assert payload["dropped"][0]["reason"] == "too_small"
    assert raw_stems(raw) == [stem]


def test_write_sample_rejects_an_illegal_box(tmp_path):
    raw = _raw(tmp_path)
    bad = YoloBox(0, 0.5, 0.5, 1.4, 0.2)
    with pytest.raises(Exception, match="outside"):
        write_sample(
            raw, split="train", scene_id="sim_0000", camera="cam_nav",
            pose_index=0, frame_index=0, image=_image(), boxes=[bad],
        )


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def _build_ok_dataset(tmp_path, *, empty_second=False):
    layout = _layout(tmp_path)
    for scene, split, camera in (
        ("sim_0000", "train", "cam_nav"),
        ("sim_0001", "val", "cam_tag"),
    ):
        boxes = [] if (empty_second and scene == "sim_0001") else [
            YoloBox.from_pixels(0, 4.0, 8.0, 20.0, 24.0, 64, 48),
            YoloBox.from_pixels(1, 30.0, 10.0, 60.0, 40.0, 64, 48),
        ]
        write_sample(
            layout, split=split, scene_id=scene, camera=camera,
            pose_index=0, frame_index=0, image=_image(), boxes=boxes,
        )
    from rover_dataset.dataset import finalize_dataset

    finalize_dataset(layout, CLASSES, [{"scene_id": "sim_0000"}, {"scene_id": "sim_0001"}], {"a": 1})
    return layout


def test_verify_passes_on_a_clean_dataset(tmp_path):
    layout = _build_ok_dataset(tmp_path)
    report = verify_dataset(layout, CLASSES)
    assert report.ok, report.problems
    assert report.scene_split == {"sim_0000": "train", "sim_0001": "val"}
    assert report.stats["splits"]["train"]["boxes"] == 2


def test_verify_counts_background_only_images(tmp_path):
    layout = _build_ok_dataset(tmp_path, empty_second=True)
    report = verify_dataset(layout, CLASSES)
    assert report.ok, report.problems
    assert report.stats["splits"]["val"]["empty_images"] == 1


def test_verify_detects_a_missing_label(tmp_path):
    layout = _build_ok_dataset(tmp_path)
    stem = os.path.splitext(os.listdir(os.path.join(layout.images_dir, "train"))[0])[0]
    os.remove(layout.label_path("train", stem))
    report = verify_dataset(layout, CLASSES)
    assert not report.ok
    assert any("without a label" in problem for problem in report.problems)


def test_verify_detects_a_class_mismatch(tmp_path):
    layout = _build_ok_dataset(tmp_path)
    from rover_dataset.labels import write_classes

    write_classes(layout.classes_path, ["rock", "hole"])
    report = verify_dataset(layout, CLASSES)
    assert not report.ok
    assert any("classes.txt" in problem for problem in report.problems)


def test_verify_detects_a_scene_leaking_across_splits(tmp_path):
    layout = _layout(tmp_path)
    for split in ("train", "val"):
        write_sample(
            layout, split=split, scene_id="sim_0000", camera="cam_nav",
            pose_index=0, frame_index=0,
            image=_image(), boxes=[YoloBox.from_pixels(0, 4.0, 8.0, 20.0, 24.0, 64, 48)],
        )
    report = verify_dataset(layout, CLASSES)
    assert not report.ok
    assert any("appears in both" in problem for problem in report.problems)


def test_verify_detects_an_out_of_range_class(tmp_path):
    layout = _layout(tmp_path)
    write_sample(
        layout, split="train", scene_id="sim_0000", camera="cam_nav",
        pose_index=0, frame_index=0,
        image=_image(), boxes=[YoloBox.from_pixels(0, 4.0, 8.0, 20.0, 24.0, 64, 48)],
    )
    from rover_dataset.dataset import finalize_dataset

    finalize_dataset(layout, CLASSES, [], {})
    with open(layout.label_path("train", image_stem("sim_0000", "cam_nav", 0, 0)), "w") as handle:
        handle.write("5 0.5 0.5 0.2 0.2\n")
    report = verify_dataset(layout, CLASSES)
    assert not report.ok
    assert any("out of range" in problem for problem in report.problems)


def test_verify_checks_the_expected_split(tmp_path):
    layout = _build_ok_dataset(tmp_path)
    report = verify_dataset(
        layout, CLASSES, expected_scene_split={"sim_0000": "val", "sim_0001": "val"}
    )
    assert not report.ok
    assert any("sim_0000" in problem for problem in report.problems)


def test_verify_duplicates_detects_identical_images(tmp_path):
    from rover_dataset.dataset import finalize_dataset

    layout = _layout(tmp_path)
    for split in ("train", "val"):
        write_sample(
            layout, split=split, scene_id=f"sim_000{split}", camera="cam_nav",
            pose_index=0, frame_index=0,
            image=_image(), boxes=[YoloBox.from_pixels(0, 4.0, 8.0, 20.0, 24.0, 64, 48)],
        )
    finalize_dataset(layout, CLASSES, [], {})
    report = verify_dataset(layout, CLASSES, check_duplicates=True)
    assert any("identical image" in problem for problem in report.problems)


def test_dataset_stats_are_per_split(tmp_path):
    layout = _build_ok_dataset(tmp_path)
    stats = dataset_stats(layout, CLASSES)
    assert stats["splits"]["train"]["images"] == 1
    assert stats["splits"]["train"]["per_class"] == {"boulder": 1, "crater": 1}
    assert stats["splits"]["train"]["per_camera"] == {"cam_nav": 1}
