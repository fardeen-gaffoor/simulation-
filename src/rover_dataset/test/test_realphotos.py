"""Real-photo import tests.

The importer is the only gate between a human's labels and the training set, so
it must reject bad class ids, illegal boxes and a mounting-height mismatch.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from rover_dataset.dataset import RawLayout, raw_stems, save_image
from rover_dataset.labels import YoloBox, write_labels
from rover_dataset.realphotos import import_real_photos


def _photo(path: str, width=64, height=48, value=120):
    save_image(path, np.full((height, width, 3), value, dtype=np.uint8), "jpg", 95)


@pytest.fixture
def photos(tmp_path):
    directory = tmp_path / "photos"
    directory.mkdir()
    for index in range(3):
        stem = f"img{index:03d}"
        _photo(str(directory / f"{stem}.jpg"), value=100 + index)
        write_labels(
            str(directory / f"{stem}.txt"),
            [YoloBox.from_pixels(0, 4.0, 8.0, 20.0, 24.0, 64, 48, object_id="b0")],
        )
    return str(directory)


def test_import_writes_staged_samples(tmp_path, cfg, calibration, photos):
    root = str(tmp_path / "dataset")
    result = import_real_photos(cfg, root, photos, None, "kitchen", "cam_nav", calibration=calibration)
    assert result.ok, result.problems
    assert result.imported == 3
    assert result.scene_id == "real_kitchen"
    raw = RawLayout(root=root)
    stems = raw_stems(raw)
    assert len(stems) == 3
    assert all(stem.startswith("real_kitchen__cam_nav__p0000__f") for stem in stems)
    assert all(os.path.isfile(raw.label_path("train", stem)) for stem in stems)
    record = result.as_scene_record()
    assert record["source"] == "real"
    assert record["images"] == 3


def test_import_records_the_calibration_height(tmp_path, cfg, calibration, photos):
    root = str(tmp_path / "dataset")
    result = import_real_photos(cfg, root, photos, None, "b1", "cam_nav", calibration=calibration)
    assert result.calibration_height_m == pytest.approx(calibration.camera("cam_nav").t_base_cam[2])


def test_import_rejects_a_mount_height_mismatch(tmp_path, cfg, calibration, photos):
    root = str(tmp_path / "dataset")
    wrong = calibration.camera("cam_nav").t_base_cam[2] + 0.10
    result = import_real_photos(
        cfg, root, photos, None, "b1", "cam_nav", mount_height_m=wrong, calibration=calibration
    )
    assert not result.ok
    assert any("mount height" in problem for problem in result.problems)
    # the photos are still imported; the human is warned before training
    assert result.imported == 3


def test_import_accepts_a_matching_mount_height(tmp_path, cfg, calibration, photos):
    root = str(tmp_path / "dataset")
    height = calibration.camera("cam_nav").t_base_cam[2]
    result = import_real_photos(
        cfg, root, photos, None, "b1", "cam_nav", mount_height_m=height, calibration=calibration
    )
    assert result.ok, result.problems


def test_import_rejects_an_unknown_camera(tmp_path, cfg, calibration, photos):
    root = str(tmp_path / "dataset")
    result = import_real_photos(cfg, root, photos, None, "b1", "cam_nope", calibration=calibration)
    assert not result.ok
    assert any("not in the calibration" in problem for problem in result.problems)
    assert result.imported == 0


def test_missing_labels_are_skipped(tmp_path, cfg, calibration, photos):
    os.remove(os.path.join(photos, "img001.txt"))
    root = str(tmp_path / "dataset")
    result = import_real_photos(cfg, root, photos, None, "b1", "cam_nav", calibration=calibration)
    assert result.imported == 2
    assert any("img001" in entry and "no label file" in entry for entry in result.skipped)


def test_unlabeled_can_be_imported_as_background(tmp_path, cfg, calibration, photos):
    os.remove(os.path.join(photos, "img001.txt"))
    root = str(tmp_path / "dataset")
    result = import_real_photos(
        cfg, root, photos, None, "b1", "cam_nav",
        calibration=calibration, unlabeled_as_negative=True,
    )
    assert result.imported == 3
    raw = RawLayout(root=root)
    empty = [stem for stem in raw_stems(raw) if os.path.getsize(raw.label_path("train", stem)) == 0]
    assert len(empty) == 1


def test_out_of_range_class_is_reported(tmp_path, cfg, calibration, photos):
    write_labels(
        os.path.join(photos, "img000.txt"),
        [YoloBox.from_pixels(7, 4.0, 8.0, 20.0, 24.0, 64, 48)],
    )
    root = str(tmp_path / "dataset")
    result = import_real_photos(cfg, root, photos, None, "b1", "cam_nav", calibration=calibration)
    assert not result.ok
    assert any("class_id 7" in problem for problem in result.problems)


def test_box_outside_the_image_is_reported(tmp_path, cfg, calibration, photos):
    path = os.path.join(photos, "img000.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("0 0.99 0.5 0.5 0.3\n")
    root = str(tmp_path / "dataset")
    result = import_real_photos(cfg, root, photos, None, "b1", "cam_nav", calibration=calibration)
    assert not result.ok
    assert any("outside" in problem for problem in result.problems)


def test_empty_directory_is_rejected(tmp_path, cfg, calibration):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = import_real_photos(
        cfg, str(tmp_path / "dataset"), str(empty), None, "b1", "cam_nav", calibration=calibration
    )
    assert not result.ok
    assert any("no photographs" in problem for problem in result.problems)


def test_missing_directory_is_rejected(tmp_path, cfg, calibration):
    result = import_real_photos(
        cfg, str(tmp_path / "dataset"), str(tmp_path / "nope"), None, "b1", "cam_nav",
        calibration=calibration,
    )
    assert not result.ok
    assert any("not found" in problem for problem in result.problems)


def test_labels_can_live_in_a_separate_directory(tmp_path, cfg, calibration, photos):
    labels = tmp_path / "labels"
    labels.mkdir()
    for name in sorted(os.listdir(photos)):
        if name.endswith(".txt"):
            os.rename(os.path.join(photos, name), str(labels / name))
    root = str(tmp_path / "dataset")
    result = import_real_photos(
        cfg, root, photos, str(labels), "b1", "cam_nav", calibration=calibration
    )
    assert result.ok, result.problems
    assert result.imported == 3


# --------------------------------------------------------------------------- #
# the CLI wrapper: a batch with problems must not reach the split
# --------------------------------------------------------------------------- #
def _import_script():
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "scripts", "import_real_photos.py")
    spec = importlib.util.spec_from_file_location("import_real_photos_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_refuses_to_register_a_batch_with_a_problem(tmp_path, photos, calibration):
    from rover_dataset.registry import IMPORTS_FILE

    root = tmp_path / "dataset"
    code = _import_script().main(
        [
            "--out-dir", str(root), "--photos", photos, "--batch", "b1",
            "--camera", "cam_nav", "--mount-height", "0.99",
            "--calibration", calibration.path,
        ]
    )
    assert code == 1
    assert not (root / IMPORTS_FILE).exists(), "a bad batch must not join the split"


def test_cli_registers_a_clean_batch(tmp_path, photos, calibration):
    from rover_dataset.registry import load_imports

    root = tmp_path / "dataset"
    code = _import_script().main(
        [
            "--out-dir", str(root), "--photos", photos, "--batch", "b1",
            "--camera", "cam_nav", "--calibration", calibration.path,
        ]
    )
    assert code == 0
    scenes = load_imports(str(root))["scenes"]
    assert [scene["scene_id"] for scene in scenes] == ["real_b1"]
    assert scenes[0]["images"] == 3
