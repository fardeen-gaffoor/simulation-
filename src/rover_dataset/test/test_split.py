"""Scene-level split and real-photo weighting tests.

The two things that must never regress: a scene may not appear in both splits,
and the real-photo target math must be right.
"""

from __future__ import annotations

import os

import pytest

from rover_dataset.config import SamplingCfg, SplitCfg
from rover_dataset.dataset import DatasetLayout, RawLayout, verify_dataset, write_sample
from rover_dataset.labels import YoloBox
from rover_dataset.split import build_split, sampling_plan, split_scenes

CLASSES = ["boulder", "crater"]


def _box():
    return YoloBox.from_pixels(0, 4.0, 8.0, 20.0, 24.0, 64, 48)


def _stage(tmp_path, scene_id: str, count: int, camera="cam_nav"):
    raw = RawLayout(root=str(tmp_path))
    for index in range(count):
        write_sample(
            raw, split="train", scene_id=scene_id, camera=camera,
            pose_index=index, frame_index=0,
            image=__import__("numpy").full((48, 64, 3), 100 + index, dtype="uint8"),
            boxes=[_box()],
        )


# --------------------------------------------------------------------------- #
# split_scenes
# --------------------------------------------------------------------------- #
def test_split_is_deterministic():
    cfg = SplitCfg(val_fraction=0.25, seed=0, seed_namespace="split", stratify_by_source=True)
    ids = [f"sim_{i:04d}" for i in range(8)]
    sources = {scene_id: "sim" for scene_id in ids}
    first = split_scenes(ids, sources, cfg)
    second = split_scenes(list(reversed(ids)), sources, cfg)
    assert first == second


def test_split_never_leaks_a_scene():
    cfg = SplitCfg(val_fraction=0.5, seed=3, seed_namespace="split", stratify_by_source=True)
    ids = [f"sim_{i:04d}" for i in range(10)]
    result = split_scenes(ids, {i: "sim" for i in ids}, cfg)
    assert set(result) == set(ids)
    assert set(result.values()) <= {"train", "val"}
    assert sum(1 for v in result.values() if v == "val") == 5


def test_split_keeps_a_single_scene_in_train():
    cfg = SplitCfg(val_fraction=0.25, seed=0, seed_namespace="split", stratify_by_source=True)
    result = split_scenes(["sim_0000"], {"sim_0000": "sim"}, cfg)
    assert result == {"sim_0000": "train"}


def test_split_never_empties_val_for_two_scenes():
    cfg = SplitCfg(val_fraction=0.01, seed=0, seed_namespace="split", stratify_by_source=True)
    ids = ["sim_0000", "sim_0001"]
    result = split_scenes(ids, {i: "sim" for i in ids}, cfg)
    assert sorted(result.values()) == ["train", "val"]


def test_split_stratifies_by_source():
    cfg = SplitCfg(val_fraction=0.5, seed=0, seed_namespace="split", stratify_by_source=True)
    ids = [f"sim_{i:04d}" for i in range(4)] + ["real_a", "real_b"]
    sources = {i: ("sim" if i.startswith("sim") else "real") for i in ids}
    result = split_scenes(ids, sources, cfg)
    real_val = [k for k, v in result.items() if v == "val" and k.startswith("real")]
    assert len(real_val) == 1  # each source contributes to val


def test_split_of_nothing_is_empty():
    cfg = SplitCfg(val_fraction=0.25, seed=0, seed_namespace="split", stratify_by_source=True)
    assert split_scenes([], {}, cfg) == {}


# --------------------------------------------------------------------------- #
# sampling_plan
# --------------------------------------------------------------------------- #
def test_sampling_plan_without_real_photos_is_honest():
    cfg = SamplingCfg(target_real_fraction=0.6, max_repeat=8, sim_repeat=1)
    plan = sampling_plan({"sim": 1000}, cfg)
    assert plan["real_repeat"] == 1
    assert plan["real_images"] == 0
    assert plan["reachable"] is False


def test_sampling_plan_reaches_the_target():
    cfg = SamplingCfg(target_real_fraction=0.6, max_repeat=20, sim_repeat=1)
    # 1000 sim, 100 real: need real*r/(1000 + real*r) >= 0.6 -> r >= 15
    plan = sampling_plan({"sim": 1000, "real": 100}, cfg)
    assert plan["real_repeat"] == 15
    assert plan["real_fraction_after"] >= 0.6
    assert plan["reachable"] is True


def test_sampling_plan_respects_max_repeat_and_reports_it():
    cfg = SamplingCfg(target_real_fraction=0.9, max_repeat=3, sim_repeat=1)
    plan = sampling_plan({"sim": 1000, "real": 10}, cfg)
    assert plan["real_repeat"] == 3
    assert plan["capped_by_max_repeat"] is True
    assert plan["reachable"] is False


def test_sampling_plan_already_past_the_target_keeps_one_copy():
    cfg = SamplingCfg(target_real_fraction=0.5, max_repeat=8, sim_repeat=1)
    plan = sampling_plan({"sim": 10, "real": 100}, cfg)
    assert plan["real_repeat"] == 1
    assert plan["reachable"] is True


# --------------------------------------------------------------------------- #
# build_split
# --------------------------------------------------------------------------- #
def test_build_split_materialises_both_splits(tmp_path, cfg):
    for index in range(4):
        _stage(tmp_path, f"sim_{index:04d}", 3)
    scenes = [
        {"scene_id": f"sim_{index:04d}", "source": "sim", "images": 3} for index in range(4)
    ]
    result = build_split(cfg, str(tmp_path), scenes)
    assert result.written["train"] + result.written["val"] == 12
    assert result.written["val"] > 0
    layout = DatasetLayout(
        root=str(tmp_path), image_format="jpg", image_quality=cfg.paths.image_quality
    )
    report = verify_dataset(layout, cfg.class_names, expected_scene_split=result.scene_split)
    assert report.ok, report.problems
    # symlinks are used by default
    train_images = os.listdir(os.path.join(layout.images_dir, "train"))
    assert train_images
    assert os.path.islink(os.path.join(layout.images_dir, "train", train_images[0]))


def test_build_split_oversamples_real_photos(tmp_path, cfg):
    for index in range(3):
        _stage(tmp_path, f"sim_{index:04d}", 20)
    _stage(tmp_path, "real_batch", 5, camera="cam_nav")
    scenes = [
        {"scene_id": f"sim_{index:04d}", "source": "sim", "images": 20} for index in range(3)
    ] + [{"scene_id": "real_batch", "source": "real", "images": 5}]
    result = build_split(cfg, str(tmp_path), scenes)
    assert result.scene_split["real_batch"] in ("train", "val")
    if result.scene_split["real_batch"] == "val":
        pytest.skip("the real batch landed in val for this seed")
    assert result.sampling["real_repeat"] > 1
    assert result.sampling["real_fraction_after"] >= min(
        0.999, result.sampling["target_real_fraction"]
    ) or result.sampling["capped_by_max_repeat"]
    # repeated copies are materialised with an __r suffix
    names = os.listdir(os.path.join(tmp_path, "images", "train"))
    assert any("__r2" in name for name in names)


def test_build_split_is_idempotent(tmp_path, cfg):
    for index in range(2):
        _stage(tmp_path, f"sim_{index:04d}", 2)
    scenes = [
        {"scene_id": f"sim_{index:04d}", "source": "sim", "images": 2} for index in range(2)
    ]
    first = build_split(cfg, str(tmp_path), scenes)
    second = build_split(cfg, str(tmp_path), scenes)
    assert first.written == second.written
    total = sum(len(files) for _, _, files in os.walk(os.path.join(tmp_path, "images")))
    assert total == 4


def test_build_split_can_copy_instead_of_linking(tmp_path, cfg):
    for index in range(2):
        _stage(tmp_path, f"sim_{index:04d}", 2)
    scenes = [
        {"scene_id": f"sim_{index:04d}", "source": "sim", "images": 2} for index in range(2)
    ]
    build_split(cfg, str(tmp_path), scenes, use_links=False)
    names = os.listdir(os.path.join(tmp_path, "images", "train"))
    assert names
    for name in names:
        path = os.path.join(tmp_path, "images", "train", name)
        assert not os.path.islink(path)
        assert os.path.isfile(path)
        # a real copy carries real bytes, so a self-contained dataset is portable
        assert os.path.getsize(path) > 0


def test_build_split_writes_data_yaml_and_split_json(tmp_path, cfg):
    _stage(tmp_path, "sim_0000", 2)
    build_split(cfg, str(tmp_path), [{"scene_id": "sim_0000", "source": "sim", "images": 2}])
    assert os.path.isfile(os.path.join(tmp_path, "data.yaml"))
    assert os.path.isfile(os.path.join(tmp_path, "split.json"))
    assert os.path.isfile(os.path.join(tmp_path, "scenes.json"))
    assert os.path.isfile(os.path.join(tmp_path, "classes.txt"))
