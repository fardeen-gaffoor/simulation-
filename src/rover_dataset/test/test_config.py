"""``dataset.yaml`` contract tests.

The dataset definition is the source of truth for collection, labelling,
augmentation and the split, so a bad value must fail here rather than in the
middle of an eight-scene collection run.
"""

from __future__ import annotations

import os

import pytest
import yaml

from rover_dataset.config import (
    GEOMETRIC_OPS,
    PHOTOMETRIC_OPS,
    DatasetConfigError,
    load_config,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(os.path.dirname(_HERE), "config", "dataset.yaml")


def _base_raw():
    with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _load_mutated(tmp_path, mutate):
    raw = _base_raw()
    mutate(raw)
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(str(path))


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_shipped_config_loads(cfg):
    assert cfg.schema_version == 1
    assert cfg.name.startswith("Caterpillar")


def test_class_order_is_a_wire_contract(cfg):
    # class index 0 IS a boulder; module 4's detector depends on this order
    assert cfg.class_names == ["boulder", "crater"]
    assert cfg.class_index("boulder") == 0
    assert cfg.class_index("crater") == 1
    assert cfg.class_by_name("crater").index == 1


def test_capture_plan_is_sane(cfg):
    assert cfg.capture.poses_per_scene > 0
    assert cfg.capture.frames_per_pose >= 1
    assert cfg.capture.cameras
    for camera in cfg.capture.cameras:
        # module 2 publishes exactly these four topics
        assert camera in ("cam_nav", "cam_tag", "cam_excavation", "cam_dump")


def test_pose_regions_are_inside_the_arena(cfg):
    for region in cfg.capture.pose_plan.regions:
        assert 0.0 <= region.x0 < region.x1 <= 9.0
        assert 0.0 <= region.y0 < region.y1 <= 5.0
        assert region.weight > 0.0


def test_pose_region_names_match_the_competition_zones(cfg):
    names = {region.name for region in cfg.capture.pose_plan.regions}
    assert names == {"obstacle", "excavation", "construction", "start"}


def test_augmentation_ops_are_known_and_bounded(cfg):
    assert set(cfg.augment.ops) <= set(PHOTOMETRIC_OPS) | set(GEOMETRIC_OPS)
    for name, op in cfg.augment.ops.items():
        assert 0.0 <= op.prob <= 1.0, name
        for key, value in op.params.items():
            assert value.lo <= value.hi, f"{name}.{key}"


def test_geometry_changing_ops_are_disabled_by_default(cfg):
    # the CSI hub is not chosen yet, so images must NOT be assumed rotated
    assert cfg.augment.ops["flip_180"].prob == 0.0
    assert cfg.augment.ops["horizontal_flip"].prob == 0.0
    assert cfg.augment.ops["flip_180"].geometric
    assert not cfg.augment.ops["gaussian_noise"].geometric


def test_scene_randomization_is_configured(cfg):
    assert cfg.scene.count >= 1
    assert cfg.scene.sun_elevation_deg.lo >= 0.0
    assert cfg.scene.sun_elevation_deg.hi <= 90.0
    assert cfg.scene.mount_rotation_deg.hi >= 0.0
    assert 0.0 <= cfg.scene.sun_cast_shadows_prob <= 1.0


def test_split_never_leaves_val_empty(cfg):
    assert 0.0 < cfg.split.val_fraction < 1.0
    assert cfg.split.stratify_by_source


def test_sampling_target_encodes_the_brief(cfg):
    # "real photos must carry most of the weight"
    assert cfg.sampling.target_real_fraction >= 0.5
    assert cfg.sampling.max_repeat >= 1


def test_config_sha256_is_stable(cfg):
    assert cfg.config_sha256() == cfg.config_sha256()
    assert len(cfg.config_sha256()) == 64


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #
def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="not found"):
        load_config(str(tmp_path / "nope.yaml"))


def test_bad_schema_version_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="schema_version"):
        _load_mutated(tmp_path, lambda raw: raw.update({"schema_version": 7}))


def test_unknown_class_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="non-empty list"):
        _load_mutated(tmp_path, lambda raw: raw.update({"classes": []}))


def test_duplicate_class_name_is_rejected(tmp_path):
    def mutate(raw):
        raw["classes"] = [{"name": "boulder"}, {"name": "boulder"}]

    with pytest.raises(DatasetConfigError, match="duplicate class"):
        _load_mutated(tmp_path, mutate)


def test_unknown_augmentation_op_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="unknown op"):
        _load_mutated(tmp_path, lambda raw: raw["augment"].update({"supernova": {"prob": 1.0}}))


def test_probability_above_one_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="prob"):
        _load_mutated(tmp_path, lambda raw: raw["augment"]["gamma"].update({"prob": 1.5}))


def test_inverted_range_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="<= max"):
        _load_mutated(
            tmp_path,
            lambda raw: raw["augment"]["gamma"].update({"value": {"min": 2.0, "max": 1.0}}),
        )


def test_degenerate_pose_region_is_rejected(tmp_path):
    def mutate(raw):
        raw["capture"]["pose_plan"]["regions"][0]["rect"] = [2.0, 2.0, 2.0, 4.0]

    with pytest.raises(DatasetConfigError, match="positive width"):
        _load_mutated(tmp_path, mutate)


def test_unknown_heading_mode_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="heading"):
        _load_mutated(tmp_path, lambda raw: raw["capture"]["pose_plan"].update({"heading": "spin"}))


def test_bad_occlusion_policy_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="occluded_policy"):
        _load_mutated(tmp_path, lambda raw: raw["labeling"].update({"occluded_policy": "maybe"}))


def test_out_of_range_visible_fraction_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="min_visible_fraction"):
        _load_mutated(tmp_path, lambda raw: raw["labeling"].update({"min_visible_fraction": 0.0}))


def test_bad_image_format_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="image_format"):
        _load_mutated(tmp_path, lambda raw: raw["paths"].update({"image_format": "bmp"}))


def test_zero_poses_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="poses_per_scene"):
        _load_mutated(tmp_path, lambda raw: raw["capture"].update({"poses_per_scene": 0}))


def test_zero_scene_count_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="scene.count"):
        _load_mutated(tmp_path, lambda raw: raw["scene"].update({"count": 0}))


def test_val_fraction_of_one_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="val_fraction"):
        _load_mutated(tmp_path, lambda raw: raw["split"].update({"val_fraction": 1.0}))


def test_missing_capture_section_is_rejected(tmp_path):
    with pytest.raises(DatasetConfigError, match="capture"):
        _load_mutated(tmp_path, lambda raw: raw.pop("capture"))


def test_unknown_camera_name_is_rejected_when_calibration_is_available(tmp_path):
    import importlib

    config_module = importlib.import_module("rover_dataset.config")
    if not config_module._known_camera_names():
        pytest.skip("rover_description calibration not available")
    with pytest.raises(DatasetConfigError, match="not a rover_description camera"):
        _load_mutated(tmp_path, lambda raw: raw["capture"].update({"cameras": ["cam_nope"]}))
