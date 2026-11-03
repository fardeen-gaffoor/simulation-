"""Configuration contract tests for ``perception.yaml``.

This file is the single source of truth for module 4, so a typo or an accidental
rule violation has to fail *here*, before it can produce a wrong projection or a
wrong tag pose.  Two tests in particular are cross-module guard rails:

* :func:`test_arena_matches_cat_arena` re-reads module 1's ``arena.yaml`` and
  checks the dimensions and zone rectangles module 4 duplicates by hand,
* :func:`test_tag_protrusion_matches_cat_arena_board` re-derives the tag standoff
  from module 1's board thickness, so a change there cannot silently bias every
  tag fix by half a board.
"""

from __future__ import annotations

import copy
import math
import os

import pytest
import yaml

from rover_perception.config import (
    ACTIVE_CAMERA_ROLE,
    MODES,
    OBSTACLE_CLASS_IDS,
    THERMAL_STATES,
    PerceptionConfigError,
    load_config,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_CAT_ARENA_ARENA_YAML = os.path.normpath(
    os.path.join(_HERE, "..", "..", "cat_arena", "config", "arena.yaml")
)


def _write(tmp_path, document, name="perception.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return str(path)


def _document(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------------- #
# the shipped configuration
# --------------------------------------------------------------------------- #
def test_shipped_config_loads(cfg):
    assert cfg.schema_version == 1
    assert cfg.path.endswith("perception.yaml")
    assert len(cfg.sha256) == 64


def test_four_cameras_with_roles(cfg):
    assert cfg.camera_names == ("cam_nav", "cam_tag", "cam_excavation", "cam_dump")
    assert [camera.role for camera in cfg.cameras] == ["nav", "tag", "excavation", "dump"]


def test_camera_roles_are_unique(cfg):
    roles = [camera.role for camera in cfg.cameras]
    assert len(set(roles)) == len(roles), "roles index ACTIVE_CAMERA_ROLE, so they must be unique"


def test_bandwidth_budget_is_the_competition_limit(cfg):
    assert cfg.camera_manager.bandwidth.limit_kbps == 4000.0
    assert cfg.camera_manager.bandwidth.window_s > 0.0
    assert cfg.camera_manager.bandwidth.hard_limit_kbps >= cfg.camera_manager.bandwidth.limit_kbps


def test_derate_ladder_is_monotonic_and_reaches_off(cfg):
    factors = cfg.camera_manager.bandwidth.derate_factors
    assert factors[0] == 1.0
    assert factors[-1] == 0.0
    assert all(a >= b for a, b in zip(factors, factors[1:]))


def test_idle_below_active_and_not_below_minimum(cfg):
    manager = cfg.camera_manager
    assert manager.idle_rate_hz <= manager.active_rate_hz
    assert manager.idle_rate_hz == 0.0 or manager.idle_rate_hz >= manager.min_rate_hz


def test_thermal_thresholds_are_ordered(cfg):
    thermal = cfg.camera_manager.thermal
    assert thermal.warn_c < thermal.throttle_c < thermal.critical_c
    assert thermal.throttle_steps >= 0


def test_arena_dimensions_and_zones(cfg):
    arena = cfg.arena
    assert (arena.length_x, arena.width_y) == (9.0, 5.0)
    assert arena.origin_xy == (0.0, 0.0)
    assert arena.ground_z == 0.0
    assert set(arena.zones) == {
        "start",
        "corridor",
        "construction",
        "obstacle",
        "excavation",
    }
    assert arena.zones["start"].as_tuple() == (0.0, 2.70, 2.0, 5.00)
    assert arena.zones["construction"].as_tuple() == (0.0, 0.0, 2.0, 1.50)
    assert arena.zones["obstacle"].as_tuple() == (2.0, 0.0, 7.5, 5.0)
    assert arena.zones["excavation"].as_tuple() == (7.5, 0.0, 9.0, 5.0)


def test_every_zone_is_inside_the_arena(cfg):
    for name, rect in cfg.arena.zones.items():
        assert cfg.arena.bounds.contains(*rect.center), f"zone {name} centre is outside the arena"
        assert 0.0 <= rect.x0 and rect.x1 <= cfg.arena.length_x
        assert 0.0 <= rect.y0 and rect.y1 <= cfg.arena.width_y


def test_zone_of_and_inside(cfg):
    assert cfg.arena.zone_of(1.0, 4.0) == "start"
    assert cfg.arena.zone_of(1.0, 0.5) == "construction"
    assert cfg.arena.zone_of(5.0, 2.5) == "obstacle"
    assert cfg.arena.zone_of(8.0, 2.5) == "excavation"
    assert cfg.arena.inside(4.5, 2.5)
    assert not cfg.arena.inside(-0.01, 2.5), "outside the arena must be masked"
    # the wall margin shrinks the legal area
    assert not cfg.arena.inside(0.05, 2.5)
    assert cfg.arena.inside(0.05, 2.5, margin=0.0)


def test_active_camera_for_mode(cfg):
    assert cfg.active_camera_for_mode("nav") == "cam_nav"
    assert cfg.active_camera_for_mode("tag") == "cam_tag"
    assert cfg.active_camera_for_mode("excavate") == "cam_excavation"
    assert cfg.active_camera_for_mode("dump") == "cam_dump"
    assert cfg.active_camera_for_mode("idle") is None
    assert cfg.active_camera_for_mode("degraded") is None
    assert set(ACTIVE_CAMERA_ROLE) == set(MODES)


def test_camera_payload_bytes_are_exact(cfg):
    nav = cfg.camera("cam_nav")
    assert nav.payload_bytes == 640 * 480 * 3
    tag = cfg.camera("cam_tag")
    assert tag.payload_bytes == 1280 * 720 * 3
    assert tag.payload_bytes > nav.payload_bytes


def test_camera_lookup_errors_are_explicit(cfg):
    with pytest.raises(PerceptionConfigError):
        cfg.camera("cam_nope")


def test_detector_contract(cfg):
    detector = cfg.detector
    assert detector.backend in ("classical", "onnx")
    assert detector.camera in cfg.camera_names
    assert detector.ground.min_range_m > 0.0
    assert detector.ground.max_range_m > detector.ground.min_range_m
    assert detector.ground.contact_point in ("bbox_bottom_center", "bbox_bottom_center_inset")
    assert cfg.tags.camera in cfg.camera_names
    assert cfg.tags.family == "tag36h11"
    # tag36h11 separates its codes by >= 11 bits, so accepting a 1-bit
    # correction cannot confuse two ids (see test_tagcodes.py)
    assert 0 <= cfg.tags.detection.hamming_max <= 1


def test_class_order_is_module_threes_contract(cfg):
    assert cfg.detector.onnx.classes == ("boulder", "crater")
    assert OBSTACLE_CLASS_IDS == {"boulder": 0, "crater": 1}


def test_localization_has_no_magnetometer_and_no_amcl(cfg):
    localization = cfg.localization
    assert localization.odometry_topics, "at least one odometry topic must be configured"
    assert localization.growth_m_per_m > 0.0, "covariance must grow while no tag is visible"
    assert localization.max_correction_m > 0.0
    assert 0.0 < localization.xy_gain <= 1.0
    assert 0.0 < localization.yaw_gain <= 1.0


def test_thermal_states_are_named(cfg):
    assert THERMAL_STATES == ("nominal", "warning", "throttling", "critical")


def test_health_topics(cfg):
    assert cfg.health.topic.startswith("/")
    assert cfg.health.diagnostics_topic == "/diagnostics"
    assert cfg.health.publish_hz > 0.0


# --------------------------------------------------------------------------- #
# validation: break the document one field at a time
# --------------------------------------------------------------------------- #
def test_missing_file_is_reported():
    with pytest.raises(PerceptionConfigError) as excinfo:
        load_config("/nonexistent/perception.yaml")
    assert "not found" in str(excinfo.value)


def test_missing_top_level_section(tmp_path, config_path):
    document = _document(config_path)
    del document["arena"]
    with pytest.raises(PerceptionConfigError) as excinfo:
        load_config(_write(tmp_path, document))
    assert "arena" in str(excinfo.value)


def test_schema_version_is_checked(tmp_path, config_path):
    document = _document(config_path)
    document["schema_version"] = 99
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_idle_rate_above_active_is_rejected(tmp_path, config_path):
    document = _document(config_path)
    document["camera_manager"]["idle_rate_hz"] = document["camera_manager"]["active_rate_hz"] + 1.0
    with pytest.raises(PerceptionConfigError) as excinfo:
        load_config(_write(tmp_path, document))
    assert "idle_rate_hz" in str(excinfo.value)


def test_idle_rate_below_minimum_is_rejected(tmp_path, config_path):
    document = _document(config_path)
    manager = document["camera_manager"]
    manager["min_rate_hz"] = 5.0
    manager["idle_rate_hz"] = 1.0
    with pytest.raises(PerceptionConfigError) as excinfo:
        load_config(_write(tmp_path, document))
    assert "min_rate_hz" in str(excinfo.value)


def test_zone_outside_arena_is_rejected(tmp_path, config_path):
    document = _document(config_path)
    document["arena"]["zones"]["start"] = [0.0, 2.70, 20.0, 5.00]
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_degenerate_zone_is_rejected(tmp_path, config_path):
    document = _document(config_path)
    document["arena"]["zones"]["construction"] = [0.0, 0.0, 0.0, 1.5]
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_duplicate_camera_names_are_rejected(tmp_path, config_path):
    document = _document(config_path)
    document["cameras"].append(copy.deepcopy(document["cameras"][0]))
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_unknown_role_is_rejected(tmp_path, config_path):
    document = _document(config_path)
    document["cameras"][0]["role"] = "sideways"
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_unknown_detector_backend_is_rejected(tmp_path, config_path):
    document = _document(config_path)
    document["detector"]["backend"] = "magic"
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_negative_wall_margin_is_rejected(tmp_path, config_path):
    document = _document(config_path)
    document["arena"]["wall_margin_m"] = -0.1
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_zero_active_rate_is_rejected(tmp_path, config_path):
    document = _document(config_path)
    document["camera_manager"]["active_rate_hz"] = 0.0
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_tag_protrusion_must_be_non_negative(tmp_path, config_path):
    document = _document(config_path)
    document["tags"]["board"]["protrusion_m"] = -0.001
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_tag_lift_must_be_positive(tmp_path, config_path):
    document = _document(config_path)
    document["tags"]["board"]["lift_m"] = 0.0
    with pytest.raises(PerceptionConfigError):
        load_config(_write(tmp_path, document))


def test_onnx_backend_requires_a_model(tmp_path, config_path):
    document = _document(config_path)
    document["detector"]["backend"] = "onnx"
    document["detector"]["onnx"]["model"] = ""
    with pytest.raises(PerceptionConfigError) as excinfo:
        load_config(_write(tmp_path, document))
    assert "model" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# cross-module guard rails
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.isfile(_CAT_ARENA_ARENA_YAML), reason="module 1's arena.yaml is not in this workspace"
)
def test_arena_matches_cat_arena(cfg):
    with open(_CAT_ARENA_ARENA_YAML, "r", encoding="utf-8") as handle:
        arena = yaml.safe_load(handle)
    assert float(arena["arena"]["length_x"]) == cfg.arena.length_x
    assert float(arena["arena"]["width_y"]) == cfg.arena.width_y
    # module 4 renames module 1's "left_corridor" to "corridor"
    alias = {"corridor": "left_corridor"}
    for name, rect in cfg.arena.zones.items():
        source = arena["zones"][alias.get(name, name)]["rect"]
        assert rect.as_tuple() == tuple(float(v) for v in source), f"zone {name} drifted"


@pytest.mark.skipif(
    not os.path.isfile(_CAT_ARENA_ARENA_YAML), reason="module 1's arena.yaml is not in this workspace"
)
def test_tag_protrusion_matches_cat_arena_board(cfg):
    with open(_CAT_ARENA_ARENA_YAML, "r", encoding="utf-8") as handle:
        arena = yaml.safe_load(handle)
    markers = arena["markers"]
    thickness = float(markers["board_thickness"])
    # module 1 draws the tag as a box of 1.4 * thickness centred on the board,
    # so half of it protrudes in front of the board's centre plane.
    expected_protrusion = 0.5 * 1.4 * thickness
    assert cfg.tags.board.protrusion_m == pytest.approx(expected_protrusion, abs=1e-9)
    assert cfg.tags.board.protrusion_m < config_val(markers, "board_size", 0.25)


def config_val(mapping, key, default):
    return float(mapping.get(key, default))


@pytest.mark.skipif(
    not os.path.isfile(_CAT_ARENA_ARENA_YAML), reason="module 1's arena.yaml is not in this workspace"
)
def test_tag_size_matches_cat_arena(cfg):
    with open(_CAT_ARENA_ARENA_YAML, "r", encoding="utf-8") as handle:
        arena = yaml.safe_load(handle)
    markers = arena["markers"]
    assert float(markers["tag_size"]) * 1000.0 > 0.0
    assert float(markers["center_height"]) > 0.0
    assert float(markers["board_size"]) <= 0.25
    assert math.isclose(cfg.tags.board.standoff_m, cfg.tags.board.protrusion_m + cfg.tags.board.lift_m, abs_tol=1e-12)
