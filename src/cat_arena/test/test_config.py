"""Configuration contract tests.

Everything in the stack reads arena geometry from ``config/arena.yaml``, so a
typo or an accidental rule violation there must fail *here*, before it can
produce a wrong world or, worse, a world that violates a competition rule.
"""

from __future__ import annotations

import math

import pytest
import yaml

from cat_arena.config import ConfigError, Rect, load_config

# --------------------------------------------------------------------------- #
# happy path: the shipped configuration
# --------------------------------------------------------------------------- #
def test_shipped_config_loads(cfg):
    assert cfg.schema_version == 1
    assert cfg.world_name == "cat_arena"


def test_competition_ground_truth_dimensions(cfg):
    # These four numbers are given by the challenge and must never drift.
    assert cfg.length_x == pytest.approx(9.0)
    assert cfg.width_y == pytest.approx(5.0)
    assert cfg.sand_depth == pytest.approx(0.30)
    assert cfg.friction.mu == pytest.approx(0.6)
    assert cfg.friction.mu2 == pytest.approx(0.6)


def test_zone_rectangles(cfg):
    assert cfg.zone("start").rect.as_list() == pytest.approx([0.0, 2.70, 2.0, 5.00])
    assert cfg.zone("construction").rect.as_list() == pytest.approx([0.0, 0.0, 2.0, 1.50])
    assert cfg.zone("obstacle").rect.as_list() == pytest.approx([2.0, 0.0, 7.5, 5.00])
    assert cfg.zone("excavation").rect.as_list() == pytest.approx([7.5, 0.0, 9.0, 5.00])
    assert cfg.zone("berm_target").rect.as_list() == pytest.approx([0.375, 0.30, 1.625, 1.20])
    # challenge: construction zone 2.0 x 1.5 m, start zone ~2 x 2.3 m
    assert cfg.zone("construction").rect.width == pytest.approx(2.0)
    assert cfg.zone("construction").rect.height == pytest.approx(1.5)
    assert cfg.zone("start").rect.width == pytest.approx(2.0)
    assert cfg.zone("start").rect.height == pytest.approx(2.3)


def test_tiling_zones_partition_the_arena(cfg):
    tiling = cfg.tiling_zones()
    assert len(tiling) == 5
    area = sum(z.rect.width * z.rect.height for z in tiling)
    assert area == pytest.approx(cfg.length_x * cfg.width_y)
    for i, a in enumerate(tiling):
        for b in tiling[i + 1:]:
            assert not a.rect.intersects(b.rect), f"{a.name} overlaps {b.name}"
    for z in tiling:
        assert cfg.footprint.covers(z.rect, tol=1e-9)


def test_no_obstacles_may_ever_spawn_in_the_start_zone(cfg):
    assert cfg.zone("start").obstacles_allowed is False
    assert "start" not in cfg.boulders.zones
    assert "start" not in cfg.craters.zones
    assert "left_corridor" not in cfg.boulders.zones
    assert "left_corridor" not in cfg.craters.zones


def test_craters_are_only_in_the_obstacle_zone(cfg):
    # challenge: craters may not be added to the excavation zone
    assert cfg.craters.zones == ["obstacle"]
    # challenge: boulders may be in the excavation zone
    assert "excavation" in cfg.boulders.zones


def test_berm_target_is_inside_the_construction_zone(cfg):
    assert cfg.zone("construction").rect.contains_rect(cfg.zone("berm_target").rect)
    target = cfg.zone("berm_target").rect
    assert target.width == pytest.approx(1.25)
    assert target.height == pytest.approx(0.90)


def test_obstacle_size_limits_match_the_rules(cfg):
    assert cfg.boulders.count_min >= 3
    assert cfg.boulders.diameter_min == pytest.approx(0.30)
    assert cfg.boulders.diameter_max == pytest.approx(0.40)
    assert cfg.craters.count_min >= 3
    assert cfg.craters.width_max <= 0.50 + 1e-9
    assert cfg.craters.depth_min > 0.0


def test_marker_limits_match_the_rules(cfg):
    assert len(cfg.markers.placements) <= 5
    assert cfg.markers.board_size <= 0.25 + 1e-9
    assert cfg.markers.center_height + 0.5 * cfg.markers.board_size <= 0.60 + 1e-9
    start_frame = [p for p in cfg.markers.placements if p.mount in ("north_wall", "west_wall")]
    assert len(start_frame) >= 2


def test_start_zone_fits_the_stowed_footprint_at_any_heading(cfg):
    worst = (cfg.start_pose.footprint_length + cfg.start_pose.footprint_width) / math.sqrt(2.0)
    rect = cfg.zone("start").rect
    assert rect.width >= worst + 2 * cfg.start_pose.margin
    assert rect.height >= worst + 2 * cfg.start_pose.margin
    assert cfg.start_pose.footprint_length == pytest.approx(1.50)
    assert cfg.start_pose.footprint_width == pytest.approx(0.75)


def test_mesh_physics_is_wired_for_triangle_terrain(cfg):
    # dartsim + bullet is the combination verified to make a triangle-mesh
    # terrain actually collide (see README "verified facts").
    assert cfg.physics.engine == "dartsim"
    assert cfg.physics.collision_detector == "bullet"
    assert cfg.terrain.cell_size >= 0.02


def test_assumptions_are_flagged(cfg):
    report = " | ".join(cfg.assumptions_report()).lower()
    assert "wall height" in report
    assert "friction" in report
    assert "organizer confirmation" in report


def test_rect_helpers():
    outer = Rect(0.0, 0.0, 2.0, 2.0)
    inner = Rect(0.5, 0.5, 1.5, 1.5)
    assert outer.contains_rect(inner)
    assert outer.contains_rect(inner, margin=0.4)
    assert not outer.contains_rect(inner, margin=0.6)
    assert outer.covers(Rect(-1e-12, 0.0, 2.0, 2.0), tol=1e-9)
    assert not outer.covers(Rect(-0.1, 0.0, 2.0, 2.0), tol=1e-9)
    assert outer.intersects(Rect(1.0, 1.0, 3.0, 3.0))
    assert not outer.intersects(Rect(2.0, 0.0, 3.0, 3.0))  # touching is not overlapping
    assert inner.distance_to_border(1.0, 1.0) == pytest.approx(0.5)
    assert outer.center == pytest.approx((1.0, 1.0))


# --------------------------------------------------------------------------- #
# rejection tests: a broken config must fail loudly at load time
# --------------------------------------------------------------------------- #
def _base_raw():
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(os.path.dirname(here), "config", "arena.yaml"), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_mutated(tmp_path, mutate):
    raw = _base_raw()
    mutate(raw)
    path = tmp_path / "arena.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(str(path))


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nope.yaml"))


def test_bad_schema_version_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="schema_version"):
        _load_mutated(tmp_path, lambda raw: raw.update({"schema_version": 2}))


def test_tiling_gap_is_rejected(tmp_path):
    def mutate(raw):
        raw["zones"]["obstacle"]["rect"] = [2.0, 0.0, 7.4, 5.0]  # 0.1 m gap

    with pytest.raises(ConfigError, match="partition|cover"):
        _load_mutated(tmp_path, mutate)


def test_overlapping_zones_are_rejected(tmp_path):
    def mutate(raw):
        # same area as the shipped rectangle (so the area check passes) but
        # shifted east, which makes it overlap the obstacle zone and leaves a
        # gap in the west
        raw["zones"]["left_corridor"]["rect"] = [0.5, 1.5, 2.5, 2.7]

    with pytest.raises(ConfigError, match="overlap"):
        _load_mutated(tmp_path, mutate)


def test_degenerate_zone_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="degenerate"):
        _load_mutated(tmp_path, lambda raw: raw["zones"]["obstacle"].update({"rect": [2.0, 0.0, 2.0, 5.0]}))


def test_missing_required_zone_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="required zone"):
        _load_mutated(tmp_path, lambda raw: raw["zones"].pop("berm_target"))


def test_obstacles_in_start_zone_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="start"):
        _load_mutated(tmp_path, lambda raw: raw["boulders"].update({"zones": ["start", "obstacle"]}))


def test_craters_in_start_zone_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="start"):
        _load_mutated(tmp_path, lambda raw: raw["craters"].update({"zones": ["start"]}))


def test_fewer_than_three_obstacles_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="count"):
        _load_mutated(tmp_path, lambda raw: raw["boulders"].update({"count": {"min": 2, "max": 3}}))


def test_non_positive_boulder_diameter_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="diameter"):
        _load_mutated(tmp_path, lambda raw: raw["boulders"].update({"diameter": {"min": 0.0, "max": 0.40}}))


def test_non_positive_crater_depth_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="depth"):
        _load_mutated(tmp_path, lambda raw: raw["craters"].update({"depth": {"min": -0.1, "max": 0.18}}))


def test_oversized_marker_board_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="0.25"):
        _load_mutated(tmp_path, lambda raw: raw["markers"].update({"board_size": 0.30, "tag_size": 0.20}))


def test_too_tall_marker_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="0.60"):
        _load_mutated(tmp_path, lambda raw: raw["markers"].update({"center_height": 0.60}))


def test_too_many_markers_are_rejected(tmp_path):
    def mutate(raw):
        raw["markers"]["max_count"] = 5
        raw["markers"]["placements"] = [
            {"id": i, "mount": "north_wall", "position": [1.0 + 0.2 * i, 5.0]} for i in range(6)
        ]

    with pytest.raises(ConfigError, match="max_count|limit"):
        _load_mutated(tmp_path, mutate)


def test_marker_not_on_its_wall_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="north_wall"):
        _load_mutated(tmp_path, lambda raw: raw["markers"]["placements"][0].update({"position": [1.0, 4.5]}))


def test_unknown_collision_detector_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="collision_detector"):
        _load_mutated(tmp_path, lambda raw: raw["physics"].update({"collision_detector": "not-a-thing"}))


def test_berm_target_outside_construction_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="berm_target"):
        _load_mutated(tmp_path, lambda raw: raw["zones"]["berm_target"].update({"rect": [0.375, 0.30, 3.0, 1.2]}))


def test_start_zone_too_small_for_the_rover_is_rejected(tmp_path):
    def mutate(raw):
        # shrink the start zone and keep the tiling partition valid
        raw["zones"]["start"]["rect"] = [0.0, 3.0, 1.6, 5.0]
        raw["zones"]["left_corridor"]["rect"] = [0.0, 1.5, 1.6, 3.0]
        raw["zones"]["obstacle"]["rect"] = [1.6, 0.0, 7.5, 5.0]
        raw["zones"]["construction"]["rect"] = [0.0, 0.0, 1.6, 1.5]
        raw["zones"]["berm_target"]["rect"] = [0.30, 0.30, 1.30, 1.20]

    with pytest.raises(ConfigError, match="too small"):
        _load_mutated(tmp_path, mutate)


def test_bad_colour_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="colour"):
        _load_mutated(tmp_path, lambda raw: raw["arena"].update({"sand_color": [2.0, 0.0, 0.0, 1.0]}))


def test_invalid_yaml_is_rejected(tmp_path):
    path = tmp_path / "arena.yaml"
    path.write_text("schema_version: 1\nzones: [this is: not a mapping\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML|mapping"):
        load_config(str(path))


def test_config_sha256_is_stable(cfg):
    assert cfg.config_sha256() == cfg.config_sha256()
    assert len(cfg.config_sha256()) == 64


# --------------------------------------------------------------------------- #
# lighting / scene (module 3 domain randomization hooks)
# --------------------------------------------------------------------------- #
def test_shipped_light_and_scene_defaults(cfg):
    # Defaults must reproduce the values that used to be hard-coded in sdf.py.
    assert cfg.scene.ambient == pytest.approx((0.55, 0.55, 0.55, 1.0))
    assert cfg.scene.background == pytest.approx((0.72, 0.80, 0.90, 1.0))
    assert cfg.scene.shadows is True
    assert cfg.light.name == "sun"
    assert cfg.light.cast_shadows is True
    assert cfg.light.diffuse == pytest.approx((0.92, 0.90, 0.86, 1.0))
    assert cfg.light.specular == pytest.approx((0.25, 0.25, 0.25, 1.0))
    assert cfg.light.direction == pytest.approx((-0.45, 0.35, -0.82))
    assert cfg.light.attenuation == pytest.approx((1000.0, 0.9, 0.01, 0.001))


def test_light_and_scene_are_overridable(tmp_path):
    def mutate(raw):
        raw["light"].update(
            {
                "direction": [-0.9, 0.05, -0.33],
                "diffuse": [1.0, 0.55, 0.3, 1.0],
                "cast_shadows": False,
            }
        )
        raw["scene"].update({"ambient": [0.2, 0.2, 0.25, 1.0], "shadows": False})

    cfg = _load_mutated(tmp_path, mutate)
    assert cfg.light.direction == pytest.approx((-0.9, 0.05, -0.33))
    assert cfg.light.diffuse == pytest.approx((1.0, 0.55, 0.3, 1.0))
    assert cfg.light.cast_shadows is False
    assert cfg.scene.ambient == pytest.approx((0.2, 0.2, 0.25, 1.0))
    assert cfg.scene.shadows is False


def test_missing_light_section_falls_back_to_defaults(tmp_path):
    def mutate(raw):
        raw.pop("light")
        raw.pop("scene")

    cfg = _load_mutated(tmp_path, mutate)
    assert cfg.light.direction == pytest.approx((-0.45, 0.35, -0.82))
    assert cfg.scene.ambient == pytest.approx((0.55, 0.55, 0.55, 1.0))


def test_zero_light_direction_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="direction"):
        _load_mutated(tmp_path, lambda raw: raw["light"].update({"direction": [0.0, 0.0, 0.0]}))


def test_bad_light_colour_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="colour"):
        _load_mutated(tmp_path, lambda raw: raw["light"].update({"diffuse": [1.5, 0.0, 0.0, 1.0]}))

