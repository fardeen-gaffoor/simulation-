"""``navigation.yaml``: the rules the loader enforces, and the geometry it derives.

The config is module 5's single source of truth, so the tests here are in two
camps:

* **what the loader refuses** - a config that asks for a rover smaller than it
  is, a zero safety limit, an unknown work mode, a zone that is not a rectangle.
  These are the mistakes that would otherwise be found halfway across the
  obstacle zone.
* **what the loader derives** - the free rectangle, the inflation, the two radii,
  the mission phase vocabulary.  These are checked against the shipped file so a
  change to a default is visible in a diff.
"""

from __future__ import annotations

import os

import pytest
import yaml

from rover_navigation.config import (
    NAV_STATES,
    PHASE_NAMES,
    WORK_MODES,
    WORK_PHASES,
    NavigationConfigError,
    Rect,
    config_from_mapping,
    default_config_path,
    load_config,
    with_work_mode,
)

from support import config_with

# --------------------------------------------------------------------------- #
# the shipped file
# --------------------------------------------------------------------------- #
def test_the_shipped_config_loads(config):
    assert config.schema_version == 1
    assert config.path.endswith("navigation.yaml")
    assert len(config.sha256) == 64
    assert config.raw  # the parsed document is kept for introspection


def test_default_config_path_resolves_to_a_real_file():
    path = default_config_path()
    assert os.path.isfile(path)
    # ... and it is the same file the fixture loads from the source tree.
    assert os.path.basename(path) == "navigation.yaml"


def test_the_default_load_matches_an_explicit_load(config):
    reloaded = load_config(None)
    assert reloaded.arena.zones["start"].as_tuple() == config.arena.zones["start"].as_tuple()
    assert reloaded.arena.length_x == config.arena.length_x
    assert reloaded.robot == config.robot


def test_missing_file_is_reported_not_swallowed():
    with pytest.raises(NavigationConfigError) as info:
        load_config("/nonexistent/navigation.yaml")
    assert "not found" in str(info.value)


def test_broken_yaml_is_reported(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("schema_version: 1\narena: [this is not a mapping\n")
    with pytest.raises(NavigationConfigError) as info:
        load_config(str(path))
    assert "YAML" in str(info.value)


# --------------------------------------------------------------------------- #
# frames, arena, zones
# --------------------------------------------------------------------------- #
def test_frames_are_named_as_the_rest_of_the_stack_names_them(config):
    assert config.frames.map == "map"
    assert config.frames.odom == "odom"
    assert config.frames.robot == "base_link"
    assert config.frames.ground_z == 0.0


def test_arena_size_matches_the_competition(config):
    assert (config.arena.length_x, config.arena.width_y) == (9.0, 5.0)


def test_every_zone_is_a_rectangle_inside_the_arena(config):
    for name, zone in config.arena.zones.items():
        assert zone.x1 > zone.x0 and zone.y1 > zone.y0, name
        assert zone.x0 >= 0.0 and zone.y0 >= 0.0, name
        assert zone.x1 <= config.arena.length_x + 1e-9, name
        assert zone.y1 <= config.arena.width_y + 1e-9, name


def test_the_named_zones_the_mission_needs_all_exist(config):
    for name in ("start", "corridor", "construction", "obstacle", "excavation", "berm_target"):
        assert name in config.arena.zones, name
    # module 1 calls the second one `left_corridor`; module 5 renames it because
    # it plans in the arena frame, not in a diagram.
    assert "left_corridor" not in config.arena.zones


def test_the_tiling_zones_partition_the_arena(config):
    """The five tiling zones must tile 9 x 5 m exactly: no gaps, no overlaps."""
    tiling = ("start", "corridor", "construction", "obstacle", "excavation")
    total = sum(config.arena.zone(name).area for name in tiling)
    assert total == pytest.approx(config.arena.length_x * config.arena.width_y, abs=1e-9)


def test_the_berm_target_is_inside_the_construction_zone(config):
    target = config.arena.zone("berm_target")
    construction = config.arena.zone("construction")
    assert construction.contains(target.x0, target.y0)
    assert construction.contains(target.x1, target.y1)


def test_the_corridor_sits_between_construction_and_start(config):
    corridor = config.arena.zone("corridor")
    assert corridor.y0 == config.arena.zone("construction").y1
    assert corridor.y1 == config.arena.zone("start").y0


def test_an_unknown_zone_is_an_error(config):
    with pytest.raises(NavigationConfigError):
        config.arena.zone("nowhere")


def test_a_zone_that_is_not_four_numbers_is_refused(tmp_path):
    with pytest.raises(NavigationConfigError):
        config_with(_shipped(), tmp_path, {"arena.zones.start": [0, 1, 2, 3, 4]})


def test_a_zone_outside_the_arena_is_refused(tmp_path):
    with pytest.raises(NavigationConfigError):
        config_with(
            _shipped(), tmp_path, {"arena.zones.excavation": [7.5, 0.0, 9.5, 5.0]}
        )


def _shipped() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "navigation.yaml")


# --------------------------------------------------------------------------- #
# robot: the two radii and the inflation
# --------------------------------------------------------------------------- #
def test_the_operational_footprint_fits_the_competition_envelope(config):
    assert config.robot.length_m <= config.robot.build_length_m + 1e-9
    assert config.robot.width_m <= config.robot.build_width_m + 1e-9
    assert (config.robot.build_length_m, config.robot.build_width_m) == (1.50, 0.75)


def test_the_planner_radius_is_the_inscribed_radius_by_default(config):
    """A radius of 0 means "half the width": what a body on a path sweeps."""
    assert config.robot.radius_m == pytest.approx(config.robot.inscribed_radius_m)
    assert config.robot.inscribed_radius_m == pytest.approx(0.5 * config.robot.width_m)


def test_the_circumscribed_radius_is_the_half_diagonal(config):
    expected = 0.5 * (config.robot.length_m**2 + config.robot.width_m**2) ** 0.5
    assert config.robot.circumscribed_radius_m == pytest.approx(expected)
    assert config.robot.circumscribed_radius_m > config.robot.inscribed_radius_m


def test_the_inflation_is_the_radius_plus_the_clearance(config):
    assert config.robot.obstacle_inflation_m == pytest.approx(
        config.robot.radius_m + config.robot.footprint_clearance_m
    )


def test_a_radius_larger_than_the_robot_is_refused(tmp_path):
    with pytest.raises(NavigationConfigError) as info:
        config_with(_shipped(), tmp_path, {"robot.radius_m": 5.0})
    assert "circumscribed" in str(info.value)


def test_a_radius_smaller_than_the_body_is_refused(tmp_path):
    with pytest.raises(NavigationConfigError) as info:
        config_with(_shipped(), tmp_path, {"robot.radius_m": 0.01})
    assert "inscribed" in str(info.value)


def test_a_footprint_larger_than_the_build_limit_is_refused(tmp_path):
    with pytest.raises(NavigationConfigError) as info:
        config_with(_shipped(), tmp_path, {"robot.length_m": 2.0})
    assert "envelope" in str(info.value)


def test_the_wall_inset_is_the_planner_radius(config):
    assert config.robot.wall_inset_m == config.robot.radius_m


def test_speed_and_rate_limits_are_sane(config):
    assert 0.0 < config.robot.max_speed_mps <= 1.2     # the drive can do ~1.2 m/s
    assert 0.0 < config.robot.max_yaw_rate_radps <= 3.0
    assert 0.0 < config.robot.max_accel_mps2
    assert 0.0 < config.robot.max_yaw_accel_radps2


# --------------------------------------------------------------------------- #
# the free rectangle
# --------------------------------------------------------------------------- #
def test_the_free_rect_is_inset_by_the_wall_margin_and_the_radius(config):
    free = config.free_rect
    inset = config.arena.wall_margin_m + config.robot.wall_inset_m
    assert free.x0 == pytest.approx(inset)
    assert free.y0 == pytest.approx(inset)
    assert free.x1 == pytest.approx(config.arena.length_x - inset)
    assert free.y1 == pytest.approx(config.arena.width_y - inset)


def test_the_free_rect_is_never_empty(config):
    free = config.free_rect
    assert free.area > 0.0
    assert free.contains(*free.center)


def test_an_absurd_wall_margin_is_refused(tmp_path):
    with pytest.raises(NavigationConfigError):
        config_with(_shipped(), tmp_path, {"arena.wall_margin_m": 4.0})


# --------------------------------------------------------------------------- #
# costmap / planner / controller / safety values
# --------------------------------------------------------------------------- #
def test_the_crater_scale_is_at_least_the_boulder_scale(config):
    """A crater is grown at least as much as a boulder: the mistakes differ in cost."""
    assert config.costmap.crater_radius_scale >= config.costmap.boulder_radius_scale


def test_the_resolution_divides_the_arena_evenly(config):
    from rover_navigation.costmap import Costmap

    costmap = Costmap(config, [])
    assert costmap.nx * costmap.resolution == pytest.approx(config.arena.length_x, abs=0.01)
    assert costmap.ny * costmap.resolution == pytest.approx(config.arena.width_y, abs=0.01)


def test_memory_can_be_turned_off_but_defaults_to_finite(config):
    assert config.costmap.obstacle_memory_s > 0.0
    assert config.costmap.stale_obstacle_s <= config.costmap.obstacle_memory_s


def test_memory_thresholds_are_set(config):
    assert config.costmap.min_observations >= 1
    assert 0.0 <= config.costmap.min_confidence <= 1.0
    assert config.costmap.memory_associate_m > 0.0


def test_smoothing_needs_at_least_one_pass(config):
    assert config.planner.smoothing_passes >= 1
    assert config.planner.snap_radius_m > 0.0


def test_the_controller_lookahead_band_is_ordered(config):
    assert config.controller.min_lookahead_m <= config.controller.lookahead_m
    assert config.controller.lookahead_m <= config.controller.max_lookahead_m
    assert 0.0 < config.controller.in_place_turn_rad < 3.1416
    assert config.controller.slowdown_distance_m > 0.0
    assert config.controller.goal_timeout_s > 0.0
    assert config.controller.replan_period_s > 0.0


def test_safety_thresholds_are_ordered(config):
    assert config.safety.sigma_xy_slow_m < config.safety.sigma_xy_stop_m
    assert config.safety.sigma_yaw_slow_rad < config.safety.sigma_yaw_stop_rad
    assert config.safety.command_timeout_s > 0.0
    assert config.safety.gate_rate_hz >= 5.0
    assert config.safety.emergency_stop_m > config.robot.goal_tolerance_m


def test_the_gate_sigma_band_covers_the_localizers_reported_accuracy(config):
    """Module 4 reports ~0.04 m after a fix; that must be comfortably 'fine'."""
    assert config.safety.sigma_xy_slow_m >= 0.10


# --------------------------------------------------------------------------- #
# mission
# --------------------------------------------------------------------------- #
def test_the_work_modes_are_exactly_the_two_the_sequencer_understands():
    assert WORK_MODES == ("delegate", "navigate_only")


def test_the_shipped_config_delegates_work(config):
    """The competition setting: module 6 is asked, and only real cycles count."""
    assert config.mission.work_mode == "delegate"
    assert config.mission.enabled is True
    assert config.mission.excavation_action == "/excavation/work_cycle"


def test_the_phase_vocabulary_matches_the_status_message(config):
    assert len(PHASE_NAMES) == 10
    assert PHASE_NAMES[0] == "idle"
    assert PHASE_NAMES[8] == "complete"
    assert PHASE_NAMES[9] == "aborted"


def test_the_navigation_state_vocabulary_matches_the_status_message():
    assert NAV_STATES == (
        "idle",
        "planning",
        "turning",
        "driving",
        "arrived",
        "holding",
        "blocked",
        "failed",
    )


def test_the_work_phases_are_the_two_the_action_accepts():
    assert WORK_PHASES == ("excavate", "dump")


def test_an_unknown_work_mode_is_refused(tmp_path):
    with pytest.raises(NavigationConfigError) as info:
        config_with(_shipped(), tmp_path, {"mission.work_mode": "teleport"})
    assert "mission.work_mode" in str(info.value)


def test_with_work_mode_returns_a_copy_and_leaves_the_original_alone(config):
    override = with_work_mode(config, "navigate_only")
    assert override.mission.work_mode == "navigate_only"
    assert config.mission.work_mode == "delegate"
    assert override.arena is config.arena  # untouched sections are shared, not copied
    assert with_work_mode(config, "delegate") is config


def test_with_work_mode_refuses_an_unknown_mode(config):
    with pytest.raises(NavigationConfigError):
        with_work_mode(config, "nope")


def test_a_zero_cycle_run_is_refused(tmp_path):
    with pytest.raises(NavigationConfigError):
        config_with(_shipped(), tmp_path, {"mission.cycles_target": 0})


def test_the_dig_heading_is_a_quarter_turn(config):
    """The excavation strip is worked along its 5 m side, i.e. north/south."""
    import math

    assert config.mission.dig.heading_rad == pytest.approx(math.pi / 2, abs=1e-6)


def test_the_dump_heading_is_a_quarter_turn(config):
    import math

    assert config.mission.dump.heading_rad == pytest.approx(math.pi / 2, abs=1e-6)


def test_the_park_pose_is_inside_the_start_zone(config):
    park = config.mission.park
    assert config.arena.zone("start").contains(park.x, park.y)


# --------------------------------------------------------------------------- #
# topics
# --------------------------------------------------------------------------- #
def test_the_topic_names_are_the_shipped_contract(config):
    topics = config.topics
    assert topics.obstacles == "/perception/obstacles"
    assert topics.localization_state == "/perception/localization_state"
    assert topics.nav_cmd_vel == "/nav/cmd_vel"
    assert topics.cmd_vel == "/cmd_vel"
    assert topics.navigate_action == "/nav/navigate_to"
    assert topics.perception_mode == "/perception/mode"
    assert topics.estop == "/estop"
    assert topics.berm_started == "/mission/berm_started"


def test_module_5_never_writes_the_gated_topic_itself(config):
    """The navigator's raw topic and the gated one must not be the same name."""
    assert config.topics.nav_cmd_vel != config.topics.cmd_vel


def test_both_odometry_topics_are_configured_with_the_ekf_first(config):
    assert config.topics.odometry_topics[0] == "/odometry/filtered"
    assert "/odom" in config.topics.odometry_topics


def test_the_perception_mode_vocabulary_is_shared_with_module_4(config):
    from rover_navigation.mission import PERCEPTION_MODES

    assert "idle" in PERCEPTION_MODES
    assert "nav" in PERCEPTION_MODES
    assert "tag" in PERCEPTION_MODES
    assert "excavate" in PERCEPTION_MODES
    assert "dump" in PERCEPTION_MODES


# --------------------------------------------------------------------------- #
# schema-level guards
# --------------------------------------------------------------------------- #
def test_a_wrong_schema_version_is_refused():
    with pytest.raises(NavigationConfigError) as info:
        config_from_mapping({"schema_version": 99})
    assert "schema_version" in str(info.value)


def test_a_missing_section_is_named():
    with pytest.raises(NavigationConfigError) as info:
        config_from_mapping({"schema_version": 1})
    assert "missing required section" in str(info.value)


def test_a_boolean_where_a_number_belongs_is_refused(config):
    document = yaml.safe_load(open(_shipped(), encoding="utf-8"))
    document["robot"]["max_speed_mps"] = True
    with pytest.raises(NavigationConfigError):
        config_from_mapping(document)


def test_the_shipped_file_carries_no_obstacle_data():
    """The competition forbids handing the autonomy a precomputed obstacle list.

    The comment at the top of the file mentions the *word* - that is the point of
    the comment - so this looks for data: a track id, a coordinate list, anything
    that reads like a seeded obstacle rather than prose.
    """
    document = yaml.safe_load(open(_shipped(), encoding="utf-8"))
    blob = str(document).lower()
    for forbidden in ("boulder_0", "crater_0", "ground_truth", "ground truth"):
        assert forbidden not in blob, forbidden
    # No section anywhere holds a list of obstacle-like mappings.
    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, str) and path and "zone" not in path:
            assert node not in ("boulder", "crater"), f"{path} looks like obstacle data"
    walk(document)


def test_rect_geometry_helpers():
    rect = Rect(1.0, 2.0, 3.0, 5.0)
    assert rect.width == 2.0 and rect.height == 3.0
    assert rect.area == 6.0
    assert rect.center == (2.0, 3.5)
    assert rect.contains(2.0, 3.0)
    assert rect.contains(1.0, 2.0)          # the corners are inside
    assert not rect.contains(0.5, 3.0)
    assert rect.inset(0.5).as_tuple() == (1.5, 2.5, 2.5, 4.5)
    assert rect.inset(-0.5).as_tuple() == (0.5, 1.5, 3.5, 5.5)
    assert rect.inset(4.0).area == 0.0      # collapsed, never negative
    assert rect.as_tuple() == (1.0, 2.0, 3.0, 5.0)
