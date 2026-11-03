"""``excavation.yaml``: defaults, derived geometry, and every way it can be wrong.

The loader is the only place a bad plan can be caught before a run, so the
rejection cases matter as much as the happy path: each one is a number that
would otherwise make the mechanism do something silly at the dig face.
"""

from __future__ import annotations

import math
import os

import pytest

from rover_excavation.config import (
    BACKEND_KINDS,
    MECHANISM_STATES,
    STAGE_NAMES,
    WORK_PHASES,
    ExcavationConfigError,
    Rect,
    config_from_mapping,
    default_config_path,
    dig_rate_m3_s,
    load_config,
    package_share_dir,
    repose_basal_radius_m,
)

from support import config_with


# --------------------------------------------------------------------------- #
# the shipped file
# --------------------------------------------------------------------------- #
def test_loads(config):
    assert config.schema_version == 1
    assert config.path.endswith("excavation.yaml")
    assert len(config.sha256) == 64


def test_default_path_exists():
    assert default_config_path().endswith("excavation.yaml")
    assert os.path.isfile(default_config_path())


def test_the_share_dir_would_survive_an_installed_run():
    """The node resolves the config with no ``--config``; that has to work.

    ``__file__`` walks up into ``local/lib/.../dist-packages`` after a
    ``--symlink-install``, which has no ``config`` directory, so the share
    directory has to come from the ament index.
    """
    share = package_share_dir()
    assert os.path.isdir(share)
    assert os.path.isfile(os.path.join(share, "config", "excavation.yaml"))


def test_frames(config):
    assert config.frames.map == "map"
    assert config.frames.base == "base_link"
    assert config.frames.map != config.frames.base


def test_arena_copy(config):
    assert config.arena.length_x == 9.0
    assert config.arena.width_y == 5.0
    assert config.arena.sand_depth == 0.30
    assert config.berm_target.as_tuple() == (0.375, 0.30, 1.625, 1.20)
    assert config.berm_target.area == pytest.approx(1.125)
    assert config.arena.zone("construction").contains(*config.berm_target.center)


def test_berm_target_volume(config):
    assert config.berm_target_volume_m3 == pytest.approx(1.125 * 0.15)


def test_robot_fits_the_envelope(config):
    assert config.robot.length_m <= config.robot.build_length_m
    assert config.robot.width_m <= config.robot.build_width_m
    assert config.robot.half_length_m == pytest.approx(0.65)
    assert config.robot.circumscribed_radius_m == pytest.approx(0.5 * math.hypot(1.30, 0.70), abs=1e-3)


def test_mechanism_limits(config):
    ladder = config.mechanism.ladder
    assert ladder.min_pitch_rad < ladder.stowed_pitch_rad < ladder.dig_pitch_rad < ladder.max_pitch_rad
    assert ladder.engaged_width_m == pytest.approx(0.40 - 2 * 0.05)
    assert config.mechanism.belt.dig_velocity_radps > 0.0
    assert config.mechanism.belt.dump_velocity_radps < 0.0
    assert config.mechanism.rate_hz > 0.0
    assert config.mechanism.period_s == pytest.approx(1.0 / config.mechanism.rate_hz)


def test_ladder_geometry_is_physical(config):
    ladder = config.mechanism.ladder
    # The pivot is measured above the sand, not above base_link: the drive wheels
    # hold the body one wheel radius up.
    assert ladder.pivot_z_above_sand_m == pytest.approx(ladder.pivot_z_m + ladder.ride_height_m)
    assert ladder.ride_height_m > 0.0
    # The shortest pitch that reaches the sand follows from pivot height/length.
    assert ladder.min_dig_pitch_rad == pytest.approx(
        math.asin(ladder.pivot_z_above_sand_m / ladder.length_m)
    )
    # Stowed, the tip is in the air; at the dig pitch it is under the surface.
    assert ladder.tip_depth_m(ladder.stowed_pitch_rad) < 0.0
    assert ladder.tip_depth_m(ladder.dig_pitch_rad) > 0.0
    # ... and never deeper than the interlock allows.
    assert ladder.tip_depth_m(ladder.dig_pitch_rad) <= config.mechanism.interlocks.max_depth_m


def test_tub_capacity_matches_the_box(config):
    tub = config.mechanism.tub
    assert tub.capacity_m3 == pytest.approx(tub.fill_fraction * tub.box_volume_m3, abs=2e-4)
    assert 0.0 < tub.fill_fraction <= 1.0


def test_travel_attitude_is_travel_ready(config):
    interlocks = config.mechanism.interlocks
    assert config.mechanism.ladder.stowed_pitch_rad <= interlocks.travel_pitch_max_rad
    assert interlocks.max_depth_m <= config.arena.sand_depth


def test_cycle_ordering(config):
    assert config.cycle.dig.min_s < config.cycle.dig.timeout_s
    assert config.cycle.dump.min_s < config.cycle.dump.timeout_s
    assert config.cycle.settle_s >= 0.0


def test_backend_is_sim_by_default(config):
    assert config.backend.kind == "sim"
    assert config.backend.ladder_cmd_topic.startswith("/")
    assert config.backend.belt_cmd_topic.startswith("/")


def test_topics(config):
    assert config.topics.work_action == "/excavation/work_cycle"
    assert config.topics.estop == "/estop"
    assert config.topics.perception_mode == "/perception/mode"


def test_constants_agree_with_the_messages():
    from rover_excavation_msgs.msg import BermStatus, MechanismStatus

    assert MECHANISM_STATES == (
        "stowed",
        "deploying",
        "digging",
        "dumping",
        "retracting",
        "fault",
    )
    assert MechanismStatus.STATE_STOWED == 0
    assert MechanismStatus.STATE_FAULT == 5
    assert len(MECHANISM_STATES) == 6
    from rover_navigation_msgs.action import WorkCycle

    assert STAGE_NAMES == ("STARTING", "MOVING", "WORKING", "RETRACTING", "DONE")
    assert WorkCycle.Feedback.STAGE_STARTING == 0
    assert WorkCycle.Feedback.STAGE_DONE == 4
    assert WORK_PHASES == ("excavate", "dump")
    assert isinstance(BermStatus(), BermStatus)


# --------------------------------------------------------------------------- #
# derived sand geometry
# --------------------------------------------------------------------------- #
def test_basal_radius_formula():
    # A cone of volume V at repose theta has r = (3V/(pi tan theta))^(1/3).
    volume = 0.0346
    theta = math.radians(34.0)
    expected = (3.0 * volume / (math.pi * math.tan(theta))) ** (1.0 / 3.0)
    assert repose_basal_radius_m(volume, theta) == pytest.approx(expected)
    assert repose_basal_radius_m(0.0, theta) == 0.0


def test_landing_is_behind_the_aft_face(config):
    landing = config.landing_aft_m
    assert landing > config.robot.half_length_m
    assert landing == pytest.approx(config.robot.half_length_m + config.pile_basal_radius_m, abs=1e-9)


def test_landing_point_is_aft_of_the_pose(config):
    x, y, yaw = 1.0, config.berm_dump_pose_y, config.nav_dump.heading_rad
    lx, ly = config.landing_point(x, y, yaw)
    # Facing +y, "behind" is -y, and the landing point stays on the centreline.
    assert lx == pytest.approx(x, abs=1e-6)
    assert ly < y
    assert config.berm_target.contains(lx, ly)


def test_dig_rate_matches_the_mechanism(config):
    rate = dig_rate_m3_s(
        config.sand.dig, config.mechanism, config.mechanism.ladder.dig_pitch_rad, config.mechanism.belt.dig_velocity_radps
    )
    assert config.sand.dig.min_rate_m3_s <= rate <= config.sand.dig.max_rate_m3_s
    # Clamped, so it can never exceed the ceiling no matter the pitch.
    huge = dig_rate_m3_s(
        config.sand.dig, config.mechanism, config.mechanism.ladder.max_pitch_rad, config.mechanism.belt.max_velocity_radps
    )
    assert huge <= config.sand.dig.max_rate_m3_s
    # No bite, no sand: stowed, or a reversing belt, both give zero.
    assert dig_rate_m3_s(config.sand.dig, config.mechanism, config.mechanism.ladder.stowed_pitch_rad, 5.0) == 0.0
    assert dig_rate_m3_s(config.sand.dig, config.mechanism, config.mechanism.ladder.dig_pitch_rad, -5.0) == 0.0


def test_dig_rate_is_zero_at_the_reach_limit(config):
    assert dig_rate_m3_s(config.sand.dig, config.mechanism, 0.0, 5.0) == 0.0


# --------------------------------------------------------------------------- #
# rejections
# --------------------------------------------------------------------------- #
def _mapping(config):
    import yaml

    with open(config.path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_rejects_non_mapping():
    with pytest.raises(ExcavationConfigError):
        config_from_mapping(["not", "a", "mapping"])


def test_rejects_bad_schema(config):
    document = _mapping(config)
    document["schema_version"] = 2
    with pytest.raises(ExcavationConfigError, match="schema_version"):
        config_from_mapping(document)


def test_rejects_missing_section(config):
    document = _mapping(config)
    del document["mechanism"]
    with pytest.raises(ExcavationConfigError, match="mechanism"):
        config_from_mapping(document)


def test_rejects_missing_zone(config):
    document = _mapping(config)
    del document["arena"]["zones"]["berm_target"]
    with pytest.raises(ExcavationConfigError, match="berm_target"):
        config_from_mapping(document)


def test_rejects_zone_outside_arena(config):
    document = _mapping(config)
    document["arena"]["zones"]["berm_target"]["rect"] = [0.375, 0.30, 20.0, 1.20]
    with pytest.raises(ExcavationConfigError, match="leaves the arena"):
        config_from_mapping(document)


def test_rejects_degenerate_rect(config):
    document = _mapping(config)
    document["arena"]["zones"]["berm_target"]["rect"] = [1.0, 0.30, 1.0, 1.20]
    with pytest.raises(ExcavationConfigError, match="degenerate"):
        config_from_mapping(document)


def test_rejects_oversized_robot(config):
    document = _mapping(config)
    document["robot"]["length_m"] = 2.0
    with pytest.raises(ExcavationConfigError, match="envelope"):
        config_from_mapping(document)


def test_rejects_wrong_circumscribed_radius(config):
    document = _mapping(config)
    document["robot"]["circumscribed_radius_m"] = 1.2
    with pytest.raises(ExcavationConfigError, match="circumscribed"):
        config_from_mapping(document)


def test_rejects_dig_pitch_that_cannot_reach(config):
    document = _mapping(config)
    document["mechanism"]["ladder"]["dig_pitch_rad"] = 0.1
    with pytest.raises(ExcavationConfigError, match="steeper"):
        config_from_mapping(document)


def test_rejects_stowed_outside_limits(config):
    document = _mapping(config)
    document["mechanism"]["ladder"]["stowed_pitch_rad"] = 2.0
    with pytest.raises(ExcavationConfigError, match="stowed_pitch_rad"):
        config_from_mapping(document)


def test_rejects_stowed_not_travel_ready(config):
    document = _mapping(config)
    document["mechanism"]["interlocks"]["travel_pitch_max_rad"] = -0.60
    with pytest.raises(ExcavationConfigError, match="travel-ready"):
        config_from_mapping(document)


def test_rejects_depth_past_the_floor(config):
    document = _mapping(config)
    document["mechanism"]["interlocks"]["max_depth_m"] = 0.40
    with pytest.raises(ExcavationConfigError, match="deeper"):
        config_from_mapping(document)


def test_rejects_belt_over_max(config):
    document = _mapping(config)
    document["mechanism"]["belt"]["dig_velocity_radps"] = 9.0
    with pytest.raises(ExcavationConfigError, match="max_velocity_radps"):
        config_from_mapping(document)


def test_positive_dump_velocity_is_taken_as_reverse(config):
    # The sign is not a tuning knob: reverse is what empties the tub, so a
    # positive number in the file is normalised rather than obeyed.
    document = _mapping(config)
    document["mechanism"]["belt"]["dump_velocity_radps"] = 3.0
    loaded = config_from_mapping(document)
    assert loaded.mechanism.belt.dump_velocity_radps == -3.0


def test_rejects_tub_capacity_not_matching_the_box(config):
    document = _mapping(config)
    document["mechanism"]["tub"]["capacity_m3"] = 0.20
    with pytest.raises(ExcavationConfigError, match="capacity_m3"):
        config_from_mapping(document)


def test_rejects_spread_far_from_the_pile(config):
    document = _mapping(config)
    document["sand"]["dump"]["spread_m"] = 2.0
    with pytest.raises(ExcavationConfigError, match="spread_m"):
        config_from_mapping(document)


def test_rejects_dead_dig_model(config):
    document = _mapping(config)
    document["sand"]["dig"]["coefficient"] = 0.0
    with pytest.raises(ExcavationConfigError, match="dig rate is zero|must be > 0"):
        config_from_mapping(document)


def test_rejects_coarse_berm_cells(config):
    document = _mapping(config)
    document["sand"]["berm"]["cell_m"] = 0.30
    with pytest.raises(ExcavationConfigError, match="cell_m"):
        config_from_mapping(document)


def test_rejects_cycle_timeout_before_min(config):
    document = _mapping(config)
    document["cycle"]["dig"]["timeout_s"] = 1.0
    with pytest.raises(ExcavationConfigError, match="timeout_s"):
        config_from_mapping(document)


def test_rejects_unknown_backend(config):
    document = _mapping(config)
    document["backend"]["kind"] = "telepathy"
    with pytest.raises(ExcavationConfigError, match="backend.kind"):
        config_from_mapping(document)
    assert set(BACKEND_KINDS) == {"sim", "hardware", "null"}


def test_rejects_relative_topic(config):
    document = _mapping(config)
    document["topics"]["status"] = "excavation/status"
    with pytest.raises(ExcavationConfigError, match="absolute"):
        config_from_mapping(document)


def test_rejects_same_frames(config):
    document = _mapping(config)
    document["frames"]["base"] = "map"
    with pytest.raises(ExcavationConfigError, match="different"):
        config_from_mapping(document)


# --------------------------------------------------------------------------- #
# overriding
# --------------------------------------------------------------------------- #
def test_config_with_applies_overrides(config_file, tmp_path):
    patched = config_with(config_file, tmp_path, {"cycle.dig.min_s": 0.1})
    assert patched.cycle.dig.min_s == 0.1
    # ... and the shipped file is untouched.
    assert load_config(config_file).cycle.dig.min_s != 0.1


def test_rect_helpers():
    rect = Rect(1.0, 2.0, 3.0, 4.0)
    assert rect.width == 2.0 and rect.height == 2.0
    assert rect.center == (2.0, 3.0)
    assert rect.area == 4.0
    assert rect.contains(2.0, 3.0)
    assert not rect.contains(0.0, 0.0)
