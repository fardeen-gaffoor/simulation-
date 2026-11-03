"""Cross-checks against the modules that own the numbers module 6 copies.

Module 6 duplicates three sets of numbers on purpose, because on the rover
computer ``cat_arena`` and ``rover_description`` are not installed and the
excavation node still has to know where the berm target is, how big the machine
is and which topics to talk on:

* the arena size, the berm target rectangle and the judged berm height against
  ``cat_arena/config/arena.yaml`` (module 1, the arena's owner),
* the mechanism's dimensions, joint names, limits and command topics against
  ``rover_description/config/rover.yaml`` (module 2, the robot's owner),
* the dump geometry module 5 uses, against
  ``rover_navigation/config/navigation.yaml`` (module 5, the mission's owner) -
  this is the one that makes "a dump lands inside the target" a testable
  statement rather than a hope.

Where a source file is not reachable the test **skips** rather than passing
quietly: a cross-check that cannot run is not evidence.
"""

from __future__ import annotations

import os

import pytest
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))


def _share(name: str, relative: str) -> str:
    """A file from another package: install space first, then the source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory

        path = os.path.join(get_package_share_directory(name), relative)
        if os.path.isfile(path):
            return path
    except Exception:  # noqa: BLE001 - not installed is a normal case in pytest
        pass
    path = os.path.join(_REPO, "src", name, relative)
    if os.path.isfile(path):
        return path
    pytest.skip(f"{name}/{relative} is not reachable from this test run")
    return ""  # pragma: no cover - pytest.skip raises


def _document(name: str, relative: str) -> dict:
    with open(_share(name, relative), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _arena() -> dict:
    return _document("cat_arena", os.path.join("config", "arena.yaml"))


def _rover() -> dict:
    return _document("rover_description", os.path.join("config", "rover.yaml"))


def _navigation() -> dict:
    return _document("rover_navigation", os.path.join("config", "navigation.yaml"))


# --------------------------------------------------------------------------- #
# module 1: the arena
# --------------------------------------------------------------------------- #
def test_arena_size_matches_module_1(config):
    arena = _arena()["arena"]
    assert config.arena.length_x == arena["length_x"]
    assert config.arena.width_y == arena["width_y"]


def test_sand_depth_matches_module_1(config):
    arena = _arena()["arena"]
    document = _arena()
    depth = arena.get("sand_depth", document.get("terrain", {}).get("sand_depth"))
    if depth is None:
        # Module 1 keeps it under terrain; accept either spelling.
        depth = document.get("sand", {}).get("depth")
    assert depth is not None, "module 1 does not declare a sand depth"
    assert config.arena.sand_depth == pytest.approx(float(depth))


def test_berm_target_matches_module_1(config):
    zones = _arena()["zones"]
    expected = [float(value) for value in zones["berm_target"]["rect"]]
    assert list(config.berm_target.as_tuple()) == pytest.approx(expected)


def test_construction_zone_matches_module_1(config):
    zones = _arena()["zones"]
    expected = [float(value) for value in zones["construction"]["rect"]]
    assert list(config.arena.zone("construction").as_tuple()) == pytest.approx(expected)


def test_berm_height_matches_module_1(config):
    document = _arena()
    height = document.get("marking", {}).get("berm_height")
    if height is None:
        height = document.get("arena", {}).get("berm_height")
    assert height is not None, "module 1 does not declare a berm height"
    assert config.sand.berm.target_height_m == pytest.approx(float(height))


def test_module_6_knows_no_zone_it_should_not(config):
    """Module 6 needs the berm target and the construction zone, nothing else."""
    assert set(config.arena.zones) == {"construction", "berm_target"}


def test_module_6_does_not_read_ground_truth():
    """The rule that module 6 may not look at the answer key, enforced in code.

    Module 1's own test suite scans the autonomy packages for ``ground_truth``;
    this is the same check from module 6's side, so it fails here too if someone
    ever reaches for the seeded obstacle layout.
    """
    import rover_excavation

    root = os.path.dirname(os.path.abspath(rover_excavation.__file__))
    offenders = []
    for directory, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                text = handle.read()
            if "ground_truth" in text:
                offenders.append(os.path.join(directory, name))
    assert offenders == []


# --------------------------------------------------------------------------- #
# module 2: the robot
# --------------------------------------------------------------------------- #
def test_the_competition_envelope_matches_module_2(config):
    limits = _rover()["limits"]["envelope"]
    assert config.robot.build_length_m == limits["length_x"]
    assert config.robot.build_width_m == limits["width_y"]


def test_the_operational_footprint_covers_the_measured_robot(config):
    """Planning or dumping for less than the real body would clip the rover."""
    try:
        from rover_description.config import load_config as load_rover_config
        from rover_description.envelope import check_envelope
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"rover_description is not importable: {exc}")

    report = check_envelope(load_rover_config(None))
    assert report.ok, report.table()
    measured = report.measured
    assert config.robot.length_m >= measured.length_x - 1e-9, report.table()
    assert config.robot.width_m >= measured.width_y - 1e-9, report.table()


def test_ladder_geometry_matches_module_2(config):
    ladder = _rover()["excavation"]["ladder"]
    ours = config.mechanism.ladder
    assert ours.length_m == pytest.approx(ladder["length"])
    assert ours.width_m == pytest.approx(ladder["width"])
    assert ours.frame_thickness_m == pytest.approx(ladder["frame_thickness"])
    assert ours.pivot_x_m == pytest.approx(ladder["pivot"][0])
    assert ours.pivot_z_m == pytest.approx(ladder["pivot"][2])
    # Above the sand, not above base_link: the body rides one wheel radius up.
    drive = _rover()["drive"]
    assert ours.ride_height_m == pytest.approx(drive["wheel_radius"])
    assert ours.pivot_z_above_sand_m == pytest.approx(ladder["pivot"][2] + drive["wheel_radius"])


def test_ladder_limits_match_module_2(config):
    import math

    ladder = _rover()["excavation"]["ladder"]
    ours = config.mechanism.ladder
    assert ours.stowed_pitch_rad == pytest.approx(math.radians(ladder["stowed_pitch_deg"]))
    assert ours.min_pitch_rad == pytest.approx(math.radians(ladder["min_pitch_deg"]))
    assert ours.max_pitch_rad == pytest.approx(math.radians(ladder["max_pitch_deg"]))
    # The working cut is inside module 2's travel limits, not beyond them.
    assert ours.min_pitch_rad <= ours.dig_pitch_rad <= ours.max_pitch_rad


def test_belt_matches_module_2(config):
    belt = _rover()["excavation"]["belt"]
    ours = config.mechanism.belt
    assert ours.roller_radius_m == pytest.approx(belt["roller_radius"])
    assert ours.max_velocity_radps <= belt["max_velocity"]
    assert abs(ours.dig_velocity_radps) <= belt["max_velocity"]
    assert abs(ours.dump_velocity_radps) <= belt["max_velocity"]


def test_tub_matches_module_2(config):
    tub = _rover()["excavation"]["tub"]
    size = list(tub["size"]) if "size" in tub else list(tub["box"])
    assert list(config.mechanism.tub.box_m) == pytest.approx(size)
    # The usable capacity cannot exceed the box it sits in.
    assert config.mechanism.tub.capacity_m3 <= config.mechanism.tub.box_volume_m3


def test_joint_names_match_module_2(config):
    """The URDF joints module 6 commands must be the ones module 2 actually builds."""
    try:
        from rover_description.config import load_config as load_rover_config
        from rover_description.generate import expected_joints
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"rover_description is not importable: {exc}")

    joints = expected_joints(load_rover_config(None))
    for joint in (config.mechanism.ladder.joint, config.mechanism.belt.joint):
        assert joint in joints, joint


def test_command_topics_match_module_2(config):
    topics = _rover()["topics"]
    assert config.backend.ladder_cmd_topic == topics["ladder_cmd"]
    assert config.backend.belt_cmd_topic == topics["belt_cmd"]
    assert config.backend.joint_states_topic == topics["joint_states"]


def test_command_topics_are_bridged_by_module_2(config):
    """The sim bridge table has to carry exactly the topics module 6 publishes."""
    bridges = _document("rover_description", os.path.join("config", "bridges.yaml"))["bridges"]
    by_topic = {entry["ros_topic"]: entry for entry in bridges}
    for topic in (config.backend.ladder_cmd_topic, config.backend.belt_cmd_topic):
        assert topic in by_topic, topic
        entry = by_topic[topic]
        assert entry["direction"] == "ros_to_gz"
        assert entry["ros_type"] == "std_msgs/msg/Float64"
        assert entry["gz_type"] == "gz.msgs.Double"
        assert entry["gz_topic"] == topic


# --------------------------------------------------------------------------- #
# module 5: the dump geometry
# --------------------------------------------------------------------------- #
def test_dump_geometry_matches_module_5(config):
    """The dump block is what decides whether a pile lands in the berm target."""
    mission = _navigation()["mission"]
    dump = mission["dump"]
    ours = config.nav_dump
    assert ours.heading_rad == pytest.approx(dump["heading_rad"])
    assert ours.standoff_m == pytest.approx(dump["standoff_m"])
    assert ours.points == int(dump["points"])
    assert ours.inset_m == pytest.approx(dump["inset_m"])


def test_module_5_uses_the_same_footprint(config):
    import math

    navigation = _navigation()
    robot = navigation["robot"]
    assert config.robot.length_m == pytest.approx(robot["length_m"])
    assert config.robot.width_m == pytest.approx(robot["width_m"])
    # Module 5 derives the circumscribed radius from the footprint; module 6
    # copies the derived number, so compare against *module 5's* derivation.
    assert config.robot.circumscribed_radius_m == pytest.approx(
        0.5 * math.hypot(robot["length_m"], robot["width_m"]), abs=1e-3
    )


def test_module_5s_dump_pose_is_where_module_6_thinks_it_is(config):
    """Recompute module 5's dump y from *module 5's* numbers and compare.

    ``excavation.yaml`` copies the derived circumscribed radius rounded to four
    decimals (0.7382), so the two agree to well under a millimetre rather than
    exactly - four orders of magnitude below anything that matters to a pile of
    sand, and the softened tolerance is what stops this test from being a
    tautology about a copied constant.
    """
    import math

    navigation = _navigation()
    robot = navigation["robot"]
    target = [float(value) for value in _arena()["zones"]["berm_target"]["rect"]]
    expected = (
        target[3]
        + navigation["mission"]["dump"]["standoff_m"]
        + 0.5 * math.hypot(robot["length_m"], robot["width_m"])
    )
    assert config.berm_dump_pose_y == pytest.approx(expected, abs=1e-3)


def test_a_dump_from_every_module_5_pose_lands_inside_the_target(config):
    """The coupling that matters: six cycles have to build the berm, not spill it.

    If this fails, module 5's dump stand-off and module 6's pile model have
    drifted apart, and the fix is to re-tune ``sand.dump`` (see docs/sand.md),
    *not* to relax this test.
    """
    yaw = config.nav_dump.heading_rad
    y = config.berm_dump_pose_y
    target = config.berm_target
    from support import dump_pose_xs

    for x in dump_pose_xs(config):
        lx, ly = config.landing_point(x, y, yaw)
        assert target.contains(lx, ly), (x, lx, ly)


def test_the_work_cycle_action_still_has_the_fields_module_6_serves():
    """Module 6 is the *server*; the action lives in module 5's message package."""
    from rover_navigation_msgs.action import WorkCycle

    goal = WorkCycle.Goal()
    assert hasattr(goal, "phase")
    assert hasattr(goal, "work_pose")
    assert hasattr(goal, "timeout_s")
    assert hasattr(goal, "target_amount")
    result = WorkCycle.Result()
    assert hasattr(result, "success")
    assert hasattr(result, "message")
    assert hasattr(result, "amount_moved")
    assert hasattr(result, "elapsed_s")
    feedback = WorkCycle.Feedback()
    assert hasattr(feedback, "stage")
    assert hasattr(feedback, "progress")
    assert hasattr(feedback, "detail")
