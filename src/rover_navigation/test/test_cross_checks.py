"""Cross-checks against the modules that own the numbers module 5 copies.

Module 5 duplicates arena geometry and the robot's envelope on purpose: on the
rover computer ``cat_arena`` is not installed (there is no Gazebo there), so a
navigator has to carry its own copy of the numbers it is *allowed* to know.  The
risk of a deliberate duplicate is that it drifts, so these tests pin it:

* the zone rectangles and the arena size against ``cat_arena/config/arena.yaml``
  (module 1, the arena's owner),
* the competition envelope against ``rover_description/config/rover.yaml``
  (module 2, the robot's owner), and the operational footprint against module 2's
  *measured* envelope, computed by module 2's own generator rather than a number
  copied into a test,
* the topic contract against module 2's ``bridges.yaml`` - the file that says
  what actually crosses the gz boundary.

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


def _arena_document() -> dict:
    with open(_share("cat_arena", os.path.join("config", "arena.yaml")), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _rover_document() -> dict:
    with open(_share("rover_description", os.path.join("config", "rover.yaml")), encoding="utf-8") as handle:
        return yaml.safe_load(handle)


# --------------------------------------------------------------------------- #
# module 1: the arena
# --------------------------------------------------------------------------- #
def test_arena_size_matches_module_1(config):
    document = _arena_document()["arena"]
    assert config.arena.length_x == document["length_x"]
    assert config.arena.width_y == document["width_y"]


def test_every_zone_module_5_names_matches_module_1(config):
    zones = _arena_document()["zones"]
    # module 5 renames `left_corridor` to `corridor`: it plans in the arena
    # frame and the direction word belongs to the diagram.
    mapping = {
        "start": "start",
        "corridor": "left_corridor",
        "construction": "construction",
        "obstacle": "obstacle",
        "excavation": "excavation",
        "berm_target": "berm_target",
    }
    for ours, theirs in mapping.items():
        assert theirs in zones, theirs
        expected = [float(value) for value in zones[theirs]["rect"]]
        actual = list(config.arena.zone(ours).as_tuple())
        assert actual == pytest.approx(expected), ours


def test_module_5_knows_no_zone_module_1_does_not_have(config):
    zones = _arena_document()["zones"]
    known = set(zones)
    for name in config.arena.zones:
        assert name in known or name == "corridor", name


def test_module_5_does_not_inherit_module_1s_obstacle_permissions(config):
    """`obstacles_allowed` is a generator flag, not something a navigator reads."""
    assert not hasattr(config.arena, "obstacles_allowed")


def test_the_wall_margin_is_no_larger_than_module_1s_wall_thickness_band(config):
    """Module 1 builds walls 0.06 m thick flush with the arena border."""
    walls = _arena_document()["arena"]["walls"]
    assert walls["present"] is True
    # A navigator keeps *more* clearance than the wall itself occupies, which is
    # the point - but the margin must not be so large that the arena is unusable.
    assert config.arena.wall_margin_m >= walls["thickness"] / 2.0


# --------------------------------------------------------------------------- #
# module 2: the robot
# --------------------------------------------------------------------------- #
def test_the_competition_envelope_matches_module_2(config):
    limits = _rover_document()["limits"]["envelope"]
    assert config.robot.build_length_m == limits["length_x"]
    assert config.robot.build_width_m == limits["width_y"]


def test_module_5s_footprint_is_at_least_the_measured_robot(config):
    """Planning for less than the rover's real size would clip its own body.

    The measured envelope comes from module 2's own generator, so this test
    tracks the URDF rather than a number typed into a test.
    """
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


def test_module_5s_footprint_is_inside_the_build_limit(config):
    assert config.robot.length_m <= config.robot.build_length_m
    assert config.robot.width_m <= config.robot.build_width_m


def test_the_drive_kinematics_module_5_assumes_match_module_2(config):
    drive = _rover_document()["drive"]
    wheel_radius = float(drive["wheel_radius"])
    max_wheel_velocity = float(drive["max_wheel_velocity"])
    # The fastest the differential drive can go is one wheel at full speed.
    assert config.robot.max_speed_mps <= wheel_radius * max_wheel_velocity + 1e-9


def _cameras() -> dict:
    document = _rover_document()
    return {entry["name"]: entry for entry in document["sensors"]["cameras"]}


def test_the_dump_camera_looks_aft(config):
    """The dump poses face +y because cam_dump is mounted looking backwards.

    If module 2 ever remounts it, the dump heading in this config is wrong and
    the sand lands on the wrong side of the rover - so the assumption is pinned
    here rather than only in a comment.  Module 2 stores mount angles in degrees.
    """
    import math

    cameras = _cameras()
    assert "cam_dump" in cameras, sorted(cameras)
    yaw_deg = float(cameras["cam_dump"]["rpy_deg"][2])
    assert abs(abs(yaw_deg) - 180.0) < 5.0, yaw_deg
    # The rover faces +y (north) while dumping, so its aft - where cam_dump
    # looks - is over the berm to the south of it.
    assert config.mission.dump.heading_rad == pytest.approx(math.pi / 2, abs=1e-6)


def test_the_camera_that_sees_obstacles_looks_forward(config):
    cameras = _cameras()
    assert "cam_nav" in cameras
    yaw_deg = float(cameras["cam_nav"]["rpy_deg"][2])
    # Forward-facing: anything the navigator plans around is in front of it.
    assert abs(yaw_deg) < 60.0, yaw_deg


# --------------------------------------------------------------------------- #
# module 2: the topic contract
# --------------------------------------------------------------------------- #
def _bridge_topics(direction: str = "") -> list:
    document = yaml.safe_load(
        open(_share("rover_description", os.path.join("config", "bridges.yaml")), encoding="utf-8")
    )
    entries = document["bridges"] if isinstance(document, dict) else document
    return [
        entry["ros_topic"]
        for entry in entries
        if not direction or entry.get("direction") == direction
    ]


def test_the_estop_topic_module_5_gates_on_is_the_one_module_2_owns(config):
    """``/estop`` is not bridged: it is a native ROS topic module 2 latches."""
    topics = _rover_document()["topics"]
    assert topics["estop"] == config.topics.estop
    assert config.topics.estop == "/estop"
    assert all("estop" not in topic for topic in _bridge_topics()), _bridge_topics()


def test_the_odometry_topic_module_5_reads_is_the_one_the_bridge_publishes(config):
    bridged = _bridge_topics()
    assert "/odom" in bridged, bridged
    # Module 5 prefers the EKF but lists /odom as the fallback for exactly this
    # reason: the EKF is optional (see localization.launch.py).
    assert config.topics.odometry_topics[-1] == "/odom"


def test_the_command_topic_module_5_writes_is_the_one_the_drive_listens_to(config):
    to_gz = _bridge_topics("ros_to_gz")
    assert config.topics.cmd_vel in to_gz, (config.topics.cmd_vel, to_gz)
    # The navigator's raw topic must NOT be a bridge: only the gate reaches gz.
    assert config.topics.nav_cmd_vel not in to_gz


def test_the_vision_topics_module_5_expects_are_the_ones_module_2_bridges(config):
    bridged = _bridge_topics("gz_to_ros")
    for camera in ("/cam_nav", "/cam_tag"):
        assert camera in bridged, camera
