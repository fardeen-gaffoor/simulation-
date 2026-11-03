"""Tests for the generated URDF: structure, sensors, joints and plugins."""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest

from rover_description.config import CAMERA_TOPICS
from rover_description.generate import expected_joints, expected_links
from rover_description.geometry import link_inertias


@pytest.fixture(scope="module")
def urdf_root(generated):
    with open(generated.urdf_path, "r", encoding="utf-8") as handle:
        return ET.fromstring(handle.read())


def test_robot_root(urdf_root):
    assert urdf_root.tag == "robot"
    assert urdf_root.get("name") == "rover"


def test_expected_links_and_joints(urdf_root, cfg):
    links = {e.get("name") for e in urdf_root.findall("link")}
    joints = {e.get("name") for e in urdf_root.findall("joint")}
    assert set(expected_links(cfg)) <= links
    assert set(expected_joints(cfg)) <= joints


def test_exactly_four_cameras_with_topics(urdf_root):
    topics = []
    for sensor in urdf_root.iter("sensor"):
        if (sensor.get("type") or "") == "camera":
            topics.append(sensor.find("topic").text)
    assert sorted(topics) == sorted(CAMERA_TOPICS)


def test_camera_info_and_optical_frames(urdf_root, cfg):
    by_name = {s.get("name"): s for s in urdf_root.iter("sensor") if s.get("type") == "camera"}
    for cam in cfg.sensors.cameras:
        sensor = by_name[cam.name]
        info = sensor.find("camera/camera_info_topic").text
        optical = sensor.find("camera/optical_frame_id").text
        assert info == cam.camera_info_topic
        assert optical == cam.optical_frame


def test_no_forbidden_sensors(urdf_root):
    allowed = {"camera", "imu"}
    for sensor in urdf_root.iter("sensor"):
        assert (sensor.get("type") or "") in allowed


def test_imu_present_and_has_no_magnetometer(urdf_root):
    sensors = {s.get("name"): s for s in urdf_root.iter("sensor") if s.get("type") == "imu"}
    assert "imu" in sensors
    assert sensors["imu"].find("topic").text == "/imu"
    # no magnetometer element anywhere under the imu sensor
    assert not any("magnetometer" in (child.tag or "") for child in sensors["imu"].iter())


def test_gz_plugins(urdf_root):
    plugins = [f"{e.get('filename')} {e.get('name')}" for e in urdf_root.iter("plugin")]
    assert any("diff-drive" in p for p in plugins)
    assert any("joint-state-publisher" in p for p in plugins)


def test_excavation_joint_controllers(urdf_root):
    """The two gz joint controllers module 6 commands through the bridge."""
    plugins = {e.get("name"): e for e in urdf_root.iter("plugin")}
    ladder = plugins.get("gz::sim::systems::JointPositionController")
    belt = plugins.get("gz::sim::systems::JointController")
    assert ladder is not None, "the ladder needs a position controller to hold an angle"
    assert belt is not None
    assert ladder.find("joint_name").text == "ladder_pitch_joint"
    assert belt.find("joint_name").text == "belt_joint"
    assert ladder.find("topic").text == "/excavation/ladder_cmd"
    assert belt.find("topic").text == "/excavation/belt_cmd"
    # The belt is a roller: velocity mode, not position.
    assert belt.find("use_force_commands").text == "false"


def test_excavation_joint_controllers_use_a_global_topic(urdf_root):
    """``<sub_topic>`` is a trap: gz scopes it to ``/model/<model_name>/...``.

    A topic built that way can never be reached by the ``ros_gz_bridge`` entry
    in ``config/bridges.yaml``, and the failure is silent - the joint simply
    never moves.  ``<topic>`` is the global form, the same one DiffDrive uses for
    ``/cmd_vel``, so this pins the choice that was verified live.
    """
    for name in (
        "gz::sim::systems::JointPositionController",
        "gz::sim::systems::JointController",
    ):
        plugin = next(e for e in urdf_root.iter("plugin") if e.get("name") == name)
        assert plugin.find("sub_topic") is None, f"{name} must not use <sub_topic>"
        assert plugin.find("topic") is not None, f"{name} must name the global topic"


def test_diff_drive_joints(urdf_root):
    plugin = next(e for e in urdf_root.iter("plugin") if "diff-drive" in (e.get("filename") or ""))
    left = [e.text for e in plugin.findall("left_joint")]
    right = [e.text for e in plugin.findall("right_joint")]
    assert left == ["wheel_front_left_joint", "wheel_rear_left_joint"]
    assert right == ["wheel_front_right_joint", "wheel_rear_right_joint"]


def test_wheel_joints_continuous(urdf_root):
    for wheel in ("wheel_front_left", "wheel_front_right", "wheel_rear_left", "wheel_rear_right"):
        joint = urdf_root.find(f"joint[@name='{wheel}_joint']")
        assert joint is not None
        assert joint.get("type") == "continuous"


def test_ladder_revolute_limits(urdf_root, cfg):
    joint = urdf_root.find("joint[@name='ladder_pitch_joint']")
    assert joint.get("type") == "revolute"
    limit = joint.find("limit")
    import math

    assert float(limit.get("lower")) == pytest.approx(math.radians(cfg.excavation.ladder.min_pitch_deg))
    assert float(limit.get("upper")) == pytest.approx(math.radians(cfg.excavation.ladder.max_pitch_deg))
    assert joint.find("axis").get("xyz") == "0 1 0"


def test_belt_joint_continuous(urdf_root):
    joint = urdf_root.find("joint[@name='belt_joint']")
    assert joint.get("type") == "continuous"
    assert joint.find("parent").get("link") == "ladder_link"


def test_tub_and_funnel_fixed(urdf_root):
    assert urdf_root.find("joint[@name='tub_joint']").get("type") == "fixed"
    assert urdf_root.find("joint[@name='funnel_joint']").get("type") == "fixed"


def test_optical_frame_rotation(urdf_root, cfg):
    for cam in cfg.sensors.cameras:
        joint = urdf_root.find(f"joint[@name='{cam.name}_optical_joint']")
        assert joint is not None
        rpy = [float(v) for v in joint.find("origin").get("rpy").split()]
        assert rpy == pytest.approx([-1.5707963267948966, 0.0, -1.5707963267948966])


def test_link_masses_sum_to_structural(cfg, urdf_root):
    total = 0.0
    for link in urdf_root.findall("link"):
        inertial = link.find("inertial")
        if inertial is not None:
            total += float(inertial.find("mass").get("value"))
    assert total == pytest.approx(cfg.structural_mass, rel=1e-6)


def test_link_inertia_values_match_table(cfg, urdf_root):
    table = link_inertias(cfg)
    for link in urdf_root.findall("link"):
        name = link.get("name")
        if name not in table:
            continue
        inertia = link.find("inertial/inertia")
        assert float(inertia.get("ixx")) == pytest.approx(table[name].ixx, rel=1e-6)
        assert float(inertia.get("izz")) == pytest.approx(table[name].izz, rel=1e-6)
