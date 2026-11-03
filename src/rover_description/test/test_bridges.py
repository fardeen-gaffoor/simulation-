"""Tests for the ros_gz_bridge topic table."""

from __future__ import annotations

import pytest

from rover_description.bridges import (
    GZ_TO_ROS,
    ROS_TO_GZ,
    bridge_arguments,
    load_bridges,
)
from rover_description.config import CAMERA_TOPICS


def test_load(bridges_path):
    bridges = load_bridges(bridges_path)
    # 4 image + 4 camera_info + imu + odom + tf + joint_states + cmd_vel
    # + ladder_cmd + belt_cmd (module 6's mechanism commands)
    assert len(bridges) == 15


def test_excavation_mechanism_bridges(bridges_path):
    # Module 6 drives the ladder and belt over these two ros_to_gz bridges; the
    # URDF's joint controllers read the same topics, so the names have to match
    # rover.yaml's `topics` exactly.
    bridges = load_bridges(bridges_path)
    by_topic = {b["ros_topic"]: b for b in bridges}
    for topic in ("/excavation/ladder_cmd", "/excavation/belt_cmd"):
        assert topic in by_topic, topic
        entry = by_topic[topic]
        assert entry["direction"] == ROS_TO_GZ
        assert entry["ros_type"] == "std_msgs/msg/Float64"
        assert entry["gz_type"] == "gz.msgs.Double"
        assert entry["gz_topic"] == topic


def test_camera_coverage(bridges_path):
    bridges = load_bridges(bridges_path)
    topics = {b["ros_topic"] for b in bridges}
    for topic in CAMERA_TOPICS:
        assert topic in topics
        assert f"{topic}/camera_info" in topics


def test_no_clock_bridge(bridges_path):
    # cat_arena's arena.launch.py owns the clock bridge
    topics = {b["ros_topic"] for b in load_bridges(bridges_path)}
    assert "/clock" not in topics


def test_types(bridges_path):
    by_topic = {b["ros_topic"]: b for b in load_bridges(bridges_path)}
    assert by_topic["/cam_nav"]["ros_type"] == "sensor_msgs/msg/Image"
    assert by_topic["/cam_nav/camera_info"]["ros_type"] == "sensor_msgs/msg/CameraInfo"
    assert by_topic["/imu"]["ros_type"] == "sensor_msgs/msg/Imu"
    assert by_topic["/odom"]["ros_type"] == "nav_msgs/msg/Odometry"
    assert by_topic["/tf"]["gz_type"] == "gz.msgs.Pose_V"
    assert by_topic["/joint_states"]["gz_type"] == "gz.msgs.Model"


def test_cmd_vel_direction(bridges_path):
    by_topic = {b["ros_topic"]: b for b in load_bridges(bridges_path)}
    assert by_topic["/cmd_vel"]["direction"] == ROS_TO_GZ


def test_argument_syntax(bridges_path):
    args = bridge_arguments(load_bridges(bridges_path))
    assert "/cam_nav@sensor_msgs/msg/Image[gz.msgs.Image" in args
    assert "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist" in args
    assert all("@" in a for a in args)


def test_direction_validation(tmp_path):
    import yaml

    doc = {
        "bridges": [
            {
                "ros_topic": "/x",
                "gz_topic": "/x",
                "ros_type": "std_msgs/msg/Bool",
                "gz_type": "gz.msgs.Boolean",
                "direction": "sideways",
            }
        ]
    }
    path = tmp_path / "bridges.yaml"
    path.write_text(yaml.safe_dump(doc))
    from rover_description.config import RoverConfigError

    with pytest.raises(RoverConfigError):
        load_bridges(str(path))


def test_mismatched_topics_rejected(tmp_path):
    import yaml

    doc = {
        "bridges": [
            {
                "ros_topic": "/x",
                "gz_topic": "/y",
                "ros_type": "std_msgs/msg/Bool",
                "gz_type": "gz.msgs.Boolean",
                "direction": GZ_TO_ROS,
            }
        ]
    }
    path = tmp_path / "bridges.yaml"
    path.write_text(yaml.safe_dump(doc))
    from rover_description.config import RoverConfigError

    with pytest.raises(RoverConfigError):
        load_bridges(str(path))
