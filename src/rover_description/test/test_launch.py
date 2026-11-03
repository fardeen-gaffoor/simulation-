"""Structural tests for the launch files.

The launch files are not importable packages, so they are loaded by path and
their declared arguments inspected.  Behaviour that only happens inside the
OpaqueFunction (which builds the nodes/actions) is checked against the source
text, so the wiring can't silently disappear.
"""

from __future__ import annotations

from loaders import declared_argument_names, launch_path, load_launch
import os


def _read(name):
    with open(launch_path(name), "r", encoding="utf-8") as handle:
        return handle.read()


def test_sim_declares_expected_args():
    module = load_launch("sim")
    names = declared_argument_names(module.generate_launch_description())
    assert {
        "seed",
        "headless",
        "headless_rendering",
        "arena_config",
        "rover_config",
        "robot_output_dir",
        "bridges",
        "estop_gate",
        "spawn_delay",
    } <= names


def test_sim_wiring_source():
    text = _read("sim")
    # includes the arena, spawns with robot plugins, starts RSP, the bridge and the stub
    assert "arena.launch.py" in text
    assert "robot_plugins" in text
    assert "robot_state_publisher" in text
    assert "parameter_bridge" in text
    assert "stub_topics.py" in text
    assert "bridge_arguments" in text


def test_robot_declares_expected_args():
    module = load_launch("robot")
    names = declared_argument_names(module.generate_launch_description())
    assert {"rover_config", "robot_output_dir"} <= names


def test_robot_is_sim_free():
    text = _read("robot")
    assert "gz sim" not in text
    assert "ros_gz_bridge" not in text
    assert "robot_state_publisher" in text
    assert "stub_topics.py" in text


def test_description_launches_rviz():
    module = load_launch("description")
    names = declared_argument_names(module.generate_launch_description())
    assert {"rover_config", "robot_output_dir", "use_joint_state_publisher"} <= names
    text = _read("description")
    assert "rviz2" in text
    assert "rover.rviz" in text


def test_rviz_config_exists():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.isfile(os.path.join(root, "rviz", "rover.rviz"))
