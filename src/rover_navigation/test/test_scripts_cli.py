"""The ``ros2 run`` entry points.

Three of the four files in ``scripts/`` are shims: three lines that forward to a
node module.  A shim cannot be run to completion in a test (it would spin), so the
test pins what a shim can get wrong - that it exposes ``main`` and that ``--help``
reaches argparse *before* ``rclpy`` is initialised, which is what makes
``ros2 run rover_navigation navigation.py --help`` usable.

The fourth, ``navigation_info.py``, has real logic and is driven end to end: it is
the tool to run before a competition run ("can the rover legally be everywhere it
is about to be told to go"), so its answer has to be correct, not merely printed.
"""

from __future__ import annotations

import json

import pytest

from rover_navigation.config import load_config
from scripts_loader import load_script

navigation = load_script("navigation")
velocity_gate = load_script("velocity_gate")
mission = load_script("mission")
navigation_info = load_script("navigation_info")

_SHIMS = ("navigation", "velocity_gate", "mission")


# =========================================================================== #
# every script is loadable
# =========================================================================== #
def test_every_script_exposes_main():
    for name in (*_SHIMS, "navigation_info"):
        module = load_script(name)
        assert callable(module.main), name


@pytest.mark.parametrize("name", _SHIMS)
def test_a_node_shim_forwards_to_the_node_main(name):
    from rover_navigation.nodes import mission_node, navigation_node, velocity_gate_node

    expected = {
        "navigation": navigation_node,
        "velocity_gate": velocity_gate_node,
        "mission": mission_node,
    }[name]
    assert load_script(name).main is expected.main


@pytest.mark.parametrize("name", _SHIMS)
def test_help_exits_before_rclpy_starts(name, capsys):
    module = load_script(name)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0
    assert "--config" in capsys.readouterr().out


def test_navigation_info_help_lists_its_flags(capsys):
    with pytest.raises(SystemExit) as excinfo:
        navigation_info.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--config", "--json", "--deps", "--no-poses", "--quiet"):
        assert flag in out


# =========================================================================== #
# navigation_info: the report
# =========================================================================== #
def test_the_report_describes_the_shipped_configuration(config):
    data = navigation_info.report(config)
    assert data["config"]["path"] == config.path
    assert data["config"]["sha256"] == config.sha256
    assert data["arena"]["length_x"] == config.arena.length_x
    assert data["arena"]["free_rect"] == config.free_rect.as_tuple()
    assert data["robot"]["operational_footprint_m"] == (config.robot.length_m, config.robot.width_m)
    assert data["robot"]["competition_envelope_m"] == (config.robot.build_length_m, config.robot.build_width_m)
    assert data["mission"]["work_mode"] == config.mission.work_mode
    assert data["topics"]["navigate_action"] == config.topics.navigate_action


def test_the_report_lists_every_generated_pose(config):
    from rover_navigation.mission import dig_poses, dump_poses

    data = navigation_info.report(config)
    labels = [pose["label"] for pose in data["poses"]]
    assert len(data["poses"]) == len(dig_poses(config)) + len(dump_poses(config)) + 1  # + park
    assert any(label.startswith("dig:") for label in labels)
    assert any(label.startswith("dump:") for label in labels)
    assert "park:park" in labels


def test_every_generated_pose_is_reported_legal_on_an_empty_map(config):
    """The tool's whole purpose: a pose the rover cannot occupy is a bug caught here."""
    data = navigation_info.report(config)
    illegal = [pose for pose in data["poses"] if not pose["legal"]]
    assert illegal == [], illegal


def test_the_pose_report_can_be_skipped(config):
    data = navigation_info.report(config, with_poses=False)
    assert data["poses"] == []


def test_the_dependency_report_lists_the_required_packages():
    lines = navigation_info.dependency_report()
    assert lines[0] == "dependencies:"
    text = "\n".join(lines)
    assert "rclpy" in text
    assert "rover_navigation_msgs" in text
    assert "rover_perception_msgs" in text
    assert "optional:" in text
    assert "nav2_bringup" in text, "the report should say module 5 does not use nav2"


# =========================================================================== #
# navigation_info: the CLI
# =========================================================================== #
def test_json_output_is_parseable_and_complete(config_path, capsys):
    assert navigation_info.main(["--config", config_path, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"]["path"] == config_path
    assert payload["robot"]["planner_radius_m"] == pytest.approx(load_config(config_path).robot.radius_m)
    assert payload["poses"], "the poses are in the JSON too"


def test_json_with_deps_adds_the_dependency_table(config_path, capsys):
    assert navigation_info.main(["--config", config_path, "--json", "--deps"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "dependencies" in payload
    assert "required" in payload["dependencies"]


def test_no_poses_empties_the_json_poses(config_path, capsys):
    assert navigation_info.main(["--config", config_path, "--json", "--no-poses"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["poses"] == []


def test_text_output_covers_the_sections(config_path, capsys):
    assert navigation_info.main(["--config", config_path]) == 0
    out = capsys.readouterr().out
    for heading in ("navigation.yaml", "frames:", "arena:", "robot:", "limits:", "costmap:", "safety:", "mission:", "topics:"):
        assert heading in out, heading
    assert "generated poses:" in out


def test_quiet_prints_only_the_poses(config_path, capsys):
    assert navigation_info.main(["--config", config_path, "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "navigation.yaml:" not in out
    assert "generated poses:" in out


def test_deps_appends_the_table_to_the_text(config_path, capsys):
    assert navigation_info.main(["--config", config_path, "--deps"]) == 0
    assert "dependencies:" in capsys.readouterr().out


def test_the_default_config_is_the_installed_one(capsys):
    """No ``--config`` resolves the installed default, which is the runtime path."""
    assert navigation_info.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert load_config(None).path == payload["config"]["path"]


def test_a_bad_config_fails_with_a_message(capsys):
    assert navigation_info.main(["--config", "/nonexistent/navigation.yaml"]) == 1
    assert "navigation_info.py:" in capsys.readouterr().err


def test_an_illegal_pose_is_reported_as_illegal(tmp_path, config_file, capsys):
    """The tool must be able to say no, not just print 'ok' for everything."""
    from support import config_with

    # Push the park pose through the wall: the report must flag it.
    cfg = config_with(config_file, tmp_path, {"mission.park.x": 0.05, "mission.park.y": 0.05})
    data = navigation_info.report(cfg)
    park = next(pose for pose in data["poses"] if pose["label"] == "park:park")
    assert park["legal"] is False
    assert park["blocked_by"]
