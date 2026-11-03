"""The launch files: the arguments they promise and the actions they produce.

A launch file is the only place the *composition* of module 4 is expressed, so
the tests execute its setup function against a real ``LaunchContext`` and assert
what would actually start: five nodes with the config wired in, no node when its
toggle is off, tag boards spawned only when asked for, and ``map -> odom`` left
off by default in the simulation (gz already publishes ``odom -> base_link``).
"""

from __future__ import annotations

import os

import pytest
import yaml

from launch import LaunchContext, LaunchDescription
from scripts_loader import load_launch

perception = load_launch("perception")
localization = load_launch("localization")


def _context(module, **values) -> LaunchContext:
    """A launch context with the file's own defaults, then ``values`` on top."""
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if action.__class__.__name__ == "DeclareLaunchArgument":
            action.execute(context)  # sets context.launch_configurations[name]
    context.launch_configurations.update({key: str(value) for key, value in values.items()})
    return context


def _nodes(actions):
    return [action for action in actions if action.__class__.__name__ == "Node"]


def _flatten(actions):
    """Top-level actions plus the contents of any TimerAction."""
    out = []
    for action in actions:
        out.append(action)
        inner = getattr(action, "actions", None)
        if isinstance(inner, (list, tuple)):
            out.extend(inner)
    return out


def _executables(actions):
    return [getattr(node, "node_executable", None) for node in _nodes(actions)]


def _perform(context, value):
    if hasattr(value, "perform"):
        return value.perform(context)
    if isinstance(value, (tuple, list)):
        return "".join(_perform(context, item) for item in value)
    return value


def _parameters(node, context):
    """Flatten a Node's parameter list into one dict, as launch would deliver it.

    launch_ros encodes a plain string parameter by ``yaml.dump``-ing it, so the
    performed text is ``"/path\\n...\\n"``; the runtime decodes it again when it
    evaluates the value.  Decoding here keeps the assertion honest about what the
    node actually receives.
    """
    flat = {}
    for entry in getattr(node, "_Node__parameters"):
        for key, value in entry.items():
            name = _perform(context, key)
            if isinstance(value, tuple):
                text = "".join(_perform(context, part) for part in value)
                try:
                    flat[name] = yaml.safe_load(text)
                except yaml.YAMLError:  # pragma: no cover - never for our values
                    flat[name] = text
            else:
                flat[name] = value
    return flat


def _log_texts(actions, context):
    texts = []
    for action in actions:
        if action.__class__.__name__ == "LogInfo":
            texts.append("".join(_perform(context, part) for part in action.msg))
    return texts


# --------------------------------------------------------------------------- #
# module shape
# --------------------------------------------------------------------------- #
def test_argument_truthiness():
    for text in ("1", "true", "TRUE", "yes", "on", " True "):
        assert perception._as_bool(text) is True
    for text in ("0", "false", "no", "off", ""):
        assert perception._as_bool(text) is False


def test_both_files_can_be_imported_as_launch_descriptions():
    for module in (perception, localization):
        description = module.generate_launch_description()
        assert isinstance(description, LaunchDescription)
        assert any(
            action.__class__.__name__ == "OpaqueFunction" for action in description.entities
        )


def test_perception_declares_its_switches():
    names = {
        getattr(action, "name", None)
        for action in perception.generate_launch_description().entities
    }
    for expected in (
        "config",
        "world",
        "markers",
        "tags_dir",
        "generate_tags",
        "spawn_tags",
        "spawn_delay",
        "use_sim_time",
        "publish_tf",
        "camera_manager",
        "tag_detector",
        "tag_localizer",
        "obstacle_detector",
        "health_monitor",
    ):
        assert expected in names


# --------------------------------------------------------------------------- #
# perception.launch.py
# --------------------------------------------------------------------------- #
def test_all_five_nodes_start_with_the_config(cfg):
    context = _context(perception, config=cfg.path, generate_tags="false", spawn_tags="false")
    actions = perception._launch_setup(context)
    assert sorted(_executables(actions)) == sorted(perception._NODES.values())
    for node in _nodes(actions):
        parameters = _parameters(node, context)
        assert parameters["config"] == cfg.path
        assert parameters["use_sim_time"] is True


def test_publish_tf_is_off_by_default(cfg):
    context = _context(perception, config=cfg.path, generate_tags="false", spawn_tags="false")
    actions = perception._launch_setup(context)
    localizer = next(
        node for node in _nodes(actions) if getattr(node, "node_executable") == "tag_localizer.py"
    )
    assert _parameters(localizer, context)["publish_tf"] is False


def test_publish_tf_can_be_turned_on(cfg):
    context = _context(
        perception, config=cfg.path, generate_tags="false", spawn_tags="false", publish_tf="true"
    )
    actions = perception._launch_setup(context)
    localizer = next(
        node for node in _nodes(actions) if getattr(node, "node_executable") == "tag_localizer.py"
    )
    assert _parameters(localizer, context)["publish_tf"] is True


def test_a_disabled_node_is_not_started(cfg):
    context = _context(
        perception,
        config=cfg.path,
        generate_tags="false",
        spawn_tags="false",
        obstacle_detector="false",
    )
    actions = perception._launch_setup(context)
    assert "obstacle_detector.py" not in _executables(actions)
    assert any("obstacle_detector disabled" in text for text in _log_texts(actions, context))


def test_tag_board_generation_and_spawn(cfg, markers_path, tmp_path):
    tags_dir = str(tmp_path / "tags")
    context = _context(
        perception,
        config=cfg.path,
        markers=markers_path,
        tags_dir=tags_dir,
        generate_tags="true",
        spawn_tags="true",
        spawn_delay="0.0",
    )
    actions = perception._launch_setup(context)
    assert os.path.isfile(os.path.join(tags_dir, "tags.yaml"))
    assert os.path.isfile(os.path.join(tags_dir, "tag_boards.yaml"))
    created = [action for action in _flatten(actions) if action.__class__.__name__ == "ExecuteProcess"]
    assert len(created) == 3, "one ros_gz_sim create per board"
    assert any(action.__class__.__name__ == "TimerAction" for action in actions)


def test_a_missing_marker_file_does_not_kill_the_launch(cfg, tmp_path):
    context = _context(
        perception,
        config=cfg.path,
        markers=str(tmp_path / "absent.yaml"),
        tags_dir=str(tmp_path / "tags"),
        generate_tags="true",
        spawn_tags="true",
    )
    actions = perception._launch_setup(context)
    assert any("tag artefacts not generated" in text for text in _log_texts(actions, context))
    assert sorted(_executables(actions)) == sorted(perception._NODES.values()), "nodes still start"


# --------------------------------------------------------------------------- #
# localization.launch.py
# --------------------------------------------------------------------------- #
def test_localization_declares_its_switches():
    names = {
        getattr(action, "name", None)
        for action in localization.generate_launch_description().entities
    }
    for expected in ("config", "ekf_config", "ekf", "publish_tf", "use_sim_time", "tag_detector"):
        assert expected in names


def test_localization_defaults_are_the_robot_defaults():
    context = _context(localization)
    assert context.launch_configurations["publish_tf"] == "true", "on the robot nothing else publishes map -> odom"
    assert context.launch_configurations["use_sim_time"] == "false"
    assert context.launch_configurations["tag_detector"] == "true"
    assert context.launch_configurations["ekf"] == "true"


def test_localization_starts_the_detector_and_localizer(cfg):
    context = _context(localization, config=cfg.path, ekf="false")
    actions = localization._launch_setup(context)
    assert "tag_detector.py" in _executables(actions)
    localizer = next(
        node for node in _nodes(actions) if getattr(node, "node_executable") == "tag_localizer.py"
    )
    assert _parameters(localizer, context)["publish_tf"] is True
    assert _parameters(localizer, context)["use_sim_time"] is False


def test_localization_can_skip_the_detector(cfg):
    context = _context(localization, config=cfg.path, ekf="false", tag_detector="false")
    actions = localization._launch_setup(context)
    assert "tag_detector.py" not in _executables(actions)
    assert "tag_localizer.py" in _executables(actions)


def test_missing_robot_localization_is_reported_not_fatal(cfg, monkeypatch):
    monkeypatch.setattr(localization, "_robot_localization_available", lambda: False)
    context = _context(localization, config=cfg.path, ekf="true")
    actions = localization._launch_setup(context)
    assert any("robot_localization is not installed" in text for text in _log_texts(actions, context))
    assert "tag_localizer.py" in _executables(actions), "the tag chain still starts"
