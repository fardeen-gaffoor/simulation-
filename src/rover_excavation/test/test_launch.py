"""The launch file: the arguments it promises and the actions it produces.

A launch file is the only place the *composition* of module 6 is expressed, so the
tests execute its setup function against a real ``LaunchContext`` and assert what
would actually start - which is the thing a typo in an argument name silently
breaks, because launch happily passes a parameter nobody reads.

Two properties matter more than the rest:

* **it starts exactly one node, and only one.**  Module 6 is one process by
  design: the mechanism, the tub and the berm must be in the same address space,
  so there is no second node to split out and nothing that could be started twice.
* **``use_sim_time`` reaches the node.**  A node running on the wall clock in a
  simulator spends a whole work cycle waiting for a timeout that already passed,
  and it looks like a mechanism bug rather than a launch bug.
"""

from __future__ import annotations

import os

import pytest
import yaml

from launch import LaunchContext, LaunchDescription
from scripts_loader import load_launch

excavation = load_launch("excavation")

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _context(module, **values) -> LaunchContext:
    """A launch context with the file's own defaults, then ``values`` on top."""
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if action.__class__.__name__ == "DeclareLaunchArgument":
            action.execute(context)
    context.launch_configurations.update({key: str(value) for key, value in values.items()})
    return context


def _nodes(actions):
    return [action for action in actions if action.__class__.__name__ == "Node"]


def _executables(actions):
    return [getattr(node, "node_executable", None) for node in _nodes(actions)]


def _performed(context, value):
    if hasattr(value, "perform"):
        return value.perform(context)
    if isinstance(value, (tuple, list)):
        return "".join(_performed(context, item) for item in value)
    return value


def _parameters(node, context):
    """Flatten a Node's parameter list into one dict, as launch would deliver it.

    launch_ros encodes a plain string parameter by ``yaml.dump``-ing it, so the
    performed text is ``"/path\\n...\\n"``; the runtime decodes it again.  Decoding
    here keeps the assertion honest about what the node actually receives.
    """
    flat = {}
    for entry in getattr(node, "_Node__parameters"):
        for key, value in entry.items():
            name = _performed(context, key)
            if isinstance(value, tuple):
                text = "".join(_performed(context, part) for part in value)
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
            texts.append("".join(_performed(context, part) for part in action.msg))
    return texts


# =========================================================================== #
# module shape
# =========================================================================== #
def test_argument_truthiness():
    for text in ("1", "true", "TRUE", "yes", "on", " True "):
        assert excavation._as_bool(text) is True
    for text in ("0", "false", "no", "off", ""):
        assert excavation._as_bool(text) is False


def test_the_file_is_a_launch_description():
    description = excavation.generate_launch_description()
    assert isinstance(description, LaunchDescription)
    assert any(action.__class__.__name__ == "OpaqueFunction" for action in description.entities)


def test_it_declares_its_switches():
    names = {getattr(action, "name", None) for action in excavation.generate_launch_description().entities}
    for expected in (
        "config",
        "use_sim_time",
        "backend",
        "rate_hz",
        "gz_berm",
        "publish_marker",
        "excavation",
    ):
        assert expected in names


def test_the_defaults_are_the_safe_ones():
    context = _context(excavation)
    assert context.launch_configurations["use_sim_time"] == "true"
    assert context.launch_configurations["backend"] == "", "empty means excavation.yaml decides"
    assert context.launch_configurations["rate_hz"] == "0.0"
    assert context.launch_configurations["gz_berm"] == "true"
    assert context.launch_configurations["publish_marker"] == "true"
    assert context.launch_configurations["excavation"] == "true"


# =========================================================================== #
# what actually starts
# =========================================================================== #
def test_it_starts_the_one_node_with_the_config(config_path):
    context = _context(excavation, config=config_path)
    actions = excavation._launch_setup(context)
    assert _executables(actions) == ["excavation.py"]
    parameters = _parameters(_nodes(actions)[0], context)
    assert parameters["config"] == config_path
    assert parameters["use_sim_time"] is True


def test_the_node_gets_the_name_the_docs_use(config_path):
    context = _context(excavation, config=config_path)
    actions = excavation._launch_setup(context)
    names = {getattr(node, "_Node__node_name", None) for node in _nodes(actions)}
    assert names == {"rover_excavation_mechanism"}


def test_use_sim_time_is_off_when_asked(config_path):
    context = _context(excavation, config=config_path, use_sim_time="false")
    actions = excavation._launch_setup(context)
    assert _parameters(_nodes(actions)[0], context)["use_sim_time"] is False


def test_no_config_path_leaves_the_parameter_to_the_installed_default():
    context = _context(excavation)
    actions = excavation._launch_setup(context)
    assert "config" not in _parameters(_nodes(actions)[0], context)


def test_the_rate_reaches_the_node(config_path):
    context = _context(excavation, config=config_path, rate_hz="20.0")
    actions = excavation._launch_setup(context)
    assert _parameters(_nodes(actions)[0], context)["rate_hz"] == pytest.approx(20.0)


def test_the_default_rate_is_left_to_the_config(config_path):
    """``rate_hz=0`` means "the yaml decides", which must *not* be declared.

    Declaring 0.0 would hand the node a parameter that overrides its own default
    and then have to be clamped to the 1 Hz floor - a control loop 10x too slow
    that nothing in the logs would explain.
    """
    context = _context(excavation, config=config_path)
    actions = excavation._launch_setup(context)
    assert "rate_hz" not in _parameters(_nodes(actions)[0], context)


def test_the_backend_override_is_only_passed_when_given(config_path):
    context = _context(excavation, config=config_path)
    actions = excavation._launch_setup(context)
    assert "backend" not in _parameters(_nodes(actions)[0], context), "the config is the source of truth"
    context = _context(excavation, config=config_path, backend="null")
    actions = excavation._launch_setup(context)
    assert _parameters(_nodes(actions)[0], context)["backend"] == "null"


def test_the_gz_berm_switch_is_passed_as_a_bool(config_path):
    context = _context(excavation, config=config_path, gz_berm="false")
    actions = excavation._launch_setup(context)
    assert _parameters(_nodes(actions)[0], context)["gz_berm"] is False
    context = _context(excavation, config=config_path, gz_berm="true")
    actions = excavation._launch_setup(context)
    assert _parameters(_nodes(actions)[0], context)["gz_berm"] is True


def test_an_empty_gz_berm_argument_defers_to_the_config(config_path):
    """``gz_berm:=''`` must mean "whatever the yaml says", not False."""
    context = _context(excavation, config=config_path, gz_berm="")
    actions = excavation._launch_setup(context)
    assert "gz_berm" not in _parameters(_nodes(actions)[0], context)


def test_the_marker_can_be_turned_off(config_path):
    context = _context(excavation, config=config_path, publish_marker="false")
    actions = excavation._launch_setup(context)
    assert _parameters(_nodes(actions)[0], context)["publish_marker"] is False


def test_the_node_can_be_left_out_entirely(config_path):
    context = _context(excavation, config=config_path, excavation="false")
    actions = excavation._launch_setup(context)
    assert _nodes(actions) == []
    assert any("excavation disabled" in text for text in _log_texts(actions, context))


def test_it_says_that_module_5_is_the_caller(config_path):
    context = _context(excavation, config=config_path)
    actions = excavation._launch_setup(context)
    assert any("work_cycle" in text for text in _log_texts(actions, context))


def test_it_starts_exactly_one_node_in_every_configuration(config_path):
    for flags in ({}, {"backend": "sim"}, {"gz_berm": "false"}, {"rate_hz": "5.0"}):
        context = _context(excavation, config=config_path, **flags)
        actions = excavation._launch_setup(context)
        assert _executables(actions).count("excavation.py") == 1


def test_the_launch_file_never_starts_module_5(config_path):
    """Module 6 must not drag the mission sequencer in: the boundary is the point."""
    context = _context(excavation, config=config_path)
    actions = excavation._launch_setup(context)
    for node in _nodes(actions):
        assert getattr(node, "node_package", None) == "rover_excavation"
