"""The launch files: the arguments they promise and the actions they produce.

A launch file is the only place the *composition* of module 5 is expressed, so the
tests execute its setup function against a real ``LaunchContext`` and assert what
would actually start - which is the thing a typo in an argument name silently
breaks, because launch happily passes a parameter nobody reads.

Two properties matter more than the rest:

* **the split is preserved.**  ``navigation.launch.py`` starts the navigator *and*
  the gate, and never the sequencer; ``mission.launch.py`` adds the sequencer and
  includes the other file.  Keeping them apart is what lets the navigator be driven
  by hand without a sequencer sending it work.
* **``use_sim_time`` reaches every node.**  A node running on the wall clock in a
  simulator spends its whole life timing out, and the failure looks like a planner
  bug rather than a launch bug.
"""

from __future__ import annotations

import os

import pytest
import yaml

from launch import LaunchContext, LaunchDescription
from scripts_loader import load_launch

navigation = load_launch("navigation")
mission = load_launch("mission")

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERCEPTION_SHARE = os.path.join(os.path.dirname(_SRC), "rover_perception")


@pytest.fixture()
def share_dir(monkeypatch):
    """Resolve package share directories to the source tree, not the install space.

    The include is a real path join, so pointing it at the source tree exercises
    exactly the code the install space would run without depending on ``colcon
    build`` having been run in this checkout.
    """
    import ament_index_python.packages as packages

    def fake(name: str) -> str:
        if name == "rover_navigation":
            return _SRC
        if name == "rover_perception":
            return _PERCEPTION_SHARE
        raise packages.PackageNotFoundError(name)

    monkeypatch.setattr(packages, "get_package_share_directory", fake)
    return fake


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


def _includes(actions):
    return [action for action in actions if action.__class__.__name__ == "IncludeLaunchDescription"]


def _include_target(include, context) -> str:
    """The file an ``IncludeLaunchDescription`` will load, as a plain path.

    ``launch_description_source.location`` stringifies its own substitution list
    (so it reads as the repr of a ``TextSubstitution``), and the substitutions are
    what carry the path.  Reaching for launch's private list is ugly, but it is the
    only way to assert *which* file is included rather than merely that one is.
    """
    source = include.launch_description_source
    parts = getattr(source, "_LaunchDescriptionSource__location")
    return "".join(part.perform(context) for part in parts)


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
        assert navigation._as_bool(text) is True
        assert mission._as_bool(text) is True
    for text in ("0", "false", "no", "off", ""):
        assert navigation._as_bool(text) is False
        assert mission._as_bool(text) is False


def test_both_files_can_be_imported_as_launch_descriptions():
    for module in (navigation, mission):
        description = module.generate_launch_description()
        assert isinstance(description, LaunchDescription)
        assert any(action.__class__.__name__ == "OpaqueFunction" for action in description.entities)


def test_navigation_declares_its_switches():
    names = {getattr(action, "name", None) for action in navigation.generate_launch_description().entities}
    for expected in (
        "config",
        "use_sim_time",
        "perception",
        "perception_delay",
        "spawn_delay",
        "publish_costmap",
        "publish_status",
        "navigation",
        "velocity_gate",
    ):
        assert expected in names


def test_mission_declares_its_switches():
    names = {getattr(action, "name", None) for action in mission.generate_launch_description().entities}
    for expected in ("config", "use_sim_time", "navigation", "perception", "autostart", "start_delay", "work_mode"):
        assert expected in names


def test_the_navigation_defaults_are_the_safe_ones():
    context = _context(navigation)
    assert context.launch_configurations["use_sim_time"] == "true"
    assert context.launch_configurations["perception"] == "false", "module 5 must not assume it is in a simulator"
    assert context.launch_configurations["publish_costmap"] == "true"
    assert context.launch_configurations["publish_status"] == "true"
    assert context.launch_configurations["navigation"] == "true"
    assert context.launch_configurations["velocity_gate"] == "true"


def test_the_mission_defaults_autostart_after_a_tag_fix():
    context = _context(mission)
    assert context.launch_configurations["autostart"] == "true"
    assert float(context.launch_configurations["start_delay"]) > 0.0, "a tag fix needs a moment"
    assert context.launch_configurations["work_mode"] == "", "empty means the config decides"
    assert context.launch_configurations["navigation"] == "true"


# =========================================================================== #
# navigation.launch.py
# =========================================================================== #
def test_the_navigation_stack_starts_two_nodes_with_the_config(config_path):
    context = _context(navigation, config=config_path)
    actions = navigation._launch_setup(context)
    assert sorted(_executables(actions)) == sorted(navigation._NODES.values())
    for node in _nodes(actions):
        parameters = _parameters(node, context)
        assert parameters["config"] == config_path
        assert parameters["use_sim_time"] is True


def test_the_navigator_and_the_gate_get_the_names_the_docs_use(config_path):
    context = _context(navigation, config=config_path)
    actions = navigation._launch_setup(context)
    names = {getattr(node, "_Node__node_name", None) for node in _nodes(actions)}
    assert names == {"rover_navigation_navigation", "rover_navigation_velocity_gate"}


def test_use_sim_time_is_off_when_asked(config_path):
    context = _context(navigation, config=config_path, use_sim_time="false")
    actions = navigation._launch_setup(context)
    for node in _nodes(actions):
        assert _parameters(node, context)["use_sim_time"] is False


def test_no_config_path_leaves_the_parameter_to_the_installed_default():
    context = _context(navigation)
    actions = navigation._launch_setup(context)
    for node in _nodes(actions):
        assert "config" not in _parameters(node, context)


def test_the_costmap_preview_can_be_turned_off(config_path):
    context = _context(navigation, config=config_path, publish_costmap="false")
    actions = navigation._launch_setup(context)
    for node in _nodes(actions):
        assert _parameters(node, context)["publish_costmap"] is False


def test_the_gate_status_can_be_turned_off_and_it_only_touches_the_gate(config_path):
    context = _context(navigation, config=config_path, publish_status="false")
    actions = navigation._launch_setup(context)
    for node in _nodes(actions):
        parameters = _parameters(node, context)
        if getattr(node, "node_executable") == "velocity_gate.py":
            assert parameters["publish_status"] is False
        else:
            assert "publish_status" not in parameters, "the navigator has no such parameter"


def test_a_disabled_node_is_not_started(config_path):
    context = _context(navigation, config=config_path, velocity_gate="false")
    actions = navigation._launch_setup(context)
    assert "velocity_gate.py" not in _executables(actions)
    assert "navigation.py" in _executables(actions)
    assert any("velocity_gate disabled" in text for text in _log_texts(actions, context))


def test_perception_is_not_included_by_default(config_path):
    context = _context(navigation, config=config_path)
    actions = navigation._launch_setup(context)
    assert _includes(actions) == []
    assert sorted(_executables(actions)) == ["navigation.py", "velocity_gate.py"]


def test_perception_can_be_included(share_dir, config_path):
    context = _context(navigation, config=config_path, perception="true", spawn_delay="3.0")
    actions = navigation._launch_setup(context)
    includes = _includes(actions)
    assert len(includes) == 1
    assert _include_target(includes[0], context) == os.path.join(
        _PERCEPTION_SHARE, "launch", "perception.launch.py"
    )
    assert any("starting module 4" in text for text in _log_texts(actions, context))
    assert sorted(_executables(actions)) == ["navigation.py", "velocity_gate.py"], "module 5 still starts"


def test_a_missing_perception_package_does_not_kill_the_launch(monkeypatch, config_path):
    import ament_index_python.packages as packages

    def boom(name: str) -> str:
        raise packages.PackageNotFoundError(name)

    monkeypatch.setattr(packages, "get_package_share_directory", boom)
    context = _context(navigation, config=config_path, perception="true")
    actions = navigation._launch_setup(context)
    assert any("could not find rover_perception" in text for text in _log_texts(actions, context))
    assert sorted(_executables(actions)) == ["navigation.py", "velocity_gate.py"]


def test_the_perception_delay_explains_the_wait(config_path):
    context = _context(navigation, config=config_path, perception="true", perception_delay="12.0")
    actions = navigation._launch_setup(context)
    assert any("hold" in text and "12s" in text for text in _log_texts(actions, context))


# =========================================================================== #
# mission.launch.py
# =========================================================================== #
def test_mission_includes_the_driving_stack(share_dir, config_path):
    context = _context(mission, config=config_path)
    actions = mission._launch_setup(context)
    includes = _includes(actions)
    assert len(includes) == 1
    assert _include_target(includes[0], context) == os.path.join(
        _SRC, "launch", "navigation.launch.py"
    )
    assert "mission.py" in _executables(actions)


def test_mission_can_run_without_the_driving_stack(share_dir, config_path):
    context = _context(mission, config=config_path, navigation="false")
    actions = mission._launch_setup(context)
    assert _includes(actions) == []
    assert _executables(actions) == ["mission.py"], "the sequencer still starts"


def test_mission_always_starts_exactly_one_sequencer(share_dir, config_path):
    for with_navigation in ("true", "false"):
        context = _context(mission, config=config_path, navigation=with_navigation)
        actions = mission._launch_setup(context)
        assert _executables(actions).count("mission.py") == 1


def test_mission_passes_autostart_and_the_delay(share_dir, config_path):
    context = _context(mission, config=config_path, autostart="false", start_delay="21.0")
    actions = mission._launch_setup(context)
    node = next(node for node in _nodes(actions) if getattr(node, "node_executable") == "mission.py")
    parameters = _parameters(node, context)
    assert parameters["autostart"] is False
    assert parameters["start_delay_s"] == pytest.approx(21.0)
    assert parameters["use_sim_time"] is True


def test_the_work_mode_is_only_overridden_when_it_is_given(share_dir, config_path):
    context = _context(mission, config=config_path)
    actions = mission._launch_setup(context)
    node = next(node for node in _nodes(actions) if getattr(node, "node_executable") == "mission.py")
    assert "work_mode_override" not in _parameters(node, context), "the config is the source of truth"


def test_a_work_mode_override_reaches_the_node(share_dir, config_path):
    context = _context(mission, config=config_path, work_mode="navigate_only")
    actions = mission._launch_setup(context)
    node = next(node for node in _nodes(actions) if getattr(node, "node_executable") == "mission.py")
    assert _parameters(node, context)["work_mode_override"] == "navigate_only"
    assert any("navigate_only" in text for text in _log_texts(actions, context))


def test_the_delegate_work_mode_is_explained_too(share_dir, config_path):
    context = _context(mission, config=config_path, work_mode="delegate")
    actions = mission._launch_setup(context)
    assert any("/excavation/work_cycle" in text for text in _log_texts(actions, context))


def test_a_missing_navigation_package_does_not_kill_the_launch(monkeypatch, config_path):
    import ament_index_python.packages as packages

    def boom(name: str) -> str:
        raise packages.PackageNotFoundError(name)

    monkeypatch.setattr(packages, "get_package_share_directory", boom)
    context = _context(mission, config=config_path)
    actions = mission._launch_setup(context)
    assert any("could not find navigation.launch.py" in text for text in _log_texts(actions, context))
    assert "mission.py" in _executables(actions)
