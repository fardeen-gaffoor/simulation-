"""Every node: it imports, it parses its flags, and it can be constructed and ticked.

The nodes are thin ROS wrappers around the tested libraries, so this file does not
re-test the algorithms.  What it *does* pin is the glue that only shows up when
the whole thing is started:

* each module exposes ``NODE_NAME`` / ``add_arguments`` / ``main``, with the node
  name the launch files use;
* the constructor accepts ``--config`` and therefore honours an override rather
  than silently using the installed default;
* a node builds its publishers, subscriptions, action servers and timers without
  a live DDS peer, and one ``_tick()`` runs to completion with nothing connected.
  That last one is worth having: it is exactly the state a node is in during the
  first seconds of a launch, and it is where a None pose or an empty path used to
  reach a division.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import pytest

from rover_navigation.nodes import mission_node, navigation_node, velocity_gate_node

_MODULES = {
    "navigation": navigation_node,
    "velocity_gate": velocity_gate_node,
    "mission": mission_node,
}

_CLASSES = {
    "navigation": ("rover_navigation.nodes.navigation_node", "NavigationNode"),
    "velocity_gate": ("rover_navigation.nodes.velocity_gate_node", "VelocityGateNode"),
    "mission": ("rover_navigation.nodes.mission_node", "MissionNode"),
}


# =========================================================================== #
# module shape
# =========================================================================== #
def test_every_module_declares_its_node_name():
    for label, module in _MODULES.items():
        assert isinstance(module.NODE_NAME, str) and module.NODE_NAME
        assert module.NODE_NAME.startswith("rover_navigation_"), label


def test_the_node_names_are_the_ones_the_launch_files_use():
    assert navigation_node.NODE_NAME == "rover_navigation_navigator"
    assert velocity_gate_node.NODE_NAME == "rover_navigation_velocity_gate"
    assert mission_node.NODE_NAME == "rover_navigation_mission"


def test_every_module_exports_the_entry_points():
    for label, module in _MODULES.items():
        assert callable(module.main), label
        assert callable(module.add_arguments), label
        node_class = next(getattr(module, name) for name in module.__all__ if name.endswith("Node"))
        assert isinstance(node_class, type), label


def test_the_node_classes_are_importable_by_name():
    for label, (module_name, class_name) in _CLASSES.items():
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name), label


# =========================================================================== #
# flags
# =========================================================================== #
def _flags(module):
    parser = argparse.ArgumentParser(prog="node")
    parser.add_argument("--config", default="")
    module.add_arguments(parser)
    return parser


def test_the_shared_config_flag_exists_with_the_default():
    for label, module in _MODULES.items():
        args = _flags(module).parse_args([])
        assert args.config == "", label


def test_every_node_takes_a_publish_rate_override():
    for label, module in _MODULES.items():
        args = _flags(module).parse_args(["--publish-hz", "7.5"])
        assert args.publish_hz == pytest.approx(7.5), label


def test_mission_flags_parse():
    parser = _flags(mission_node)
    assert parser.parse_args([]).autostart is None, "no flag must leave the parameter default alone"
    assert parser.parse_args(["--autostart"]).autostart is True
    assert parser.parse_args(["--no-autostart"]).autostart is False
    args = parser.parse_args(["--start-delay", "12.5", "--work-mode", "navigate_only"])
    assert args.start_delay == pytest.approx(12.5)
    assert args.work_mode == "navigate_only"


def test_mission_rejects_an_unknown_work_mode(capsys):
    with pytest.raises(SystemExit):
        _flags(mission_node).parse_args(["--work-mode", "dig_everything"])


def test_the_argument_split_keeps_ros_args_for_rclpy():
    from rover_navigation import nodeutil

    ros, own = nodeutil.split_node_arguments(
        ["--config", "/x.yaml", "--ros-args", "-r", "__node:=n", "--params-file", "/p.yaml"]
    )
    assert own == ["--config", "/x.yaml"]
    assert ros == ["--ros-args", "-r", "__node:=n", "--params-file", "/p.yaml"]

    ros, own = nodeutil.split_node_arguments(["--publish-hz", "2.0"])
    assert ros is None
    assert own == ["--publish-hz", "2.0"]


def test_sys_argv_carrying_ros_args_is_split_too(monkeypatch):
    from rover_navigation import nodeutil

    monkeypatch.setattr(sys, "argv", ["navigation.py", "--ros-args", "-r", "__node:=navigator"])
    ros, own = nodeutil.split_node_arguments(None)
    assert own == []
    assert ros == ["--ros-args", "-r", "__node:=navigator"]


def test_no_ros_args_leaves_own_args_alone(monkeypatch):
    from rover_navigation import nodeutil

    monkeypatch.setattr(sys, "argv", ["mission.py", "--no-autostart"])
    ros, own = nodeutil.split_node_arguments(None)
    assert ros is None
    assert own == ["--no-autostart"]


# =========================================================================== #
# construction
# =========================================================================== #
@pytest.fixture()
def built(rclpy_session):
    """Construct a node, then always destroy it so the DDS graph stays clean."""
    made = []

    def _build(factory):
        node = factory()
        made.append(node)
        return node

    yield _build
    for node in made:
        node.destroy_node()


def test_navigation_node_builds(built, config_path):
    node = built(lambda: navigation_node.NavigationNode(config_override=config_path))
    assert node.config.path == config_path, "the --config override is honoured"
    assert node.publish_hz > 0.0
    snapshot = node.snapshot()
    assert snapshot["state"] == "idle"
    assert snapshot["active"] is False
    assert snapshot["pose"] is None, "no odometry and no tag fix yet"
    assert snapshot["obstacles"] == 0
    assert snapshot["path_points"] == 0
    assert node.costmap_publisher is not None


def test_navigation_node_ticks_with_nothing_connected(built, config_path):
    node = built(lambda: navigation_node.NavigationNode(config_override=config_path))
    node._tick()
    assert node.state == "idle"
    assert (node.command.linear, node.command.angular) == (0.0, 0.0)
    assert node.costmap is not None, "the costmap is built whether or not there is a goal"
    assert node.costmap.free_cells() > 0


def test_the_costmap_preview_can_be_turned_off(built, config_path):
    node = built(lambda: navigation_node.NavigationNode(config_override=config_path))
    assert node.publish_costmap is True
    node.publish_costmap = False
    node.costmap_publisher = None
    node._publish_costmap()          # must not raise with no publisher


def test_velocity_gate_node_builds(built, config_path):
    node = built(lambda: velocity_gate_node.VelocityGateNode(config_override=config_path))
    assert node.config.path == config_path
    snapshot = node.snapshot()
    assert snapshot["estop"] is False
    assert snapshot["allowed"] is False, "nothing has been commanded yet"
    assert snapshot["ticks"] == 0


def test_the_gate_stops_the_rover_on_its_first_tick(built, config_path):
    node = built(lambda: velocity_gate_node.VelocityGateNode(config_override=config_path))
    node._tick()
    assert node.ticks == 1
    assert node.stops == 1
    assert (node.decision.linear, node.decision.angular) == (0.0, 0.0)
    assert node.decision.reason, "a stop with no reason is indistinguishable from a bug"


def test_the_gate_commands_nothing_until_it_has_a_map(built, config_path):
    """'Starting up' is not 'clear to drive': without a fix the reason says so."""
    from geometry_msgs.msg import Twist

    node = built(lambda: velocity_gate_node.VelocityGateNode(config_override=config_path))
    command = Twist()
    command.linear.x = 0.3
    node._on_command(command)        # fresh, but the map has no fix yet
    node._tick()
    assert node.decision.allowed is False
    assert "localization" in node.decision.reason


def test_mission_node_builds(built, config_path):
    node = built(lambda: mission_node.MissionNode(config_override=config_path))
    assert node.config.path == config_path
    snapshot = node.snapshot()
    assert snapshot["started"] is False
    assert snapshot["phase"] == 0
    assert snapshot["cycles_completed"] == 0
    assert snapshot["work_mode"] == node.config.mission.work_mode
    assert snapshot["mode"] == "", "no mode has been published yet"


def test_mission_generates_its_poses_when_the_run_starts(built, config_path):
    node = built(lambda: mission_node.MissionNode(config_override=config_path, autostart=False))
    assert node.snapshot()["dig_poses"] == 0, "nothing is generated until it is needed"
    node.sequencer.start(0.0)
    snapshot = node.snapshot()
    assert snapshot["dig_poses"] > 0
    assert snapshot["dump_poses"] > 0


def test_mission_ticks_with_nothing_connected(built, config_path):
    from rover_navigation.mission import PHASE_IDLE

    node = built(lambda: mission_node.MissionNode(config_override=config_path, autostart=False))
    node._tick()
    assert node.started is False
    assert node.sequencer.phase == PHASE_IDLE
    node._tick()                     # and again, still without a start


def test_mission_autostart_starts_the_run_after_the_delay(built, config_path):
    node = built(
        lambda: mission_node.MissionNode(config_override=config_path, autostart=True, start_delay_s=0.0)
    )
    assert node.autostart is True
    node._tick()
    assert node.started is True
    assert node.sequencer.running is True


def test_mission_autostart_waits_out_the_delay(built, config_path):
    node = built(
        lambda: mission_node.MissionNode(config_override=config_path, autostart=True, start_delay_s=30.0)
    )
    node._tick()
    assert node.started is False, "30 s have not passed"


def test_mission_republishes_the_perception_mode_every_tick(built, config_path):
    """Module 4 subscribes volatile, so a one-shot publish would be missed by a
    perception stack that starts after the sequencer - the camera manager would
    then sit at its idle rate for the whole run."""
    node = built(lambda: mission_node.MissionNode(config_override=config_path, autostart=False))

    class _Recorder:
        def __init__(self):
            self.values = []

        def publish(self, message):
            self.values.append(message.data)

    recorder = _Recorder()
    node.mode_publisher = recorder
    node._publish_mode("nav")
    node._publish_mode("nav")
    node._publish_mode("excavate")
    assert recorder.values == ["nav", "nav", "excavate"]


def test_mission_work_mode_can_be_overridden_on_the_command_line(built, config_path):
    node = built(
        lambda: mission_node.MissionNode(config_override=config_path, work_mode_override="navigate_only")
    )
    assert node.config.mission.work_mode == "navigate_only"


def test_mission_leaves_the_config_mode_alone_without_an_override(built, config_path):
    from rover_navigation.config import load_config

    node = built(lambda: mission_node.MissionNode(config_override=config_path))
    assert node.config.mission.work_mode == load_config(config_path).mission.work_mode


def test_a_bad_config_path_fails_loudly(built):
    from rover_navigation.config import NavigationConfigError

    for label, factory in (
        ("navigation", lambda: navigation_node.NavigationNode(config_override="/nonexistent/nav.yaml")),
        ("velocity_gate", lambda: velocity_gate_node.VelocityGateNode(config_override="/nonexistent/nav.yaml")),
        ("mission", lambda: mission_node.MissionNode(config_override="/nonexistent/nav.yaml")),
    ):
        with pytest.raises((NavigationConfigError, FileNotFoundError, ValueError)):
            built(factory)


# =========================================================================== #
# the goal lifecycle
# =========================================================================== #
class _FakeGoalHandle:
    """The slice of ``ServerGoalHandle`` the navigator's execute callback touches.

    It refuses a second terminal transition, which is exactly what rclpy raises
    on - so a double abort shows up here as a failed test rather than as a log
    line nobody reads during a run.
    """

    def __init__(self, goal):
        self.request = goal
        self.active = True
        self.terminal = None
        self.is_cancel_requested = False
        self.feedback = []

    def publish_feedback(self, message):
        self.feedback.append(message)

    def _finalise(self, state):
        if not self.active:
            raise RuntimeError(f"goal handle transitioned twice ({self.terminal} -> {state})")
        self.active = False
        self.terminal = state

    def abort(self):
        self._finalise("aborted")

    def succeed(self):
        self._finalise("succeeded")

    def canceled(self):
        self._finalise("cancelled")


def _goal_request(x: float, y: float, yaw: float = 0.0):
    import math

    from geometry_msgs.msg import PoseStamped
    from rover_navigation_msgs.action import NavigateTo

    goal = NavigateTo.Goal()
    goal.goal = PoseStamped()
    goal.goal.header.frame_id = "map"
    goal.goal.pose.position.x = float(x)
    goal.goal.pose.position.y = float(y)
    goal.goal.pose.orientation.z = math.sin(0.5 * yaw)
    goal.goal.pose.orientation.w = math.cos(0.5 * yaw)
    return goal


def test_a_goal_inside_a_wall_is_refused_and_finalised_once(built, config_path):
    import asyncio

    node = built(lambda: navigation_node.NavigationNode(config_override=config_path))
    node.records = []
    handle = _FakeGoalHandle(_goal_request(0.05, 0.05))  # inside the wall inset

    result = asyncio.run(node._execute(handle))

    assert result.success is False
    assert result.message.startswith("goal refused:")
    assert handle.terminal == "aborted", "a refused goal must be aborted exactly once"
    assert node.goal_rejections == 1


def test_a_newer_goal_supersedes_the_one_that_is_running(built, config_path, monkeypatch):
    import asyncio

    node = built(lambda: navigation_node.NavigationNode(config_override=config_path))
    node.records = []
    # ``rclpy.task.Future`` is only awaitable inside rclpy's own task machinery;
    # under a plain event loop it has to be an asyncio future.
    monkeypatch.setattr(navigation_node, "Future", asyncio.Future)

    async def scenario():
        first = _FakeGoalHandle(_goal_request(5.0, 2.5))
        second = _FakeGoalHandle(_goal_request(3.0, 2.5))
        first_task = asyncio.ensure_future(node._execute(first))
        await asyncio.sleep(0)  # let the first goal reach its wait
        second_task = asyncio.ensure_future(node._execute(second))
        await asyncio.sleep(0)  # let the second supersede it
        first_result = await first_task

        # The superseded goal is finished by the newer one, exactly once...
        assert first.terminal == "aborted"
        assert first_result.success is False
        assert "superseded" in first_result.message
        # ...and the newer goal is still the active one.
        assert node._goal_handle is second
        node._finish_goal("succeeded", "done")
        second_result = await second_task
        assert second_result.success is True
        assert second.terminal == "succeeded"
        return node._goal_handle

    assert asyncio.run(scenario()) is None


def test_a_goals_path_length_accumulates_across_replans(built, config_path, monkeypatch):
    from rover_navigation import nodeutil
    from rover_navigation.geometry import Pose3

    node = built(lambda: navigation_node.NavigationNode(config_override=config_path))
    node.records = []
    node._goal_handle = _FakeGoalHandle(_goal_request(6.0, 2.5))
    node._goal_pose = Pose3.from_xy_yaw(6.0, 2.5, 0.0)
    node._goal_started = nodeutil.now_seconds(node)
    node._goal_timeout = 60.0
    node._goal_tolerance = 0.15
    node._goal_yaw_tolerance = 0.30
    node._goal_max_speed = 0.45
    monkeypatch.setattr(
        type(node), "pose", property(lambda self: Pose3.from_xy_yaw(2.0, 2.5, 0.0))
    )

    node._tick()
    after_first = node._goal_path_length
    assert after_first > 0.0

    # Force the next tick to replan rather than reuse the plan it already has.
    node._last_plan_time = 0.0
    node._last_plan_obstacles = -1
    node._tick()
    assert node._goal_path_length > after_first, "a replan must add to the total, not replace it"


# =========================================================================== #
# the shims point at the same entry points
# =========================================================================== #
def test_the_scripts_forward_to_the_node_main():
    from scripts_loader import load_script

    assert load_script("navigation").main is navigation_node.main
    assert load_script("velocity_gate").main is velocity_gate_node.main
    assert load_script("mission").main is mission_node.main


@pytest.mark.parametrize("name", ("navigation", "velocity_gate", "mission"))
def test_help_exits_before_rclpy_starts(name, capsys):
    from scripts_loader import load_script

    module = load_script(name)
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0
    assert "--config" in capsys.readouterr().out
