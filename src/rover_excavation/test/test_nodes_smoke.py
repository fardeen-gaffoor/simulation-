"""The node: it imports, it parses its flags, and it can be built and ticked.

The node is a thin ROS wrapper around the tested libraries, so this file does not
re-test the algorithms.  What it *does* pin is the glue that only shows up when
the whole thing is started:

* the module exposes ``NODE_NAME`` / ``add_arguments`` / ``main``;
* the constructor honours ``--config`` rather than silently using the installed
  default;
* the node builds its publishers, subscriptions, action server and timer without
  a live DDS peer, and one ``_tick()`` runs to completion with nothing connected -
  which is exactly the state it is in for the first seconds of a launch;
* the ``WorkCycle`` goal lifecycle finalises each goal exactly once, refuses a
  bad goal, supersedes an old one and honours a cancel.

Time is faked throughout: the node reads its clock through
``nodeutil.now_seconds``, so a test can run an eleven-second work cycle in a
millisecond and still exercise the real slew-limited mechanism.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys

import pytest

from rover_excavation.nodes import excavation_node


# =========================================================================== #
# module shape
# =========================================================================== #
def test_the_module_declares_its_node_name():
    assert isinstance(excavation_node.NODE_NAME, str)
    assert excavation_node.NODE_NAME == "rover_excavation_mechanism"


def test_the_module_exports_the_entry_points():
    assert callable(excavation_node.main)
    assert callable(excavation_node.add_arguments)
    assert isinstance(excavation_node.ExcavationNode, type)


def test_the_node_class_is_importable_by_name():
    module = importlib.import_module("rover_excavation.nodes.excavation_node")
    assert hasattr(module, "ExcavationNode")


# =========================================================================== #
# flags
# =========================================================================== #
def _flags():
    parser = argparse.ArgumentParser(prog="node")
    parser.add_argument("--config", default="")
    excavation_node.add_arguments(parser)
    return parser


def test_the_shared_config_flag_exists_with_the_default():
    assert _flags().parse_args([]).config == ""


def test_the_backend_and_rate_flags_parse():
    args = _flags().parse_args(["--backend", "null", "--rate-hz", "20.0"])
    assert args.backend == "null"
    assert args.rate_hz == pytest.approx(20.0)


def test_the_gz_berm_flag_is_tri_state():
    parser = _flags()
    assert parser.parse_args([]).gz_berm is None, "no flag must leave the parameter alone"
    assert parser.parse_args(["--gz-berm"]).gz_berm is True
    assert parser.parse_args(["--no-gz-berm"]).gz_berm is False


def test_the_argument_split_keeps_ros_args_for_rclpy():
    from rover_excavation import nodeutil

    ros, own = nodeutil.split_node_arguments(
        ["--config", "/x.yaml", "--ros-args", "-r", "__node:=n"]
    )
    assert own == ["--config", "/x.yaml"]
    assert ros == ["--ros-args", "-r", "__node:=n"]
    ros, own = nodeutil.split_node_arguments(["--backend", "null"])
    assert ros is None
    assert own == ["--backend", "null"]


def test_sys_argv_carrying_ros_args_is_split_too(monkeypatch):
    from rover_excavation import nodeutil

    monkeypatch.setattr(sys, "argv", ["excavation.py", "--ros-args", "-r", "__node:=digger"])
    ros, own = nodeutil.split_node_arguments(None)
    assert own == []
    assert ros == ["--ros-args", "-r", "__node:=digger"]


# =========================================================================== #
# a fake clock
# =========================================================================== #
class _Clock:
    """A clock the node believes, so a whole work cycle runs in a millisecond."""

    def __init__(self, start: float = 0.0):
        self.t = float(start)

    def now(self, node=None) -> float:
        return self.t

    def advance(self, dt: float = 0.05) -> None:
        self.t += float(dt)


@pytest.fixture()
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(excavation_node.nodeutil, "now_seconds", fake.now)
    return fake


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


def _node(built, config_path, **kwargs):
    kwargs.setdefault("backend_override", "null")
    kwargs.setdefault("gz_berm_override", False)
    return built(lambda: excavation_node.ExcavationNode(config_override=config_path, **kwargs))


def _drive(node, clock, limit_s: float = 60.0, dt: float = 0.05):
    """Tick the node until its work cycle finishes (or the budget runs out)."""
    guard = int(limit_s / dt) + 10
    while node.cycle is not None and guard > 0:
        clock.advance(dt)
        node._tick()
        guard -= 1
    return node.cycle is None


# =========================================================================== #
# construction and a bare tick
# =========================================================================== #
def test_the_node_builds(built, config_path):
    node = _node(built, config_path)
    assert node.config.path == config_path, "the --config override is honoured"
    assert node.rate_hz == pytest.approx(node.config.mechanism.rate_hz)
    assert node.io.describe().startswith("null")
    assert node.cycle is None
    assert node.berm.placed_volume_m3 == 0.0
    assert node.tub.is_empty
    assert node.mechanism.travel_ready


def test_the_node_ticks_with_nothing_connected(built, config_path, clock):
    node = _node(built, config_path)
    node._tick()
    assert node.cycle is None
    assert node.mechanism.travel_ready
    snapshot = node.snapshot_dict()
    assert snapshot["cycle"] is None
    assert snapshot["mechanism"]["state_name"] == "STOWED"
    assert snapshot["berm"]["dumps"] == 0


def test_the_node_publishes_status_and_berm(built, config_path, clock):
    node = _node(built, config_path)

    class _Recorder:
        def __init__(self):
            self.messages = []

        def publish(self, message):
            self.messages.append(message)

    status = _Recorder()
    berm = _Recorder()
    node.status_publisher = status
    node.berm_publisher = berm
    node._tick()
    assert len(status.messages) == 1
    assert len(berm.messages) == 1
    message = status.messages[0]
    assert message.state_name == "STOWED"
    assert message.travel_ready is True
    assert message.tub_capacity_m3 == pytest.approx(node.config.mechanism.tub.capacity_m3)
    assert berm.messages[0].target_volume_m3 == pytest.approx(node.config.berm_target_volume_m3)


def test_the_rate_can_be_overridden(built, config_path):
    node = _node(built, config_path, rate_hz_override=25.0)
    assert node.rate_hz == pytest.approx(25.0)


def test_the_backend_can_be_overridden(built, config_path):
    node = _node(built, config_path, backend_override="sim")
    assert node.io.describe().startswith("sim")


def test_a_bad_backend_override_is_refused(config_path):
    with pytest.raises(SystemExit):
        excavation_node.ExcavationNode(config_override=config_path, backend_override="telepathy")


def test_gz_berm_can_be_turned_off(built, config_path):
    node = _node(built, config_path)
    assert node.visual.enabled is False


def test_a_bad_config_path_fails_loudly(built):
    from rover_excavation.config import ExcavationConfigError

    with pytest.raises((ExcavationConfigError, FileNotFoundError, ValueError)):
        built(lambda: excavation_node.ExcavationNode(config_override="/nonexistent/dig.yaml"))


def test_sim_backend_publishes_the_two_joint_targets(built, config_path, clock):
    node = _node(built, config_path, backend_override="sim")

    class _Recorder:
        def __init__(self):
            self.values = []

        def publish(self, message):
            self.values.append(message.data)

    ladder = _Recorder()
    belt = _Recorder()
    node.ladder_publisher = ladder
    node.belt_publisher = belt
    node._tick()
    assert ladder.values == [pytest.approx(node.mechanism.pitch_rad)]
    assert belt.values == [pytest.approx(node.mechanism.belt_velocity_radps)]


# =========================================================================== #
# the e-stop
# =========================================================================== #
def test_the_estop_stops_the_commands(built, config_path, clock):
    node = _node(built, config_path, backend_override="sim")

    class _Recorder:
        def __init__(self):
            self.values = []

        def publish(self, message):
            self.values.append(message.data)

    ladder = _Recorder()
    node.ladder_publisher = ladder
    node._tick()
    assert ladder.values, "the mechanism is driven when it is not e-stopped"
    node._on_estop(type("M", (), {"data": True})())
    node._tick()
    assert ladder.values == [pytest.approx(node.mechanism.pitch_rad)], (
        "no actuator command may be published while the e-stop is latched"
    )
    assert node.estop is True
    # The model still stows, so what it reports stays truthful.
    assert node.mechanism.target.ladder_pitch_rad == pytest.approx(
        node.config.mechanism.ladder.stowed_pitch_rad
    )


def test_the_estop_reports_stowed_not_a_fault(built, config_path, clock):
    node = _node(built, config_path)
    node._on_estop(type("M", (), {"data": True})())
    node._tick()
    assert node.mechanism.snapshot().state_name == "STOWED"
    node._on_estop(type("M", (), {"data": False})())
    assert node.estop is False


# =========================================================================== #
# the joint tracking check
# =========================================================================== #
def test_a_disagreeing_joint_is_warned_about_once(built, config_path, clock):
    from sensor_msgs.msg import JointState

    node = _node(built, config_path)
    message = JointState()
    message.name = [node.config.mechanism.ladder.joint]
    message.position = [0.9]  # wildly different from the stowed -0.44
    node._on_joint_states(message)
    assert node._joint_pitch == pytest.approx(0.9)
    for _ in range(200):
        clock.advance(0.05)
        node._tick()
    # LogOnce only emits a key once, so the key is the evidence that it fired.
    assert "joint-mismatch" in node._log._seen


def test_joint_states_without_the_ladder_joint_are_ignored(built, config_path):
    from sensor_msgs.msg import JointState

    node = _node(built, config_path)
    message = JointState()
    message.name = ["wheel_front_left_joint"]
    message.position = [0.0]
    node._on_joint_states(message)
    assert node._joint_pitch is None


# =========================================================================== #
# the goal lifecycle
# =========================================================================== #
class _FakeGoalHandle:
    """The slice of ``ServerGoalHandle`` the execute callback touches.

    It refuses a second terminal transition, which is what rclpy raises on - so a
    double abort shows up here as a failed test rather than as a log line nobody
    reads during a run.
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


def _goal(phase: str = "excavate", x: float = 8.0, y: float = 2.5, yaw: float = 1.570796, timeout_s: float = 0.0):
    import math

    from geometry_msgs.msg import PoseStamped
    from rover_navigation_msgs.action import WorkCycle

    goal = WorkCycle.Goal()
    goal.phase = phase
    goal.work_pose = PoseStamped()
    goal.work_pose.header.frame_id = "map"
    goal.work_pose.pose.position.x = float(x)
    goal.work_pose.pose.position.y = float(y)
    goal.work_pose.pose.orientation.z = math.sin(0.5 * yaw)
    goal.work_pose.pose.orientation.w = math.cos(0.5 * yaw)
    goal.timeout_s = float(timeout_s)
    goal.target_amount = 0.0
    return goal


def test_an_unknown_phase_is_refused_and_finalised_once(built, config_path):
    node = _node(built, config_path)
    handle = _FakeGoalHandle(_goal(phase="teleport"))
    result = asyncio.run(node._execute(handle))
    assert result.success is False
    assert "unknown phase" in result.message
    assert handle.terminal == "aborted"
    assert node.cycle is None


def test_a_goal_during_an_estop_is_refused(built, config_path):
    node = _node(built, config_path)
    node.estop = True
    handle = _FakeGoalHandle(_goal())
    result = asyncio.run(node._execute(handle))
    assert result.success is False
    assert "e-stop" in result.message
    assert handle.terminal == "aborted"


def test_a_goal_during_a_fault_is_refused(built, config_path):
    node = _node(built, config_path)
    node.mechanism.raise_fault("injected")
    handle = _FakeGoalHandle(_goal())
    result = asyncio.run(node._execute(handle))
    assert result.success is False
    assert "fault" in result.message
    assert handle.terminal == "aborted"


def test_a_full_dig_cycle_succeeds_through_the_action(built, config_path, clock, monkeypatch):
    node = _node(built, config_path)
    monkeypatch.setattr(excavation_node, "Future", asyncio.Future)
    handle = _FakeGoalHandle(_goal(phase="excavate"))

    async def scenario():
        task = asyncio.ensure_future(node._execute(handle))
        await asyncio.sleep(0)
        assert node.cycle is not None
        assert node.cycle.phase == "excavate"
        assert _drive(node, clock), "the dig cycle did not finish"
        await asyncio.sleep(0)
        return await task

    result = asyncio.run(scenario())
    assert result.success is True
    assert result.amount_moved > 0.0
    assert result.elapsed_s > 0.0
    assert handle.terminal == "succeeded"
    assert node.tub.fraction >= node.config.cycle.dig.target_fill
    assert node.mechanism.travel_ready
    assert node.cycles_served == 1


def test_a_running_cycle_publishes_feedback_for_the_sequencer(built, config_path, clock, monkeypatch):
    """Module 5's sequencer logs ``WorkCycle`` feedback; without it a run is blind."""
    node = _node(built, config_path)
    monkeypatch.setattr(excavation_node, "Future", asyncio.Future)
    handle = _FakeGoalHandle(_goal(phase="excavate"))

    async def scenario():
        task = asyncio.ensure_future(node._execute(handle))
        await asyncio.sleep(0)
        _drive(node, clock)
        await asyncio.sleep(0)
        return await task

    result = asyncio.run(scenario())
    assert result.success is True
    assert handle.feedback, "no feedback was published during the cycle"
    stages = {message.stage for message in handle.feedback}
    assert {1, 2, 3} <= stages, f"the whole cycle should be visible, got stages {sorted(stages)}"
    assert all(0.0 <= message.progress <= 1.0 for message in handle.feedback)
    assert max(message.progress for message in handle.feedback) > 0.5
    assert all(message.detail for message in handle.feedback)


def test_a_dig_then_a_dump_builds_the_berm_through_the_action(built, config_path, clock, monkeypatch):
    node = _node(built, config_path)
    monkeypatch.setattr(excavation_node, "Future", asyncio.Future)

    def run(handle):
        async def scenario():
            task = asyncio.ensure_future(node._execute(handle))
            await asyncio.sleep(0)
            _drive(node, clock)
            await asyncio.sleep(0)
            return await task

        return asyncio.run(scenario())

    dig = run(_FakeGoalHandle(_goal(phase="excavate")))
    assert dig.success is True
    dump = run(_FakeGoalHandle(_goal(phase="dump", x=1.0, y=node.config.berm_dump_pose_y, yaw=1.570796)))
    assert dump.success is True
    assert dump.amount_moved > 0.0
    assert node.berm.dumps == 1
    assert node.berm.placed_volume_m3 > 0.0
    assert node.berm.spilled_volume_m3 >= 0.0


def test_a_newer_goal_supersedes_the_one_that_is_running(built, config_path, clock, monkeypatch):
    node = _node(built, config_path)
    monkeypatch.setattr(excavation_node, "Future", asyncio.Future)

    async def scenario():
        first = _FakeGoalHandle(_goal(phase="excavate"))
        second = _FakeGoalHandle(_goal(phase="dump"))
        first_task = asyncio.ensure_future(node._execute(first))
        await asyncio.sleep(0)
        second_task = asyncio.ensure_future(node._execute(second))
        await asyncio.sleep(0)
        first_result = await first_task
        # The superseded goal is finished exactly once, by the newer one...
        assert first.terminal == "aborted"
        assert first_result.success is False
        assert "superseded" in first_result.message
        # ... and the newer goal owns the mechanism now.
        assert node.cycle is not None
        assert node.cycle.phase == "dump"
        _drive(node, clock)
        await asyncio.sleep(0)
        second_result = await second_task
        assert second.terminal == "succeeded", second_result.message
        assert node.cycle is None

    asyncio.run(scenario())


def test_a_cancel_stops_the_cycle_and_stows(built, config_path, clock, monkeypatch):
    node = _node(built, config_path)
    monkeypatch.setattr(excavation_node, "Future", asyncio.Future)
    handle = _FakeGoalHandle(_goal(phase="excavate"))

    async def scenario():
        task = asyncio.ensure_future(node._execute(handle))
        await asyncio.sleep(0)
        assert node.cycle is not None
        # A cancel arrives from the executor while the goal is waiting.
        response = node._on_cancel(handle)
        await asyncio.sleep(0)
        result = await task
        return response, result

    from rclpy.action import CancelResponse

    response, result = asyncio.run(scenario())
    assert response == CancelResponse.ACCEPT
    assert handle.terminal == "cancelled"
    assert result.success is False
    assert "canceled" in result.message
    assert node.cycle is None
    assert node.mechanism.target.ladder_pitch_rad == pytest.approx(
        node.config.mechanism.ladder.stowed_pitch_rad
    )


def test_a_cancel_with_no_cycle_is_harmless(built, config_path):
    from rclpy.action import CancelResponse

    node = _node(built, config_path)
    assert node._on_cancel(_FakeGoalHandle(_goal())) == CancelResponse.ACCEPT
    assert node.cycle is None


def test_an_estop_during_a_cycle_fails_it(built, config_path, clock, monkeypatch):
    node = _node(built, config_path)
    monkeypatch.setattr(excavation_node, "Future", asyncio.Future)
    handle = _FakeGoalHandle(_goal(phase="excavate"))

    async def scenario():
        task = asyncio.ensure_future(node._execute(handle))
        await asyncio.sleep(0)
        clock.advance(0.5)
        node._tick()
        node._on_estop(type("M", (), {"data": True})())
        await asyncio.sleep(0)
        return await task

    result = asyncio.run(scenario())
    assert result.success is False
    assert "e-stop" in result.message
    assert node.cycle is None


def test_the_perception_mode_is_recorded_for_logging(built, config_path):
    node = _node(built, config_path)
    node._on_mode(type("M", (), {"data": "excavate"})())
    assert node.perception_mode == "excavate"


def test_the_marker_is_only_republished_when_the_berm_changes(built, config_path, clock):
    node = _node(built, config_path)
    node.publish_marker = True

    class _Recorder:
        def __init__(self):
            self.count = 0

        def publish(self, message):
            self.count += 1
            self.last = message

    recorder = _Recorder()
    node.marker_publisher = recorder
    clock.advance(0.05)
    node._tick()
    assert recorder.count == 1
    clock.advance(0.05)
    node._tick()
    assert recorder.count == 1, "nothing changed, so nothing to redraw"
    # ... but a late rviz still gets one every couple of seconds.  The tick is
    # clamped, so this takes a few of them.
    for _ in range(6):
        clock.advance(0.5)
        node._tick()
    assert recorder.count == 2
    assert len(recorder.last.markers) == node.berm.cells_total


def test_marker_publishing_can_be_turned_off(built, config_path, clock):
    node = _node(built, config_path)
    node.publish_marker = False
    node.marker_publisher = None
    node._tick()  # must not raise


def test_a_backend_send_failure_faults_the_mechanism(built, config_path, clock):
    node = _node(built, config_path)

    def _boom(command):
        raise RuntimeError("serial link went away")

    node.io.send = _boom
    node._tick()
    assert node.mechanism.fault is not None
    assert "serial link went away" in node.mechanism.fault


def test_the_gz_berm_is_not_touched_when_disabled(built, config_path, clock):
    node = _node(built, config_path)
    assert node.visual.enabled is False
    node._update_gz_berm()  # must not raise and must not shell out


def test_the_gz_visual_warns_once_when_gz_is_missing(built, config_path):
    from rover_excavation.backend import GzBermVisual

    node = _node(built, config_path, gz_berm_override=True)
    target = node.config.berm_target
    node.visual = GzBermVisual(
        node.config.visual,
        (target.x0, target.y0, target.x1, target.y1),
        node.config.sand.berm.target_height_m,
        run=lambda command: 1,  # a machine with no gz on it
    )
    node._update_gz_berm()
    assert node.visual.enabled is False
    assert node.visual.note
    assert node._visual_warned is True
    # It does not complain again on every tick.
    node._update_gz_berm()
    assert node._visual_warned is True


# =========================================================================== #
# the shim points at the same entry point
# =========================================================================== #
def test_the_script_forwards_to_the_node_main():
    from scripts_loader import load_script

    assert load_script("excavation").main is excavation_node.main


def test_help_exits_before_rclpy_starts(capsys):
    from scripts_loader import load_script

    module = load_script("excavation")
    with pytest.raises(SystemExit) as excinfo:
        module.main(["--help"])
    assert excinfo.value.code == 0
    assert "--config" in capsys.readouterr().out
