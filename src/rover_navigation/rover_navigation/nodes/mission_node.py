"""``mission``: the run's state machine, and the only writer of ``/perception/mode``.

Started by ``mission.launch.py`` as ``rover_navigation_mission``.  It owns the
order of the run and nothing else: it never computes a wheel speed (that is the
navigator) and never decides what is safe (that is the gate).  What it does is
decide *what the rover is trying to do*, one tick at a time, and then wire that
decision to three outputs:

* a ``NavigateTo`` goal, when the sequencer wants to be somewhere;
* a ``WorkCycle`` goal, when it wants sand moved - answered by module 6;
* ``/perception/mode``, so module 4 rates its cameras for the phase the rover is
  actually in (looking for a tag, driving, digging, dumping).

Every decision is taken by :class:`~rover_navigation.mission.MissionSequencer`,
which is ROS-free and unit-tested; this node is only the wiring.  The two things
that wiring has to get right are that a nav goal and a work cycle are each sent
**once** per request (the sequencer says "go here", the node sends an action
goal, the result comes back asynchronously) and that nothing in flight is
abandoned without being cancelled.
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Bool, String

from rover_navigation_msgs.action import NavigateTo, WorkCycle
from rover_navigation_msgs.msg import MissionStatus
from rover_perception_msgs.msg import LocalizationState

from .. import nodeutil
from ..config import NavigationConfig, with_work_mode
from ..mission import PHASE_COMPLETE, MissionSequencer

__all__ = ["MissionNode", "main", "add_arguments"]

NODE_NAME = "rover_navigation_mission"

#: Consent time for a goal the navigator has not answered yet, on top of the
#: goal's own timeout.  A watchdog rather than a policy: if it fires, something
#: has gone wrong with the action layer itself.
_RESULT_WATCHDOG_S = 10.0


class MissionNode(Node):
    """Drives the sequencer from topics and actions."""

    def __init__(
        self,
        config_override: str = "",
        publish_hz_override: float = 0.0,
        autostart: Optional[bool] = None,
        start_delay_s: Optional[float] = None,
        work_mode_override: str = "",
    ) -> None:
        super().__init__(NODE_NAME)
        self.config: NavigationConfig = nodeutil.load_node_config(self, override=config_override)
        self.publish_hz = float(
            nodeutil.declare(self, "publish_hz", publish_hz_override or 5.0, "sequencer tick rate")
        )
        # The command-line flags become the *defaults* of the parameters, so an
        # explicit ``-p autostart:=...`` from a launch file still wins.
        self.autostart = bool(
            nodeutil.declare(
                self, "autostart", True if autostart is None else autostart, "start the run on launch"
            )
        )
        self.start_delay_s = float(
            nodeutil.declare(
                self, "start_delay_s", 5.0 if start_delay_s is None else start_delay_s, "delay before autostart"
            )
        )
        work_mode = str(
            nodeutil.declare(
                self,
                "work_mode_override",
                work_mode_override,
                "override mission.work_mode (empty = config)",
            )
        ).strip()
        if work_mode:
            self.config = with_work_mode(self.config, work_mode)
            self.get_logger().warn(
                f"mission.work_mode overridden on the command line to '{self.config.mission.work_mode}'"
            )

        self.sequencer = MissionSequencer(self.config)
        self.tracker = nodeutil.PoseTracker(preferred_topic=self.config.topics.odometry_topics[0], node=self)
        self.localized = False
        self.estop = False
        self.started = False
        self.last_mode = ""
        self.berm_started = False
        self._nav_handle: Any = None
        self._nav_sent_at = 0.0
        self._work_handle: Any = None
        self._work_sent_at = 0.0
        self._boot_time = nodeutil.now_seconds(self)

        self._group = ReentrantCallbackGroup()
        topics = self.config.topics
        self.status_publisher = self.create_publisher(MissionStatus, topics.mission_status, 10)
        self.mode_publisher = self.create_publisher(String, topics.perception_mode, 10)
        self.berm_publisher = self.create_publisher(Bool, topics.berm_started, nodeutil.latched_qos(1))

        self.nav_client = ActionClient(self, NavigateTo, topics.navigate_action, callback_group=self._group)
        self.work_client = ActionClient(
            self, WorkCycle, self.config.mission.excavation_action, callback_group=self._group
        )

        self.localization_subscription = self.create_subscription(
            LocalizationState, topics.localization_state, self._on_localization, nodeutil.sensor_qos(5)
        )
        self.odom_subscriptions = [
            self.create_subscription(
                Odometry, topic, partial(self._on_odometry, topic=topic), nodeutil.sensor_qos(20)
            )
            for topic in topics.odometry_topics
        ]
        self.estop_subscription = self.create_subscription(
            Bool, topics.estop, self._on_estop, nodeutil.latched_qos(1)
        )

        self.timer = self.create_timer(1.0 / max(0.5, self.publish_hz), self._tick)
        self._log = nodeutil.LogOnce(self, period_s=10.0)
        self._work_warned = False
        self.get_logger().info(
            f"{NODE_NAME} up: run {'auto-starts' if self.autostart else 'waits for a start'}, "
            f"work mode '{self.config.mission.work_mode}', "
            f"{self.config.mission.cycles_target} cycles, "
            f"navigation {topics.navigate_action}, excavation {self.config.mission.excavation_action}"
        )
        if self.config.mission.work_mode == "navigate_only":
            self.get_logger().warn(
                "mission.work_mode is 'navigate_only': every leg will be driven but no work cycle "
                "will be counted or claimed.  Set it to 'delegate' for the competition."
            )

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    def _on_localization(self, message: LocalizationState) -> None:
        self.localized = bool(message.localized)
        self.tracker.on_localization(message)

    def _on_odometry(self, message: Odometry, topic: str) -> None:
        self.tracker.on_odometry(message, topic)

    def _on_estop(self, message: Bool) -> None:
        value = bool(message.data)
        if value and not self.estop:
            self.get_logger().error("e-stop latched: the run is over until a human restarts it")
            self.sequencer.fail("e-stop latched")
            self._cancel_in_flight()
        self.estop = value

    # ------------------------------------------------------------------ #
    # the tick
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        now = nodeutil.now_seconds(self)
        if not self.started:
            if not self.autostart:
                self._publish_status(now, None)
                return
            if now - self._boot_time < self.start_delay_s:
                self._publish_status(now, None)
                return
            self.get_logger().info("starting the run")
            self.sequencer.start(now)
            self.started = True

        self._check_watchdogs(now)
        work_available = self.work_client.server_is_ready()
        if self.config.mission.work_mode == "delegate" and not work_available and not self._work_warned:
            self._work_warned = True
            self._log.warn(
                "no-work-server",
                f"no WorkCycle server on {self.config.mission.excavation_action} yet; "
                "module 6 provides it.  The sequencer will hold at the first dig.",
                now,
            )

        command = self.sequencer.tick(
            now, pose=self.tracker.pose, localized=self.localized, work_available=work_available
        )

        if command.goal is not None and self._nav_handle is None:
            self._send_goal(command.goal)
        if command.work is not None and self._work_handle is None:
            self._send_work(command.work)
        if command.finished:
            self._cancel_in_flight()
            if self.sequencer.phase == PHASE_COMPLETE:
                self.get_logger().info(f"run complete: {self.sequencer.detail}")

        self._publish_mode(command.mode)
        if command.cycles_completed >= 1 and not self.berm_started:
            # Tell the navigator to treat the berm's footprint as a keep-out from
            # now on: it is sand, so the camera will not see it as an obstacle.
            self.berm_started = True
            self.berm_publisher.publish(Bool(data=True))
            self.get_logger().info("first dump done: the berm footprint is now a keep-out")
        self._publish_status(now, command)

    def _check_watchdogs(self, now: float) -> None:
        if self._nav_handle is not None:
            limit = self.config.controller.goal_timeout_s + _RESULT_WATCHDOG_S
            if now - self._nav_sent_at > limit:
                self.get_logger().error("navigator did not answer the goal in time; dropping it")
                self._nav_handle = None
                self.sequencer.nav_finished(now, False, "no result from the navigator")
        if self._work_handle is not None:
            limit = self.config.mission.work_timeout_s + _RESULT_WATCHDOG_S
            if now - self._work_sent_at > limit:
                self.get_logger().error("work cycle did not answer in time; dropping it")
                self._work_handle = None
                self.sequencer.work_finished(now, False, "no result from the excavation server")

    # ------------------------------------------------------------------ #
    # actions
    # ------------------------------------------------------------------ #
    def _send_goal(self, waypoint: Any) -> None:
        if not self.nav_client.server_is_ready():
            self._log.warn(
                "no-navigator",
                f"no NavigateTo server on {self.config.topics.navigate_action}; "
                "is rover_navigation's navigation node running?",
            )
            self.sequencer.nav_finished(nodeutil.now_seconds(self), False, "no navigator available")
            return
        goal = NavigateTo.Goal()
        pose = PoseStamped()
        pose.header = nodeutil.make_header(self, self.config.frames.map)
        pose.pose.position.x = float(waypoint.x)
        pose.pose.position.y = float(waypoint.y)
        pose.pose.orientation.z = float(math.sin(0.5 * waypoint.yaw))
        pose.pose.orientation.w = float(math.cos(0.5 * waypoint.yaw))
        goal.goal = pose
        self._nav_sent_at = nodeutil.now_seconds(self)
        future = self.nav_client.send_goal_async(goal, feedback_callback=self._on_feedback)
        future.add_done_callback(self._on_goal_response)
        self.get_logger().info(f"sending goal: {waypoint.label or 'goal'}")

    def _on_goal_response(self, future: Any) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # pragma: no cover - transport failure
            self.sequencer.nav_finished(nodeutil.now_seconds(self), False, f"goal send failed: {exc}")
            return
        if handle is None or not handle.accepted:
            self.sequencer.nav_finished(nodeutil.now_seconds(self), False, "goal refused by the navigator")
            return
        self._nav_handle = handle
        handle.get_result_async().add_done_callback(self._on_result)

    def _on_result(self, future: Any) -> None:
        self._nav_handle = None
        try:
            wrapped = future.result()
            result = wrapped.result
            success = bool(result.success)
            message = str(result.message)
        except Exception as exc:  # pragma: no cover - transport failure
            success, message = False, f"result failed: {exc}"
        self.sequencer.nav_finished(nodeutil.now_seconds(self), success, message)

    def _on_feedback(self, feedback: Any) -> None:
        # Feedback is deliberately not acted on: the sequencer runs on the
        # navigator's *result*.  It is logged once per period so a stuck run is
        # visible in the log without a topic echo.
        self._log.info(
            f"feedback-{feedback.feedback.state}",
            f"nav feedback: state {feedback.feedback.state}, "
            f"{feedback.feedback.distance_remaining_m:.2f} m to go",
        )

    def _send_work(self, request: tuple) -> None:
        phase, x, y, yaw = request
        if not self.work_client.server_is_ready():
            # Lost a race with the server going away: say so, or the sequencer
            # would wait on a work cycle that was never sent.
            self.sequencer.work_finished(
                nodeutil.now_seconds(self), False, "no WorkCycle server available"
            )
            return
        goal = WorkCycle.Goal()
        goal.phase = str(phase)
        pose = PoseStamped()
        pose.header = nodeutil.make_header(self, self.config.frames.map)
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = float(math.sin(0.5 * yaw))
        pose.pose.orientation.w = float(math.cos(0.5 * yaw))
        goal.work_pose = pose
        goal.timeout_s = float(self.config.mission.work_timeout_s)
        goal.target_amount = 0.0
        self._work_sent_at = nodeutil.now_seconds(self)
        future = self.work_client.send_goal_async(goal, feedback_callback=self._on_work_feedback)
        future.add_done_callback(self._on_work_response)
        self.get_logger().info(f"sending work cycle: {phase} at ({x:.2f}, {y:.2f})")

    def _on_work_response(self, future: Any) -> None:
        try:
            handle = future.result()
        except Exception as exc:  # pragma: no cover - transport failure
            self.sequencer.work_finished(nodeutil.now_seconds(self), False, f"work send failed: {exc}")
            return
        if handle is None or not handle.accepted:
            self.sequencer.work_finished(nodeutil.now_seconds(self), False, "work cycle refused")
            return
        self._work_handle = handle
        handle.get_result_async().add_done_callback(self._on_work_result)

    def _on_work_result(self, future: Any) -> None:
        self._work_handle = None
        try:
            wrapped = future.result()
            result = wrapped.result
            success = bool(result.success)
            message = str(result.message)
            amount = float(result.amount_moved)
        except Exception as exc:  # pragma: no cover - transport failure
            success, message, amount = False, f"result failed: {exc}", 0.0
        self.sequencer.work_finished(nodeutil.now_seconds(self), success, message, amount)

    def _on_work_feedback(self, feedback: Any) -> None:
        self._log.info(
            f"work-feedback-{feedback.feedback.stage}",
            f"work feedback: stage {feedback.feedback.stage}, "
            f"{feedback.feedback.progress:.0%} {feedback.feedback.detail}",
        )

    def _cancel_in_flight(self) -> None:
        if self._nav_handle is not None:
            self._nav_handle.cancel_goal_async()
            self._nav_handle = None
        if self._work_handle is not None:
            self._work_handle.cancel_goal_async()
            self._work_handle = None

    # ------------------------------------------------------------------ #
    # outputs
    # ------------------------------------------------------------------ #
    def _publish_mode(self, mode: str) -> None:
        # Published every tick, not only on change: module 4 subscribes with the
        # default volatile QoS, so a rate-set once at startup would never reach a
        # perception stack that (re)started after it - and a camera manager that
        # misses "nav" leaves the forward camera at its idle rate for the whole
        # run.  Republishing is one short string at the status rate; the log line
        # is still only emitted when the mode actually changes.
        changed = mode != self.last_mode
        self.last_mode = mode
        self.mode_publisher.publish(String(data=mode))
        if changed:
            self.get_logger().info(f"/perception/mode -> '{mode}'")

    def _publish_status(self, now: float, command: Any) -> None:
        message = MissionStatus()
        message.header = nodeutil.make_header(self, self.config.frames.map)
        if command is None:
            message.phase = MissionStatus.PHASE_IDLE
            message.phase_name = "idle"
            message.running = False
            message.detail = "waiting to start" if not self.started else self.sequencer.detail
            message.cycles_target = int(self.config.mission.cycles_target)
        else:
            message.phase = int(command.phase)
            message.phase_name = str(command.phase_name)
            message.running = bool(command.running)
            message.detail = str(command.detail)
            message.cycles_completed = int(command.cycles_completed)
            message.cycles_target = int(command.cycles_target)
            message.cycle_progress = float(command.progress)
        message.excavation_action = self.config.mission.excavation_action
        message.excavation_available = bool(self.work_client.server_is_ready())
        message.perception_mode = self.last_mode
        self.status_publisher.publish(message)

    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        return {
            "started": self.started,
            "phase": self.sequencer.phase,
            "detail": self.sequencer.detail,
            "cycles_completed": self.sequencer.cycles_completed,
            "cycles_target": self.config.mission.cycles_target,
            "work_mode": self.config.mission.work_mode,
            "work_available": self.work_client.server_is_ready(),
            "nav_in_flight": self._nav_handle is not None,
            "work_in_flight": self._work_handle is not None,
            "localized": self.localized,
            "estop": self.estop,
            "mode": self.last_mode,
            "berm_started": self.berm_started,
            "dig_poses": len(self.sequencer.dig_poses),
            "dump_poses": len(self.sequencer.dump_poses),
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publish-hz", type=float, default=0.0, help="sequencer tick rate (0 = 5 Hz)")
    parser.add_argument(
        "--autostart", dest="autostart", action="store_true", default=None, help="start the run on launch"
    )
    parser.add_argument(
        "--no-autostart", dest="autostart", action="store_false", help="do not start the run on launch"
    )
    parser.add_argument("--start-delay", type=float, default=None, help="seconds before autostart")
    parser.add_argument(
        "--work-mode", default="", choices=["", "delegate", "navigate_only"], help="override the config"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    def build(args: argparse.Namespace) -> MissionNode:
        return MissionNode(
            publish_hz_override=args.publish_hz,
            autostart=args.autostart,
            start_delay_s=args.start_delay,
            work_mode_override=args.work_mode,
        )

    return nodeutil.node_main(
        argv,
        "module 5 mission sequencer: acquire, excavate, haul, dump, park",
        build=build,
        add_arguments=add_arguments,
        epilog=(
            "Drives /perception/mode and the NavigateTo action, and delegates work cycles to "
            "module 6 over WorkCycle.  With mission.work_mode 'navigate_only' it drives every "
            "leg without claiming any sand moved."
        ),
    )
