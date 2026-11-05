"""``excavation``: the ``WorkCycle`` server, manual dig/dump, the berm.

Started by ``excavation.launch.py`` as ``rover_excavation_mechanism``.  It is the
only rclpy process in module 6, and it is pure wiring:

* it *serves* the ``WorkCycle`` action module 5 calls;
* it accepts manual commands on ``/excavation/manual`` (``dig``/``dump``/
  ``stop``/``reset``) so a human can run the mechanism from the teleop (``V``
  digs, ``B`` dumps) without a mission sequencer;
* it owns one :class:`~rover_excavation.mechanism.Mechanism`, one
  :class:`~rover_excavation.sand.Tub` and one
  :class:`~rover_excavation.sand.BermField`, i.e. the physical state of the
  digger and of the berm;
* once per tick it steps the mechanism, advances the active
  :class:`~rover_excavation.cycle.WorkCycle`, sends the actuator targets through
  the configured backend and publishes the two status messages;
* it stops everything, and lets the ladder stow, when ``/estop`` is latched;
* it keeps ``MechanismStatus.travel_ready`` truthful, because that is the one
  fact module 5 cannot derive for itself.

It never sends a wheel command, never reads ground truth and never decides where
the rover should be: module 5 owns all three.
"""

from __future__ import annotations

import math
import subprocess
import sys
from typing import Any, Dict, Optional

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String
from visualization_msgs.msg import MarkerArray

from rover_excavation_msgs.msg import BermStatus, MechanismStatus
from rover_navigation_msgs.action import WorkCycle

from .. import nodeutil
from ..backend import GzBermVisual, GzParticles, MechanismIO, MechanismIOError, build_io
from ..config import WORK_PHASES, ExcavationConfig
from ..cycle import STAGE_DONE, WorkCycle as WorkCycleRunner
from ..mechanism import STATE_DIGGING, STATE_DUMPING, Mechanism
from ..sand import BermField, Tub

__all__ = ["ExcavationNode", "main", "add_arguments"]

NODE_NAME = "rover_excavation_mechanism"

#: A tick longer than this is treated as this long.  A paused simulator must not
#: let the mechanism slew through half its travel in one step.
_MAX_TICK_S = 0.5

#: How often the berm marker is republished even when nothing has changed, so a
#: late rviz still gets one.
_MARKER_REFRESH_S = 2.0


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Planar heading from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ExcavationNode(Node):
    """Serves ``WorkCycle`` and runs the mechanism."""

    def __init__(
        self,
        config_override: str = "",
        backend_override: str = "",
        rate_hz_override: float = 0.0,
        gz_berm_override: Optional[bool] = None,
        start_stowed: bool = True,
    ) -> None:
        super().__init__(NODE_NAME)
        self.config: ExcavationConfig = nodeutil.load_node_config(self, override=config_override)

        backend_kind = str(
            nodeutil.declare(self, "backend", backend_override, "override backend.kind (empty = config)")
        ).strip()
        if backend_kind:
            self.config = _with_backend_kind(self.config, backend_kind)
            self.get_logger().warn(f"backend.kind overridden on the command line to '{backend_kind}'")

        rate_hz = float(
            nodeutil.declare(self, "rate_hz", rate_hz_override or self.config.mechanism.rate_hz, "control loop rate")
        )
        self.rate_hz = max(1.0, rate_hz)
        if gz_berm_override is not None:
            self.config = _with_gz_berm(self.config, bool(gz_berm_override))
        self.publish_marker = bool(
            nodeutil.declare(self, "publish_marker", self.config.visual.publish_marker, "publish the berm marker")
        )
        # Manual work commands (V = dig, B = dump, N = stop in the teleop).
        # The topic is a std_msgs/String: "dig"/"dump"/"stop"/"reset".
        self.manual_topic = str(
            nodeutil.declare(self, "manual_topic", "/excavation/manual", "manual dig/dump command topic")
        )
        self.odom_topic = str(nodeutil.declare(self, "odom_topic", "/odom", "wheel odometry topic"))
        self.initial_pose_topic = str(
            nodeutil.declare(
                self, "initial_pose_topic", "/cat_arena/initial_pose", "latched world start pose topic"
            )
        )

        # -- state ---------------------------------------------------------- #
        self.mechanism = Mechanism(self.config.mechanism)
        self.tub = Tub(self.config.mechanism.tub)
        self.berm = BermField(self.config)
        self.estop = False
        self.perception_mode = ""
        self.cycle: Optional[WorkCycleRunner] = None
        self.cycles_served = 0

        self._cycle_seq = 0
        self._goal_handle: Optional[Any] = None
        self._finish: Optional[Future] = None
        self._outcome: Optional[tuple] = None
        self._cancel_requested = False
        self._last_tick = 0.0
        self._marker_age = 0.0
        self._berm_dumps = -1
        self._joint_pitch: Optional[float] = None
        self._joint_mismatch_since: Optional[float] = None
        self._log = nodeutil.LogOnce(self, period_s=10.0)

        # -- world pose for manual dumping ---------------------------------- #
        # /odom starts at the spawn pose, so the world pose is the latched
        # /cat_arena/initial_pose composed with the odometry since spawn.  It
        # is only used to decide where a manually triggered dump lands; the
        # mechanism itself never needs it.
        self._init_pose: Optional[tuple] = None
        self._odom_pose = (0.0, 0.0, 0.0)
        self._have_odom = False
        self._manual_log_age = 0.0

        # -- publications --------------------------------------------------- #
        topics = self.config.topics
        self.status_publisher = self.create_publisher(MechanismStatus, topics.status, 10)
        self.berm_publisher = self.create_publisher(BermStatus, topics.berm, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, self.config.visual.marker_topic, 1)
        self.ladder_publisher = self.create_publisher(Float64, self.config.backend.ladder_cmd_topic, 10)
        self.belt_publisher = self.create_publisher(Float64, self.config.backend.belt_cmd_topic, 10)

        # -- subscriptions -------------------------------------------------- #
        self.estop_subscription = self.create_subscription(
            Bool, topics.estop, self._on_estop, nodeutil.latched_qos(1)
        )
        self.mode_subscription = self.create_subscription(
            String, topics.perception_mode, self._on_mode, nodeutil.sensor_qos(5)
        )
        self.joint_subscription = self.create_subscription(
            JointState, self.config.backend.joint_states_topic, self._on_joint_states, nodeutil.sensor_qos(10)
        )
        self.manual_subscription = self.create_subscription(
            String, self.manual_topic, self._on_manual, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, nodeutil.sensor_qos(10)
        )
        self.initial_pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self._on_initial_pose,
            nodeutil.latched_qos(1),
        )

        # -- actions -------------------------------------------------------- #
        self._group = ReentrantCallbackGroup()
        self.action_server = ActionServer(
            self,
            WorkCycle,
            topics.work_action,
            execute_callback=self._execute,
            cancel_callback=self._on_cancel,
            callback_group=self._group,
        )

        # -- backend + visual ----------------------------------------------- #
        self.io: MechanismIO = build_io(self.config.backend, publish=self._publish_float)
        try:
            self.io.start()
        except MechanismIOError as exc:
            self.get_logger().error(f"mechanism backend failed to start: {exc}")
            raise
        target = self.config.berm_target
        self.visual = GzBermVisual(
            self.config.visual,
            (target.x0, target.y0, target.x1, target.y1),
            self.config.sand.berm.target_height_m,
            run=self._run_gz,
        )
        # The rover's dig/dump dust emitters live in the robot SDF; this is the
        # only thing that turns them on and off.
        self.gz_robot_model = str(
            nodeutil.declare(self, "gz_robot_model", "rover", "spawned rover model name for particle effects")
        )
        self.particles = GzParticles(
            self.gz_robot_model,
            (("ladder_link", "dig_dust"), ("base_link", "dump_sand")),
            run=self._run_gz,
        )
        if start_stowed:
            self.mechanism.stow()

        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f"{NODE_NAME} up: backend {self.io.describe()}, {self.rate_hz:g} Hz, "
            f"WorkCycle on {topics.work_action}, berm target "
            f"{target.width:g}x{target.height:g} m x {self.config.sand.berm.target_height_m:g} m "
            f"({self.config.berm_target_volume_m3 * 1000.0:.1f} L), "
            f"tub {self.config.mechanism.tub.capacity_m3 * 1000.0:.1f} L, "
            f"gz berm {'on' if self.visual.enabled else 'off'}, "
            f"manual commands on {self.manual_topic}"
        )

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    def _on_estop(self, message: Bool) -> None:
        value = bool(message.data)
        if value and not self.estop:
            self.get_logger().error("e-stop latched: stopping the mechanism and stowing")
        elif self.estop and not value:
            self.get_logger().warn("e-stop released; the mechanism is stowed and idle")
        self.estop = value
        if value and self.cycle is not None:
            self._stop_cycle("e-stop latched during the work cycle")

    def _on_mode(self, message: String) -> None:
        self.perception_mode = str(message.data)

    def _on_joint_states(self, message: JointState) -> None:
        """Track the *physical* ladder pitch, to catch a broken sim bridge.

        The model is authoritative for the sand, but if the Gazebo joint does not
        follow the command the rover is not actually digging, and it is better to
        hear about it once than never.
        """
        names = list(message.name)
        joint = self.config.mechanism.ladder.joint
        if joint not in names:
            return
        self._joint_pitch = float(message.position[names.index(joint)])

    def _on_odom(self, message: Odometry) -> None:
        pose = message.pose.pose
        self._odom_pose = (
            float(pose.position.x),
            float(pose.position.y),
            _yaw_from_quaternion(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        )
        self._have_odom = True

    def _on_initial_pose(self, message: PoseWithCovarianceStamped) -> None:
        pose = message.pose.pose
        self._init_pose = (
            float(pose.position.x),
            float(pose.position.y),
            _yaw_from_quaternion(
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        )

    def _world_pose(self) -> tuple:
        """Rover pose in the arena/map frame, for the dump landing point."""
        ox, oy, oyaw = self._odom_pose
        if self._init_pose is None:
            return (ox, oy, oyaw)
        ix, iy, iyaw = self._init_pose
        c, s = math.cos(iyaw), math.sin(iyaw)
        return (ix + c * ox - s * oy, iy + s * ox + c * oy, _wrap_angle(iyaw + oyaw))

    def _on_manual(self, message: String) -> None:
        """Manual dig/dump/stop from the teleop (``V`` / ``B`` / ``N``).

        ``dig`` and ``dump`` are toggles: pressing the same key again stops the
        running work cycle.  Pressing the other key switches to it.  ``stop``
        stops and stows; ``reset`` also clears a mechanism fault.
        """
        command = str(message.data).strip().lower()
        now = nodeutil.now_seconds(self)
        if command in ("dig", "excavate", "v"):
            self._manual_cycle("excavate", now)
        elif command in ("dump", "drop", "b"):
            self._manual_cycle("dump", now)
        elif command in ("stop", "stow", "n"):
            self._manual_stop()
        elif command in ("reset", "clear"):
            self._manual_reset()
        else:
            self.get_logger().warn(
                f"unknown manual command {message.data!r}; use dig, dump, stop or reset"
            )

    def _manual_cycle(self, phase: str, now: float) -> None:
        if self.estop:
            self.get_logger().warn(f"manual {phase} refused: e-stop is latched")
            return
        if self.cycle is not None and self.cycle.phase == phase:
            self._manual_stop()
            return
        if self.mechanism.fault is not None:
            self.get_logger().warn(
                f"manual {phase} refused: {self.mechanism.fault} (send 'reset' to clear)"
            )
            return
        if self.cycle is not None:
            self.get_logger().warn(f"manual {phase} supersedes the running '{self.cycle.phase}' cycle")
            self._stop_cycle("superseded by a manual command")

        pose = self._world_pose()
        if phase == "dump" and self.tub.is_empty:
            self.get_logger().warn("manual dump: the tub is empty, so only the mechanism will move")
        if self._init_pose is None:
            self._log.warn(
                "manual-pose",
                "world start pose not received yet; the dump landing point uses odometry only",
                now,
            )

        self.cycle = WorkCycleRunner(
            self.config,
            phase,
            pose,
            self.mechanism,
            self.tub,
            self.berm,
            cycle_id=self.cycles_served,
            timeout_s=0.0,
            target_amount=0.0,
            now=now,
        )
        self._manual_log_age = 0.0
        self.get_logger().info(
            f"manual {phase} started at ({pose[0]:.2f}, {pose[1]:.2f}, yaw {pose[2]:.2f}); "
            f"tub {self.tub.fraction * 100.0:.0f}% full"
        )

    def _manual_stop(self) -> None:
        if self.cycle is not None:
            self._stop_cycle("stopped manually")
        else:
            self.mechanism.stow()
        self.get_logger().info("manual stop: mechanism stowing, belt off")

    def _manual_reset(self) -> None:
        if self.cycle is not None:
            self._stop_cycle("reset")
        self.mechanism.clear_fault()
        self.mechanism.stow()
        self.get_logger().info("manual reset: fault cleared, mechanism stowing")

    # ------------------------------------------------------------------ #
    # the control loop
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        now = nodeutil.now_seconds(self)
        dt = 0.0 if self._last_tick <= 0.0 else max(0.0, min(_MAX_TICK_S, now - self._last_tick))
        self._last_tick = now

        if self.estop:
            self.mechanism.stow()
        self.mechanism.step(dt)

        if self.cycle is not None and self.cycle.stage != STAGE_DONE:
            self.cycle.step(now)
            self._publish_work_feedback()
            self._log_manual(dt)
            if self.cycle.done:
                self._complete_cycle()
        elif self.cycle is None:
            self._idle_state()

        self._send_command()
        self._check_joint_tracking(now)
        self._publish_status(now)
        if self.publish_marker:
            self._publish_marker(dt)
        self._update_gz_berm()
        self.particles.set_all(
            (self.mechanism.state == STATE_DIGGING, self.mechanism.state == STATE_DUMPING)
        )

    def _idle_state(self) -> None:
        self.mechanism.settle_idle()

    def _log_manual(self, dt: float) -> None:
        """A once-a-second progress line for a manually triggered cycle."""
        if self.cycle is None or self._goal_handle is not None:
            return
        self._manual_log_age += dt
        if self._manual_log_age < 1.0:
            return
        self._manual_log_age = 0.0
        self.get_logger().info(f"manual {self.cycle.phase}: {self.cycle.stage_name} - {self.cycle.detail}")

    def _send_command(self) -> None:
        target = self.mechanism.target
        if self.estop:
            # Never drive an actuator while the e-stop is latched; the model
            # still stows so the reported state stays truthful.
            return
        try:
            self.io.send(target)
        except Exception as exc:  # pragma: no cover - backend failure
            self.mechanism.raise_fault(f"backend send failed: {exc}")
            self._log.error("backend-send", f"mechanism backend send failed: {exc}")

    def _publish_work_feedback(self) -> None:
        """One ``WorkCycle`` feedback message per tick while a cycle is running.

        Module 5's sequencer logs it and a human watching a run reads it, so it
        is the only live view into the dig: which stage the mechanism is in and
        how far through the cycle it is.
        """
        cycle = self.cycle
        handle = self._goal_handle
        if cycle is None or handle is None:
            return
        feedback = WorkCycle.Feedback()
        feedback.stage = int(cycle.stage)
        feedback.progress = float(cycle.progress)
        feedback.detail = cycle.detail
        handle.publish_feedback(feedback)

    def _complete_cycle(self) -> None:
        cycle = self.cycle
        assert cycle is not None
        self.cycles_served += 1
        self._outcome = (cycle.success, cycle.message, cycle.amount_moved, cycle.elapsed_s)
        self.cycle = None
        finish = self._finish
        self._finish = None
        if finish is not None and not finish.done():
            finish.set_result(True)
        level = "info" if cycle.success else "warn"
        getattr(self.get_logger(), level)(
            f"work cycle '{cycle.phase}' finished: success={cycle.success} "
            f"amount={cycle.amount_moved:.4f} m3 elapsed={cycle.elapsed_s:.1f}s ({cycle.message})"
        )
        if cycle.phase == "dump" and cycle.amount_moved > 0.0:
            self.get_logger().info(f"berm now {self.berm.summary()}")

    def _stop_cycle(self, reason: str) -> None:
        """Abort the running cycle (e-stop or a cancel) and stow."""
        cycle = self.cycle
        if cycle is None:
            return
        cycle.done = True
        cycle.success = False
        cycle.message = reason
        self.mechanism.stow()
        self.cycle = None
        self._outcome = (False, reason, cycle.amount_moved, cycle.elapsed_s)
        finish = self._finish
        self._finish = None
        if finish is not None and not finish.done():
            finish.set_result(True)

    def _check_joint_tracking(self, now: float) -> None:
        if self._joint_pitch is None:
            return
        expected = self.mechanism.pitch_rad
        if abs(self._joint_pitch - expected) <= 0.20:
            self._joint_mismatch_since = None
            return
        if self._joint_mismatch_since is None:
            self._joint_mismatch_since = now
            return
        if now - self._joint_mismatch_since > 3.0:
            self._log.warn(
                "joint-mismatch",
                f"{self.config.mechanism.ladder.joint} reports {self._joint_pitch:.2f} rad but the "
                f"model commands {expected:.2f} rad; the Gazebo joint controllers or their bridges "
                "are not working, so the rover is not physically digging",
                now,
            )
            self._joint_mismatch_since = None

    # ------------------------------------------------------------------ #
    # outputs
    # ------------------------------------------------------------------ #
    def _publish_float(self, topic: str, value: float) -> None:
        message = Float64()
        message.data = float(value)
        if topic == self.config.backend.ladder_cmd_topic:
            self.ladder_publisher.publish(message)
        elif topic == self.config.backend.belt_cmd_topic:
            self.belt_publisher.publish(message)
        else:  # pragma: no cover - guarded by the config loader
            self.get_logger().warn(f"backend published to an unknown topic {topic!r}")

    def _publish_status(self, now: float) -> None:
        snapshot = self.mechanism.snapshot()
        self.status_publisher.publish(
            nodeutil.mechanism_status_msg(self, self.config, snapshot, self.tub.volume_m3, self.tub.fraction)
        )
        self.berm_publisher.publish(nodeutil.berm_status_msg(self, self.config, self.berm))

    def _publish_marker(self, dt: float) -> None:
        self._marker_age += dt
        changed = self.berm.dumps != self._berm_dumps
        if not changed and self._marker_age < _MARKER_REFRESH_S:
            return
        self._berm_dumps = self.berm.dumps
        self._marker_age = 0.0
        self.marker_publisher.publish(nodeutil.berm_marker_array(self, self.config, self.berm))

    def _update_gz_berm(self) -> None:
        if not self.visual.enabled:
            return
        spoke = self.visual.update(self.berm.mean_height_m)
        if not spoke and self.visual.note and not getattr(self, "_visual_warned", False):
            self._visual_warned = True
            self.get_logger().warn(self.visual.note)

    def _run_gz(self, command: list) -> int:
        try:
            completed = subprocess.run(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5.0, check=False
            )
        except Exception as exc:  # pragma: no cover - no gz on the machine
            self._log.warn("gz-missing", f"cannot run '{command[0]}': {exc}")
            return 1
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
            self._log.warn("gz-failed", f"gz command failed: {detail[-1] if detail else completed.returncode}")
        return int(completed.returncode)

    # ------------------------------------------------------------------ #
    # the action
    # ------------------------------------------------------------------ #
    def _on_cancel(self, goal_handle: Any) -> Any:
        self._cancel_requested = True
        self._stop_cycle("canceled by the caller")
        return CancelResponse.ACCEPT

    async def _execute(self, goal_handle: Any) -> Any:
        """Validate, supersede, then hold the goal open while the timer drives it.

        The shape mirrors module 5's navigator, for the same reason: rclpy calls
        this once per accepted goal and aborts the handle itself afterwards if it
        is still active, so every terminal transition has to happen here.  Two
        goals can be accepted in the same control tick, so a goal carries its own
        sequence number and a superseded goal never touches state the newer one
        owns.
        """
        request = goal_handle.request
        phase = str(request.phase).strip()
        if phase not in WORK_PHASES:
            self.get_logger().warn(f"refusing work cycle: unknown phase {phase!r}")
            goal_handle.abort()
            return self._result(False, f"unknown phase {phase!r}; expected one of {list(WORK_PHASES)}", 0.0, 0.0)
        if self.estop:
            self.get_logger().warn(f"refusing work cycle '{phase}': e-stop is latched")
            goal_handle.abort()
            return self._result(False, "e-stop latched", 0.0, 0.0)
        if self.mechanism.fault is not None:
            reason = f"mechanism fault: {self.mechanism.fault}"
            self.get_logger().warn(f"refusing work cycle '{phase}': {reason}")
            goal_handle.abort()
            return self._result(False, reason, 0.0, 0.0)

        # One cycle at a time: the newest wins.  The superseded goal's callback is
        # waiting on its own future, so aborting its handle and resolving that
        # future lets it return; the sequence check there tells it not to touch
        # the mechanism the newer goal now commands.  Every terminal transition
        # happens in this method, which is why the abort is here rather than left
        # to rclpy's "still active after the callback returned" fallback.
        if self._goal_handle is not None:
            try:
                self._goal_handle.abort()
            except Exception:  # pragma: no cover - already terminal
                pass
        if self.cycle is not None:
            self.get_logger().warn(f"work cycle '{phase}' supersedes the running '{self.cycle.phase}'")
            self._stop_cycle("superseded by a newer work cycle")
        self._cycle_seq += 1
        sequence = self._cycle_seq
        self._cancel_requested = False
        self._goal_handle = goal_handle

        pose = nodeutil.pose_stamped_to_xy_yaw(request.work_pose)
        self.cycle = WorkCycleRunner(
            self.config,
            phase,
            pose,
            self.mechanism,
            self.tub,
            self.berm,
            cycle_id=self.cycles_served,
            timeout_s=float(request.timeout_s),
            target_amount=float(request.target_amount),
            now=nodeutil.now_seconds(self),
        )
        finish = self._finish = Future()
        self._outcome = None
        self.get_logger().info(
            f"work cycle accepted: {phase} at ({pose[0]:.2f}, {pose[1]:.2f}, yaw {pose[2]:.2f}), "
            f"timeout {self.cycle.timeout_s:.0f}s, tub {self.tub.fraction * 100.0:.0f}%"
        )

        await finish

        if sequence != self._cycle_seq:
            return self._result(False, "superseded by a newer work cycle", 0.0, 0.0)

        outcome = self._outcome or (False, "work cycle ended without an outcome", 0.0, 0.0)
        self._outcome = None
        if self._goal_handle is goal_handle:
            self._goal_handle = None
        success, message, amount, elapsed = outcome
        if self._cancel_requested:
            self._cancel_requested = False
            goal_handle.canceled()
            self.get_logger().warn(f"work cycle canceled: {message}")
            return self._result(False, message, amount, elapsed)
        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(f"work cycle '{phase}' -> {'succeeded' if success else 'failed'}: {message}")
        return self._result(success, message, amount, elapsed)

    def _result(self, success: bool, message: str, amount_moved: float, elapsed_s: float) -> Any:
        result = WorkCycle.Result()
        result.success = bool(success)
        result.message = str(message)
        result.amount_moved = float(amount_moved)
        result.elapsed_s = float(elapsed_s)
        return result

    # ------------------------------------------------------------------ #
    # diagnostics
    # ------------------------------------------------------------------ #
    def snapshot_dict(self) -> Dict[str, Any]:
        """A plain snapshot for tools and tests."""
        return {
            "cycle": None if self.cycle is None else {"phase": self.cycle.phase, "stage": self.cycle.stage_name},
            "mechanism": self.mechanism.snapshot().__dict__,
            "tub_fraction": self.tub.fraction,
            "berm": {
                "placed_m3": self.berm.placed_volume_m3,
                "spilled_m3": self.berm.spilled_volume_m3,
                "completion": self.berm.completion,
                "coverage": self.berm.coverage,
                "mean_height_m": self.berm.mean_height_m,
                "peak_height_m": self.berm.peak_height_m,
                "dumps": self.berm.dumps,
            },
        }


def _with_backend_kind(config: ExcavationConfig, kind: str) -> ExcavationConfig:
    import dataclasses

    from ..config import BACKEND_KINDS

    if kind not in BACKEND_KINDS:
        raise SystemExit(f"--backend must be one of {list(BACKEND_KINDS)}, got {kind!r}")
    return dataclasses.replace(config, backend=dataclasses.replace(config.backend, kind=kind))


def _with_gz_berm(config: ExcavationConfig, enabled: bool) -> ExcavationConfig:
    import dataclasses

    return dataclasses.replace(config, visual=dataclasses.replace(config.visual, gz_berm=enabled))


def add_arguments(parser: Any) -> None:
    parser.add_argument("--backend", default="", help="override backend.kind: sim | hardware | null")
    parser.add_argument("--rate-hz", type=float, default=0.0, help="control loop rate (0 = config)")
    parser.add_argument("--gz-berm", dest="gz_berm", action="store_true", default=None, help="draw the berm in Gazebo")
    parser.add_argument(
        "--no-gz-berm", dest="gz_berm", action="store_false", default=None, help="do not draw the berm in Gazebo"
    )


def main(argv: Optional[list] = None) -> int:
    """``ros2 run rover_excavation excavation.py``."""

    def build(args: Any) -> ExcavationNode:
        return ExcavationNode(
            config_override=args.config,
            backend_override=args.backend,
            rate_hz_override=args.rate_hz,
            gz_berm_override=args.gz_berm,
        )

    return nodeutil.node_main(
        argv,
        "module 6: the excavation mechanism and the berm (serves /excavation/work_cycle)",
        build,
        add_arguments,
        epilog=(
            "Serves rover_navigation_msgs/action/WorkCycle for module 5's mission sequencer, "
            "publishes MechanismStatus and BermStatus, and draws the berm in rviz and (optionally) Gazebo."
        ),
    )


if __name__ == "__main__":  # pragma: no cover - the scripts/ shim is the entry point
    sys.exit(main())
