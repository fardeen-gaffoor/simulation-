"""``velocity_gate``: the single writer of ``/cmd_vel``, and the safety authority.

Started by ``navigation.launch.py`` as ``rover_navigation_velocity_gate``.  The
navigator publishes what it *wants* on ``/nav/cmd_vel``; this node publishes what
is *allowed* on ``/cmd_vel``.  Splitting them is the point: there is exactly one
place in the whole stack that can move the rover, so there is exactly one answer
to "why is it not moving", and it is in ``GateStatus.reason``.

It publishes every tick whether or not it has anything to say.  That matters
because Gazebo's diff-drive holds the last velocity it was given: if this node
went quiet, the rover would keep driving at whatever speed it last heard.  Silence
on ``/cmd_vel`` is not a thing; zero is.

The four reasons it can stop the rover are documented on
:class:`~rover_navigation.safety.VelocityGate`.  This node adds the I/O: the
``/estop`` latch, module 4's ``LocalizationState``, the live obstacle list for the
reactive brake, and ``/nav/gate_status`` so the decision is visible.
"""

from __future__ import annotations

import argparse
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool

from rover_navigation_msgs.msg import GateStatus
from rover_perception_msgs.msg import LocalizationState, ObstacleArray

from .. import nodeutil
from ..config import NavigationConfig
from ..costmap import ObstacleMemory, ObstacleRecord
from ..safety import GateDecision, LocalizationSnapshot, VelocityGate

__all__ = ["VelocityGateNode", "main", "add_arguments"]

NODE_NAME = "rover_navigation_velocity_gate"


class VelocityGateNode(Node):
    """Turns ``/nav/cmd_vel`` into ``/cmd_vel`` - or into a stop."""

    def __init__(self, config_override: str = "", publish_hz_override: float = 0.0) -> None:
        super().__init__(NODE_NAME)
        self.config: NavigationConfig = nodeutil.load_node_config(self, override=config_override)
        self.rate_hz = float(
            nodeutil.declare(
                self,
                "publish_hz",
                publish_hz_override or self.config.safety.gate_rate_hz,
                "gate tick rate; the rover is stopped at this rate while held",
            )
        )
        self.publish_status = bool(nodeutil.declare(self, "publish_status", True, "publish /nav/gate_status"))

        self.gate = VelocityGate(self.config)
        self.memory = ObstacleMemory(self.config)
        self.tracker = nodeutil.PoseTracker(preferred_topic=self.config.topics.odometry_topics[0])
        self.records: List[ObstacleRecord] = []
        self.localization = LocalizationSnapshot()
        self.estop = False
        self.command: Optional[tuple] = None
        self.last_command_time: Optional[float] = None
        self.last_command_source = 0
        self.decision = GateDecision(allowed=False, linear=0.0, angular=0.0, reason="starting up")
        self.ticks = 0
        self.stops = 0
        self.last_tick = nodeutil.now_seconds(self)

        topics = self.config.topics
        self.cmd_publisher = self.create_publisher(Twist, topics.cmd_vel, 10)
        self.status_publisher = (
            self.create_publisher(GateStatus, topics.gate_status, 10) if self.publish_status else None
        )
        self.command_subscription = self.create_subscription(
            Twist, topics.nav_cmd_vel, self._on_command, nodeutil.sensor_qos(1)
        )
        self.estop_subscription = self.create_subscription(
            Bool, topics.estop, self._on_estop, nodeutil.latched_qos(1)
        )
        self.localization_subscription = self.create_subscription(
            LocalizationState, topics.localization_state, self._on_localization, nodeutil.sensor_qos(5)
        )
        self.obstacle_subscription = self.create_subscription(
            ObstacleArray, topics.obstacles, self._on_obstacles, nodeutil.sensor_qos(5)
        )
        self.odom_subscriptions = [
            self.create_subscription(
                Odometry, topic, partial(self._on_odometry, topic=topic), nodeutil.sensor_qos(20)
            )
            for topic in topics.odometry_topics
        ]
        self.timer = self.create_timer(1.0 / max(1.0, self.rate_hz), self._tick)

        self._log = nodeutil.LogOnce(self, period_s=5.0)
        self.get_logger().info(
            f"{NODE_NAME} up: {topics.nav_cmd_vel} -> {topics.cmd_vel} at {self.rate_hz:g} Hz, "
            f"localization gating {'on' if self.config.safety.require_localized else 'off'}, "
            f"emergency brake at {self.config.safety.emergency_stop_m:.2f} m"
        )

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    def _on_command(self, message: Twist) -> None:
        self.command = (float(message.linear.x), float(message.angular.z))
        self.last_command_time = nodeutil.now_seconds(self)
        self.last_command_source += 1

    def _on_estop(self, message: Bool) -> None:
        value = bool(message.data)
        if value != self.estop:
            self.get_logger().warn(f"e-stop {'LATCHED' if value else 'released'}")
        self.estop = value

    def _on_odometry(self, message: Odometry, topic: str) -> None:
        # The gate keeps its own pose so the reactive brake does not depend on
        # the navigator being alive - which is the entire point of the brake.
        self.tracker.on_odometry(message, topic)

    def _on_localization(self, message: LocalizationState) -> None:
        self.tracker.on_localization(message)
        self.localization = LocalizationSnapshot(
            has_state=True,
            localized=bool(message.localized),
            sigma_xy_m=float(message.sigma_xy_m),
            sigma_yaw_rad=float(message.sigma_yaw_rad),
            time_since_fix_s=float(message.time_since_fix_s),
        )

    def _on_obstacles(self, message: ObstacleArray) -> None:
        now = nodeutil.now_seconds(self)
        self.memory.update(
            [
                ObstacleRecord(
                    identifier=obstacle.id or "unnamed",
                    kind=int(obstacle.kind),
                    x=float(obstacle.pose.pose.position.x),
                    y=float(obstacle.pose.pose.position.y),
                    radius=float(obstacle.radius),
                    confidence=float(obstacle.confidence),
                    observations=int(obstacle.observations),
                    conservative=bool(obstacle.conservative),
                )
                for obstacle in message.obstacles
            ],
            now,
        )
        self.records = self.memory.records

    # ------------------------------------------------------------------ #
    # the tick
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        now = nodeutil.now_seconds(self)
        dt = max(0.0, now - self.last_tick)
        self.last_tick = now
        decision = self.gate.decide(
            command=self.command,
            now=now,
            last_command_time=self.last_command_time,
            estop=self.estop,
            snapshot=self.localization,
            pose=self.tracker.pose,
            records=self.records,
            dt=dt,
        )
        self.decision = decision
        self.ticks += 1
        if not decision.allowed:
            self.stops += 1
            if decision.reason:
                self._log.warn(decision.reason[:32], f"holding: {decision.reason}", now)

        twist = Twist()
        twist.linear.x = float(decision.linear)
        twist.angular.z = float(decision.angular)
        self.cmd_publisher.publish(twist)
        self._publish_status(twist)

    def _publish_status(self, twist: Twist) -> None:
        if self.status_publisher is None:
            return
        decision = self.decision
        message = GateStatus()
        message.header = nodeutil.make_header(self, self.config.frames.map)
        message.estop = bool(self.estop)
        message.allowed = bool(decision.allowed)
        message.stale_command = bool(decision.stale)
        message.emergency_brake = bool(decision.emergency)
        message.localized = bool(self.localization.localized)
        message.speed_scale = float(decision.speed_scale)
        message.sigma_xy_m = float(self.localization.sigma_xy_m)
        message.reason = decision.reason
        message.command = twist
        message.max_speed_mps = float(decision.max_speed)
        message.max_yaw_rate_radps = float(decision.max_yaw_rate)
        self.status_publisher.publish(message)

    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        return {
            "estop": self.estop,
            "allowed": self.decision.allowed,
            "reason": self.decision.reason,
            "speed_scale": round(self.decision.speed_scale, 4),
            "localized": self.localization.localized,
            "sigma_xy_m": round(self.localization.sigma_xy_m, 4),
            "sigma_yaw_rad": round(self.localization.sigma_yaw_rad, 4),
            "time_since_fix_s": round(self.localization.time_since_fix_s, 3),
            "obstacles": len(self.records),
            "pose": None
            if self.tracker.pose is None
            else (
                round(self.tracker.pose.x, 3),
                round(self.tracker.pose.y, 3),
                round(self.tracker.pose.yaw, 3),
            ),
            "ticks": self.ticks,
            "stops": self.stops,
            "rate_hz": self.rate_hz,
            "command": (round(self.decision.linear, 4), round(self.decision.angular, 4)),
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publish-hz", type=float, default=0.0, help="gate rate (0 = gate_rate_hz)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return nodeutil.node_main(
        argv,
        "module 5 velocity gate: the only writer of /cmd_vel (e-stop, staleness, localisation, brake)",
        build=lambda args: VelocityGateNode(publish_hz_override=args.publish_hz),
        add_arguments=add_arguments,
        epilog=(
            "Subscribes to /nav/cmd_vel and republishes it on /cmd_vel only when it is safe: "
            "not e-stopped, not stale, with a tag-anchored map, and with nothing within the "
            "emergency stop distance of the front bumper."
        ),
    )
