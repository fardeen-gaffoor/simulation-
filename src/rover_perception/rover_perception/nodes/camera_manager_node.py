"""``camera_manager``: the rclpy wrapper around :class:`PhaseController`.

Started by ``perception.launch.py`` as ``perception_camera_manager``.  It does
three things and nothing else:

1. **counts bytes.**  One best-effort ``sensor_msgs/Image`` subscription per
   managed camera; the *payload length* is what feeds the 4000 kbps competition
   budget.  The frames are never decoded, so the meter costs nothing measurable.
2. **decides a phase.**  ``/perception/mode`` (``std_msgs/String``, driven by
   module 5) switches the perception phase; the controller turns that plus the
   measured bandwidth plus the SoC temperature into per-camera rates, and the
   node publishes them as ``rover_perception_msgs/CameraPhase``.
3. **reports the facts.**  ``CameraStatusArray`` carries what was actually
   observed (frames, bytes, measured rate, streaming or not), which is what
   ``health_monitor`` forwards and what an operator looks at when the link is
   the problem.

In simulation the Gazebo sensors cannot be re-rated at runtime, so the phase is
advisory there and the *measured* numbers are the ones that matter - the node
says so in the log at startup rather than silently pretending.  On the Pi the
same message is the contract ``rover_camera_driver`` obeys, and nothing outside
those two nodes knows which CSI hub is fitted.
"""

from __future__ import annotations

import argparse
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from rover_perception_msgs.msg import CameraPhase, CameraStatus, CameraStatusArray

from .. import nodeutil
from ..bandwidth import ByteRateMeter
from ..camera_manager import PhaseController
from ..config import MODES
from ..thermal import ThermalMonitor

__all__ = ["CameraManagerNode", "main", "add_arguments"]

NODE_NAME = "perception_camera_manager"


class CameraManagerNode(Node):
    """Phase-aware frame-rate manager, bandwidth meter and thermal watchdog."""

    def __init__(self, config_override: str = "", publish_hz_override: float = 0.0):
        super().__init__(NODE_NAME)
        self.config = nodeutil.load_node_config(self, override=config_override)
        cfg = self.config.camera_manager
        self.publish_hz = float(nodeutil.declare(self, "publish_hz", publish_hz_override or cfg.publish_hz))
        if self.publish_hz <= 0.0:
            self.publish_hz = cfg.publish_hz

        self.meter = ByteRateMeter(self.config.camera_names, cfg.bandwidth.window_s)
        self.thermal = ThermalMonitor(cfg.thermal)
        self.reading = self.thermal.read()
        self.controller = PhaseController(
            cfg,
            self.config.camera_names,
            self.config.active_camera_for_mode,
            initial_thermal_state=self.reading.state,
        )
        self._log = nodeutil.LogOnce(self)
        self._last_phase_key: Optional[tuple] = None

        self.phase_publisher = self.create_publisher(CameraPhase, cfg.phase_topic, 10)
        self.status_publisher = self.create_publisher(CameraStatusArray, cfg.status_topic, 10)
        self.mode_subscription = self.create_subscription(
            String, cfg.mode_topic, self._on_mode, 10
        )
        self.image_subscriptions = [
            self.create_subscription(
                Image,
                camera.topic,
                partial(self._on_image, camera.name),
                nodeutil.sensor_qos(),
            )
            for camera in self.config.cameras
        ]
        self.create_timer(1.0 / self.publish_hz, self._on_publish)
        if cfg.thermal.enabled:
            self.create_timer(1.0 / cfg.thermal.poll_hz, self._on_thermal_poll)

        self.get_logger().info(
            f"managing {len(self.config.cameras)} cameras "
            f"({', '.join(self.config.camera_names)}); mode '{self.controller.mode}', "
            f"budget {cfg.bandwidth.limit_kbps:.0f} kbps over {cfg.bandwidth.window_s:.0f} s"
        )
        self.get_logger().info(
            "phase commands are ADVISORY in simulation (gz sensors cannot be re-rated); "
            "on the rover they are the rover_camera_driver contract"
        )
        if not cfg.thermal.enabled:
            self.get_logger().warn("thermal monitoring is disabled in perception.yaml")
        elif self.reading.temperature_c is None:
            self.get_logger().warn(
                "no readable thermal zone on this machine; temperature de-rating is inert"
            )
        else:
            self.get_logger().info(
                f"thermal: {self.reading.temperature_c:.1f} C on {self.reading.zone} "
                f"({self.reading.state_name})"
            )

    # -- callbacks ---------------------------------------------------------- #
    def _on_image(self, name: str, message: Any) -> None:
        """Count one frame's payload towards this camera's rate."""
        payload = len(message.data) if message.data is not None else 0
        if payload <= 0:
            payload = int(message.height) * int(message.step)
        self.meter.observe(name, payload, nodeutil.now_seconds(self))

    def _on_mode(self, message: Any) -> None:
        mode = str(getattr(message, "data", "") or "").strip()
        if mode not in MODES:
            self._log.warn(
                f"bad_mode_{mode}",
                f"ignoring unknown perception mode '{mode}' (known: {list(MODES)})",
            )
            return
        previous = self.controller.mode
        self.controller.set_mode(mode)
        if mode != previous:
            self.get_logger().info(f"perception mode {previous} -> {mode}")

    def _on_thermal_poll(self) -> None:
        previous = self.reading.state
        self.reading = self.thermal.read()
        if self.reading.state != previous:
            level = self.reading.state_name
            self.get_logger().warn(
                f"thermal state {level}: {self.reading.temperature_c} C on {self.reading.zone}"
            )

    def _on_publish(self) -> None:
        now = nodeutil.now_seconds(self)
        self.meter.tick(now)
        measured = self.meter.total_kbps(now)
        command = self.controller.update(measured, self.reading.state)
        self._publish_phase(command)
        self._publish_status(now, command)

        key = (
            command.mode,
            command.active_camera,
            tuple(round(item.rate_hz, 3) for item in command.commands),
            command.derate_index,
            command.thermal_state,
        )
        if key != self._last_phase_key:
            self._last_phase_key = key
            rates = ", ".join(
                f"{item.name}={item.rate_hz:.1f}Hz" + ("" if item.enabled else " (off)")
                for item in command.commands
            )
            self.get_logger().info(
                f"phase '{command.mode}': {rates} | measured {measured:.0f} kbps, "
                f"derate x{command.derate_factor:.2f}"
            )
        for reason in command.reasons:
            self.get_logger().warn(reason)

    # -- publishing --------------------------------------------------------- #
    def _publish_phase(self, command: Any) -> None:
        message = CameraPhase()
        message.header = nodeutil.make_header(self, self.config.arena.frame_id)
        message.mode = command.mode
        message.active_camera = command.active_camera
        message.active_rate_hz = float(command.active_rate_hz)
        message.idle_rate_hz = float(command.idle_rate_hz)
        message.cameras = list(self.config.camera_names)
        message.rate_hz = [float(item.rate_hz) for item in command.commands]
        message.enabled = [bool(item.enabled) for item in command.commands]
        message.bandwidth_ok = bool(command.bandwidth_ok)
        message.measured_kbps = float(command.measured_kbps)
        message.limit_kbps = float(command.limit_kbps)
        self.phase_publisher.publish(message)

    def _publish_status(self, now: float, command: Any) -> None:
        cfg = self.config.camera_manager
        message = CameraStatusArray()
        message.header = nodeutil.make_header(self, self.config.arena.frame_id)
        statuses: List[CameraStatus] = []
        for item in command.commands:
            status = CameraStatus()
            status.header = message.header
            status.name = item.name
            status.streaming = self.meter.is_streaming(item.name, now, cfg.stale_after_s)
            status.active = bool(item.active)
            status.measured_rate_hz = float(self.meter.rate_hz(item.name, now))
            status.measured_kbps = float(self.meter.rate_kbps(item.name, now))
            status.frames = int(self.meter.frames(item.name))
            status.bytes = float(self.meter.bytes(item.name))
            statuses.append(status)
        message.cameras = statuses
        message.total_kbps = float(command.measured_kbps)
        message.limit_kbps = float(command.limit_kbps)
        message.window_s = float(cfg.bandwidth.window_s)
        message.throttled = bool(command.throttled)
        message.over_budget = bool(command.measured_kbps > command.limit_kbps)
        self.status_publisher.publish(message)

    # -- introspection (tests) ---------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": self.config.path,
            "publish_hz": self.publish_hz,
            "phase_topic": self.config.camera_manager.phase_topic,
            "status_topic": self.config.camera_manager.status_topic,
            "mode_topic": self.config.camera_manager.mode_topic,
            "cameras": list(self.config.camera_names),
            "controller": self.controller.snapshot(),
            "thermal": self.reading.state_name,
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags specific to this node (``--config`` comes from ``nodeutil``)."""
    parser.add_argument(
        "--publish-hz", type=float, default=0.0, help="override camera_manager.publish_hz"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    def build(args: argparse.Namespace) -> Node:
        return CameraManagerNode(args.config, float(getattr(args, "publish_hz", 0.0) or 0.0))

    return nodeutil.node_main(
        argv,
        "camera manager: phase rates, bandwidth budget, thermal de-rating",
        build,
        add_arguments=add_arguments,
        epilog=(
            "phase commands go out on camera_manager.phase_topic as "
            "rover_perception_msgs/CameraPhase; measured traffic goes out on "
            "camera_manager.status_topic"
        ),
    )
