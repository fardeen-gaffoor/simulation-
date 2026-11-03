"""``health_monitor``: the numbers the judges ask about, once a second.

Started by ``perception.launch.py`` as ``perception_health_monitor``.  It owns no
perception of its own: it subscribes to what the other four nodes already
publish, adds this machine's SoC temperature, and republishes the lot as

* ``rover_perception_msgs/PerceptionHealth`` on ``health.topic`` - the numbers
  module 5 and the run log care about (inference fps, CPU, temperature,
  bandwidth, obstacle count, de-rating state), and
* ``diagnostic_msgs/DiagnosticArray`` on ``health.diagnostics_topic`` - the same
  facts in the standard form, so ``rqt_runtime_monitor`` and any ROS 2
  diagnostics tooling work without a custom plugin.

Two deliberate choices: the fps/latency/CPU numbers are read back from the
detector's own stats topic (a process cannot measure another process's CPU use
without ``psutil``, and module 4 does not add a dependency for that), and the
thermal state is classified here from ``/sys/class/thermal`` rather than
reported by the camera manager, so the temperature the judges see is measured,
not inferred from a command.
"""

from __future__ import annotations

import argparse
import socket
from typing import Any, Dict, List, Optional, Sequence

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from rover_perception_msgs.msg import (
    CameraPhase,
    CameraStatusArray,
    LocalizationState,
    ObstacleArray,
    PerceptionHealth,
)

from .. import nodeutil
from ..cpu import CpuMeter
from .obstacle_detector_node import STATS_LAYOUT
from ..thermal import ThermalMonitor

__all__ = ["HealthMonitorNode", "main", "add_arguments"]

NODE_NAME = "perception_health_monitor"


def _stats_lookup(message: Float32MultiArray) -> Dict[str, float]:
    """Read a detector-stats array by *label*, falling back to positional order.

    Using the published labels means the health monitor keeps working if a field
    is added to ``STATS_LAYOUT`` (the layout is the contract), and the positional
    fallback keeps it working against an older publisher.
    """
    values = [float(value) for value in message.data]
    labels: List[str] = []
    try:
        if message.layout.dim:
            labels = [part for part in str(message.layout.dim[0].label).split(",") if part]
    except Exception:  # pragma: no cover - defensive against odd layouts
        labels = []
    if len(labels) == len(values) and list(labels) == list(STATS_LAYOUT):
        return dict(zip(labels, values))
    if len(values) >= len(STATS_LAYOUT):
        return dict(zip(STATS_LAYOUT, values[: len(STATS_LAYOUT)]))
    return {}


class HealthMonitorNode(Node):
    """Aggregates perception health and mirrors it into diagnostics."""

    def __init__(self, config_override: str = "", publish_hz_override: float = 0.0):
        super().__init__(NODE_NAME)
        self.config = nodeutil.load_node_config(self, override=config_override)
        cfg = self.config.health
        self.publish_hz = float(
            nodeutil.declare(self, "publish_hz", float(publish_hz_override or cfg.publish_hz))
        )
        if self.publish_hz <= 0.0:
            self.publish_hz = cfg.publish_hz
        self.thermal = ThermalMonitor(self.config.camera_manager.thermal)
        self.reading = self.thermal.read()
        self.cpu = CpuMeter()

        self.bandwidth_kbps = 0.0
        self.bandwidth_limit_kbps = float(self.config.camera_manager.bandwidth.limit_kbps)
        self.throttled = False
        self.streaming = 0
        self.camera_count = len(self.config.cameras)
        self.stats: Dict[str, float] = {}
        self.obstacles = 0
        self.localized = False
        self.sigma_xy_m = 0.0
        self.last_fix_age_s = float("inf")
        self.published = 0

        self.health_publisher = self.create_publisher(PerceptionHealth, cfg.topic, 10)
        self.diagnostics_publisher = self.create_publisher(DiagnosticArray, cfg.diagnostics_topic, 10)
        self.status_subscription = self.create_subscription(
            CameraStatusArray, self.config.camera_manager.status_topic, self._on_camera_status, 10
        )
        self.phase_subscription = self.create_subscription(
            CameraPhase, self.config.camera_manager.phase_topic, self._on_phase, 10
        )
        self.stats_subscription = self.create_subscription(
            Float32MultiArray, self.config.detector.stats_topic, self._on_detector_stats, 10
        )
        self.obstacles_subscription = self.create_subscription(
            ObstacleArray, self.config.detector.obstacles_topic, self._on_obstacles, 10
        )
        self.localization_subscription = self.create_subscription(
            LocalizationState, self.config.localization.state_topic, self._on_localization, 10
        )
        self.thermal_timer = None
        if self.config.camera_manager.thermal.enabled:
            self.thermal_timer = self.create_timer(
                1.0 / self.config.camera_manager.thermal.poll_hz, self._on_thermal_poll
            )
        self.create_timer(1.0 / max(self.publish_hz, 1e-3), self._on_publish)

        self.get_logger().info(
            f"health: {cfg.topic} + {cfg.diagnostics_topic} at {self.publish_hz:.1f} Hz, "
            f"watching {self.config.detector.stats_topic}, "
            f"{self.config.camera_manager.status_topic}, "
            f"{self.config.localization.state_topic}"
        )

    # -- inputs ------------------------------------------------------------- #
    def _on_camera_status(self, message: Any) -> None:
        self.bandwidth_kbps = float(message.total_kbps)
        self.bandwidth_limit_kbps = float(message.limit_kbps)
        self.throttled = bool(message.throttled)
        self.streaming = sum(1 for camera in message.cameras if camera.streaming)

    def _on_phase(self, message: Any) -> None:
        self.throttled = self.throttled or bool(message.measured_kbps > message.limit_kbps)

    def _on_detector_stats(self, message: Any) -> None:
        self.stats = _stats_lookup(message)

    def _on_obstacles(self, message: Any) -> None:
        self.obstacles = len(message.obstacles)

    def _on_localization(self, message: Any) -> None:
        self.localized = bool(message.localized)
        self.sigma_xy_m = float(message.sigma_xy_m)
        self.last_fix_age_s = float(message.time_since_fix_s)

    def _on_thermal_poll(self) -> None:
        previous = self.reading.state
        self.reading = self.thermal.read()
        if self.reading.state != previous:
            self.get_logger().warn(
                f"thermal {self.reading.state_name}: "
                f"{self.reading.temperature_c} C on {self.reading.zone}"
            )

    # -- outputs ------------------------------------------------------------ #
    def _on_publish(self) -> None:
        now = nodeutil.now_seconds(self)
        own_cpu = float(self.cpu.sample(now))
        health = PerceptionHealth()
        health.header = nodeutil.make_header(self, self.config.arena.frame_id)
        health.detector_backend = self.config.detector.backend if self.config.detector.enabled else "off"
        health.fps = float(self.stats.get("fps", 0.0))
        health.latency_ms = float(self.stats.get("latency_ms", 0.0))
        # the detector measures its own CPU use; this process only reports it,
        # and falls back to its own reading while the detector is still starting
        health.cpu_percent = float(self.stats.get("cpu_percent", own_cpu))
        health.temperature_c = float(self.reading.temperature_c or 0.0)
        health.thermal_zone = self.reading.zone if self.reading.temperature_c is not None else "none"
        health.bandwidth_kbps = float(self.bandwidth_kbps)
        health.bandwidth_limit_kbps = float(self.bandwidth_limit_kbps)
        health.frames = int(self.stats.get("frames_processed", 0.0))
        health.obstacles = int(self.obstacles)
        health.throttled = bool(self.throttled)
        health.thermal_state = int(self.reading.state)
        self.health_publisher.publish(health)
        self.diagnostics_publisher.publish(self._build_diagnostics(health, now))
        self.published += 1

        if self.published % max(1, int(round(self.publish_hz))) == 1:
            self.get_logger().info(
                f"{health.detector_backend}: {health.fps:.1f} fps, {health.latency_ms:.1f} ms, "
                f"cpu {health.cpu_percent:.0f}%, {health.temperature_c:.1f} C "
                f"({health.thermal_zone}, {self.reading.state_name}), "
                f"{health.bandwidth_kbps:.0f}/{health.bandwidth_limit_kbps:.0f} kbps, "
                f"{health.obstacles} obstacles, "
                f"{'localized' if self.localized else 'NOT localized'}"
                + (f" (fix {self.last_fix_age_s:.1f} s ago)" if self.localized else "")
                + (", throttled" if health.throttled else "")
            )

    def _build_diagnostics(self, health: PerceptionHealth, now: float) -> DiagnosticArray:
        cfg = self.config.health
        array = DiagnosticArray()
        array.header = health.header
        status = DiagnosticStatus()
        status.name = "rover_perception"
        status.hardware_id = socket.gethostname()
        level = DiagnosticStatus.OK
        notes: List[str] = []
        if health.thermal_state >= 2:
            level = DiagnosticStatus.ERROR if health.thermal_state >= 3 else DiagnosticStatus.WARN
            notes.append(f"thermal {self.reading.state_name}")
        elif health.temperature_c >= cfg.thermal_warn_c:
            level = DiagnosticStatus.WARN
            notes.append(f"temperature {health.temperature_c:.0f} C")
        if health.cpu_percent >= cfg.cpu_warn_pct:
            level = DiagnosticStatus.WARN
            notes.append(f"cpu {health.cpu_percent:.0f}%")
        if health.bandwidth_limit_kbps > 0.0 and health.bandwidth_kbps > health.bandwidth_limit_kbps:
            level = DiagnosticStatus.WARN
            notes.append(
                f"over budget {health.bandwidth_kbps:.0f}/{health.bandwidth_limit_kbps:.0f} kbps"
            )
        if self.config.localization.enabled and not self.localized:
            level = DiagnosticStatus.WARN
            notes.append("no tag fix yet")
        elif self.config.localization.enabled and self.last_fix_age_s > 30.0:
            level = DiagnosticStatus.WARN
            notes.append(f"last fix {self.last_fix_age_s:.0f} s ago")
        if self.streaming == 0:
            level = DiagnosticStatus.WARN
            notes.append("no camera streaming")
        if level == DiagnosticStatus.OK and self.stats:
            notes.append("all checks passed")
        elif level == DiagnosticStatus.OK:
            notes.append("waiting for detector stats")
        status.level = level
        status.message = "; ".join(notes)
        status.values = [
            KeyValue(key="detector_backend", value=str(health.detector_backend)),
            KeyValue(key="fps", value=f"{health.fps:.2f}"),
            KeyValue(key="latency_ms", value=f"{health.latency_ms:.2f}"),
            KeyValue(key="cpu_percent", value=f"{health.cpu_percent:.1f}"),
            KeyValue(key="temperature_c", value=f"{health.temperature_c:.1f}"),
            KeyValue(key="thermal_zone", value=str(health.thermal_zone)),
            KeyValue(key="thermal_state", value=self.reading.state_name),
            KeyValue(key="bandwidth_kbps", value=f"{health.bandwidth_kbps:.1f}"),
            KeyValue(key="bandwidth_limit_kbps", value=f"{health.bandwidth_limit_kbps:.1f}"),
            KeyValue(key="cameras_streaming", value=f"{self.streaming}/{self.camera_count}"),
            KeyValue(key="obstacles", value=str(health.obstacles)),
            KeyValue(key="localized", value=str(bool(self.localized))),
            KeyValue(key="sigma_xy_m", value=f"{self.sigma_xy_m:.3f}"),
            KeyValue(key="time_since_fix_s", value=f"{min(self.last_fix_age_s, 1e6):.1f}"),
            KeyValue(key="throttled", value=str(bool(health.throttled))),
        ]
        array.status = [status]
        return array

    # -- introspection (tests) ---------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": self.config.path,
            "health_topic": self.config.health.topic,
            "diagnostics_topic": self.config.health.diagnostics_topic,
            "stats": dict(self.stats),
            "bandwidth_kbps": self.bandwidth_kbps,
            "obstacles": self.obstacles,
            "localized": self.localized,
            "thermal": {"zone": self.reading.zone, "state": self.reading.state_name},
            "published": self.published,
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--publish-hz", type=float, default=0.0, help="override health.publish_hz"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    def build(args: argparse.Namespace) -> Node:
        override = float(getattr(args, "publish_hz", 0.0) or 0.0)
        return HealthMonitorNode(args.config, publish_hz_override=override)

    return nodeutil.node_main(
        argv,
        "perception health: fps, CPU, temperature, bandwidth, obstacles -> /diagnostics",
        build,
        add_arguments=add_arguments,
        epilog="publish_hz override only changes the report rate, never the measurements",
    )
