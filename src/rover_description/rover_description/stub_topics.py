"""Interface stubs declared by module 2: ``/estop`` and ``/battery/energy_wh``.

Both topics are part of the stack-wide contract and must exist with stable types
in the simulator AND on the real rover, so they are created here even though the
real battery monitor and the hardware e-stop chain arrive with the hardware.

    /estop             : std_msgs/Bool     (published + subscribed, latched)
    /battery/energy_wh : std_msgs/Float64  (published)

``/estop`` model
  The node is the *owner* of the latched e-stop state: it publishes it
  (transient-local, so any late subscriber sees the current value) and also
  subscribes to it.  Any other node may publish ``true`` to request a stop; the
  node mirrors the value back onto the latched topic, so the state stays a single
  consistent value.  When the hardware e-stop chain exists it either replaces
  this node or publishes to ``/estop`` (the mirror keeps both consistent).

  While an e-stop is latched the node also publishes zero ``/cmd_vel`` at
  ``gate_rate_hz`` - the safety-relevant half of the contract, and the reason the
  stub is more than a counter.

``/battery/energy_wh``
  Drained at ``battery.idle_draw_w`` (+ an optional ``load_w``) so downstream code
  sees a plausibly decreasing number.  It stops draining while stopped.

The node reads its topic names and battery parameters from ``rover.yaml`` so the
simulator and the robot can never disagree about the interface.
"""

from __future__ import annotations

from typing import Optional, Sequence

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float64

from .config import load_config

__all__ = ["StubTopics", "main"]


def _latched() -> QoSProfile:
    return QoSProfile(
        depth=1,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


class StubTopics(Node):
    def __init__(self) -> None:
        super().__init__("stub_topics")

        self.declare_parameter("config", "")
        self.declare_parameter("estop_gate", True)
        self.declare_parameter("load_w", 0.0)
        self.declare_parameter("gate_rate_hz", 20.0)

        config_path = self.get_parameter("config").get_parameter_value().string_value or None
        self._cfg = load_config(config_path)
        self._estop_gate = bool(self.get_parameter("estop_gate").value)
        self._load_w = float(self.get_parameter("load_w").value)
        gate_rate = max(float(self.get_parameter("gate_rate_hz").value), 1.0)

        self._energy = self._cfg.battery.initial_energy_wh
        self._estop = False

        self._energy_pub = self.create_publisher(
            Float64, self._cfg.topics["battery_energy"], 10
        )
        self._estop_pub = self.create_publisher(Bool, self._cfg.topics["estop"], _latched())
        self._cmd_pub = (
            self.create_publisher(Twist, self._cfg.topics["cmd_vel"], 10)
            if self._estop_gate
            else None
        )

        self.create_subscription(Bool, self._cfg.topics["estop"], self._on_estop, _latched())
        self.create_timer(1.0, self._tick_battery)
        if self._cmd_pub is not None:
            self.create_timer(1.0 / gate_rate, self._tick_gate)

        self._publish_estop()  # seed the latched state
        self.get_logger().info(
            f"stub_topics up: {self._cfg.topics['battery_energy']} (pub), "
            f"{self._cfg.topics['estop']} (pub+sub, latched), estop_gate={self._estop_gate}"
        )

    # ------------------------------------------------------------------ #
    def _publish_estop(self) -> None:
        msg = Bool()
        msg.data = self._estop
        self._estop_pub.publish(msg)

    def _drain_w(self) -> float:
        return 0.0 if self._estop else (self._cfg.battery.idle_draw_w + self._load_w)

    def _tick_battery(self) -> None:
        # 1 Hz: consume Wh equal to watts / 3600
        self._energy = max(0.0, self._energy - self._drain_w() / 3600.0)
        msg = Float64()
        msg.data = self._energy
        self._energy_pub.publish(msg)

    def _tick_gate(self) -> None:
        if self._estop and self._cmd_pub is not None:
            self._cmd_pub.publish(Twist())  # zero velocity while latched

    def _on_estop(self, msg: Bool) -> None:
        value = bool(msg.data)
        if value != self._estop:
            self._estop = value
            self._publish_estop()
            self.get_logger().warn(
                f"{self._cfg.topics['estop']} -> {'ACTIVE' if self._estop else 'clear'}"
            )


def main(argv: Optional[Sequence[str]] = None) -> None:
    rclpy.init(args=list(argv) if argv is not None else None)
    node = StubTopics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
