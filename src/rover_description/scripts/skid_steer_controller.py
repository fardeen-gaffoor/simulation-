#!/usr/bin/env python3
"""``ros2 run rover_sim_tools skid_steer_controller``

Stage 2 of PROMPT_ROVER_SIM.md: skid-steer controller that maps ``/cmd_vel``
to four wheel velocity commands.

Reads wheel radius, track, effective-track correction and velocity/torque
limits from ``rover.yaml`` (via the rover_description config).  For each
``/cmd_vel`` (Twist) it computes per-wheel angular velocities:

    v_left  = vx - wz * effective_track / 2
    v_right = vx + wz * effective_track / 2
    wheel_vel = v_wheel / wheel_radius

All four wheels get the same left/right velocity (front-left = rear-left,
front-right = rear-right).  Commands are clamped to the configured
``max_wheel_velocity`` and published as ``std_msgs/Float64`` on:

    /wheel/wheel_front_left/cmd_vel
    /wheel/wheel_front_right/cmd_vel
    /wheel/wheel_rear_left/cmd_vel
    /wheel/wheel_rear_right/cmd_vel

These are bridged to Gazebo joint controllers (velocity mode) via
``config/bridges.yaml``.

No differential-drive plugin is used: per-wheel joint velocity control keeps
the wheels visible and individually controllable.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

from rover_description.config import load_config, default_config_path


class SkidSteerController(Node):
    def __init__(self) -> None:
        super().__init__("skid_steer_controller")

        # ---- parameters ---------------------------------------------------
        self.declare_parameter("rover_config", "")
        self.declare_parameter("effective_track_correction", "2.0")
        self.declare_parameter("max_wheel_velocity", "12.0")

        config_path = self.get_parameter("rover_config").value
        if not config_path:
            config_path = default_config_path()

        from rover_description.config import load_config as _load_config
        cfg = _load_config(config_path if config_path else None)

        d = cfg.drive
        self.wheel_radius = float(d.wheel_radius)
        self.track = float(d.track)
        self.effective_track_correction = float(
            self.get_parameter("effective_track_correction").value
        )
        self.max_wheel_velocity = float(self.get_parameter("max_wheel_velocity").value)

        effective_track = self.track * self.effective_track_correction
        self.get_logger().info(
            f"skid_steer_controller: wheel_radius={self.wheel_radius}, "
            f"track={self.track}, effective_track={effective_track:.3f}, "
            f"max_wheel_velocity={self.max_wheel_velocity}"
        )

        # ---- subscribers / publishers ------------------------------------
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_cb, 10
        )

        self.wheel_pubs: dict[str, rclpy.publisher.Publisher] = {}
        for name in ("wheel_front_left", "wheel_front_right",
                     "wheel_rear_left", "wheel_rear_right"):
            self.wheel_pubs[name] = self.create_publisher(
                Float64, f"/wheel/{name}/cmd_vel", 10
            )

    # ------------------------------------------------------------------ #
    def _cmd_vel_cb(self, msg: Twist) -> None:
        vx = msg.linear.x
        wz = msg.angular.z

        effective_track = self.track * self.effective_track_correction
        v_left = vx - wz * effective_track / 2.0
        v_right = vx + wz * effective_track / 2.0

        wl = self._clamp(v_left / self.wheel_radius)
        wr = self._clamp(v_right / self.wheel_radius)

        # front = rear on each side (skid steer)
        cmd = Float64()
        cmd.data = wl
        self.wheel_pubs["wheel_front_left"].publish(cmd)
        self.wheel_pubs["wheel_rear_left"].publish(cmd)
        cmd.data = wr
        self.wheel_pubs["wheel_front_right"].publish(cmd)
        self.wheel_pubs["wheel_rear_right"].publish(cmd)

    def _clamp(self, v: float) -> float:
        return max(-self.max_wheel_velocity, min(self.max_wheel_velocity, v))


def main() -> None:
    rclpy.init()
    node = SkidSteerController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
