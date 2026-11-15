#!/usr/bin/env python3
"""Relay node: subscribes to /model/rover/odometry (from Gazebo) and
republishes as /odom for downstream consumers."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class OdomRelay(Node):
    """Simple relay from /model/rover/odometry to /odom."""

    def __init__(self) -> None:
        super().__init__("odom_relay")
        self.sub = self.create_subscription(
            Odometry, "/model/rover/odometry", self._cb, 10
        )
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.get_logger().info("odom_relay: bridging /model/rover/odometry -> /odom")

    def _cb(self, msg: Odometry) -> None:
        # Ensure the frame IDs are correct for downstream consumers
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_footprint"
        self.pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = OdomRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()