#!/usr/bin/env python3
"""Simple hardcoded drive node for demo.

Drives the rover to the excavation (dig) zone, waits, then drives to the berm
construction (dump) zone and waits.  No perception or planner required - just
odometry + cmd_vel.  Good enough for a live demo.
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


# Arena geometry (from cat_arena/config/arena.yaml)
DIG_X = 8.0        # excavation zone, east side
DIG_Y = 2.5
DUMP_X = 1.0       # berm target, construction zone
DUMP_Y = 0.75

DRIVE_SPEED = 0.30     # m/s
TURN_SPEED = 0.6       # rad/s
ARRIVE_DIST = 0.45     # m, "close enough" to a target
YAW_TOL = 0.25         # rad, heading tolerance before driving forward

# Phase timing (seconds)
DIG_WAIT = 6.0
DUMP_WAIT = 6.0


class SimpleDrive(Node):
    """Hardcoded drive sequence: dig zone -> dig -> dump zone -> dump."""

    def __init__(self) -> None:
        super().__init__("simple_drive")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._on_odom, 10
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False

        # State machine
        # 0 = drive to dig zone, 1 = dig wait, 2 = drive to dump zone,
        # 3 = dump wait, 4 = done
        self.state = 0
        self.state_start = self.get_clock().now()

        self.timer = self.create_timer(0.1, self._tick)
        self.get_logger().info("simple_drive: waiting for odometry...")

    # -- callbacks --------------------------------------------------------- #
    def _on_odom(self, msg: Odometry) -> None:
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)
        self.odom_received = True

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _wrap(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _elapsed(self) -> float:
        return (self.get_clock().now() - self.state_start).nanoseconds / 1e9

    def _twist(self, linear: float, angular: float) -> Twist:
        t = Twist()
        t.linear.x = float(linear)
        t.angular.z = float(angular)
        return t

    # -- tick -------------------------------------------------------------- #
    def _tick(self) -> None:
        if not self.odom_received:
            return

        twist = Twist()  # zero by default

        if self.state == 0:
            # Drive to dig zone
            dx = DIG_X - self.x
            dy = DIG_Y - self.y
            dist = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            yaw_err = self._wrap(target_yaw - self.yaw)

            self.get_logger().debug(
                f"state=DRIVE_TO_DIG dist={dist:.2f} yaw_err={yaw_err:.2f}"
            )

            if abs(yaw_err) > YAW_TOL:
                # Turn in place to face the dig zone
                twist = self._twist(
                    0.0, TURN_SPEED if yaw_err > 0 else -TURN_SPEED
                )
            elif dist > ARRIVE_DIST:
                # Drive forward toward the dig zone
                twist = self._twist(DRIVE_SPEED, 0.0)
            else:
                self.get_logger().info("Arrived at dig zone - starting dig")
                self.state = 1
                self.state_start = self.get_clock().now()

        elif self.state == 1:
            # Dig wait
            self.get_logger().debug(f"state=DIG_WAIT elapsed={self._elapsed():.1f}s")
            if self._elapsed() >= DIG_WAIT:
                self.get_logger().info("Dig complete - driving to dump zone")
                self.state = 2
                self.state_start = self.get_clock().now()

        elif self.state == 2:
            # Drive to dump zone
            dx = DUMP_X - self.x
            dy = DUMP_Y - self.y
            dist = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            yaw_err = self._wrap(target_yaw - self.yaw)

            self.get_logger().debug(
                f"state=DRIVE_TO_DUMP dist={dist:.2f} yaw_err={yaw_err:.2f}"
            )

            if abs(yaw_err) > YAW_TOL:
                twist = self._twist(
                    0.0, TURN_SPEED if yaw_err > 0 else -TURN_SPEED
                )
            elif dist > ARRIVE_DIST:
                twist = self._twist(DRIVE_SPEED, 0.0)
            else:
                self.get_logger().info("Arrived at dump zone - starting dump")
                self.state = 3
                self.state_start = self.get_clock().now()

        elif self.state == 3:
            # Dump wait
            self.get_logger().debug(f"state=DUMP_WAIT elapsed={self._elapsed():.1f}s")
            if self._elapsed() >= DUMP_WAIT:
                self.get_logger().info("Dump complete - mission finished")
                self.state = 4

        elif self.state == 4:
            # Done - hold stopped
            pass

        self.cmd_pub.publish(twist)


def main() -> None:
    rclpy.init()
    node = SimpleDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
