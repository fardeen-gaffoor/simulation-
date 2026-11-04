#!/usr/bin/env python3
"""Simple hardcoded drive node for demo.

Drives the rover to the excavation (dig) zone, waits, then drives to the berm
construction (dump) zone and waits.  No perception or planner required - just
odometry + cmd_vel.  Good enough for a live demo.

State machine (each leg: TURN -> DRIVE -> WAIT):
  0 = turn toward dig zone
  1 = drive to dig zone
  2 = dig wait
  3 = turn toward dump zone
  4 = drive to dump zone
  5 = dump wait
  6 = done (hold stopped)
"""

import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


# Arena geometry (from cat_arena/config/arena.yaml)
DIG_X = 8.0        # excavation zone, east side
DIG_Y = 2.5
DUMP_X = 1.0       # berm target, construction zone
DUMP_Y = 0.75

DRIVE_SPEED = 0.25     # m/s
TURN_SPEED = 0.5       # rad/s
ARRIVE_DIST = 0.40     # m, "close enough" to a target
YAW_TOL = 0.10         # rad, heading tolerance before driving forward

# Phase timing (seconds)
DIG_WAIT = 5.0
DUMP_WAIT = 5.0
TURN_TIMEOUT = 30.0    # safety: don't turn forever
DRIVE_TIMEOUT = 60.0   # safety: don't drive forever


class SimpleDrive(Node):
    """Hardcoded drive sequence: dig zone -> dig -> dump zone -> dump."""

    def __init__(self) -> None:
        super().__init__("simple_drive")

        # Sensor data QoS for odom (matches ros_gz_bridge defaults)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self._on_odom, qos
        )

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False

        # State machine:
        # 0 = turn toward dig zone, 1 = drive to dig zone, 2 = dig wait,
        # 3 = turn toward dump zone, 4 = drive to dump zone, 5 = dump wait,
        # 6 = done
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

    def _stop(self) -> Twist:
        return self._twist(0.0, 0.0)

    # -- tick -------------------------------------------------------------- #
    def _tick(self) -> None:
        if not self.odom_received:
            # Still waiting for first odom - publish zero to be safe
            self.cmd_pub.publish(self._stop())
            return

        now = self._elapsed()
        twist = Twist()  # zero by default

        # ---- state 0: turn toward dig zone ------------------------------- #
        if self.state == 0:
            dx = DIG_X - self.x
            dy = DIG_Y - self.y
            target_yaw = math.atan2(dy, dx)
            yaw_err = self._wrap(target_yaw - self.yaw)
            self.get_logger().info(
                f"[TURN->DIG] pos=({self.x:.2f},{self.y:.2f}) yaw={self.yaw:.2f} "
                f"target_yaw={target_yaw:.2f} err={yaw_err:.2f} t={now:.1f}s"
            )
            if now > TURN_TIMEOUT:
                self.get_logger().warn("turn timeout - forcing drive")
                self.state = 1
                self.state_start = self.get_clock().now()
            elif abs(yaw_err) > YAW_TOL:
                twist = self._twist(0.0, TURN_SPEED if yaw_err > 0 else -TURN_SPEED)
            else:
                self.get_logger().info("facing dig zone - driving")
                self.state = 1
                self.state_start = self.get_clock().now()

        # ---- state 1: drive to dig zone ---------------------------------- #
        elif self.state == 1:
            dx = DIG_X - self.x
            dy = DIG_Y - self.y
            dist = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            yaw_err = self._wrap(target_yaw - self.yaw)
            self.get_logger().info(
                f"[DRIVE->DIG] dist={dist:.2f} yaw_err={yaw_err:.2f} t={now:.1f}s"
            )
            if now > DRIVE_TIMEOUT:
                self.get_logger().warn("drive timeout - starting dig wait")
                self.state = 2
                self.state_start = self.get_clock().now()
            elif abs(yaw_err) > YAW_TOL * 2.0:
                # Drifted too far - re-align
                self.get_logger().info("re-aligning heading")
                self.state = 0
                self.state_start = self.get_clock().now()
            elif dist > ARRIVE_DIST:
                twist = self._twist(DRIVE_SPEED, 0.0)
            else:
                self.get_logger().info("arrived at dig zone - digging")
                self.state = 2
                self.state_start = self.get_clock().now()

        # ---- state 2: dig wait ------------------------------------------- #
        elif self.state == 2:
            self.get_logger().info(f"[DIG] t={now:.1f}s")
            if now >= DIG_WAIT:
                self.get_logger().info("dig complete - turning to dump zone")
                self.state = 3
                self.state_start = self.get_clock().now()

        # ---- state 3: turn toward dump zone ------------------------------ #
        elif self.state == 3:
            dx = DUMP_X - self.x
            dy = DUMP_Y - self.y
            target_yaw = math.atan2(dy, dx)
            yaw_err = self._wrap(target_yaw - self.yaw)
            self.get_logger().info(
                f"[TURN->DUMP] pos=({self.x:.2f},{self.y:.2f}) yaw={self.yaw:.2f} "
                f"target_yaw={target_yaw:.2f} err={yaw_err:.2f} t={now:.1f}s"
            )
            if now > TURN_TIMEOUT:
                self.get_logger().warn("turn timeout - forcing drive")
                self.state = 4
                self.state_start = self.get_clock().now()
            elif abs(yaw_err) > YAW_TOL:
                twist = self._twist(0.0, TURN_SPEED if yaw_err > 0 else -TURN_SPEED)
            else:
                self.get_logger().info("facing dump zone - driving")
                self.state = 4
                self.state_start = self.get_clock().now()

        # ---- state 4: drive to dump zone --------------------------------- #
        elif self.state == 4:
            dx = DUMP_X - self.x
            dy = DUMP_Y - self.y
            dist = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)
            yaw_err = self._wrap(target_yaw - self.yaw)
            self.get_logger().info(
                f"[DRIVE->DUMP] dist={dist:.2f} yaw_err={yaw_err:.2f} t={now:.1f}s"
            )
            if now > DRIVE_TIMEOUT:
                self.get_logger().warn("drive timeout - starting dump wait")
                self.state = 5
                self.state_start = self.get_clock().now()
            elif abs(yaw_err) > YAW_TOL * 2.0:
                self.get_logger().info("re-aligning heading")
                self.state = 3
                self.state_start = self.get_clock().now()
            elif dist > ARRIVE_DIST:
                twist = self._twist(DRIVE_SPEED, 0.0)
            else:
                self.get_logger().info("arrived at dump zone - dumping")
                self.state = 5
                self.state_start = self.get_clock().now()

        # ---- state 5: dump wait ------------------------------------------ #
        elif self.state == 5:
            self.get_logger().info(f"[DUMP] t={now:.1f}s")
            if now >= DUMP_WAIT:
                self.get_logger().info("dump complete - mission finished")
                self.state = 6

        # ---- state 6: done ----------------------------------------------- #
        elif self.state == 6:
            pass  # hold stopped

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
