"""Publishes the randomly sampled start pose on ``/cat_arena/initial_pose``.

The pose is a pure function of the run seed (see
:func:`cat_arena.start_pose.sample_start_pose`), so the launch file, the world
generator and this node all agree without passing a file around.

Consumers (module 4 localization, module 5 navigation, the judges' display) can
rely on exactly one topic:

* ``/cat_arena/initial_pose`` - ``geometry_msgs/msg/PoseWithCovarianceStamped``,
  transient-local, so a subscriber that starts later still receives it.

With ``also_publish_initialpose`` the same pose is mirrored on ``/initialpose``
for RViz's "2D Pose Estimate" tool and any tooling that expects the nav2 name.
"""

from __future__ import annotations

import math
import sys
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseWithCovariance, PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cat_arena.config import ConfigError, load_config
from cat_arena.start_pose import sample_start_pose

try:  # rclpy >= Humble has this; older/older-free environments may not
    from rclpy.executors import ExternalShutdownException as _ExternalShutdown
except Exception:  # pragma: no cover - defensive

    class _ExternalShutdown(Exception):  # type: ignore[no-redef]
        """Fallback so the except clause always exists."""


_INITIAL_POSE_TOPIC = "/cat_arena/initial_pose"
_NAV2_INITIAL_POSE_TOPIC = "/initialpose"


def _yaw_to_quaternion(yaw: float) -> tuple:
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


class StartPosePublisher(Node):
    """Samples one start pose per run and publishes it (latched)."""

    def __init__(self) -> None:
        super().__init__("cat_arena_start_pose")
        self.declare_parameter("seed", 0)
        self.declare_parameter("config", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("yaw_override_deg", 999.0)
        self.declare_parameter("also_publish_initialpose", True)
        self.declare_parameter("repeat_period_sec", 5.0)

        seed = int(self.get_parameter("seed").value)
        config_path = str(self.get_parameter("config").value) or None
        frame_id = str(self.get_parameter("frame_id").value)
        yaw_override = float(self.get_parameter("yaw_override_deg").value)
        repeat = float(self.get_parameter("repeat_period_sec").value)

        try:
            cfg = load_config(config_path)
        except ConfigError as exc:
            self.get_logger().fatal(f"cannot load arena configuration: {exc}")
            raise

        override = None if abs(yaw_override) > 360.0 else yaw_override
        self._pose = sample_start_pose(cfg, seed, yaw_override_deg=override)
        self._frame_id = frame_id

        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(PoseWithCovarianceStamped, _INITIAL_POSE_TOPIC, qos)
        self._pub_nav2: Optional[object] = None
        if bool(self.get_parameter("also_publish_initialpose").value):
            self._pub_nav2 = self.create_publisher(
                PoseWithCovarianceStamped, _NAV2_INITIAL_POSE_TOPIC, qos
            )

        self.get_logger().info(
            f"start pose: x={self._pose.x:.4f} y={self._pose.y:.4f} z={self._pose.z:.4f} "
            f"yaw={self._pose.yaw_deg:.2f} deg (seed {seed}, zone '{self._pose.zone}', "
            f"declared forward {self._pose.declared_forward_deg} deg)"
        )
        self._publish()
        if repeat > 0.0:
            self.create_timer(repeat, self._publish)

    def _publish(self) -> None:
        message = PoseWithCovarianceStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id
        pose = Pose()
        pose.position.x = float(self._pose.x)
        pose.position.y = float(self._pose.y)
        pose.position.z = float(self._pose.z)
        qx, qy, qz, qw = _yaw_to_quaternion(self._pose.yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        message.pose = PoseWithCovariance()
        message.pose.pose = pose
        # The simulator knows the spawn pose exactly; small but non-zero covariance
        # keeps downstream filters from treating it as an infinite-precision prior.
        covariance = [0.0] * 36
        covariance[0] = 0.0025
        covariance[7] = 0.0025
        covariance[14] = 0.0025
        covariance[21] = 0.0004
        covariance[28] = 0.0004
        covariance[35] = 0.0004
        message.pose.covariance = covariance
        self._pub.publish(message)
        if self._pub_nav2 is not None:
            self._pub_nav2.publish(message)  # type: ignore[attr-defined]

    @property
    def pose(self):
        return self._pose


def main(argv: Optional[list] = None) -> int:
    rclpy.init(args=argv)
    try:
        node = StartPosePublisher()
    except ConfigError:
        if rclpy.ok():
            rclpy.shutdown()
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        print(f"cat_arena_start_pose: {exc}", file=sys.stderr)
        if rclpy.ok():
            rclpy.shutdown()
        return 1
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except _ExternalShutdown:
        # launched via `ros2 launch`: the SIGINT handler already shut rclpy down
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
