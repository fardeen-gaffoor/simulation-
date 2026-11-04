"""ROS glue shared by module 5's nodes - and the only place in the package that may.

The algorithm modules (``config`` .. ``mission``) import without ROS so their
tests run in-process on any machine; this module is the deliberate exception
that needs ``rclpy`` and the message packages, because that is exactly its job:
parameter declaration, clock handling, message <-> :class:`Pose3` conversion and
the QoS choices.

Three decisions are worth recording, because they are the ones that bite:

* **Subscriptions are best-effort.**  A best-effort subscriber matches both a
  best-effort and a reliable publisher, whereas a reliable subscriber silently
  receives nothing from a best-effort one (which is what ``ros_gz_bridge``
  produces).  Since module 5 subscribes to whatever is upstream, best-effort is
  the only safe default.
* **``/estop`` is the exception.**  Module 2 publishes it transient-local so a
  late subscriber still sees a latched stop.  A volatile subscriber would miss
  precisely the latched value, which is the one that matters, so that
  subscription is reliable + transient-local.
* **Time comes from the node clock.**  Every node runs with ``use_sim_time`` set
  by the launch file in the simulator, so ``now_seconds`` follows ``/clock`` and
  every timestamp in a run is comparable with the arena's ground truth.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import NavigationConfig, load_config

try:
    from geometry_msgs.msg import PoseWithCovarianceStamped
except ImportError:
    PoseWithCovarianceStamped = None

__all__ = [
    "declare",
    "declare_all",
    "load_node_config",
    "now_seconds",
    "stamp_to_seconds",
    "seconds_to_stamp",
    "make_header",
    "pose3_to_msg",
    "pose_to_pose_stamped",
    "msg_to_pose3",
    "pose_stamped_to_pose3",
    "PoseTracker",
    "sensor_qos",
    "latched_qos",
    "LogOnce",
    "split_node_arguments",
    "spin_until_shutdown",
    "node_main",
]


def declare(node: Any, name: str, default: Any, description: str = "") -> Any:
    """Declare a parameter on a node and return the effective value."""
    node.declare_parameter(name, default)
    return node.get_parameter(name).value


def declare_all(node: Any, specs: Sequence[Tuple[str, Any, str]]) -> Dict[str, Any]:
    """``declare`` several parameters, returning ``{name: value}``."""
    return {name: declare(node, name, default, description) for name, default, description in specs}


def load_node_config(node: Any, param: str = "config", override: str = "") -> NavigationConfig:
    """Load ``navigation.yaml`` using a node parameter (empty = installed default).

    ``override`` is what a command-line ``--config`` carried.  It becomes the
    parameter's *default*, so an explicit ``-p config:=...`` from a launch file
    still wins, while ``ros2 run rover_navigation navigation.py --config X`` does
    what it says.
    """
    value = str(declare(node, param, override or "") or "")
    config = load_config(value or None)
    node.get_logger().info(
        f"navigation.yaml: {config.path} (sha256 {config.sha256[:12]}, "
        f"arena {config.arena.length_x:g}x{config.arena.width_y:g} m, "
        f"footprint {config.robot.length_m:g}x{config.robot.width_m:g} m)"
    )
    return config


# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #
def now_seconds(node: Any) -> float:
    """The node's clock in seconds (sim time when ``use_sim_time`` is set)."""
    return float(node.get_clock().now().nanoseconds) * 1e-9


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def seconds_to_stamp(seconds: float) -> Any:
    """Build a ``builtin_interfaces/Time`` from seconds (clamped, never negative)."""
    from builtin_interfaces.msg import Time

    value = max(0.0, float(seconds))
    whole = int(math.floor(value))
    stamp = Time()
    stamp.sec = whole
    stamp.nanosec = int(round((value - whole) * 1e9))
    if stamp.nanosec >= 1000000000:  # pragma: no cover - float rounding
        stamp.sec += 1
        stamp.nanosec -= 1000000000
    return stamp


def make_header(node: Any, frame_id: str, stamp_seconds: Optional[float] = None) -> Any:
    """A ``std_msgs/Header`` on the node clock (or an explicit stamp)."""
    from std_msgs.msg import Header

    header = Header()
    header.frame_id = frame_id
    header.stamp = seconds_to_stamp(now_seconds(node) if stamp_seconds is None else stamp_seconds)
    return header


# --------------------------------------------------------------------------- #
# poses
# --------------------------------------------------------------------------- #
def pose3_to_msg(pose: Any) -> Any:
    """:class:`~rover_navigation.geometry.Pose3` -> ``geometry_msgs/Pose``."""
    from geometry_msgs.msg import Pose, Quaternion

    out = Pose()
    translation = pose.translation
    out.position.x = float(translation[0])
    out.position.y = float(translation[1])
    out.position.z = float(translation[2])
    qx, qy, qz, qw = pose.quaternion_xyzw
    out.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
    return out


def pose_to_pose_stamped(node: Any, pose: Any, frame_id: str, stamp_seconds: Optional[float] = None) -> Any:
    """``geometry_msgs/PoseStamped`` from a :class:`Pose3`."""
    from geometry_msgs.msg import PoseStamped

    out = PoseStamped()
    out.header = make_header(node, frame_id, stamp_seconds)
    out.pose = pose3_to_msg(pose)
    return out


def msg_to_pose3(message: Any) -> Any:
    """``geometry_msgs/Pose`` (or anything with position/orientation) -> ``Pose3``."""
    from .geometry import Pose3, quaternion_to_matrix

    position = message.position
    orientation = message.orientation
    return Pose3(
        quaternion_to_matrix((orientation.x, orientation.y, orientation.z, orientation.w)),
        (float(position.x), float(position.y), float(position.z)),
    )


def pose_stamped_to_pose3(message: Any) -> Any:
    """``geometry_msgs/PoseStamped`` -> ``Pose3``."""
    return msg_to_pose3(message.pose)


# --------------------------------------------------------------------------- #
# the rover's pose in the map frame
# --------------------------------------------------------------------------- #
class PoseTracker:
    """``map_T_base`` from module 4's ``map -> odom`` and the drive's odometry.

    Module 4 anchors the arena frame with the wall AprilTags and publishes the
    correction; the drive publishes ``odom -> base_link``.  Neither alone is the
    rover's pose, and the composition is the same one module 4's obstacle
    detector does - which is what makes the obstacles and the rover's own
    position agree.

    Without a correction the pose is ``None`` rather than a guess: an obstacle
    map in a frame the rover cannot locate itself in is worse than no map.

    On startup, the arena publishes the exact start pose on
    ``/cat_arena/initial_pose`` (``PoseWithCovarianceStamped`` in the map frame).
    We use this to seed the ``map -> odom`` transform so the rover has a valid
    pose from the beginning, before any tag fixes arrive.
    """

    def __init__(self, preferred_topic: str = "", node: Any = None) -> None:
        self.odom: Any = None
        self.odom_topic = ""
        self.map_to_odom: Any = None
        self.pose: Any = None
        self.pose_stamp = 0.0
        self._preferred = preferred_topic
        self._node = node
        self._initial_pose_received = False
        if node is not None and PoseWithCovarianceStamped is not None:
            self._initial_pose_sub = node.create_subscription(
                PoseWithCovarianceStamped,
                "/cat_arena/initial_pose",
                self._on_initial_pose,
                1,  # transient-local not needed; it's published once at startup
            )
        else:
            self._initial_pose_sub = None

    def _on_initial_pose(self, message: Any) -> None:
        """Initialize map->odom from the arena's published start pose."""
        if self._initial_pose_received:
            return
        # The initial pose is in the map frame. At startup, odom frame = base_link
        # at the start pose, so map->odom = initial_pose.
        self.map_to_odom = msg_to_pose3(message.pose.pose)
        self._initial_pose_received = True
        if self._node is not None:
            self._node.get_logger().info(
                f"Initialized map->odom from /cat_arena/initial_pose: "
                f"({self.map_to_odom.translation[0]:.2f}, "
                f"{self.map_to_odom.translation[1]:.2f}, "
                f"yaw {self.map_to_odom.yaw:.2f})"
            )
        if self.odom is not None:
            self.pose = self.map_to_odom.compose(self.odom)
            self.pose_stamp = now_seconds(self._node) if self._node is not None else 0.0

    def on_odometry(self, message: Any, topic: str) -> None:
        # The first configured topic that actually publishes wins; an explicit
        # preference is honoured, otherwise whoever speaks first keeps the slot.
        if self.odom_topic and self.odom_topic != topic:
            if self.odom_topic != self._preferred:
                return
            if topic != self._preferred:
                return
        self.odom = msg_to_pose3(message.pose.pose)
        self.odom_topic = topic
        if self.map_to_odom is None:
            self.pose = None
            return
        self.pose = self.map_to_odom.compose(self.odom)
        self.pose_stamp = stamp_to_seconds(message.header.stamp)

    def on_localization(self, message: Any) -> None:
        # Module 4's tag_localizer publishes a correction.  This supersedes the
        # initial pose once tags are visible.  Ignore identity transforms which
        # indicate "no valid tag fix yet" - the tag_localizer publishes identity
        # before any tags are detected.
        new_map_to_odom = msg_to_pose3(message.map_to_odom.pose)
        # Check if the transform is identity (no rotation, zero translation)
        # within a small tolerance - this indicates "no fix yet" from the tag localizer.
        trans = new_map_to_odom.translation
        if abs(trans[0]) < 1e-3 and abs(trans[1]) < 1e-3 and abs(trans[2]) < 1e-3:
            q = new_map_to_odom.quaternion_xyzw
            # Identity quaternion is (0, 0, 0, 1)
            if abs(q[0]) < 1e-3 and abs(q[1]) < 1e-3 and abs(q[2]) < 1e-3 and abs(q[3] - 1.0) < 1e-3:
                # Identity transform - no valid tag fix yet, ignore
                return
        # Valid correction - apply it
        self.map_to_odom = new_map_to_odom
        if self.odom is not None:
            self.pose = self.map_to_odom.compose(self.odom)


# --------------------------------------------------------------------------- #
# qos
# --------------------------------------------------------------------------- #
def sensor_qos(depth: int = 5) -> Any:
    """Best-effort, keep-last-``depth``: matches any publisher module 5 talks to."""
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        depth=max(1, int(depth)),
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def latched_qos(depth: int = 1) -> Any:
    """Reliable + transient-local: required to see a latched ``/estop``."""
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        depth=max(1, int(depth)),
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
class LogOnce:
    """Emit each distinct message once (or once per ``period_s``)."""

    def __init__(self, node: Any, period_s: float = 0.0):
        self.node = node
        self.period_s = float(period_s)
        self._seen: Dict[str, float] = {}

    def _should_log(self, key: str, now: Optional[float]) -> bool:
        timestamp = now_seconds(self.node) if now is None else float(now)
        previous = self._seen.get(key)
        if previous is not None and (self.period_s <= 0.0 or timestamp - previous < self.period_s):
            return False
        self._seen[key] = timestamp
        return True

    def warn(self, key: str, message: str, now: Optional[float] = None) -> None:
        if self._should_log(key, now):
            self.node.get_logger().warn(message)

    def info(self, key: str, message: str, now: Optional[float] = None) -> None:
        if self._should_log(key, now):
            self.node.get_logger().info(message)

    def error(self, key: str, message: str, now: Optional[float] = None) -> None:
        if self._should_log(key, now):
            self.node.get_logger().error(message)

    def reset(self) -> None:
        self._seen.clear()


# --------------------------------------------------------------------------- #
# process plumbing
# --------------------------------------------------------------------------- #
def split_node_arguments(argv: Optional[Sequence[str]]) -> Tuple[Optional[List[str]], List[str]]:
    """``(ros_args, own_args)``: keep ``--ros-args ...`` for rclpy, the rest for argparse.

    A module-5 node is normally started by a launch file with ``--ros-args -p``,
    but it must also be runnable by hand with plain flags, and ``rclpy.init``
    rejects flags it does not know.  Splitting them here keeps both working -
    including the ``argv is None`` case, where the arguments come from
    ``sys.argv`` and a launch file has put ``--ros-args`` there.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in arguments:
        index = arguments.index("--ros-args")
        return arguments[index:], arguments[:index]
    return None, arguments


def spin_until_shutdown(node: Any) -> None:
    """``rclpy.spin`` with a KeyboardInterrupt-safe shutdown."""
    import rclpy

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def node_main(
    argv: Optional[Sequence[str]],
    description: str,
    build: Any,
    add_arguments: Any = None,
    epilog: str = "",
) -> int:
    """Standard entry point for a module-5 node.

    ``build(args) -> rclpy.node.Node`` is called after ``rclpy.init`` so the node
    can log why it failed; ``add_arguments(parser)`` adds node-specific flags.
    Every node gets ``--config`` for free.
    """
    import argparse

    import rclpy

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]) or "node",
        description=description,
        epilog=epilog or None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="navigation.yaml (empty = installed default)")
    if add_arguments is not None:
        add_arguments(parser)
    ros_args, own_args = split_node_arguments(argv)
    args = parser.parse_args(own_args)

    rclpy.init(args=ros_args)
    node = None
    try:
        node = build(args)
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return 0
    except Exception as exc:
        print(f"{parser.prog}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
