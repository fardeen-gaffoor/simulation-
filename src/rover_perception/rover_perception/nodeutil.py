"""ROS glue shared by module 4's nodes - and the only place in the package that may.

The algorithm modules (``config`` .. ``localizer``) import without ROS so their
tests can run in-process on any machine; this module is the deliberate exception
that needs ``rclpy`` and the message packages, because that is exactly its job:
parameter declaration, clock handling, message <-> :class:`Pose3` conversion and
the QoS choices the camera topics need.

Keeping it in one place means the five nodes agree on all of it.  Two decisions
are worth recording:

* **Camera subscriptions are best-effort.**  The images come over
  ``ros_gz_bridge``, and the bridge's image publishers are best-effort; a
  best-effort subscriber also matches a reliable publisher, so this works either
  way, whereas a reliable subscriber drops frames against a best-effort gz topic.
* **Time comes from the node clock.**  Every node runs with ``use_sim_time`` set
  by the launch file in the simulator, so ``now_seconds`` follows ``/clock`` and
  every timestamp in a run is comparable with the arena's ground truth.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .config import PerceptionConfig, load_config
from .geometry import Pose3

__all__ = [
    "declare",
    "declare_all",
    "load_node_config",
    "now_seconds",
    "stamp_to_seconds",
    "seconds_to_stamp",
    "pose3_to_msg",
    "pose_with_covariance",
    "msg_to_pose3",
    "make_header",
    "sensor_qos",
    "json_qos",
    "LogOnce",
    "default_artifacts_dir",
    "resolve_tags_path",
    "spin_until_shutdown",
    "split_node_arguments",
    "node_main",
]


def declare(node: Any, name: str, default: Any, description: str = "") -> Any:
    """Declare a parameter on a node and return the effective value."""
    node.declare_parameter(name, default)
    return node.get_parameter(name).value


def declare_all(node: Any, specs: Sequence[Tuple[str, Any, str]]) -> Dict[str, Any]:
    """``declare`` several parameters, returning ``{name: value}``."""
    return {name: declare(node, name, default, description) for name, default, description in specs}


def load_node_config(node: Any, param: str = "config", override: str = "") -> PerceptionConfig:
    """Load ``perception.yaml`` using a node parameter (empty = installed default).

    ``override`` is what a command-line ``--config`` carried.  It becomes the
    parameter's *default*, so an explicit ``-p config:=...`` from a launch file
    still wins, while ``ros2 run rover_perception camera_manager.py --config X``
    does what it says.
    """
    value = str(declare(node, param, override or "") or "")
    config = load_config(value or None)
    node.get_logger().info(
        f"perception.yaml: {config.path} (sha256 {config.sha256[:12]}, "
        f"cameras {', '.join(config.camera_names)})"
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
def pose3_to_msg(pose: Pose3) -> Any:
    """:class:`Pose3` -> ``geometry_msgs/Pose``."""
    from geometry_msgs.msg import Pose, Quaternion

    out = Pose()
    translation = pose.translation
    out.position.x = float(translation[0])
    out.position.y = float(translation[1])
    out.position.z = float(translation[2])
    qx, qy, qz, qw = pose.quaternion_xyzw
    out.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
    return out


def pose_with_covariance(pose: Pose3, covariance: Optional[Sequence[float]] = None) -> Any:
    """``geometry_msgs/PoseWithCovariance`` from a pose (identity covariance by default)."""
    from geometry_msgs.msg import PoseWithCovariance

    message = PoseWithCovariance()
    message.pose = pose3_to_msg(pose)
    if covariance is not None:
        message.covariance = [float(value) for value in covariance]
    return message


def msg_to_pose3(message: Any) -> Pose3:
    """``geometry_msgs/Pose`` (or anything with ``position``/``orientation``) -> :class:`Pose3`."""
    from .geometry import quaternion_to_matrix

    position = message.position
    orientation = message.orientation
    return Pose3(
        quaternion_to_matrix((orientation.x, orientation.y, orientation.z, orientation.w)),
        (float(position.x), float(position.y), float(position.z)),
    )


# --------------------------------------------------------------------------- #
# qos
# --------------------------------------------------------------------------- #
def sensor_qos(depth: int = 5) -> Any:
    """Best-effort, keep-last-``depth``: the right profile for camera streams."""
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        depth=max(1, int(depth)),
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def json_qos(depth: int = 1) -> Any:
    """Reliable, transient-local-friendly profile for small status messages."""
    from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

    return QoSProfile(
        depth=max(1, int(depth)),
        history=QoSHistoryPolicy.KEEP_LAST,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
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

    def warn(self, key: str, message: str, now: Optional[float] = None) -> None:
        timestamp = now_seconds(self.node) if now is None else float(now)
        previous = self._seen.get(key)
        if previous is not None and (self.period_s <= 0.0 or timestamp - previous < self.period_s):
            return
        self._seen[key] = timestamp
        self.node.get_logger().warn(message)

    def info(self, key: str, message: str, now: Optional[float] = None) -> None:
        timestamp = now_seconds(self.node) if now is None else float(now)
        previous = self._seen.get(key)
        if previous is not None and (self.period_s <= 0.0 or timestamp - previous < self.period_s):
            return
        self._seen[key] = timestamp
        self.node.get_logger().info(message)

    def reset(self) -> None:
        self._seen.clear()


# --------------------------------------------------------------------------- #
# artefact paths
# --------------------------------------------------------------------------- #
def default_artifacts_dir() -> str:
    """Where generated tag artefacts live when nothing else is configured."""
    override = os.environ.get("CAT_PERCEPTION_ARTIFACTS", "")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(os.path.expanduser("~"), "cat", "perception")


def resolve_tags_path(cfg: PerceptionConfig) -> str:
    """Where ``tags.yaml`` is (``artifacts.output_dir`` + ``artifacts.tags_file``)."""
    directory = cfg.artifacts.output_dir or default_artifacts_dir()
    return os.path.join(os.path.abspath(os.path.expanduser(directory)), cfg.artifacts.tags_file)


def spin_until_shutdown(node: Any) -> None:
    """``rclpy.spin`` with a KeyboardInterrupt-safe shutdown (script boilerplate)."""
    import rclpy

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def split_node_arguments(argv: Optional[Sequence[str]]) -> Tuple[Optional[List[str]], List[str]]:
    """``(ros_args, own_args)``: keep ``--ros-args ...`` for rclpy, the rest for argparse.

    A module-4 node is normally started by a launch file with ``--ros-args -p``
    (and ``--params-file``), but it must also be runnable by hand with plain
    flags, and ``rclpy.init`` rejects flags it does not know.  Splitting them here
    keeps both working - **including the ``argv is None`` case**, where the
    arguments come from ``sys.argv``: a launch file puts ``--ros-args`` there, so
    that path must be split exactly like an explicit ``argv``.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--ros-args" in arguments:
        index = arguments.index("--ros-args")
        return arguments[index:], arguments[:index]
    return None, arguments


def node_main(
    argv: Optional[Sequence[str]],
    description: str,
    build: Any,
    add_arguments: Any = None,
    epilog: str = "",
) -> int:
    """Standard entry point for a module-4 node.

    ``build(args) -> rclpy.node.Node`` is called after ``rclpy.init`` so the
    node can log why it failed; ``add_arguments(parser)`` adds node-specific
    flags.  Every node gets ``--config`` for free.
    """
    import argparse

    import rclpy

    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]) or "node",
        description=description,
        epilog=epilog or None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="perception.yaml (empty = installed default)")
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
