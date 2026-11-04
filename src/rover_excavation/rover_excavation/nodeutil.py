"""ROS glue for module 6's node - and the only place in the package that may be.

The algorithm modules (``config`` .. ``backend``) import without ROS, so their
tests run in-process on any machine; this module is the deliberate exception
that needs ``rclpy`` and the message packages, because that is its job: parameter
declaration, the clock, and turning module 6's plain records into messages.

Two decisions are inherited from module 5 and for the same reasons:

* **Subscriptions are best-effort.**  A best-effort subscriber matches both a
  best-effort and a reliable publisher, whereas a reliable subscriber silently
  receives nothing from a best-effort one - which is what ``ros_gz_bridge``
  produces.  ``/estop`` is the exception: module 2 publishes it transient-local
  so a late subscriber still sees a latched stop.
* **Time comes from the node clock**, so ``use_sim_time`` makes every stamp in a
  run comparable with the arena's ground truth.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import ExcavationConfig, load_config
from .mechanism import MechanismSnapshot
from .sand import BermField

__all__ = [
    "declare",
    "declare_all",
    "load_node_config",
    "now_seconds",
    "stamp_to_seconds",
    "seconds_to_stamp",
    "make_header",
    "quaternion_to_yaw",
    "pose_stamped_to_xy_yaw",
    "mechanism_status_msg",
    "berm_status_msg",
    "berm_marker_array",
    "sensor_qos",
    "latched_qos",
    "LogOnce",
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


def load_node_config(node: Any, param: str = "config", override: str = "") -> ExcavationConfig:
    """Load ``excavation.yaml`` using a node parameter (empty = installed default)."""
    value = str(declare(node, param, override or "") or "")
    config = load_config(value or None)
    node.get_logger().info(
        f"excavation.yaml: {config.path} (sha256 {config.sha256[:12]}, "
        f"backend {config.backend.kind}, berm target {config.berm_target.width:g}x"
        f"{config.berm_target.height:g} m x {config.sand.berm.target_height_m:g} m)"
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
# the goal's work pose
# --------------------------------------------------------------------------- #
def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw of a planar rotation; module 6 only ever needs the heading."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pose_stamped_to_xy_yaw(message: Any) -> Tuple[float, float, float]:
    """``geometry_msgs/PoseStamped`` -> ``(x, y, yaw)`` in its own frame."""
    position = message.pose.position
    orientation = message.pose.orientation
    return (
        float(position.x),
        float(position.y),
        quaternion_to_yaw(
            float(orientation.x), float(orientation.y), float(orientation.z), float(orientation.w)
        ),
    )


# --------------------------------------------------------------------------- #
# messages
# --------------------------------------------------------------------------- #
def mechanism_status_msg(
    node: Any,
    config: ExcavationConfig,
    snapshot: MechanismSnapshot,
    tub_fill_m3: float,
    tub_fraction: float,
) -> Any:
    """``rover_excavation_msgs/MechanismStatus`` from a mechanism snapshot."""
    from rover_excavation_msgs.msg import MechanismStatus

    message = MechanismStatus()
    message.header = make_header(node, config.frames.base)
    message.state = int(snapshot.state)
    message.state_name = snapshot.state_name
    message.travel_ready = bool(snapshot.travel_ready)
    message.ladder_pitch_rad = float(snapshot.ladder_pitch_rad)
    message.ladder_depth_m = float(snapshot.ladder_depth_m)
    message.belt_velocity_radps = float(snapshot.belt_velocity_radps)
    message.belt_linear_mps = float(snapshot.belt_linear_mps)
    message.tub_gate = float(snapshot.tub_gate)
    message.tub_fill_m3 = float(tub_fill_m3)
    message.tub_capacity_m3 = float(config.mechanism.tub.capacity_m3)
    message.tub_fraction = float(tub_fraction)
    message.message = snapshot.message
    return message


def berm_status_msg(node: Any, config: ExcavationConfig, berm: BermField) -> Any:
    """``rover_excavation_msgs/BermStatus`` from the height field."""
    from rover_excavation_msgs.msg import BermStatus

    message = BermStatus()
    message.header = make_header(node, config.frames.map)
    message.target_volume_m3 = float(berm.target_volume_m3)
    message.placed_volume_m3 = float(berm.placed_volume_m3)
    message.spilled_volume_m3 = float(berm.spilled_volume_m3)
    message.completion = float(berm.completion)
    message.mean_height_m = float(berm.mean_height_m)
    message.peak_height_m = float(berm.peak_height_m)
    message.coverage = float(berm.coverage)
    message.cells_total = int(berm.cells_total)
    message.cells_filled = int(berm.cells_filled)
    message.target_height_m = float(berm.target_height_m)
    message.dumps = int(berm.dumps)
    message.message = berm.summary()
    return message


def berm_marker_array(node: Any, config: ExcavationConfig, berm: BermField) -> Any:
    """One cube per height-field cell, so rviz shows the berm the way it is.

    A ``CUBE_LIST`` cannot vary its cubes' size, so each cell is its own marker
    with ``scale.z`` set to its own height.  That is 25 x 18 = 450 markers over
    the target, published only when the berm has changed - cheap enough for
    rviz, and it is the same height field ``BermStatus`` reports.
    """
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA
    from visualization_msgs.msg import Marker, MarkerArray

    array = MarkerArray()
    dx = berm.dx
    dy = berm.dy
    target_height = max(1e-6, berm.target_height_m)
    for index, (x, y, height) in enumerate(berm.cells()):
        marker = Marker()
        marker.header = make_header(node, config.frames.map)
        marker.ns = "berm"
        marker.id = index
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=float(x), y=float(y), z=0.5 * float(max(height, 0.0)))
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(dx)
        marker.scale.y = float(dy)
        marker.scale.z = float(max(height, 1e-3))
        fraction = min(1.0, float(height) / (2.0 * target_height))
        # Barely-there cells are nearly transparent; a finished berm is solid.
        marker.color = ColorRGBA(
            r=0.84 - 0.30 * fraction,
            g=0.78 - 0.28 * fraction,
            b=0.58 - 0.30 * fraction,
            a=0.25 + 0.75 * min(1.0, fraction * 2.0),
        )
        array.markers.append(marker)
    return array


# --------------------------------------------------------------------------- #
# qos
# --------------------------------------------------------------------------- #
def sensor_qos(depth: int = 5) -> Any:
    """Best-effort, keep-last-``depth``: matches any publisher module 6 talks to."""
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
    """``(ros_args, own_args)``: keep ``--ros-args ...`` for rclpy, the rest for argparse."""
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
    """Standard entry point for a module-6 node.

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
    parser.add_argument("--config", default="", help="excavation.yaml (empty = installed default)")
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
