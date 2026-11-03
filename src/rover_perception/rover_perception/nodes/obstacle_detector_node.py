"""``obstacle_detector``: boulders and craters on ``cam_nav`` -> ``ObstacleArray``.

Started by ``perception.launch.py`` as ``perception_obstacle_detector``.  This is
the node that produces the only obstacle knowledge module 5 is allowed to have -
live camera perception, in the map frame, with the evidence attached.  The
competition forbids handing the autonomy a precomputed obstacle list, and it also
forbids using the arena walls as sensors, so two things are structural here, not
optional:

* **every obstacle comes from a frame that was actually seen.**  The pipeline is
  one function call per stage, each in its own tested module -
  ``backend.detect`` (image space) -> ``groundproj.observe_detection`` (flat
  ground, map frame) -> ``tracking.Tracker.update`` (one track per object) ->
  ``obstacles.mask_tracks`` (the wall mask) -> ``obstacles.build_obstacle_payload``
  - and each stage's rejections are counted, not swallowed.
* **no fix, no map.**  without a ``map -> odom`` correction the rover's pose in
  the arena is unknown, so the frame is dropped and counted as
  ``frames_without_fix`` instead of publishing obstacles at invented coordinates.

The three pose inputs the projection needs (``map -> odom`` from
``tag_localizer``, ``odom -> base_link`` from odometry, and the camera extrinsics
from ``calibration.yaml``) are deliberately kept separate here, so a failure can
be attributed to the right one.
"""

from __future__ import annotations

import argparse
import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from rover_perception_msgs.msg import LocalizationState, Obstacle, ObstacleArray

from .. import imagecodec, nodeutil, obstacles as obstacle_module
from ..calibration_ref import load_calibration
from ..config import PerceptionConfigError
from ..cpu import CpuMeter, FpsMeter, LatencyMeter
from ..detector import annotate, create_backend
from ..geometry import Pose3
from ..groundproj import GroundObservation, Rejection, observe_detection
from ..localizer import OdomBuffer
from ..obstacles import build_obstacle_payload, mask_tracks, pointcloud_dtype, pointcloud_payload
from ..tracking import Tracker

__all__ = ["ObstacleDetectorNode", "main", "add_arguments", "STATS_LAYOUT"]

NODE_NAME = "perception_obstacle_detector"

#: Field names of the ``Float32MultiArray`` published on ``detector.stats_topic``.
#: ``health_monitor`` reads this; keeping the names here means the layout is in
#: one place instead of being an index convention two nodes have to remember.
STATS_LAYOUT: Tuple[str, ...] = (
    "fps",
    "latency_ms",
    "cpu_percent",
    "detections_last_frame",
    "accepted_last_frame",
    "rejected_last_frame",
    "tracks_live",
    "frames_processed",
)


class ObstacleDetectorNode(Node):
    """Camera-only boulder/crater detector with tracking and the arena-wall mask."""

    def __init__(self, config_override: str = "", debug_image: bool = False):
        super().__init__(NODE_NAME)
        self.config = nodeutil.load_node_config(self, override=config_override)
        cfg = self.config.detector
        self.arena = self.config.arena
        if not cfg.enabled:
            raise PerceptionConfigError("detector.enabled is false; do not start this node")

        calibration = load_calibration(self.config.artifacts.calibration)
        self.camera = calibration.camera(cfg.camera)
        self.backend = create_backend(cfg, self.camera)
        self.tracker = Tracker(self.config.tracking)
        self._log = nodeutil.LogOnce(self)
        self.publish_hz = float(
            nodeutil.declare(self, "publish_hz", float(cfg.publish_hz))
        )
        if self.publish_hz <= 0.0:
            self.publish_hz = cfg.publish_hz
        self.debug_image = bool(nodeutil.declare(self, "debug_image", bool(debug_image)))

        self.map_to_odom = Pose3.identity()
        self.localization_available = False
        self.odom_buffer = OdomBuffer(capacity=64)
        self.active_odom_topic: Optional[str] = None
        self.tracks: List[Any] = []
        self.counters: Dict[str, int] = {
            "frames_processed": 0,
            "dropped_wall": 0,
            "dropped_range": 0,
            "frames_without_fix": 0,
        }
        self.rejection_reasons: Dict[str, int] = {}
        self.frames_seen = 0
        self.detections_seen = 0
        self.accepted_seen = 0
        self.detections_last_frame = 0
        self.accepted_last_frame = 0
        self.rejected_last_frame = 0
        self.last_detections: List[Any] = []
        self.fps = FpsMeter(window_s=2.0)
        self.latency = LatencyMeter(window_s=5.0)
        self.cpu = CpuMeter()

        self.obstacles_publisher = self.create_publisher(ObstacleArray, cfg.obstacles_topic, 10)
        self.stats_publisher = self.create_publisher(Float32MultiArray, cfg.stats_topic, 10)
        self.cloud_publisher = (
            self.create_publisher(PointCloud2, cfg.cloud_topic, 10) if cfg.cloud_enabled else None
        )
        self.image_subscription = self.create_subscription(
            Image, self.camera.topic, self._on_image, nodeutil.sensor_qos()
        )
        self.odom_subscriptions = [
            self.create_subscription(Odometry, topic, partial(self._on_odometry, topic), 20)
            for topic in self.config.localization.odometry_topics
        ]
        self.localization_subscription = self.create_subscription(
            LocalizationState,
            self.config.localization.state_topic,
            self._on_localization,
            10,
        )
        self.create_timer(1.0 / self.publish_hz, self._on_publish)
        self.create_timer(5.0, self._log_stats)
        if self.debug_image:
            self.get_logger().warn(
                "--debug-image is accepted but module 4 publishes no obstacle debug image; "
                "use 'ros2 run rover_dataset preview_labels.py' or rqt_image_view on the "
                "tag debug topic instead"
            )

        self.get_logger().info(
            f"detector '{self.backend.name}' on {self.camera.topic} "
            f"({self.camera.width}x{self.camera.height}, fx {self.camera.fx:.1f}) -> "
            f"{cfg.obstacles_topic} at {self.publish_hz:.0f} Hz"
        )
        for key, value in self.backend.describe().items():
            self.get_logger().debug(f"  {key}: {value}")
        self.get_logger().info(
            f"projection: {cfg.ground.contact_point}, {cfg.ground.min_range_m}"
            f"..{cfg.ground.max_range_m} m, wall margin {self.arena.wall_margin_m} m; "
            f"map frame '{self.arena.frame_id}' from {self.config.localization.state_topic}"
        )

    # -- inputs ------------------------------------------------------------- #
    def _on_localization(self, message: Any) -> None:
        self.map_to_odom = nodeutil.msg_to_pose3(message.map_to_odom.pose)
        if not self.localization_available:
            self.localization_available = True
            self.get_logger().info(
                f"map -> odom correction received (localized={bool(message.localized)}); "
                "obstacles will be published in the map frame"
            )

    def _on_odometry(self, topic: str, message: Any) -> None:
        if self.active_odom_topic is None:
            self.active_odom_topic = topic
            self.get_logger().info(f"using odometry from {topic}")
        elif topic != self.active_odom_topic:
            return
        stamp = nodeutil.stamp_to_seconds(message.header.stamp)
        self.odom_buffer.add(stamp, nodeutil.msg_to_pose3(message.pose.pose))

    # -- the pipeline ------------------------------------------------------- #
    def _on_image(self, message: Any) -> None:
        now = nodeutil.now_seconds(self)
        self.frames_seen += 1
        stamp = nodeutil.stamp_to_seconds(message.header.stamp)

        if bool(self.config.localization.enabled) and not self.localization_available:
            self.counters["frames_without_fix"] += 1
            self._log.warn(
                "no_fix",
                f"no map -> odom correction yet on {self.config.localization.state_topic}; "
                "frames are dropped rather than reported at unknown coordinates",
            )
            return
        if now - stamp > self.config.detector.max_frame_age_s and stamp > 0.0:
            self._log.warn(
                "stale_frame",
                f"dropping a {now - stamp:.2f} s old frame "
                f"(max_frame_age_s {self.config.detector.max_frame_age_s})",
            )
            return

        odom_pose = self.odom_buffer.nearest(stamp) if stamp > 0.0 else self.odom_buffer.latest()
        if odom_pose is None:
            self.counters["frames_without_fix"] += 1
            self._log.warn(
                "no_odometry",
                "no odometry yet on "
                f"{list(self.config.localization.odometry_topics)}; frames are dropped",
            )
            return
        pose_map_base = self.map_to_odom.compose(odom_pose)

        try:
            rgb = imagecodec.to_rgb(message)
        except Exception as exc:
            self._log.warn("decode_failure", f"cannot decode '{message.encoding}': {exc}")
            return

        started = time.perf_counter()
        detections = self.backend.detect(rgb)
        self.latency.add(time.perf_counter() - started, now)
        self.fps.tick(now)
        self.last_detections = list(detections)
        self.detections_seen += len(detections)
        self.detections_last_frame = len(detections)

        observations: List[GroundObservation] = []
        rejected = 0
        for detection in detections:
            result = observe_detection(
                detection, self.camera, pose_map_base, self.config.detector.ground, self.arena
            )
            if isinstance(result, Rejection):
                rejected += 1
                self.rejection_reasons[result.reason] = self.rejection_reasons.get(result.reason, 0) + 1
                self.counters["dropped_range"] += 1
                continue
            observations.append(result)
        self.accepted_seen += len(observations)
        self.accepted_last_frame = len(observations)
        self.rejected_last_frame = rejected

        tracks = self.tracker.update(observations, now, source=self.backend.name)
        kept, dropped = mask_tracks(tracks, self.arena)
        if dropped:
            self.counters["dropped_wall"] += len(dropped)
            for track, reason in dropped:
                self._log.warn(
                    f"wall_{reason}",
                    f"masked a track at ({track.x:.2f}, {track.y:.2f}) as '{reason}' "
                    "(walls are not sensors)",
                )
        self.tracks = kept
        self.counters["frames_processed"] += 1

    # -- outputs ------------------------------------------------------------ #
    def _on_publish(self) -> None:
        if self.counters["frames_processed"] == 0 and not self.tracks:
            return
        now = nodeutil.now_seconds(self)
        frame_id = self.arena.frame_id
        payload = build_obstacle_payload(self.tracks, now, frame_id, self.counters)
        message = ObstacleArray()
        message.header = nodeutil.make_header(self, frame_id)
        message.obstacles = [self._build_obstacle(fields, message.header) for fields in payload["obstacles"]]
        message.frames_processed = int(payload["frames_processed"])
        message.dropped_wall = int(payload["dropped_wall"])
        message.dropped_range = int(payload["dropped_range"])
        message.frames_without_fix = int(payload["frames_without_fix"])
        self.obstacles_publisher.publish(message)
        if self.cloud_publisher is not None:
            self.cloud_publisher.publish(self._build_cloud(payload["obstacles"], message.header))
        self.stats_publisher.publish(self._build_stats(now, len(message.obstacles)))

    def _build_obstacle(self, fields: Dict[str, Any], header: Any) -> Obstacle:
        message = Obstacle()
        message.header = header
        message.id = str(fields["id"])
        message.kind = int(fields["kind"])
        message.class_id = int(fields["class_id"])
        message.pose = nodeutil.pose_with_covariance(
            Pose3.from_xy_yaw(float(fields["x"]), float(fields["y"]), 0.0, z=float(fields["z"])),
            fields["covariance"],
        )
        message.radius = float(fields["radius"])
        message.height = float(fields["height"])
        message.depth = float(fields["depth"])
        message.confidence = float(fields["confidence"])
        message.score = float(fields["score"])
        message.observations = int(fields["observations"])
        message.age_s = float(fields["age_s"])
        message.last_seen_age_s = float(fields["last_seen_age_s"])
        message.source = str(fields["source"])
        message.conservative = bool(fields["conservative"])
        return message

    def _build_cloud(self, fields_list: Sequence[Dict[str, Any]], header: Any) -> PointCloud2:
        points, point_step = pointcloud_payload(fields_list, self.arena.frame_id)
        message = PointCloud2()
        message.header = header
        message.height = 1
        message.width = int(points.shape[0])
        message.fields = []
        for name, offset, datatype in obstacle_module.POINT_FIELDS:
            field = PointField()
            field.name = name
            field.offset = int(offset)
            field.datatype = int(datatype)
            field.count = 1
            message.fields.append(field)
        message.is_bigendian = False
        message.point_step = int(point_step)
        message.row_step = int(point_step) * int(points.shape[0])
        message.is_dense = True
        message.data = points.tobytes()
        return message

    def _build_stats(self, now: float, track_count: int) -> Float32MultiArray:
        message = Float32MultiArray()
        layout = MultiArrayLayout()
        dimension = MultiArrayDimension()
        dimension.label = ",".join(STATS_LAYOUT)
        dimension.size = len(STATS_LAYOUT)
        dimension.stride = len(STATS_LAYOUT)
        layout.dim = [dimension]
        layout.data_offset = 0
        message.layout = layout
        values = [
            float(self.fps.fps(now)),
            float(self.latency.mean_ms()),
            float(self.cpu.sample(now)),
            float(self.detections_last_frame),
            float(self.accepted_last_frame),
            float(self.rejected_last_frame),
            float(track_count),
            float(self.counters["frames_processed"]),
        ]
        message.data = values
        return message

    def _log_stats(self) -> None:
        if self.frames_seen == 0:
            self._log.warn("no_frames", f"no frames received on {self.camera.topic} yet")
            return
        now = nodeutil.now_seconds(self)
        reasons = ", ".join(
            f"{reason}={count}" for reason, count in sorted(self.rejection_reasons.items())
        )
        self.get_logger().info(
            f"{self.fps.fps(now):.1f} fps, {self.latency.mean_ms():.1f} ms/frame, "
            f"{self.cpu.sample(now):.0f}% cpu, {len(self.tracks)} tracks, "
            f"frames {self.counters['frames_processed']}/{self.frames_seen} "
            f"(no fix {self.counters['frames_without_fix']}, wall {self.counters['dropped_wall']})"
            + (f" | rejections: {reasons}" if reasons else "")
        )

    # -- introspection (tests) ---------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": self.config.path,
            "backend": self.backend.name,
            "camera": self.camera.name,
            "obstacles_topic": self.config.detector.obstacles_topic,
            "stats_topic": self.config.detector.stats_topic,
            "cloud_topic": self.config.detector.cloud_topic if self.config.detector.cloud_enabled else "",
            "frames_seen": self.frames_seen,
            "counters": dict(self.counters),
            "tracks": len(self.tracks),
            "localization_available": self.localization_available,
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--debug-image",
        action="store_true",
        help="acknowledge the module-4 debug-image convention (no image is published)",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    def build(args: argparse.Namespace) -> Node:
        return ObstacleDetectorNode(args.config, debug_image=bool(getattr(args, "debug_image", False)))

    return nodeutil.node_main(
        argv,
        "boulder/crater detector: cam_nav -> map-frame ObstacleArray",
        build,
        add_arguments=add_arguments,
        epilog="obstacles are published in the map frame with per-obstacle confidence and covariance",
    )
