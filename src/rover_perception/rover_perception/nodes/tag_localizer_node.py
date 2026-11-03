"""``tag_localizer``: tag sightings -> ``map -> odom``, with growing covariance.

Started by ``perception.launch.py`` as ``perception_tag_localizer``.  This is the
module that replaces AMCL: the arena frame is anchored by the wall AprilTags, so
a tag sighting is a direct measurement of where the rover is, and
:mod:`rover_perception.localizer` solves it exactly (no particle filter, no scan
matching - there is no lidar in this build either).

What the node adds on top of the solver:

* **odometry selection.**  ``localization.odometry_topics`` is a list, and the
  first topic that actually delivers a message wins for the whole run, so the
  node works with ``robot_localization`` fitted *or* absent (``/odometry/filtered``
  then ``/odom``).
* **time alignment.**  the odometry samples are buffered and the one nearest the
  tag frame's stamp is used (see :class:`rover_perception.localizer.OdomBuffer`).
* **publishing.**  ``LocalizationState`` at ``publish_hz`` (so the covariance
  keeps growing visibly while no tag is in view - that is the signal module 5
  needs), the applied correction on ``fix_topic``, and optionally a
  ``map -> odom`` TF.

The TF is **off by default in simulation**: gz's diff-drive plugin already
publishes ``odom -> base_link`` on the bridged ``/tf``, and a second publisher of
the same edge is a tf2 error.  On the robot, where ``robot_localization`` runs
with ``publish_tf: true`` for ``odom -> base_link``, the same is true - so the
launch file exposes ``publish_tf`` and the default is off unless asked for.
"""

from __future__ import annotations

import argparse
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from rover_perception_msgs.msg import LocalizationState, TagDetectionArray

from .. import nodeutil
from ..calibration_ref import load_calibration
from ..geometry import Pose3
from ..localizer import Localizer, OdomBuffer, TagObservation
from ..tags import load_tags

__all__ = ["TagLocalizerNode", "main", "add_arguments"]

NODE_NAME = "perception_tag_localizer"


class TagLocalizerNode(Node):
    """Maintains and publishes the ``map -> odom`` correction."""

    def __init__(
        self,
        config_override: str = "",
        publish_tf: bool = False,
        publish_hz_override: float = 0.0,
        tags_file: str = "",
    ):
        super().__init__(NODE_NAME)
        self.config = nodeutil.load_node_config(self, override=config_override)
        cfg = self.config.localization
        if not cfg.enabled:
            raise ValueError("localization.enabled is false; do not start this node")
        self.publish_hz = float(
            nodeutil.declare(self, "publish_hz", float(publish_hz_override or cfg.publish_hz))
        )
        if self.publish_hz <= 0.0:
            self.publish_hz = cfg.publish_hz
        self.publish_tf = bool(nodeutil.declare(self, "publish_tf", bool(publish_tf)))
        self.tags_file = str(nodeutil.declare(self, "tags_file", str(tags_file)) or "")
        self._log = nodeutil.LogOnce(self)

        calibration = load_calibration(self.config.artifacts.calibration)
        self.camera = calibration.camera(self.config.tags.camera)
        self.catalogue = load_tags(self.tags_file or nodeutil.resolve_tags_path(self.config))
        self.localizer = Localizer(cfg, self.catalogue, self.camera)

        self.odom_buffer = OdomBuffer(capacity=64)
        self.active_odom_topic: Optional[str] = None
        self.odom_frames = 0
        self.frames_without_odom = 0
        self.tag_frames = 0
        self.detections_seen = 0
        self.fixes_published = 0
        self.odom_pose: Optional[Pose3] = None

        self.state_publisher = self.create_publisher(LocalizationState, cfg.state_topic, 10)
        self.fix_publisher = self.create_publisher(PoseWithCovarianceStamped, cfg.fix_topic, 10)
        self.odom_subscriptions = [
            self.create_subscription(Odometry, topic, partial(self._on_odometry, topic), 20)
            for topic in cfg.odometry_topics
        ]
        self.detection_subscription = self.create_subscription(
            TagDetectionArray, self.config.tags.topic, self._on_detections, 10
        )
        self.create_timer(1.0 / self.publish_hz, self._on_publish)

        self.tf_broadcaster = None
        if self.publish_tf:
            from tf2_ros import TransformBroadcaster

            self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f"localizer: {self.catalogue.count} tags from {self.catalogue.path}, "
            f"odometry candidates {list(cfg.odometry_topics)}, publishing "
            f"{cfg.state_topic} at {self.publish_hz:.0f} Hz"
        )
        if self.publish_tf:
            self.get_logger().warn(
                f"publishing TF '{self.config.arena.frame_id}' -> '{self.config.arena.odom_frame}'; "
                "make sure nothing else publishes that same edge"
            )
        else:
            self.get_logger().info(
                "TF publishing is off; the correction is on "
                f"{cfg.fix_topic} and {cfg.state_topic} (module 5 composes them)"
            )

    # -- odometry ----------------------------------------------------------- #
    def _on_odometry(self, topic: str, message: Any) -> None:
        if self.active_odom_topic is None:
            self.active_odom_topic = topic
            self.get_logger().info(f"using odometry from {topic}")
        elif topic != self.active_odom_topic:
            return
        try:
            pose = nodeutil.msg_to_pose3(message.pose.pose)
        except Exception as exc:  # pragma: no cover - malformed message
            self._log.warn("bad_odometry", f"ignoring unreadable odometry: {exc}")
            return
        stamp = nodeutil.stamp_to_seconds(message.header.stamp)
        self.odom_buffer.add(stamp, pose)
        self.odom_pose = pose
        self.odom_frames += 1

    # -- fixes -------------------------------------------------------------- #
    def _on_detections(self, message: Any) -> None:
        self.tag_frames += 1
        stamp = nodeutil.stamp_to_seconds(getattr(message.header, "stamp", None)) if message.header else 0.0
        odom_pose = self.odom_buffer.nearest(stamp) if stamp > 0.0 else self.odom_buffer.latest()
        if odom_pose is None:
            self.frames_without_odom += 1
            self._log.warn(
                "no_odometry",
                "tag detections arrived before any odometry; no fix is possible "
                f"(waiting on {list(self.config.localization.odometry_topics)})",
            )
            return

        observations: List[TagObservation] = []
        for detection in message.detections:
            if not getattr(detection, "pose_valid", False):
                continue
            observations.append(
                TagObservation(
                    tag_id=int(detection.id),
                    pose_camera=nodeutil.msg_to_pose3(detection.pose_camera.pose),
                    size_m=float(detection.size_m),
                    decision_margin=float(detection.decision_margin),
                    hamming=float(detection.hamming),
                    stamp=stamp,
                )
            )
        self.detections_seen += len(observations)
        if not observations:
            return
        now = nodeutil.now_seconds(self)
        before = self.localizer.fixes_total
        solution = self.localizer.update(observations, odom_pose, now)
        if solution.accepted:
            self.fixes_published += 1
            if before == 0:
                estimate = solution.proposed_map_to_odom
                self.get_logger().info(
                    f"localized from tag(s) {list(solution.used_tags)}: map->odom "
                    f"established at ({estimate.x:.3f}, {estimate.y:.3f}, "
                    f"{estimate.yaw:.4f} rad) - odometry starts at the rover's boot "
                    f"pose, so the first fix establishes the transform, it does not "
                    f"correct one"
                )
            else:
                self.get_logger().debug(
                    f"tag fix from {list(solution.used_tags)}: corrected by "
                    f"{solution.correction_xy_m * 1000.0:.1f} mm, "
                    f"{solution.correction_yaw_rad:.4f} rad"
                )
            self._publish_fix(message, solution)
        else:
            reasons = ", ".join(f"tag {fix.tag_id}: {fix.reason}" for fix in solution.fixes)
            self._log.warn(
                f"rejected_{solution.reason}",
                f"tag fix rejected ({solution.reason}) - {reasons or 'no usable tag'}",
            )

    def _publish_fix(self, source: Any, solution: Any) -> None:
        message = PoseWithCovarianceStamped()
        message.header = nodeutil.make_header(self, self.config.arena.frame_id)
        stamp = getattr(getattr(source, "header", None), "stamp", None)
        if stamp is not None and (stamp.sec or stamp.nanosec):
            message.header.stamp = stamp
        message.pose = nodeutil.pose_with_covariance(
            solution.proposed_map_to_odom,
            self._planar_covariance(
                self.localizer.sigma_xy_m, self.localizer.sigma_yaw_rad
            ),
        )
        self.fix_publisher.publish(message)

    # -- periodic state ----------------------------------------------------- #
    def _on_publish(self) -> None:
        now = nodeutil.now_seconds(self)
        if self.odom_pose is not None:
            # keep the reported uncertainty current even when no tag is in view
            self.localizer.propagate(self.odom_pose, now)
        fields = self.localizer.state_fields(now)
        message = LocalizationState()
        message.header = nodeutil.make_header(self, self.config.arena.frame_id)
        message.localized = bool(fields["localized"])
        message.fixes_total = int(fields["fixes_total"])
        message.fix_age_frames = int(fields["fix_age_frames"])
        message.time_since_fix_s = float(min(fields["time_since_fix_s"], 1e6))
        message.sigma_xy_m = float(fields["sigma_xy_m"])
        message.sigma_yaw_rad = float(fields["sigma_yaw_rad"])
        message.map_to_odom = nodeutil.pose_with_covariance(
            fields["map_to_odom"],
            self._planar_covariance(fields["sigma_xy_m"], fields["sigma_yaw_rad"]),
        )
        message.last_tag_id = int(fields["last_tag_id"])
        message.last_fix_xy_m = float(fields["last_fix_xy_m"])
        message.last_fix_yaw_rad = float(fields["last_fix_yaw_rad"])
        self.state_publisher.publish(message)
        if self.tf_broadcaster is not None:
            self._broadcast_tf(fields["map_to_odom"], message.header.stamp)

    def _broadcast_tf(self, map_to_odom: Pose3, stamp: Any) -> None:
        from geometry_msgs.msg import TransformStamped

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self.config.arena.frame_id
        transform.child_frame_id = self.config.arena.odom_frame
        transform.transform.translation.x = float(map_to_odom.x)
        transform.transform.translation.y = float(map_to_odom.y)
        transform.transform.translation.z = float(map_to_odom.z)
        qx, qy, qz, qw = map_to_odom.quaternion_xyzw
        transform.transform.rotation.x = float(qx)
        transform.transform.rotation.y = float(qy)
        transform.transform.rotation.z = float(qz)
        transform.transform.rotation.w = float(qw)
        self.tf_broadcaster.sendTransform(transform)

    @staticmethod
    def _planar_covariance(sigma_xy_m: float, sigma_yaw_rad: float) -> List[float]:
        covariance = [0.0] * 36
        variance = float(max(sigma_xy_m, 1e-6)) ** 2
        covariance[0] = variance
        covariance[7] = variance
        covariance[14] = variance
        covariance[35] = float(max(sigma_yaw_rad, 1e-6)) ** 2
        return covariance

    # -- introspection (tests) ---------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": self.config.path,
            "catalogue": self.catalogue.summary(),
            "active_odom_topic": self.active_odom_topic,
            "odom_frames": self.odom_frames,
            "tag_frames": self.tag_frames,
            "detections_seen": self.detections_seen,
            "fixes_published": self.fixes_published,
            "frames_without_odom": self.frames_without_odom,
            "localized": self.localizer.localized,
            "sigma_xy_m": self.localizer.sigma_xy_m,
            "map_to_odom": self.localizer.map_to_odom.as_rpy(),
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publish-tf", action="store_true", help="broadcast map -> odom")
    parser.add_argument("--publish-hz", type=float, default=0.0, help="override localization.publish_hz")
    parser.add_argument(
        "--tags-file", default="", help="tags.yaml path (empty = artifacts.output_dir + tags_file)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    def build(args: argparse.Namespace) -> Node:
        return TagLocalizerNode(
            args.config,
            publish_tf=bool(getattr(args, "publish_tf", False)),
            publish_hz_override=float(getattr(args, "publish_hz", 0.0) or 0.0),
            tags_file=str(getattr(args, "tags_file", "") or ""),
        )

    return nodeutil.node_main(
        argv,
        "tag localizer: AprilTag fixes -> map -> odom correction (no AMCL)",
        build,
        add_arguments=add_arguments,
        epilog="leave publish_tf off when gz or robot_localization already publishes odom -> base_link",
    )
