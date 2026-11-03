"""``tag_detector``: AprilTags on ``cam_tag`` -> ``TagDetectionArray``.

Started by ``perception.launch.py`` as ``perception_tag_detector``.  The node is
thin: :mod:`rover_perception.tagdetect` does the work and
:mod:`rover_perception.tags` supplies the catalogue, so what is left here is
subscribing to the camera, keeping the effective intrinsics honest, and filling
in a message that ``tag_localizer`` can consume without guessing.

Two details that matter for a correct *metric* pose:

* **The intrinsics must match the pixels actually received.**  The tag pose is
  solved from pixels and the physical tag size, so if the image that arrives is
  not the resolution ``calibration.yaml`` describes (a bridge can resize, a
  config can change), the intrinsics are scaled to the received size and a
  warning is logged once.  Silently mixing them would bias every fix.
* **``pose_camera`` is in the camera optical frame** (REP-103: +z forward), which
  is what :mod:`rover_perception.localizer` composes with the module-2 extrinsics.
  Nothing here converts to the map frame: that is the localizer's single job.

The catalogue (``tags.yaml``) is loaded lazily and reloaded if it appears later,
so bringing up the detector before ``generate_tags.py`` has finished is not a
failure - it just logs that it is waiting.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import replace
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from rover_perception_msgs.msg import TagDetection, TagDetectionArray

from .. import imagecodec, nodeutil
from ..calibration_ref import CalibrationError, CameraModel, load_calibration
from ..config import PerceptionConfigError
from ..cpu import FpsMeter
from ..tags import TagsError, load_tags
from ..tagdetect import TagDetector, TagDetectorError, TagDetectionResult, available, install_hint

__all__ = ["TagDetectorNode", "main", "add_arguments"]

NODE_NAME = "perception_tag_detector"

#: Cycled per tag id so a debug image distinguishes ids without drawing text.
_TAG_COLOURS: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),
    (0, 160, 255),
    (255, 128, 0),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
)


def tag_colour(tag_id: int) -> Tuple[int, int, int]:
    """Stable debug colour for a tag id."""
    return _TAG_COLOURS[int(tag_id) % len(_TAG_COLOURS)]


class TagDetectorNode(Node):
    """Detects AprilTags and publishes them, best-first, with their poses."""

    def __init__(
        self,
        config_override: str = "",
        fallback_tag_size: float = 0.18,
        tags_file: str = "",
    ):
        super().__init__(NODE_NAME)
        self.config = nodeutil.load_node_config(self, override=config_override)
        cfg = self.config.tags
        self.enabled = bool(cfg.enabled)
        self.fallback_tag_size = float(
            nodeutil.declare(self, "fallback_tag_size", float(fallback_tag_size))
        )
        self.tags_file = str(nodeutil.declare(self, "tags_file", str(tags_file)) or "")
        self._log = nodeutil.LogOnce(self)

        calibration = load_calibration(self.config.artifacts.calibration)
        self.base_camera: CameraModel = calibration.camera(cfg.camera)
        self.camera: CameraModel = self.base_camera
        self.camera_info: Optional[Any] = None

        self.catalogue: Optional[Any] = None
        self.sizes: Dict[int, float] = {}
        self._load_catalogue(required=False)

        if not self.enabled:
            raise PerceptionConfigError("tags.enabled is false; start nothing instead")
        if not available():
            raise TagDetectorError("dt-apriltags is not installed; " + install_hint())
        self.detector = TagDetector(
            cfg.detection,
            self.camera,
            self.sizes,
            family=cfg.family,
            default_size_m=self.fallback_tag_size,
        )

        self.publisher = self.create_publisher(TagDetectionArray, cfg.topic, 10)
        self.debug_publisher = (
            self.create_publisher(Image, cfg.detection.publish_debug_topic, 2)
            if cfg.detection.debug_image
            else None
        )
        self.image_subscription = self.create_subscription(
            Image, self.base_camera.topic, self._on_image, nodeutil.sensor_qos()
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo, self.base_camera.camera_info_topic, self._on_camera_info, nodeutil.sensor_qos()
        )

        self.frames = 0
        self.detections_total = 0
        self.last_detections: List[TagDetectionResult] = []
        self.fps = FpsMeter(window_s=2.0)
        self.create_timer(5.0, self._log_stats)

        self.get_logger().info(
            f"tag detector: {cfg.family} on {self.base_camera.topic} "
            f"({self.base_camera.width}x{self.base_camera.height}, fx {self.base_camera.fx:.1f}) "
            f"-> {cfg.topic}"
        )
        for key, value in self.detector.describe().items():
            self.get_logger().debug(f"  {key}: {value}")
        if self.sizes:
            self.get_logger().info(
                f"catalogue: {len(self.sizes)} tags {sorted(self.sizes)} "
                f"(size {max(self.sizes.values()):.3f} m)"
            )
        else:
            self.get_logger().warn(
                "no tag catalogue yet; tag poses will use the fallback size "
                f"{self.fallback_tag_size:.3f} m until tags.yaml appears"
            )

    # -- setup helpers ------------------------------------------------------ #
    def _load_catalogue(self, required: bool = True) -> bool:
        """Load ``tags.yaml`` (id -> size + map pose); safe to call repeatedly."""
        path = self.tags_file or nodeutil.resolve_tags_path(self.config)
        try:
            catalogue = load_tags(path)
        except TagsError as exc:
            if required:
                raise
            self._catalogue_error = str(exc)
            return False
        self.catalogue = catalogue
        self.sizes = {tag_id: tag.size_m for tag_id, tag in catalogue.tags.items()}
        # the effective detector owns the size table; keep it in sync
        if getattr(self, "detector", None) is not None:
            self.detector.sizes = dict(self.sizes)
        self._catalogue_error = ""
        return True

    def _on_camera_info(self, message: Any) -> None:
        """Keep the effective camera model consistent with the pixels we get."""
        self.camera_info = message
        width, height = int(message.width), int(message.height)
        if width <= 0 or height <= 0:
            return
        if (width, height) == (self.base_camera.width, self.base_camera.height):
            return
        scale_x = width / float(self.base_camera.width)
        scale_y = height / float(self.base_camera.height)
        self.camera = replace(
            self.base_camera,
            width=width,
            height=height,
            fx=self.base_camera.fx * scale_x,
            fy=self.base_camera.fy * scale_y,
            cx=self.base_camera.cx * scale_x,
            cy=self.base_camera.cy * scale_y,
        )
        self.detector.camera = self.camera
        self._log.warn(
            f"camera_info_size_{width}x{height}",
            f"cam_tag is publishing {width}x{height} but calibration.yaml says "
            f"{self.base_camera.width}x{self.base_camera.height}; intrinsics scaled for the "
            "received image so tag poses stay metric",
        )

    # -- detection ---------------------------------------------------------- #
    def _on_image(self, message: Any) -> None:
        if not self.sizes and self.frames % 30 == 0:
            if self._load_catalogue(required=False):
                self.get_logger().info(
                    f"tag catalogue loaded: ids {sorted(self.sizes)}"
                )
        try:
            gray = imagecodec.to_gray(message)
        except Exception as exc:
            self._log.warn("decode_failure", f"cannot decode cam_tag encoding '{message.encoding}': {exc}")
            return
        now = nodeutil.now_seconds(self)
        self.frames += 1
        self.fps.tick(now)
        # the frame's own size is authoritative for the intrinsics
        height, width = gray.shape[:2]
        if (width, height) != (self.camera.width, self.camera.height) and self.camera_info is None:
            self.camera = replace(
                self.base_camera,
                width=width,
                height=height,
                fx=self.base_camera.fx * width / float(self.base_camera.width),
                fy=self.base_camera.fy * height / float(self.base_camera.height),
                cx=self.base_camera.cx * width / float(self.base_camera.width),
                cy=self.base_camera.cy * height / float(self.base_camera.height),
            )
            self.detector.camera = self.camera
            self._log.warn(
                f"image_size_{width}x{height}",
                f"cam_tag frames are {width}x{height} but calibration.yaml says "
                f"{self.base_camera.width}x{self.base_camera.height}; intrinsics scaled",
            )

        detections = self.detector.detect(gray)
        self.last_detections = detections
        self.detections_total += len(detections)
        self.publisher.publish(self._build_message(message, detections, now))
        if self.debug_publisher is not None:
            self.debug_publisher.publish(self._build_debug(message, gray, detections))

    def _build_message(
        self, source: Any, detections: Sequence[TagDetectionResult], now: float
    ) -> TagDetectionArray:
        message = TagDetectionArray()
        message.header = nodeutil.make_header(self, self.base_camera.frame_id, now)
        message.camera = self.base_camera.name
        message.frame_width = int(self.camera.width)
        message.frame_height = int(self.camera.height)
        detected_stamp = getattr(getattr(source, "header", None), "stamp", None)
        if detected_stamp is not None and (detected_stamp.sec or detected_stamp.nanosec):
            # publish with the frame's stamp so the fix can be time-aligned
            message.header.stamp = detected_stamp
        out: List[TagDetection] = []
        for detection in detections:
            item = TagDetection()
            item.header = message.header
            item.id = int(detection.tag_id)
            item.family = detection.family
            item.size_m = float(detection.size_m)
            item.decision_margin = float(detection.decision_margin)
            item.hamming = float(detection.hamming)
            item.corners = [float(value) for value in detection.corners]
            if detection.pose_camera is not None:
                item.pose_camera = nodeutil.pose_with_covariance(
                    detection.pose_camera,
                    self._pose_covariance(detection),
                )
                item.pose_valid = True
            else:
                item.pose_valid = False
            out.append(item)
        message.detections = out
        return message

    def _pose_covariance(self, detection: TagDetectionResult) -> List[float]:
        """A documented 6x6 covariance for one monocular tag pose.

        The dominant term is the *range*: one pixel of corner error is worth
        ``range^2 / (fx * tag_size)`` metres of depth, so the translation block
        uses that with a one-pixel corner assumption.  The rotation block is a
        flat small-angle floor - apriltag resolves the in-plane rotation from the
        payload, so yaw is far better than roll/pitch, and this only has to be
        honest, not precise (the localizer weights by range, not by this).
        """
        range_m = max(detection.range_m, 1e-3)
        size_m = max(detection.size_m, 1e-3)
        sigma = range_m * range_m / (max(self.camera.fx, 1e-6) * size_m)
        covariance = [0.0] * 36
        for axis in range(3):
            covariance[axis * 7] = float(sigma * sigma)
        covariance[21] = 0.02
        covariance[28] = 0.02
        covariance[35] = 0.005
        return covariance

    def _build_debug(
        self, source: Any, gray: np.ndarray, detections: Sequence[TagDetectionResult]
    ) -> Any:
        """Annotated RGB frame: each tag boxed and cross-hair'd (no text)."""
        canvas = np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8)
        for detection in detections:
            colour = tag_colour(detection.tag_id)
            if len(detection.corners) >= 8:
                xs = detection.corners[0::2]
                ys = detection.corners[1::2]
                imagecodec.draw_rect(canvas, min(xs), min(ys), max(xs), max(ys), colour, 2)
            imagecodec.draw_cross(canvas, detection.center[0], detection.center[1], colour, 5)
        stamp = nodeutil.make_header(self, self.base_camera.frame_id).stamp
        detected_stamp = getattr(getattr(source, "header", None), "stamp", None)
        if detected_stamp is not None and (detected_stamp.sec or detected_stamp.nanosec):
            stamp = detected_stamp
        return imagecodec.build_image(canvas, "rgb8", self.base_camera.frame_id, stamp, Image)

    def _log_stats(self) -> None:
        if self.frames == 0:
            self._log.warn("no_frames", f"no frames received on {self.base_camera.topic} yet")
            return
        now = nodeutil.now_seconds(self)
        self.get_logger().info(
            f"{self.fps.fps(now):.1f} fps over {self.frames} frames, "
            f"{self.detections_total} tag detections, "
            f"{len(self.last_detections)} in the last frame"
        )

    # -- introspection (tests) ---------------------------------------------- #
    def snapshot(self) -> Dict[str, Any]:
        return {
            "config": self.config.path,
            "topic": self.config.tags.topic,
            "camera": self.base_camera.name,
            "camera_size": list(self.base_camera.image_size),
            "effective_size": list(self.camera.image_size),
            "frames": self.frames,
            "detections_total": self.detections_total,
            "catalogue_ids": sorted(self.sizes),
            "detector": self.detector.describe(),
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fallback-tag-size",
        type=float,
        default=0.18,
        help="black-square size used until tags.yaml is available",
    )
    parser.add_argument(
        "--tags-file", default="", help="tags.yaml path (empty = artifacts.output_dir + tags_file)"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    def build(args: argparse.Namespace) -> Node:
        return TagDetectorNode(
            args.config,
            float(getattr(args, "fallback_tag_size", 0.18) or 0.18),
            str(getattr(args, "tags_file", "") or ""),
        )

    return nodeutil.node_main(
        argv,
        "AprilTag detector: cam_tag -> rover_perception_msgs/TagDetectionArray",
        build,
        add_arguments=add_arguments,
        epilog="poses are published in the camera optical frame; tag_localizer turns them into a map fix",
    )
