"""The data-collection driver and its ROS node.

:class:`CollectionDriver` contains every piece of orchestration and takes its
collaborators as arguments (a gz backend and a frame source), so the whole loop
is unit-testable with fakes -- no simulator and no ROS required.

:class:`CollectorNode` is the thin ROS wrapper: it subscribes to the four
camera topics, answers frame requests by spinning, and hands the driver a real
:class:`GzCliBackend`.  It exits when the collection is finished, which is how
``ros2 launch`` is told to shut the scene down (``on_exit=Shutdown()``).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .augment import augment_image, transform_boxes
from .calibration_ref import CalibrationDoc, CameraModel
from .capture import GzBackend, GzCliBackend
from .config import DatasetConfig
from .dataset import RawLayout, write_sample
from .groundtruth import GroundTruth
from .labels import YoloBox
from .poseplan import CapturePose, plan_poses
from .projection import label_frame
from .scene import SceneSpec
from .seeding import rng_numpy

__all__ = ["FrameSource", "CollectionReport", "CollectionDriver", "CollectorNode", "main"]


class FrameSource(Protocol):
    """Something that can hand over one fresh frame per requested camera."""

    def wait_frames(self, timeout: float) -> Dict[str, np.ndarray]:
        ...


@dataclass
class CollectionReport:
    scene_id: str
    source: str
    ok: bool = True
    poses_planned: int = 0
    poses_captured: int = 0
    images_written: int = 0
    boxes_written: int = 0
    dropped: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    teleport_failures: int = 0
    frame_timeouts: int = 0
    per_class: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "source": self.source,
            "ok": self.ok,
            "poses_planned": self.poses_planned,
            "poses_captured": self.poses_captured,
            "images_written": self.images_written,
            "boxes_written": self.boxes_written,
            "dropped": self.dropped,
            "per_class": self.per_class,
            "teleport_failures": self.teleport_failures,
            "frame_timeouts": self.frame_timeouts,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def _scaled(camera: CameraModel, width: int, height: int) -> CameraModel:
    """The camera model adjusted to an actual delivered frame size."""
    if (width, height) == (camera.width, camera.height):
        return camera
    sx = width / float(camera.width)
    sy = height / float(camera.height)
    return CameraModel(
        name=camera.name,
        topic=camera.topic,
        camera_info_topic=camera.camera_info_topic,
        frame_id=camera.frame_id,
        width=width,
        height=height,
        fx=camera.fx * sx,
        fy=camera.fy * sy,
        cx=camera.cx * sx,
        cy=camera.cy * sy,
        t_base_cam=camera.t_base_cam,
        r_base_optical=camera.r_base_optical,
    )


class CollectionDriver:
    """Capture loop: teleport, settle, read back, label, augment, write."""

    def __init__(
        self,
        cfg: DatasetConfig,
        scene: SceneSpec,
        ground_truth: GroundTruth,
        calibration: CalibrationDoc,
        raw_layout: RawLayout,
        backend: GzBackend,
        frame_source: FrameSource,
        *,
        master_seed: int,
        poses: Optional[Sequence[CapturePose]] = None,
        poses_per_scene: Optional[int] = None,
        cameras: Optional[Sequence[str]] = None,
        frames_per_pose: Optional[int] = None,
        settle_seconds: Optional[float] = None,
        frame_timeout: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda message: None,
    ) -> None:
        self.cfg = cfg
        self.scene = scene
        self.ground_truth = ground_truth
        self.calibration = calibration
        self.raw = raw_layout
        self.backend = backend
        self.frames = frame_source
        self.master_seed = int(master_seed)
        self.cameras = list(cameras or cfg.capture.cameras)
        self.frames_per_pose = int(
            frames_per_pose if frames_per_pose is not None else cfg.capture.frames_per_pose
        )
        self.settle_seconds = float(
            settle_seconds if settle_seconds is not None else cfg.capture.settle_seconds
        )
        self.frame_timeout = float(frame_timeout)
        self.sleep = sleep
        self.log = log
        self.class_indices = {cls.name: cls.index for cls in cfg.classes}
        self.poses = list(
            poses
            if poses is not None
            else plan_poses(
                ground_truth,
                cfg.capture,
                self.master_seed,
                scene.scene_id,
                poses_per_scene=poses_per_scene,
            )
        )

    # -- helpers ----------------------------------------------------------- #
    def _teleport(self, pose: CapturePose) -> bool:
        for attempt in range(self.cfg.capture.pose_gz_attempts):
            try:
                if self.backend.set_pose(
                    self.cfg.capture.model_name, pose.x, pose.y, self.cfg.capture.teleport_z, pose.yaw
                ):
                    return True
            except Exception as exc:  # pragma: no cover - transport failure
                self.log(f"teleport attempt {attempt + 1} failed: {exc}")
        return False

    # -- main loop --------------------------------------------------------- #
    def run(self) -> CollectionReport:
        report = CollectionReport(scene_id=self.scene.scene_id, source=self.scene.source)
        report.poses_planned = len(self.poses)
        self.raw.ensure_dirs()
        for name in self.cfg.class_names:
            report.per_class.setdefault(name, 0)

        for pose in self.poses:
            if not self._teleport(pose):
                report.teleport_failures += 1
                report.warnings.append(f"pose {pose.index}: teleport failed")
                continue
            self.sleep(self.settle_seconds)
            try:
                actual = self.backend.get_pose(self.cfg.capture.model_name)
            except Exception as exc:
                report.errors.append(f"pose {pose.index}: cannot read pose back ({exc})")
                report.ok = False
                continue
            rover_pose = actual.as_rover_pose()
            self.log(
                f"pose {pose.index}: ({actual.x:.2f}, {actual.y:.2f}, {actual.z:.2f}) "
                f"yaw {np.degrees(actual.yaw):.1f} deg"
            )

            captured_any = False
            for frame_index in range(self.frames_per_pose):
                images = self.frames.wait_frames(self.frame_timeout)
                if not images:
                    report.frame_timeouts += 1
                    report.warnings.append(f"pose {pose.index}: no frame within timeout")
                    continue
                for camera_name in self.cameras:
                    image = images.get(camera_name)
                    if image is None:
                        report.warnings.append(f"pose {pose.index}: missing {camera_name} frame")
                        continue
                    height, width = int(image.shape[0]), int(image.shape[1])
                    model = _scaled(self.calibration.camera(camera_name), width, height)
                    if (width, height) != (
                        self.calibration.camera(camera_name).width,
                        self.calibration.camera(camera_name).height,
                    ):
                        report.warnings.append(
                            f"{camera_name}: frame {width}x{height} != calibration "
                            f"{self.calibration.camera(camera_name).width}x"
                            f"{self.calibration.camera(camera_name).height}; intrinsics scaled"
                        )
                    projection = label_frame(
                        self.ground_truth,
                        model,
                        rover_pose,
                        self.cfg.labeling,
                        self.class_indices,
                        image_size=(width, height),
                    )
                    rng = rng_numpy(
                        self.master_seed,
                        self.cfg.augment.seed_namespace,
                        self.scene.scene_id,
                        pose.index,
                        camera_name,
                        frame_index,
                    )
                    outcome = augment_image(image, self.cfg.augment, rng)
                    boxes: List[YoloBox] = transform_boxes(
                        projection.boxes, outcome.flip_h, outcome.flip_v
                    )
                    write_sample(
                        self.raw,
                        split="train",
                        scene_id=self.scene.scene_id,
                        camera=camera_name,
                        pose_index=pose.index,
                        frame_index=frame_index,
                        image=outcome.image,
                        boxes=boxes,
                        dropped=projection.sidecar_dropped(),
                        meta={
                            "source": self.scene.source,
                            "region": pose.region,
                            "pose_commanded": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
                            "pose_actual": {
                                "x": actual.x,
                                "y": actual.y,
                                "z": actual.z,
                                "roll": actual.roll,
                                "pitch": actual.pitch,
                                "yaw": actual.yaw,
                            },
                            "augment_applied": outcome.applied,
                            "augment_params": outcome.params,
                            "objects_dropped": len(projection.dropped),
                        },
                    )
                    report.images_written += 1
                    report.boxes_written += len(boxes)
                    for box in boxes:
                        name = self.cfg.class_names[box.class_id]
                        report.per_class[name] = report.per_class.get(name, 0) + 1
                    for reason in projection.dropped:
                        report.dropped[reason.reason] = report.dropped.get(reason.reason, 0) + 1
                    captured_any = True
            if captured_any:
                report.poses_captured += 1

        if report.images_written == 0:
            report.ok = False
            report.errors.append("no images were written")
        return report


# --------------------------------------------------------------------------- #
# ROS wrapper
# --------------------------------------------------------------------------- #
class CollectorNode:  # pragma: no cover - exercised by the live acceptance run
    """rclpy node that feeds camera frames to a :class:`CollectionDriver`."""

    def __init__(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image as RosImage

        self._rclpy = rclpy
        self._RosImage = RosImage
        self._qos = qos_profile_sensor_data

        node = Node("rover_dataset_collector")
        self.node = node
        defaults = {
            "scene_id": "sim_0000",
            "scene_index": 0,
            "source": "sim",
            "master_seed": 1000,
            "ground_truth": "",
            "calibration": "",
            "dataset_root": "dataset",
            "scenes_dir": "_scenes",
            "report": "",
            "cameras": ["cam_nav"],
            "poses_per_scene": 0,
            "frames_per_pose": 0,
            "settle_seconds": -1.0,
            "frame_timeout": 5.0,
            "dry_run": False,
        }
        for name, value in defaults.items():
            node.declare_parameter(name, value)
        self.params = {name: node.get_parameter(name).value for name in defaults}

        self._latest: Dict[str, Any] = {name: None for name in self.params["cameras"]}
        self._counts: Dict[str, int] = {name: 0 for name in self.params["cameras"]}
        for camera in self.params["cameras"]:
            topic = camera if camera.startswith("/") else f"/{camera}"
            node.create_subscription(
                RosImage, topic, self._make_callback(camera), self._qos
            )

    def _make_callback(self, camera: str):
        def _callback(message) -> None:
            self._latest[camera] = message
            self._counts[camera] = self._counts.get(camera, 0) + 1

        return _callback

    def spin_once(self, timeout: float = 0.05) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=timeout)

    def wait_for_cameras(self, timeout: float = 90.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.spin_once(0.1)
            if all(self._counts.get(camera, 0) > 0 for camera in self._latest):
                return True
        return False

    def make_frame_source(self) -> "FrameSource":
        node = self

        class _RosFrameSource:
            def wait_frames(self, timeout: float) -> Dict[str, np.ndarray]:
                # drop anything buffered before the teleport settled: an image
                # from the previous pose would be labelled with this pose's maths
                node.clear_frames()
                frames = {name: None for name in node._latest}
                deadline = time.time() + timeout
                while time.time() < deadline:
                    node.spin_once(0.05)
                    for name in node._latest:
                        message = node._latest[name]
                        if message is not None and frames[name] is None:
                            frames[name] = message
                    if all(value is not None for value in frames.values()):
                        break
                out: Dict[str, np.ndarray] = {}
                for name, message in frames.items():
                    if message is None:
                        continue
                    channel = 4 if message.encoding == "rgba8" else 3
                    array = np.frombuffer(bytes(message.data), dtype=np.uint8)
                    expected = message.height * message.width * channel
                    if array.size < expected:
                        continue
                    out[name] = array[:expected].reshape(message.height, message.width, channel)[:, :, :3]
                return out

        return _RosFrameSource()

    def clear_frames(self) -> None:
        for name in self._latest:
            self._latest[name] = None


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - live path
    import rclpy

    from .calibration_ref import load_calibration
    from .groundtruth import load_ground_truth

    rclpy.init(args=argv)
    node = CollectorNode()
    report_path = str(node.params["report"] or "")
    try:
        if not node.wait_for_cameras(timeout=120.0):
            raise RuntimeError("cameras never published; is the simulation running?")
        cfg = _load_dataset_config()
        calibration = load_calibration(str(node.params["calibration"] or "") or None)
        ground_truth = load_ground_truth(str(node.params["ground_truth"]))
        raw = RawLayout(
            root=str(node.params["dataset_root"]),
            image_format=cfg.paths.image_format,
            image_quality=cfg.paths.image_quality,
            write_sidecar=cfg.paths.write_sidecar,
        )
        scene = SceneSpec(
            index=int(node.params["scene_index"]),
            scene_id=str(node.params["scene_id"]),
            source=str(node.params["source"]),
            seed=int(node.params["master_seed"]),
            arena_yaml={},
            rover_yaml={},
        )
        backend = GzCliBackend(world=cfg.capture.world_name)
        driver = CollectionDriver(
            cfg,
            scene,
            ground_truth,
            calibration,
            raw,
            backend,
            node.make_frame_source(),
            master_seed=int(node.params["master_seed"]),
            poses_per_scene=int(node.params["poses_per_scene"]) or None,
            frames_per_pose=int(node.params["frames_per_pose"]) or None,
            settle_seconds=(
                float(node.params["settle_seconds"]) if float(node.params["settle_seconds"]) >= 0 else None
            ),
            frame_timeout=float(node.params["frame_timeout"]),
            sleep=lambda seconds: _spin_sleep(node, seconds),
            log=lambda message: node.node.get_logger().info(message),
        )
        report = driver.run()
        payload = report.as_dict()
        node.node.get_logger().info(
            f"collected {report.images_written} images, {report.boxes_written} boxes "
            f"from {report.poses_captured}/{report.poses_planned} poses"
        )
    except Exception as exc:
        payload = {"ok": False, "errors": [str(exc)], "scene_id": str(node.params["scene_id"])}
        node.node.get_logger().error(f"collection failed: {exc}")
    finally:
        if report_path:
            os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
                handle.write("\n")
        node.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 1 if not payload.get("ok", False) else 0


def _spin_sleep(node: "CollectorNode", seconds: float) -> None:
    """Sleep while still servicing subscriptions (keeps the frame queue fresh)."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        node.spin_once(min(0.05, max(0.0, deadline - time.time())))


def _load_dataset_config() -> DatasetConfig:
    from .config import load_config

    return load_config()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
