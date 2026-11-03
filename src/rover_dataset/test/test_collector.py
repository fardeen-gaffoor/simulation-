"""End-to-end collection-loop tests.

The whole capture loop runs against a fake gz backend and a fake frame source,
so the teleport -> settle -> read-back -> label -> augment -> write path is
covered without a simulator.  The live path is exercised separately by the
acceptance run.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from rover_dataset.capture import FakeGzBackend
from rover_dataset.collector import CollectionDriver
from rover_dataset.dataset import RawLayout, raw_stems
from rover_dataset.labels import read_labels
from rover_dataset.poseplan import CapturePose
from rover_dataset.projection import RoverPose, camera_pose_in_arena
from rover_dataset.scene import SceneSpec

from conftest import make_ground_truth


class FakeFrames:
    """Hands over a fresh synthetic frame for every requested camera."""

    def __init__(self, cameras, width=640, height=480, value=110):
        self.cameras = list(cameras)
        self.width = width
        self.height = height
        self.value = value
        self.calls = 0

    def wait_frames(self, timeout):
        self.calls += 1
        return {
            camera: np.full((self.height, self.width, 3), self.value, dtype=np.uint8)
            for camera in self.cameras
        }


class TimeoutFrames:
    def wait_frames(self, timeout):
        return {}


def _scene(scene_id="sim_0000", source="sim", seed=1000):
    return SceneSpec(
        index=0, scene_id=scene_id, source=source, seed=seed, arena_yaml={}, rover_yaml={}
    )


def _boulder_on_the_axis(calibration, x, y, z, yaw, radius=0.18):
    """A boulder whose centre projects exactly to the image centre of cam_nav."""
    camera = calibration.camera("cam_nav")
    r_ao, t_ao = camera_pose_in_arena(camera, RoverPose(x, y, z, 0.0, 0.0, yaw))
    axis = r_ao[:, 2]
    assert axis[2] < -0.05
    distance = (radius - t_ao[2]) / axis[2]
    centre = t_ao + distance * axis
    return (float(centre[0]), float(centre[1]), float(centre[2]), radius)


def _driver(cfg, calibration, tmp_path, ground_truth, frames, *, poses=None, backend=None,
            cameras=None, frames_per_pose=1, scene=None):
    return CollectionDriver(
        cfg,
        scene or _scene(),
        ground_truth,
        calibration,
        RawLayout(root=str(tmp_path), image_format="jpg", image_quality=90),
        backend or FakeGzBackend(),
        frames,
        master_seed=1000,
        poses=poses or [CapturePose(0, 2.0, 2.5, 0.0, "obstacle")],
        cameras=cameras,
        frames_per_pose=frames_per_pose,
        settle_seconds=0.0,
        sleep=lambda seconds: None,
    )


def test_collects_images_and_labels(cfg, calibration, tmp_path):
    sample = _boulder_on_the_axis(calibration, 2.0, 2.5, cfg.capture.teleport_z, 0.0)
    ground_truth = make_ground_truth(boulders=[sample])
    driver = _driver(cfg, calibration, tmp_path, ground_truth, FakeFrames(["cam_nav"]))
    report = driver.run()

    assert report.ok, report.errors
    assert report.poses_captured == 1
    assert report.images_written == 1
    assert report.boxes_written == 1
    assert report.per_class["boulder"] == 1

    raw = RawLayout(root=str(tmp_path))
    stems = raw_stems(raw)
    assert len(stems) == 1
    assert stems[0].startswith("sim_0000__cam_nav__p0000__f00")
    boxes = read_labels(raw.label_path("train", stems[0]))
    assert len(boxes) == 1
    assert boxes[0].class_id == 0
    assert boxes[0].cx == pytest.approx(0.5, abs=0.08)
    assert boxes[0].cy == pytest.approx(0.5, abs=0.08)


def test_sidecar_records_the_real_pose(cfg, calibration, tmp_path):
    sample = _boulder_on_the_axis(calibration, 2.0, 2.5, cfg.capture.teleport_z, 0.0)
    ground_truth = make_ground_truth(boulders=[sample])
    driver = _driver(cfg, calibration, tmp_path, ground_truth, FakeFrames(["cam_nav"]))
    driver.run()
    raw = RawLayout(root=str(tmp_path))
    payload = json.loads(open(raw.meta_path("train", raw_stems(raw)[0]), encoding="utf-8").read())
    assert payload["camera"] == "cam_nav"
    assert payload["source"] == "sim"
    assert payload["pose_actual"]["x"] == pytest.approx(2.0, abs=1e-6)
    assert payload["region"] == "obstacle"
    assert isinstance(payload["augment_applied"], list)


def test_multiple_frames_per_pose_and_cameras(cfg, calibration, tmp_path):
    sample = _boulder_on_the_axis(calibration, 2.0, 2.5, cfg.capture.teleport_z, 0.0)
    ground_truth = make_ground_truth(boulders=[sample])
    driver = _driver(
        cfg, calibration, tmp_path, ground_truth,
        FakeFrames(["cam_nav", "cam_tag"], width=1280, height=720),
        cameras=["cam_nav", "cam_tag"], frames_per_pose=2,
    )
    report = driver.run()
    assert report.ok, report.errors
    # cam_nav is 640x480 in the calibration while cam_tag is 1280x720; both were
    # delivered as 1280x720, so cam_nav (and only cam_nav) is flagged as resized
    assert report.images_written == 4
    assert any("cam_nav" in warning and "!=" in warning for warning in report.warnings)
    assert not any("cam_tag" in warning and "!=" in warning for warning in report.warnings)
    raw = RawLayout(root=str(tmp_path))
    assert len(raw_stems(raw)) == 4


def test_teleport_failure_is_counted(cfg, calibration, tmp_path):
    ground_truth = make_ground_truth(boulders=[(4.0, 2.5, 0.18, 0.18)])
    driver = _driver(
        cfg, calibration, tmp_path, ground_truth, FakeFrames(["cam_nav"]),
        backend=FakeGzBackend(error_every=1),
    )
    report = driver.run()
    assert report.teleport_failures == 1
    assert report.images_written == 0
    assert not report.ok


def test_frame_timeout_is_counted(cfg, calibration, tmp_path):
    ground_truth = make_ground_truth(boulders=[(4.0, 2.5, 0.18, 0.18)])
    driver = _driver(cfg, calibration, tmp_path, ground_truth, TimeoutFrames())
    report = driver.run()
    assert report.frame_timeouts == 1
    assert report.images_written == 0
    assert not report.ok


def test_collection_is_reproducible(cfg, calibration, tmp_path):
    sample = _boulder_on_the_axis(calibration, 2.0, 2.5, cfg.capture.teleport_z, 0.0)
    ground_truth = make_ground_truth(boulders=[sample])
    first_root = tmp_path / "a"
    second_root = tmp_path / "b"
    for root in (first_root, second_root):
        driver = _driver(cfg, calibration, root, ground_truth, FakeFrames(["cam_nav"]))
        assert driver.run().ok
    first = RawLayout(root=str(first_root))
    second = RawLayout(root=str(second_root))
    assert raw_stems(first) == raw_stems(second)
    stem = raw_stems(first)[0]
    assert open(first.label_path("train", stem)).read() == open(second.label_path("train", stem)).read()
    assert np.array_equal(
        np.asarray(__import__("PIL.Image").Image.open(first.image_path("train", stem))),
        np.asarray(__import__("PIL.Image").Image.open(second.image_path("train", stem))),
    )


def test_empty_ground_truth_writes_background_negatives(cfg, calibration, tmp_path):
    # a body with no objects at all: a pure negative, which must still be written
    ground_truth = make_ground_truth()
    driver = _driver(cfg, calibration, tmp_path, ground_truth, FakeFrames(["cam_nav"]))
    report = driver.run()
    assert report.ok, report.errors
    assert report.images_written == 1
    assert report.boxes_written == 0
    raw = RawLayout(root=str(tmp_path))
    assert os.path.getsize(raw.label_path("train", raw_stems(raw)[0])) == 0


def test_scaled_intrinsics_are_reported_when_the_stream_resizes(cfg, calibration, tmp_path):
    camera = calibration.camera("cam_nav")
    frames = FakeFrames(["cam_nav"], width=camera.width // 2, height=camera.height // 2)
    ground_truth = make_ground_truth()
    driver = _driver(cfg, calibration, tmp_path, ground_truth, frames)
    report = driver.run()
    assert report.ok
    assert any("!=" in warning for warning in report.warnings)
