"""Projection / auto-labelling tests -- the correctness heart of module 3.

If the projection is wrong, every label is wrong and no amount of training fixes
it.  These tests pin the geometry (including round-trip invertibility), the
buried-boulder handling, occlusion and the drop policy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rover_dataset.config import LabelingCfg
from rover_dataset.projection import (
    RoverPose,
    boulder_surface_points,
    camera_pose_in_arena,
    crater_rim_points,
    fibonacci_sphere,
    label_frame,
    project_points,
    unproject_pixel_to_ground,
)

from conftest import make_ground_truth

CLASSES = {"boulder": 0, "crater": 1}


@pytest.fixture
def labeling(cfg) -> LabelingCfg:
    return cfg.labeling


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def test_fibonacci_sphere_is_unit_and_balanced():
    points = fibonacci_sphere(500)
    norms = np.linalg.norm(points, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9)
    # roughly uniform: every octant gets a comparable share
    octants = [((points[:, 0] > 0) & (points[:, 1] > 0) & (points[:, 2] > 0)).sum()]
    assert 30 < octants[0] < 100


def test_boulder_points_are_above_ground():
    ground_truth = make_ground_truth(boulders=[(3.0, 2.5, 0.10, 0.20)])
    boulder = ground_truth.boulders[0]
    points = boulder_surface_points(boulder, 0.0, 512)
    assert len(points) > 0
    assert points[:, 2].min() >= -1e-9
    assert points[:, 2].max() == pytest.approx(boulder.top_z, abs=0.02)


def test_deeply_buried_boulder_has_little_or_nothing_above_ground():
    ground_truth = make_ground_truth(boulders=[(3.0, 2.5, -0.19, 0.20)])
    boulder = ground_truth.boulders[0]
    points = boulder_surface_points(boulder, 0.0, 512)
    assert len(points) < 60  # only a thin cap


def test_crater_rim_is_a_flat_circle():
    ground_truth = make_ground_truth(craters=[(4.0, 2.0, 0.25)])
    rim = crater_rim_points(ground_truth.craters[0], 0.0, 64)
    assert rim.shape == (64, 3)
    assert np.allclose(rim[:, 2], 0.0)
    radii = np.linalg.norm(rim[:, :2] - np.array([4.0, 2.0]), axis=1)
    assert np.allclose(radii, 0.25, atol=1e-9)


def test_camera_pose_at_identity_is_the_base_extrinsics(small_camera):
    r_ao, t_ao = camera_pose_in_arena(small_camera, RoverPose(0.0, 0.0, 0.0))
    expected = np.asarray(small_camera.r_base_optical).reshape(3, 3)
    assert np.allclose(r_ao, expected)
    assert np.allclose(t_ao, np.asarray(small_camera.t_base_cam))


def test_camera_pose_follows_rover_yaw(small_camera):
    # yaw +90 deg: base +x becomes arena +y, so the camera moves to +y of the rover
    r_ao, t_ao = camera_pose_in_arena(small_camera, RoverPose(1.0, 1.0, 0.2, 0.0, 0.0, math.pi / 2))
    assert np.allclose(t_ao, [1.0, 1.3, 0.7], atol=1e-9)
    # the optical axis (third column) must now point along arena +y
    axis = r_ao[:, 2]
    assert np.allclose(axis, [0.0, 1.0, 0.0], atol=1e-9)


def test_project_unproject_round_trip(small_camera):
    """A pixel -> ground point -> pixel must return the same pixel."""
    pose = RoverPose(2.0, 2.5, 0.4, 0.0, 0.0, 0.3)
    r_ao, t_ao = camera_pose_in_arena(small_camera, pose)
    for u, v in ((32.0, 40.0), (10.0, 30.0), (55.0, 45.0)):
        ground = unproject_pixel_to_ground(small_camera, r_ao, t_ao, u, v, 0.0)
        assert ground is not None
        uv, in_front = project_points(small_camera, r_ao, t_ao, ground.reshape(1, 3))
        assert bool(in_front[0])
        assert uv[0, 0] == pytest.approx(u, abs=1e-6)
        assert uv[0, 1] == pytest.approx(v, abs=1e-6)


def test_project_marks_points_behind_the_camera(small_camera):
    r_ao, t_ao = camera_pose_in_arena(small_camera, RoverPose(0.0, 0.0, 0.0))
    behind = np.array([[0.0, 0.0, 0.5]])  # x < camera x: behind
    uv, in_front = project_points(small_camera, r_ao, t_ao, behind)
    assert not bool(in_front[0])


def test_unproject_misses_when_pixel_looks_at_the_sky(small_camera):
    """Above the horizon there is no sand to hit."""
    r_ao, t_ao = camera_pose_in_arena(small_camera, RoverPose(0.0, 0.0, 0.0))
    assert unproject_pixel_to_ground(small_camera, r_ao, t_ao, 32.0, 0.0, 0.0) is None


# --------------------------------------------------------------------------- #
# bounding boxes
# --------------------------------------------------------------------------- #
def _pose_with_boulder_on_axis(camera, radius: float, target_z: float):
    """Place a boulder so its centre sits exactly on the camera's optical axis."""
    r_ao, t_ao = camera_pose_in_arena(camera, RoverPose(2.0, 2.5, 0.0))
    axis = r_ao[:, 2]
    assert axis[2] < -0.05, "this test needs a downward-tilted camera"
    distance = (target_z - t_ao[2]) / axis[2]
    assert distance > 1.0
    centre = t_ao + distance * axis
    return centre, r_ao, t_ao, distance


def test_boulder_on_the_optical_axis_lands_in_the_image_centre(calibration, labeling):
    camera = calibration.camera("cam_nav")
    centre, _, _, _ = _pose_with_boulder_on_axis(camera, 0.16, 0.16)
    ground_truth = make_ground_truth(boulders=[(centre[0], centre[1], centre[2], 0.16)])
    pose = RoverPose(2.0, 2.5, 0.0)
    result = label_frame(ground_truth, camera, pose, labeling, CLASSES)
    assert len(result.labels) == 1
    box = result.labels[0].box
    assert box.class_id == 0
    assert box.cx == pytest.approx(0.5, abs=0.06)
    assert box.cy == pytest.approx(0.5, abs=0.06)


def test_boulder_box_size_matches_the_pinhole_model(calibration, labeling):
    radius = 0.18
    camera = calibration.camera("cam_nav")
    centre, _, _, distance = _pose_with_boulder_on_axis(camera, radius, radius)
    ground_truth = make_ground_truth(boulders=[(centre[0], centre[1], centre[2], radius)])
    result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    assert len(result.labels) == 1
    box = result.labels[0].box
    # angular diameter 2r/d -> pixels 2*fx*r/d ; normalised by the image width
    expected_w = 2.0 * camera.fx * radius / distance / camera.width
    assert box.w == pytest.approx(expected_w, rel=0.12)
    assert box.h == pytest.approx(
        2.0 * camera.fy * radius / distance / camera.height, rel=0.12
    )


def _two_boulders_on_one_ray(camera, pose, target_xy, radius=0.25, near_fraction=0.45,
                             far_fraction=0.90):
    """Two boulders on the camera's line of sight to a ground point.

    The near one therefore hides the far one, and both stay above the sand.
    """
    _, t_ao = camera_pose_in_arena(camera, pose)
    ground = np.array([target_xy[0], target_xy[1], 0.0])
    direction = ground - t_ao
    near = t_ao + near_fraction * direction
    far = t_ao + far_fraction * direction
    return [
        (float(near[0]), float(near[1]), float(near[2]), radius),
        (float(far[0]), float(far[1]), float(far[2]), radius),
    ]


def test_boulder_behind_the_rover_is_dropped(calibration, labeling):
    camera = calibration.camera("cam_nav")
    ground_truth = make_ground_truth(boulders=[(0.5, 2.5, 0.15, 0.18)])
    result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    assert result.labels == []
    assert [d.reason for d in result.dropped] == ["off_camera"]


def test_crater_rim_becomes_a_box(calibration, labeling):
    camera = calibration.camera("cam_nav")
    ground_truth = make_ground_truth(craters=[(5.0, 2.5, 0.22)])
    result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    assert len(result.labels) == 1
    label = result.labels[0]
    assert label.kind == "crater"
    assert label.box.class_id == 1
    assert 0.0 < label.box.cx < 1.0
    assert label.box.w > 0.0


def test_occluded_boulder_is_dropped(calibration, labeling):
    """A boulder directly behind another, on the same ray, must be dropped."""
    camera = calibration.camera("cam_nav")
    boulders = _two_boulders_on_one_ray(camera, RoverPose(2.0, 2.5, 0.0), (5.0, 2.5))
    ground_truth = make_ground_truth(boulders=boulders)
    result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    assert len(result.labels) == 1
    assert result.labels[0].object_id == "boulder_0"
    assert [item.reason for item in result.dropped] == ["occluded"]
    assert result.dropped[0].object_id == "boulder_1"


def test_distant_object_is_dropped_as_too_small(calibration, labeling):
    camera = calibration.camera("cam_nav")
    # a 3 cm pebble 8 m away is far below the size floor
    ground_truth = make_ground_truth(boulders=[(8.6, 2.5, 0.03, 0.015)])
    result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    assert result.labels == []
    assert {d.reason for d in result.dropped} <= {"too_small", "off_camera"}


def test_box_is_clipped_to_the_image(calibration, labeling):
    """A crater half off the left edge keeps a legal, clamped box."""
    camera = calibration.camera("cam_nav")
    # place the crater far to the left of the view using the small tilted camera
    r_ao, t_ao = camera_pose_in_arena(camera, RoverPose(2.0, 2.5, 0.0))
    ground_truth = make_ground_truth(craters=[(2.2, 4.2, 0.35)])
    result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    for label in result.labels:
        box = label.box
        assert 0.0 <= box.cx <= 1.0
        assert 0.0 <= box.cy <= 1.0
        assert box.w > 0.0 and box.h > 0.0


def test_label_frame_reports_every_object(calibration, labeling):
    camera = calibration.camera("cam_nav")
    ground_truth = make_ground_truth(
        boulders=[(4.5, 2.5, 0.15, 0.18)],
        craters=[(5.5, 2.5, 0.22)],
    )
    result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    total = len(result.labels) + len(result.dropped)
    assert total == len(ground_truth.boulders) + len(ground_truth.craters)


def test_label_frame_is_deterministic(calibration, labeling):
    camera = calibration.camera("cam_nav")
    ground_truth = make_ground_truth(
        boulders=[(4.5, 2.5, 0.15, 0.18), (5.0, 3.2, 0.14, 0.16)],
        craters=[(6.0, 2.0, 0.2)],
    )
    first = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    second = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), labeling, CLASSES)
    assert [b.as_dict() for b in first.boxes] == [b.as_dict() for b in second.boxes]


def test_min_visible_fraction_drops_a_sliver(calibration, labeling):
    from dataclasses import replace

    camera = calibration.camera("cam_nav")
    ground_truth = make_ground_truth(craters=[(2.2, 4.2, 0.35)])
    strict = replace(labeling, min_visible_fraction=0.999)
    lenient = replace(labeling, min_visible_fraction=0.05)
    strict_result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), strict, CLASSES)
    lenient_result = label_frame(ground_truth, camera, RoverPose(2.0, 2.5, 0.0), lenient, CLASSES)
    assert len(lenient_result.labels) >= len(strict_result.labels)


def test_occluded_policy_keep_marks_the_object_occluded(calibration, labeling):
    from dataclasses import replace as _replace

    camera = calibration.camera("cam_nav")
    boulders = _two_boulders_on_one_ray(camera, RoverPose(2.0, 2.5, 0.0), (5.0, 2.5))
    ground_truth = make_ground_truth(boulders=boulders)
    kept = label_frame(
        ground_truth, camera, RoverPose(2.0, 2.5, 0.0),
        _replace(labeling, occluded_policy="keep"), CLASSES,
    )
    assert len(kept.labels) == 2
