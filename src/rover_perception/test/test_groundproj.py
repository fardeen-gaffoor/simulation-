"""Flat-ground projection: pixels to arena coordinates, with no depth sensor.

This is the module's central physics assumption.  A monocular camera cannot
measure distance; what it *can* do is intersect the ray through a pixel with the
sand plane, because the camera's height and attitude are known from module 2's
calibration and the sand is flat.  The closed-form results are therefore checked
against hand calculations (the principal-point ray, the horizon row) rather than
against the code's own outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rover_perception.calibration_ref import load_calibration
from rover_perception.geometry import Pose3
from rover_perception.groundproj import (
    GroundObservation,
    Rejection,
    camera_position_map,
    contact_pixel,
    height_for_range,
    horizon_row,
    observe_detection,
    project_points,
    r_map_optical,
    unproject_pixel,
    unproject_to_ground,
)
from support import detection

#: Hand-computed: cam_nav is 0.58 m up and pitched 18 degrees down, so the
#: principal-point ray meets the sand at 0.58 / tan(18 deg) ahead of it.
_NAV_HEIGHT = 0.58
_NAV_PITCH_RAD = math.radians(18.0)
_CENTRE_RANGE_M = _NAV_HEIGHT / math.tan(_NAV_PITCH_RAD)
_NAV_FORWARD_M = 0.36


@pytest.fixture(scope="module")
def nav(calibration_path):
    return load_calibration(calibration_path).camera("cam_nav")


@pytest.fixture(scope="module")
def tag_camera(calibration_path):
    return load_calibration(calibration_path).camera("cam_tag")


def _identity_map_base():
    return Pose3.identity()


# --------------------------------------------------------------------------- #
# the two closed-form results
# --------------------------------------------------------------------------- #
def test_camera_position_in_the_map(nav):
    position = camera_position_map(nav, _identity_map_base())
    assert position == pytest.approx((_NAV_FORWARD_M, 0.0, _NAV_HEIGHT), abs=1e-6)


def test_principal_point_ray_hits_the_sand_at_the_known_range(nav):
    hit = unproject_pixel(nav, _identity_map_base(), nav.cx, nav.cy)
    assert hit is not None
    assert hit.range_m == pytest.approx(_CENTRE_RANGE_M, rel=1e-3)
    assert hit.point[0] == pytest.approx(_NAV_FORWARD_M + _CENTRE_RANGE_M, rel=1e-3)
    assert hit.point[1] == pytest.approx(0.0, abs=1e-6)
    assert hit.point[2] == pytest.approx(0.0, abs=1e-9), "the hit is on the sand plane"


def test_horizon_row_matches_the_hand_calculation(nav):
    expected = nav.cy - nav.fy * math.tan(_NAV_PITCH_RAD)
    assert horizon_row(nav) == pytest.approx(expected, rel=1e-6)
    assert horizon_row(nav) == pytest.approx(152.8, abs=1.0)


def test_above_the_horizon_there_is_no_ground(nav):
    row = horizon_row(nav)
    assert unproject_pixel(nav, _identity_map_base(), nav.cx, row - 10.0) is None
    assert unproject_pixel(nav, _identity_map_base(), nav.cx, row + 10.0) is not None


def test_lower_pixels_are_closer(nav):
    """The flat-ground model must be monotonic: closer to the bottom is nearer."""
    ranges = []
    for row in (200.0, 260.0, 340.0, 440.0):
        hit = unproject_pixel(nav, _identity_map_base(), nav.cx, row)
        assert hit is not None
        ranges.append(hit.range_m)
    assert ranges == sorted(ranges, reverse=True)


def test_rays_diverge_to_the_sides(nav):
    left = unproject_pixel(nav, _identity_map_base(), 100.0, 400.0)
    right = unproject_pixel(nav, _identity_map_base(), 540.0, 400.0)
    assert left is not None and right is not None
    # optical +x (image right) is base -y (REP-103: +y is the robot's left), so a
    # pixel right of centre lands on the robot's right, which is -y
    assert left.point[1] > 0.0 > right.point[1]
    assert left.point[1] == pytest.approx(-right.point[1], abs=1e-6)


# --------------------------------------------------------------------------- #
# forward vs inverse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("point", [(2.0, 0.0, 0.0), (3.5, 0.8, 0.0), (5.0, -0.6, 0.0)])
def test_project_then_unproject_round_trips(nav, point):
    uv, in_front = project_points(nav, _identity_map_base(), np.array([point]))
    assert bool(in_front[0])
    hit = unproject_pixel(nav, _identity_map_base(), float(uv[0, 0]), float(uv[0, 1]))
    assert hit is not None
    assert tuple(hit.point) == pytest.approx(point, abs=1e-6)


def test_points_behind_the_camera_are_flagged(nav):
    uv, in_front = project_points(nav, _identity_map_base(), np.array([[-1.0, 0.0, 0.0]]))
    assert not bool(in_front[0])
    assert np.isnan(uv[0, 0])


def test_r_map_optical_is_a_rotation(nav):
    rotation = r_map_optical(nav, Pose3.from_xy_yaw(2.0, 1.0, 0.3))
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
    assert float(np.linalg.det(rotation)) == pytest.approx(1.0)


def test_unproject_to_ground_shortcut(nav):
    point = unproject_to_ground(nav, _identity_map_base(), nav.cx, 400.0)
    assert point is not None
    assert unproject_to_ground(nav, _identity_map_base(), nav.cx, 10.0) is None


def test_contact_pixel_is_the_bottom_centre(nav):
    box = detection("boulder", x0=100.0, y0=200.0, x1=140.0, y1=260.0)
    assert contact_pixel(box) == (120.0, 260.0)
    assert box.contact_point == (120.0, 260.0)
    assert box.width == 40.0 and box.height == 60.0 and box.area == 2400.0
    assert box.center == (120.0, 230.0)


# --------------------------------------------------------------------------- #
# height without depth
# --------------------------------------------------------------------------- #
def test_height_for_range_recovers_a_known_height(nav):
    target = (3.0, 0.0)
    horizontal = math.hypot(target[0] - _NAV_FORWARD_M, target[1])
    for height in (0.10, 0.25, 0.40):
        uv, in_front = project_points(nav, _identity_map_base(), np.array([(target[0], target[1], height)]))
        assert bool(in_front[0])
        estimate = height_for_range(
            nav, _identity_map_base(), float(uv[0, 0]), float(uv[0, 1]), horizontal
        )
        assert estimate == pytest.approx(height, abs=1e-6)


def test_height_for_range_needs_a_range(nav):
    assert height_for_range(nav, _identity_map_base(), 320.0, 400.0, 0.0) is None


# --------------------------------------------------------------------------- #
# observe_detection: accept
# --------------------------------------------------------------------------- #
def test_observation_of_a_point_on_the_sand(nav, cfg):
    target = (3.0, 1.5)
    uv, _ = project_points(nav, _identity_map_base(), np.array([(target[0], target[1], 0.0)]))
    box = detection("boulder", x0=float(uv[0, 0]) - 20.0, y0=float(uv[0, 1]) - 25.0,
                   x1=float(uv[0, 0]) + 20.0, y1=float(uv[0, 1]))
    result = observe_detection(box, nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(result, GroundObservation)
    assert result.x == pytest.approx(target[0], abs=0.02)
    assert result.y == pytest.approx(target[1], abs=0.02)
    assert result.kind == "boulder"
    assert result.class_id == 0
    assert result.range_m == pytest.approx(math.hypot(target[0] - _NAV_FORWARD_M, target[1]), rel=0.02)
    assert result.pixel == (float(uv[0, 0]), float(uv[0, 1]))
    # radius = half the box width, scaled by the pinhole depth
    expected_radius = 0.5 * 40.0 * result.optical_depth_m / nav.fx
    assert result.radius == pytest.approx(expected_radius, rel=1e-9)
    assert result.range_sigma_m > 0.0
    # bearing is measured from the rover's base origin, not the camera, so a
    # downstream consumer can compare it with the rover's own heading
    assert result.bearing_rad == pytest.approx(math.atan2(target[1], target[0]), abs=1e-6)


def test_crater_radius_scale_is_conservative(nav, cfg):
    target = (3.0, 1.5)
    uv, _ = project_points(nav, _identity_map_base(), np.array([(target[0], target[1], 0.0)]))
    common = dict(x0=float(uv[0, 0]) - 20.0, y0=float(uv[0, 1]) - 25.0,
                  x1=float(uv[0, 0]) + 20.0, y1=float(uv[0, 1]))
    boulder = observe_detection(detection("boulder", **common), nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    crater = observe_detection(detection("crater", **common), nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(boulder, GroundObservation) and isinstance(crater, GroundObservation)
    assert crater.radius > boulder.radius
    assert crater.radius == pytest.approx(boulder.radius * cfg.detector.ground.crater_radius_scale / cfg.detector.ground.boulder_radius_scale)
    assert crater.depth == 0.0, "a monocular camera cannot measure crater depth"


def test_observation_is_pose_aware(nav, cfg):
    """Turning the rover must move the reported obstacle, not the pixel."""
    target = (3.0, 1.5)
    pose = Pose3.from_xy_yaw(0.0, 0.0, math.radians(10.0))
    uv, _ = project_points(nav, pose, np.array([(target[0], target[1], 0.0)]))
    box = detection("boulder", x0=float(uv[0, 0]) - 20.0, y0=float(uv[0, 1]) - 25.0,
                    x1=float(uv[0, 0]) + 20.0, y1=float(uv[0, 1]))
    result = observe_detection(box, nav, pose, cfg.detector.ground, cfg.arena)
    assert isinstance(result, GroundObservation)
    assert (result.x, result.y) == pytest.approx(target, abs=0.02)


# --------------------------------------------------------------------------- #
# observe_detection: every rejection reason
# --------------------------------------------------------------------------- #
def _box_at(camera, pose, target, width=40.0, height=25.0, kind="boulder"):
    uv, _ = project_points(camera, pose, np.array([(target[0], target[1], 0.0)]))
    return detection(
        kind,
        x0=float(uv[0, 0]) - width / 2.0,
        y0=float(uv[0, 1]) - height,
        x1=float(uv[0, 0]) + width / 2.0,
        y1=float(uv[0, 1]),
    )


def test_rejects_a_contact_pixel_off_the_image(nav, cfg):
    box = detection("boulder", x0=10.0, y0=10.0, x1=50.0, y1=600.0)
    result = observe_detection(box, nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(result, Rejection)
    assert result.reason == "contact_pixel_outside_image"
    assert result.pixel == (30.0, 600.0)


def test_rejects_above_the_horizon(nav, cfg):
    row = horizon_row(nav)
    box = detection("boulder", x0=300.0, y0=row - 40.0, x1=340.0, y1=row - 5.0)
    result = observe_detection(box, nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(result, Rejection)
    assert result.reason == "above_horizon"


def test_rejects_too_near(nav, cfg):
    """Use a stricter ground policy so a mid-range point actually trips the gate."""
    import dataclasses

    ground = dataclasses.replace(cfg.detector.ground, min_range_m=2.0)
    result = observe_detection(
        _box_at(nav, _identity_map_base(), (2.0, 0.8)), nav, _identity_map_base(), ground, cfg.arena
    )
    assert isinstance(result, Rejection)
    assert result.reason == "too_near"


def test_rejects_too_far(nav, cfg):
    result = observe_detection(_box_at(nav, _identity_map_base(), (8.6, 0.5)), nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(result, Rejection)
    assert result.reason == "too_far"


def test_rejects_too_small(nav, cfg):
    box = _box_at(nav, _identity_map_base(), (3.0, 1.5), width=6.0)
    result = observe_detection(box, nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(result, Rejection)
    assert result.reason == "too_small"


def test_rejects_too_large(nav, cfg):
    box = _box_at(nav, _identity_map_base(), (3.0, 1.5), width=400.0)
    result = observe_detection(box, nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(result, Rejection)
    assert result.reason == "too_large"


def test_rejects_outside_the_arena(nav, cfg):
    """A wall, a reflection or a mis-projection - never an obstacle."""
    result = observe_detection(_box_at(nav, _identity_map_base(), (3.0, -0.5)), nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(result, Rejection)
    assert result.reason == "outside_arena"


def test_rejects_a_point_outside_the_north_wall(nav, cfg):
    """Facing north from mid-arena, a point past y = 5 m is off the sand."""
    pose = Pose3.from_xy_yaw(1.0, 1.0, math.radians(90.0))
    result = observe_detection(_box_at(nav, pose, (1.0, 5.6)), nav, pose, cfg.detector.ground, cfg.arena)
    assert isinstance(result, Rejection)
    assert result.reason == "outside_arena"


def test_rejection_detail_is_auditable(nav, cfg):
    box = detection("boulder", x0=10.0, y0=10.0, x1=50.0, y1=600.0)
    result = observe_detection(box, nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(result, Rejection)
    assert isinstance(result.detail, float)
    assert result.reason in {
        "contact_pixel_outside_image",
        "above_horizon",
        "too_near",
        "too_far",
        "too_small",
        "too_large",
        "outside_arena",
    }


def test_sigma_grows_with_range(nav, cfg):
    near = observe_detection(_box_at(nav, _identity_map_base(), (1.5, 1.0)), nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    far = observe_detection(_box_at(nav, _identity_map_base(), (5.5, 1.0)), nav, _identity_map_base(), cfg.detector.ground, cfg.arena)
    assert isinstance(near, GroundObservation) and isinstance(far, GroundObservation)
    assert far.range_sigma_m > near.range_sigma_m
