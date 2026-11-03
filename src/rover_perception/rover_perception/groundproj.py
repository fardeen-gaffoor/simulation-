"""Pixel -> flat-ground -> map-frame point: the geometric heart of module 4.

There is no depth sensor.  A detection is a box in a single RGB image, so the
only way to put it on the map is the flat-ground assumption that module 3 used
to draw the training labels:

    1. the box's **bottom-centre** is the object's ground-contact pixel (for a
       boulder that is where the sphere meets the sand; for a crater it is the
       near rim of the projected rim ellipse - one rule, two meanings),
    2. that pixel defines a ray through the pinhole,
    3. the ray is intersected with the flat ``z = ground_z`` sand plane in the
       map frame, using module 2's calibration and the rover pose.

The same convention in the labeller and in perception is deliberate: a model
trained on module 3's boxes must be consumed with the same ground rule, or the
boxes and the map would disagree by a systematic offset.

What this module deliberately does **not** do is invent a crater depth.  A
monocular camera looking at a uniformly sandy surface cannot measure how deep a
pit is; module 4 measures where the crater is and how wide it is, and marks it
lethal so module 5 routes around it.  ``depth`` is therefore reported as 0.0 and
that is documented in docs/detector.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .calibration_ref import CameraModel
from .config import ArenaCfg, GroundCfg
from .geometry import Pose3

__all__ = [
    "GroundHit",
    "Rejection",
    "GroundObservation",
    "r_map_optical",
    "camera_position_map",
    "unproject_pixel",
    "unproject_to_ground",
    "project_points",
    "contact_pixel",
    "horizon_row",
    "height_for_range",
    "observe_detection",
]

#: Rays shallower than this against the sand plane are treated as "no ground".
_MIN_RAY_Z = 1e-9


@dataclass(frozen=True)
class GroundHit:
    """Where a pixel's ray meets the flat sand plane."""

    point: np.ndarray  # map frame, z == ground_z
    depth_optical: float  # z in the optical frame (pinhole depth, metres)
    range_m: float  # horizontal distance from the camera's ground point
    t: float  # ray parameter (metres per unit ray)


@dataclass(frozen=True)
class Rejection:
    """A detection that could not be placed on the map, and why."""

    reason: str
    detail: float = 0.0
    pixel: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class GroundObservation:
    """A detection localised on the flat sand plane, ready to be tracked."""

    kind: str
    class_id: int
    x: float
    y: float
    radius: float
    height: float
    depth: float
    score: float
    range_m: float
    bearing_rad: float
    optical_depth_m: float
    pixel: Tuple[float, float]
    bearing_sigma_rad: float
    range_sigma_m: float
    conservative: bool

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


def r_map_optical(camera: CameraModel, pose_map_base: Pose3) -> np.ndarray:
    """``R_map_optical``: the optical-frame axes expressed in the map frame."""
    return pose_map_base.rotation @ camera.r_base_optical_matrix


def camera_position_map(camera: CameraModel, pose_map_base: Pose3) -> np.ndarray:
    """The camera optical centre in the map frame."""
    return pose_map_base.transform_point(camera.t_base_cam)


def unproject_pixel(
    camera: CameraModel,
    pose_map_base: Pose3,
    u: float,
    v: float,
    ground_z: float = 0.0,
) -> Optional[GroundHit]:
    """Intersect the ray through pixel ``(u, v)`` with the sand plane."""
    rotation = r_map_optical(camera, pose_map_base)
    position = camera_position_map(camera, pose_map_base)
    direction = rotation @ camera.pixel_to_ray(u, v)
    if abs(direction[2]) < _MIN_RAY_Z:
        return None
    t = (ground_z - position[2]) / direction[2]
    if t <= 0.0:
        return None
    point = position + t * direction
    ground_point = np.array([position[0], position[1], 0.0], dtype=float)
    horizontal = math.hypot(float(point[0] - ground_point[0]), float(point[1] - ground_point[1]))
    optical_depth = float(t * direction @ rotation[:, 2])
    return GroundHit(point=point, depth_optical=optical_depth, range_m=horizontal, t=t)


def unproject_to_ground(
    camera: CameraModel,
    pose_map_base: Pose3,
    u: float,
    v: float,
    ground_z: float = 0.0,
) -> Optional[np.ndarray]:
    """:func:`unproject_pixel` reduced to the map-frame point (or ``None``)."""
    hit = unproject_pixel(camera, pose_map_base, u, v, ground_z)
    return None if hit is None else hit.point


def project_points(
    camera: CameraModel,
    pose_map_base: Pose3,
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward projection, map -> pixels.

    Returns ``(uv, in_front)``; pixels of points behind the camera are NaN and
    flagged False.  Used by the tests and by the synthetic-image helpers, so the
    forward and inverse paths are verified against each other rather than
    independently trusted.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    rotation = r_map_optical(camera, pose_map_base)
    position = camera_position_map(camera, pose_map_base)
    optical = (pts - position) @ rotation  # row-wise R^T (p - t)
    depth = optical[:, 2]
    in_front = depth > 1e-6
    uv = np.full((pts.shape[0], 2), np.nan, dtype=float)
    if np.any(in_front):
        z = depth[in_front]
        uv[in_front, 0] = camera.fx * optical[in_front, 0] / z + camera.cx
        uv[in_front, 1] = camera.fy * optical[in_front, 1] / z + camera.cy
    return uv, in_front


def contact_pixel(detection: Any) -> Tuple[float, float]:
    """The ground-contact pixel of a detection: bbox bottom-centre."""
    return (0.5 * (float(detection.x0) + float(detection.x1)), float(detection.y1))


def horizon_row(camera: CameraModel) -> Optional[float]:
    """Image row where the view is horizontal (``None`` when it is not in frame).

    Above this row there is no sand, so the classical segmenter ignores it: the
    flat-ground model stops being meaningful there.  Derived from the camera's
    extrinsics alone (no terrain knowledge), using
    ``ray_base_z = 0`` at the principal column:
    ``v = cy - fy * R[2, 2] / R[2, 1]``.
    """
    rotation = camera.r_base_optical_matrix
    if abs(float(rotation[2, 1])) < _MIN_RAY_Z:
        return None
    row = camera.cy - camera.fy * float(rotation[2, 2]) / float(rotation[2, 1])
    if not np.isfinite(row):
        return None
    return float(row)


def height_for_range(
    camera: CameraModel,
    pose_map_base: Pose3,
    u: float,
    v: float,
    horizontal_range_m: float,
    ground_z: float = 0.0,
) -> Optional[float]:
    """Height above the sand of the point at ``(u, v)`` at a known range.

    For a boulder, the top of its box is roughly above the centre of its
    footprint, so combining the top pixel's ray with the centre's ground range
    gives an exposed-height estimate without any depth sensor.  Returns ``None``
    when the ray has no horizontal component.
    """
    if horizontal_range_m <= 0.0:
        return None
    rotation = r_map_optical(camera, pose_map_base)
    position = camera_position_map(camera, pose_map_base)
    direction = rotation @ camera.pixel_to_ray(u, v)
    horizontal_norm = math.hypot(float(direction[0]), float(direction[1]))
    if horizontal_norm < _MIN_RAY_Z:
        return None
    t = horizontal_range_m / horizontal_norm
    z = float(position[2] + t * direction[2])
    return z - ground_z


def _range_sigma(range_m: float, cfg: GroundCfg) -> float:
    """A simple, documented range uncertainty model (metres, 1 sigma).

    The flat-ground unprojection error grows with range: a 3-pixel pixel error
    is worth ``3 * range^2 / (fx * h)`` on the ground for a camera at height
    ``h``, which is the usual pinhole geometry.  We fold that into a single
    linear-in-range model so the covariance published with every obstacle is
    honest about the fact that distant boxes are much less precise.
    """
    return float(cfg.default_sigma_xy_m * (0.5 + range_m / max(cfg.max_range_m, 1e-6)))


def observe_detection(
    detection: Any,
    camera: CameraModel,
    pose_map_base: Pose3,
    ground_cfg: GroundCfg,
    arena: ArenaCfg,
) -> Union[GroundObservation, Rejection]:
    """Turn one detector output into a map-frame observation, or reject it.

    Every rejection carries a machine-readable reason so the sidecar/log can be
    audited instead of a detection silently disappearing.
    """
    u, v = contact_pixel(detection)
    if v < 0.0 or v >= camera.height or u < 0.0 or u >= camera.width:
        return Rejection("contact_pixel_outside_image", 0.0, (u, v))
    hit = unproject_pixel(camera, pose_map_base, u, v, arena.ground_z)
    if hit is None:
        return Rejection("above_horizon", 0.0, (u, v))
    if hit.range_m < ground_cfg.min_range_m:
        return Rejection("too_near", hit.range_m, (u, v))
    if hit.range_m > ground_cfg.max_range_m:
        return Rejection("too_far", hit.range_m, (u, v))

    kind = str(getattr(detection, "kind", "unknown"))
    class_id = int(getattr(detection, "class_id", 255))
    scale = (
        ground_cfg.crater_radius_scale
        if kind == "crater"
        else ground_cfg.boulder_radius_scale
    )
    width_px = max(1.0, float(detection.x1) - float(detection.x0))
    radius = 0.5 * width_px * hit.depth_optical / camera.fx * scale
    if radius < ground_cfg.min_radius_m:
        return Rejection("too_small", radius, (u, v))
    if radius > ground_cfg.max_radius_m:
        return Rejection("too_large", radius, (u, v))

    x, y = float(hit.point[0]), float(hit.point[1])
    if not arena.inside(x, y, margin=0.0):
        # outside the arena rectangle entirely: a wall, a reflection, or a
        # mis-projection.  The published wall mask uses a margin on top of this.
        return Rejection("outside_arena", hit.range_m, (u, v))

    height = 0.0
    if kind == "boulder":
        estimate = height_for_range(camera, pose_map_base, u, float(detection.y0), hit.range_m, arena.ground_z)
        if estimate is not None and estimate > 0.0:
            height = float(min(estimate, 2.0 * radius))

    bearing = math.atan2(y - pose_map_base.y, x - pose_map_base.x)
    bearing_sigma = float(math.atan2(0.5 * width_px, camera.fx))
    return GroundObservation(
        kind=kind,
        class_id=class_id,
        x=x,
        y=y,
        radius=radius,
        height=height,
        depth=0.0,
        score=float(getattr(detection, "score", 0.0)),
        range_m=hit.range_m,
        bearing_rad=bearing,
        optical_depth_m=hit.depth_optical,
        pixel=(u, v),
        bearing_sigma_rad=bearing_sigma,
        range_sigma_m=_range_sigma(hit.range_m, ground_cfg),
        conservative=bool(getattr(detection, "conservative", False)),
    )
