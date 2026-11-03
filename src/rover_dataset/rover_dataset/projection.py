"""Project module-1 ground truth through a module-2 camera into YOLO boxes.

The whole automatic labelling pipeline is here:

    arena point  --R_arena_optical-->  optical frame  --pinhole-->  pixel

Nothing is approximated away silently:

* a **boulder** is a sphere that is partly buried; its label is the bounding box
  of the sampled, above-ground part of that sphere (so burial is respected),
* a **crater** is a circular depression; its label is the bounding box of the
  projected rim circle,
* objects that straddle the camera plane (points both in front of and behind the
  camera) are *dropped*, because a pinhole projection of them is meaningless,
* objects hidden behind a boulder are dropped (``occluded_policy: drop``),
* boxes are clipped to the image and dropped when too small or too little of
  them is visible.

Every dropped object is reported with a reason so the sidecar can be audited.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration_ref import CameraModel
from .config import LabelingCfg
from .groundtruth import Boulder, Crater, GroundTruth
from .labels import YoloBox

__all__ = [
    "RoverPose",
    "ProjectionResult",
    "ObjectLabel",
    "DroppedObject",
    "fibonacci_sphere",
    "boulder_surface_points",
    "crater_rim_points",
    "camera_pose_in_arena",
    "project_points",
    "unproject_pixel_to_ground",
    "label_frame",
]

#: Objects closer than this to the camera plane are not projected.
_MIN_DEPTH_M = 1e-6


@dataclass(frozen=True)
class RoverPose:
    """The rover base_link pose in the arena frame (roll/pitch/yaw, radians)."""

    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    def matrix(self) -> np.ndarray:
        from .geometry import rpy_matrix

        return rpy_matrix(self.roll, self.pitch, self.yaw)

    def as_tuple(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)


@dataclass(frozen=True)
class ObjectLabel:
    """One labelled object plus the geometry that produced the box."""

    kind: str
    object_id: str
    class_id: int
    box: YoloBox
    raw_box_px: Tuple[float, float, float, float]
    visible_fraction: float
    truncated: bool


@dataclass(frozen=True)
class DroppedObject:
    kind: str
    object_id: str
    class_id: int
    reason: str
    detail: float = 0.0


@dataclass
class ProjectionResult:
    labels: List[ObjectLabel] = field(default_factory=list)
    dropped: List[DroppedObject] = field(default_factory=list)

    @property
    def boxes(self) -> List[YoloBox]:
        return [label.box for label in self.labels]

    def dropped_summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for item in self.dropped:
            out[item.reason] = out.get(item.reason, 0) + 1
        return out

    def sidecar_dropped(self) -> List[Dict[str, Any]]:
        return [
            {"kind": d.kind, "id": d.object_id, "class_id": d.class_id,
             "reason": d.reason, "detail": d.detail}
            for d in self.dropped
        ]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def fibonacci_sphere(n: int) -> np.ndarray:
    """``n`` near-uniform unit vectors (deterministic, no RNG)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    i = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


def boulder_surface_points(boulder: Boulder, ground_z: float, samples: int) -> np.ndarray:
    """Points on the *above-ground* part of a (partly buried) boulder sphere."""
    dirs = fibonacci_sphere(samples)
    center = np.asarray(boulder.center, dtype=float)
    pts = center + boulder.radius * dirs
    return pts[pts[:, 2] >= ground_z - 1e-9]


def crater_rim_points(crater: Crater, ground_z: float, samples: int) -> np.ndarray:
    """Points on the rim circle of a crater (a flat ring at the sand level)."""
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    cx, cy = crater.center
    return np.column_stack(
        [cx + crater.radius * np.cos(angles), cy + crater.radius * np.sin(angles),
         np.full(samples, ground_z)]
    )


def camera_pose_in_arena(camera: CameraModel, pose: RoverPose) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(R_arena_optical, t_arena_cam)`` for the given rover pose."""
    r_arena_base = pose.matrix()
    r_base_optical = np.asarray(camera.r_base_optical, dtype=float).reshape(3, 3)
    r_arena_optical = r_arena_base @ r_base_optical
    t_base_cam = np.asarray(camera.t_base_cam, dtype=float)
    t_arena_cam = pose.position() + r_arena_base @ t_base_cam
    return r_arena_optical, t_arena_cam


def project_points(
    camera: CameraModel,
    r_arena_optical: np.ndarray,
    t_arena_cam: np.ndarray,
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project arena-frame points.  Returns ``(uv, in_front)``.

    ``uv`` is (N, 2) in pixels; ``in_front`` marks points with positive depth.
    Pixels of points behind the camera are meaningless and must be masked out.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    optical = (pts - t_arena_cam) @ r_arena_optical  # row-wise R^T (p - t)
    depth = optical[:, 2]
    in_front = depth > _MIN_DEPTH_M
    uv = np.full((pts.shape[0], 2), np.nan, dtype=float)
    safe = in_front
    if np.any(safe):
        z = depth[safe]
        uv[safe, 0] = camera.fx * optical[safe, 0] / z + camera.cx
        uv[safe, 1] = camera.fy * optical[safe, 1] / z + camera.cy
    return uv, in_front


def unproject_pixel_to_ground(
    camera: CameraModel,
    r_arena_optical: np.ndarray,
    t_arena_cam: np.ndarray,
    u: float,
    v: float,
    ground_z: float = 0.0,
) -> Optional[np.ndarray]:
    """Inverse of the projection for a point on the flat sand plane.

    Returns ``None`` when the ray is parallel to, or points away from, the plane.
    Used by module 4 and by this module's round-trip tests.
    """
    ray_optical = np.array([(u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy, 1.0])
    ray_arena = r_arena_optical @ ray_optical
    if abs(ray_arena[2]) < 1e-12:
        return None
    t = (ground_z - t_arena_cam[2]) / ray_arena[2]
    if t <= 0.0:
        return None
    return t_arena_cam + t * ray_arena


def _segment_point_distance(p0: np.ndarray, p1: np.ndarray, c: np.ndarray) -> Tuple[float, float]:
    """Distance from ``c`` to the segment ``p0->p1`` and the closest parameter."""
    d = p1 - p0
    dd = float(d @ d)
    if dd <= 1e-12:
        return float(np.linalg.norm(c - p0)), 0.0
    t = float((c - p0) @ d) / dd
    tc = min(1.0, max(0.0, t))
    return float(np.linalg.norm(c - (p0 + tc * d))), t


def _occluder(
    target_point: np.ndarray,
    cam_pos: np.ndarray,
    boulders: Sequence[Boulder],
    exclude_id: str,
    fraction: float,
) -> Optional[str]:
    """Return the id of a boulder that blocks the camera's view of ``target_point``."""
    for other in boulders:
        if other.id == exclude_id:
            continue
        dist, t = _segment_point_distance(cam_pos, target_point, np.asarray(other.center, dtype=float))
        if 0.0 < t < 1.0 and dist < other.radius * fraction:
            return other.id
    return None


# --------------------------------------------------------------------------- #
# the labeler
# --------------------------------------------------------------------------- #
def _box_from_points(
    kind: str,
    object_id: str,
    class_id: int,
    points: np.ndarray,
    camera: CameraModel,
    r_arena_optical: np.ndarray,
    t_arena_cam: np.ndarray,
    cfg: LabelingCfg,
    image_size: Tuple[int, int],
) -> Tuple[Optional[ObjectLabel], Optional[DroppedObject]]:
    width, height = image_size
    uv, in_front = project_points(camera, r_arena_optical, t_arena_cam, points)
    n_total = int(uv.shape[0])
    n_front = int(np.count_nonzero(in_front))

    def drop(reason: str, detail: float = 0.0):
        return None, DroppedObject(kind, object_id, class_id, reason, detail)

    if n_front < 2:
        return drop("off_camera", float(n_front))
    if n_front < n_total:
        # Some silhouette points are behind the camera: a pinhole projection of
        # the object is not defined, so it is not a valid training sample.
        return drop("straddles_camera_plane", float(n_front) / float(n_total))

    vis = uv[in_front]
    x0 = float(vis[:, 0].min())
    x1 = float(vis[:, 0].max())
    y0 = float(vis[:, 1].min())
    y1 = float(vis[:, 1].max())
    raw_w = x1 - x0
    raw_h = y1 - y0
    if raw_w <= 0.0 or raw_h <= 0.0:
        return drop("degenerate")

    cx0 = min(max(x0, 0.0), float(width))
    cx1 = min(max(x1, 0.0), float(width))
    cy0 = min(max(y0, 0.0), float(height))
    cy1 = min(max(y1, 0.0), float(height))
    clipped_w = cx1 - cx0
    clipped_h = cy1 - cy0
    area = clipped_w * clipped_h
    if clipped_w < cfg.min_box_width_px or clipped_h < cfg.min_box_height_px:
        return drop("too_small", area)
    if area < cfg.min_box_area_px:
        return drop("too_small", area)
    visible = area / (raw_w * raw_h)
    if visible < cfg.min_visible_fraction:
        return drop("not_visible_enough", visible)

    m = cfg.truncation_margin_px
    truncated = bool(
        x0 <= m or y0 <= m or x1 >= width - m or y1 >= height - m
    )
    if truncated and not cfg.keep_truncated:
        return drop("truncated", visible)

    box = YoloBox.from_pixels(
        class_id, cx0, cy0, cx1, cy1, width, height,
        object_id=object_id, kind=kind, visible_fraction=visible, truncated=truncated,
    )
    return (
        ObjectLabel(kind, object_id, class_id, box, (x0, y0, x1, y1), visible, truncated),
        None,
    )


def label_frame(
    ground_truth: GroundTruth,
    camera: CameraModel,
    pose: RoverPose,
    cfg: LabelingCfg,
    class_indices: Dict[str, int],
    image_size: Optional[Tuple[int, int]] = None,
) -> ProjectionResult:
    """Label every ground-truth object visible in one camera frame."""
    size = image_size if image_size is not None else (camera.width, camera.height)
    r_ao, t_ao = camera_pose_in_arena(camera, pose)
    cam_pos = t_ao

    result = ProjectionResult()
    for kind, obj in ground_truth.objects():
        if kind not in class_indices:
            continue
        class_id = class_indices[kind]
        if kind == "boulder":
            target_point = np.asarray(obj.center, dtype=float)
        else:
            # a crater has no z in the ground truth; its rim sits at the sand level
            target_point = np.array([obj.center[0], obj.center[1], cfg.ground_z], dtype=float)
        occluder = _occluder(
            target_point, cam_pos, ground_truth.boulders, obj.id, cfg.occlusion_fraction
        )
        if occluder is not None and cfg.occluded_policy == "drop":
            result.dropped.append(DroppedObject(kind, obj.id, class_id, "occluded"))
            continue

        if kind == "boulder":
            points = boulder_surface_points(obj, cfg.ground_z, cfg.boulder_silhouette_samples)
        else:
            points = crater_rim_points(obj, cfg.ground_z, cfg.crater_rim_samples)

        label, drop = _box_from_points(
            kind, obj.id, class_id, points, camera, r_ao, t_ao, cfg, size
        )
        if label is not None:
            result.labels.append(label)
        elif drop is not None:
            result.dropped.append(drop)
    return result
