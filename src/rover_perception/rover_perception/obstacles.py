"""Confidence-weighted obstacles and the arena-wall mask.

This is the boundary between "what perception saw" and "what the rest of the
rover is allowed to act on", so it is deliberately explicit:

* the **arena-wall mask** rejects any track whose centre is not inside the arena
  rectangle with the configured margin.  Walls are known geometry, not
  obstacles: without this mask every wall the forward camera can see would be
  published as a rock, and a rover that mistrusts the walls cannot drive to the
  excavation zone.  Rejections are counted, not silently dropped.
* the payload is a plain mapping, so it is unit-testable without ROS, and the
  node's only job is to copy fields onto the message.
* a ``PointCloud2`` view of the same obstacles is produced as well, because
  nav2's obstacle layer consumes a point cloud natively through
  ``observation_sources`` - module 5 then needs no custom costmap plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ArenaCfg, OBSTACLE_CLASS_IDS
from .tracking import Track

__all__ = [
    "KIND_UNKNOWN",
    "wall_mask_reason",
    "mask_tracks",
    "obstacle_fields",
    "covariance_from_sigma",
    "build_obstacle_payload",
    "pointcloud_dtype",
    "pointcloud_payload",
    "POINT_FIELDS",
]

KIND_UNKNOWN = 0

#: Nominal orientation variance (rad^2) published for obstacles.
#:
#: A round footprint has no observable heading, so a floor is published instead
#: of a fake confident zero: consumers that care only about position (nav2's
#: obstacle layer) ignore it, and consumers that reason about shapes are told
#: honestly that the orientation is unknown.
ORIENTATION_VARIANCE_FLOOR = 0.25

#: The PointCloud2 layout: (name, offset, datatype) with datatype 7 = FLOAT32.
POINT_FIELDS: Tuple[Tuple[str, int, int], ...] = (
    ("x", 0, 7),
    ("y", 4, 7),
    ("z", 8, 7),
    ("intensity", 12, 7),
)


def covariance_from_sigma(sigma_xy_m: float, sigma_z_m: float = 0.0) -> List[float]:
    """A 6x6 row-major covariance (36 floats) for a tracked obstacle."""
    sigma_xy = max(1e-6, float(sigma_xy_m))
    sigma_z = max(1e-6, float(sigma_z_m)) if sigma_z_m > 0.0 else sigma_xy
    covariance = [0.0] * 36
    covariance[0] = sigma_xy * sigma_xy
    covariance[7] = sigma_xy * sigma_xy
    covariance[14] = sigma_z * sigma_z
    covariance[21] = ORIENTATION_VARIANCE_FLOOR
    covariance[28] = ORIENTATION_VARIANCE_FLOOR
    covariance[35] = ORIENTATION_VARIANCE_FLOOR
    return covariance


def wall_mask_reason(track: Track, arena: ArenaCfg) -> Optional[str]:
    """``None`` when the track is a legal in-arena obstacle, else the reason."""
    if not arena.inside(track.x, track.y):
        # Either the object is outside the arena (impossible for a real
        # obstacle), or a wall / wall shadow / reflection was detected.  The
        # spec forbids using the walls as sensors, and the walls are known
        # geometry, so this is not an obstacle.
        bounds = arena.bounds
        if track.x < bounds.x0 or track.x > bounds.x1:
            return "outside_arena_x"
        if track.y < bounds.y0 or track.y > bounds.y1:
            return "outside_arena_y"
        return "inside_wall_margin"
    return None


def mask_tracks(
    tracks: Sequence[Track], arena: ArenaCfg
) -> Tuple[List[Track], List[Tuple[Track, str]]]:
    """Split tracks into legal obstacles and wall-mask rejections."""
    kept: List[Track] = []
    dropped: List[Tuple[Track, str]] = []
    for track in tracks:
        reason = wall_mask_reason(track, arena)
        if reason is None:
            kept.append(track)
        else:
            dropped.append((track, reason))
    return kept, dropped


def obstacle_fields(
    track: Track,
    now: float,
    frame_id: str,
) -> Dict[str, Any]:
    """Every numeric field of one ``rover_perception_msgs/Obstacle``."""
    kind_map = {"boulder": 1, "crater": 2}
    return {
        "id": track.id,
        "kind": kind_map.get(track.kind, KIND_UNKNOWN),
        "class_id": int(track.class_id) if track.class_id >= 0 else 255,
        "x": float(track.x),
        "y": float(track.y),
        "z": float(track.height) if track.kind == "boulder" else 0.0,
        "radius": float(track.radius),
        "height": float(track.height) if track.kind == "boulder" else 0.0,
        "depth": float(track.depth) if track.kind == "crater" else 0.0,
        "confidence": float(track.confidence),
        "score": float(track.score),
        "observations": int(track.observations),
        "age_s": float(track.age_s(now)),
        "last_seen_age_s": float(track.last_seen_age_s(now)),
        "source": track.source,
        "conservative": bool(track.conservative),
        "covariance": covariance_from_sigma(track.sigma_xy_m),
        "frame_id": frame_id,
    }


def build_obstacle_payload(
    tracks: Sequence[Track],
    now: float,
    frame_id: str,
    counters: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """The ``ObstacleArray`` payload (list of field mappings + counters)."""
    payload: Dict[str, Any] = {
        "frame_id": frame_id,
        "obstacles": [obstacle_fields(track, now, frame_id) for track in tracks],
    }
    for key in ("frames_processed", "dropped_wall", "dropped_range", "frames_without_fix"):
        payload[key] = int((counters or {}).get(key, 0))
    return payload


def pointcloud_dtype() -> np.dtype:
    """The structured dtype of the obstacle point cloud."""
    return np.dtype([(name, "<f4") for name, _offset, _datatype in POINT_FIELDS])


def pointcloud_payload(
    obstacles: Sequence[Dict[str, Any]],
    frame_id: str,
) -> Tuple[np.ndarray, int]:
    """Pack obstacles into a PointCloud2-ready structured array.

    ``x``/``y`` are the footprint centre, ``z`` is the boulder's exposed top (or
    the sand for a crater) and ``intensity`` is the confidence, so a consumer can
    threshold the cloud (``intensity >= 0.5``) without a custom message.
    Returns ``(points, point_step)``.
    """
    points = np.zeros(len(obstacles), dtype=pointcloud_dtype())
    for index, obstacle in enumerate(obstacles):
        points["x"][index] = float(obstacle["x"])
        points["y"][index] = float(obstacle["y"])
        points["z"][index] = float(obstacle["z"])
        points["intensity"][index] = float(obstacle["confidence"])
    return points, points.dtype.itemsize
