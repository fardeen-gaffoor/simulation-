"""Camera calibration: the ONE file the ground projection depends on.

Module 4 projects a detected boulder/crater ground-contact pixel to arena
coordinates using the intrinsics and extrinsics here plus a flat-ground
assumption.  The file is generated from ``rover.yaml`` so the URDF and the
calibration can never drift apart, and it is emitted in the same shape for the
simulated cameras and (after a one-off measurement) the real CSI cameras.

Optical frames follow REP-103/REP-105: +z forward, +x right, +y down, related to
the camera link frame (which looks along +x) by ``rpy = (-pi/2, 0, -pi/2)``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from .config import Camera, RoverConfig
from .geometry import rpy_matrix, rotate

__all__ = ["optical_frame_rpy", "camera_calibration", "calibration_document"]

#: rpy (rad) of a camera *optical* frame expressed in the camera *link* frame.
OPTICAL_FRAME_RPY: Tuple[float, float, float] = (-math.pi / 2.0, 0.0, -math.pi / 2.0)

DISTORTION_MODEL = "plumb_bob"


def optical_frame_rpy() -> Tuple[float, float, float]:
    return OPTICAL_FRAME_RPY


def _round(value: float, places: int = 9) -> float:
    return round(float(value), places)


def camera_calibration(cam: Camera) -> Dict[str, Any]:
    """Full calibration record for one camera."""
    rpy = cam.rpy_rad
    r_link = rpy_matrix(*rpy)
    r_optical = rpy_matrix(*OPTICAL_FRAME_RPY)
    # R_base_optical = R_base_link @ R_link_optical
    r_base_optical = [
        [sum(r_link[i][k] * r_optical[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    # camera viewing direction (+x of the link) in base_link
    forward = rotate(r_link, (1.0, 0.0, 0.0))
    return {
        "name": cam.name,
        "topic": cam.topic,
        "camera_info_topic": cam.camera_info_topic,
        "role": cam.role,
        "frame_id": cam.optical_frame,
        "parent_frame": cam.frame,
        "model": "pinhole",
        "distortion_model": DISTORTION_MODEL,
        "image": {"width": cam.width, "height": cam.height, "encoding": "rgb8"},
        "update_rate": cam.update_rate,
        "hfov_deg": cam.hfov_deg,
        "vfov_deg": _round(cam.vfov_deg),
        "intrinsics": {
            "fx": _round(cam.fx),
            "fy": _round(cam.fy),
            "cx": _round(cam.cx),
            "cy": _round(cam.cy),
            "skew": 0.0,
        },
        "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "extrinsics": {
            "translation_xyz": [_round(v) for v in cam.origin],
            "rotation_rpy_rad": [_round(v) for v in rpy],
            "rotation_rpy_deg": [_round(math.degrees(v), 6) for v in rpy],
            "rotation_matrix_base_camera": [[_round(r_link[i][j]) for j in range(3)] for i in range(3)],
            "view_direction_base": [_round(v) for v in forward],
            "optical_frame_rpy_rad": [_round(v) for v in OPTICAL_FRAME_RPY],
            "rotation_matrix_base_optical": [
                [_round(r_base_optical[i][j]) for j in range(3)] for i in range(3)
            ],
        },
    }


def calibration_document(cfg: RoverConfig, seed: int = 0) -> Dict[str, Any]:
    """The complete ``calibration.yaml`` payload (one file, all cameras)."""
    cameras: List[Dict[str, Any]] = [camera_calibration(cam) for cam in cfg.sensors.cameras]
    return {
        "schema_version": 1,
        "source": "rover_description.generate (derived from config/rover.yaml)",
        "config_sha256": cfg.config_sha256(),
        "frame": "base_link",
        "declared_forward": cfg.declared_forward,
        "notes": [
            "intrinsics assume square pixels and a centred principal point; replace with "
            "measured values for the real CSI cameras (same keys).",
            "extrinsics are the camera link pose in base_link; the optical frame is a child "
            "with rpy = (-pi/2, 0, -pi/2).",
            "ground projection: pixel -> ray in optical frame -> rotate by "
            "rotation_matrix_base_optical -> intersect the z=0 sand plane.",
        ],
        "cameras": {cam["name"]: cam for cam in cameras},
    }
