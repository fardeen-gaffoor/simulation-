"""Arena fiducial marker boards: geometry, world poses and the module-4 handoff.

The challenge allows at most five markers, only on the start-zone frame
(2 sides) plus one wall marker, at most 25x25 cm and at most 60 cm high.

``cat_arena`` is responsible for the *physical boards* and for their exact
ground-truth poses.  Module 4 draws the AprilTag bitmaps into the dark inner
square and runs the detector: to make that handoff painless, the emitted
``markers.yaml`` also carries an ``apriltag_ros``-shaped block (position +
quaternion + tag size) for every board.

Everything here is visual-only in the world; boards never collide, so a rover
grazing a wall does not "hit" a marker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import ArenaConfig
from .sdf import board_rpy, rpy_matrix, rotation_matrix_to_rpy

__all__ = ["MarkerPose", "marker_poses", "rpy_to_quaternion", "quaternion_to_rpy"]

_MOUNT_NORMALS: Dict[str, Tuple[float, float]] = {
    "north_wall": (0.0, -1.0),
    "south_wall": (0.0, 1.0),
    "west_wall": (1.0, 0.0),
    "east_wall": (-1.0, 0.0),
}


@dataclass(frozen=True)
class MarkerPose:
    """World pose of one marker board (the board's front face faces ``normal``)."""

    marker_id: int
    family: str
    mount: str
    note: str
    board_size: float
    tag_size: float
    position: Tuple[float, float, float]
    rpy: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    top_z: float

    @property
    def quaternion_xyzw(self) -> Tuple[float, float, float, float]:
        return rpy_to_quaternion(*self.rpy)

    def to_dict(self) -> Dict[str, object]:
        qx, qy, qz, qw = self.quaternion_xyzw
        return {
            "id": self.marker_id,
            "family": self.family,
            "mount": self.mount,
            "note": self.note,
            "board_size": round(self.board_size, 6),
            "tag_size": round(self.tag_size, 6),
            "position": [round(v, 6) for v in self.position],
            "rpy": [round(v, 9) for v in self.rpy],
            "quaternion_xyzw": [round(v, 9) for v in (qx, qy, qz, qw)],
            "normal": [round(v, 9) for v in self.normal],
            "top_z": round(self.top_z, 6),
        }


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """ZYX Euler angles to ``(x, y, z, w)`` unit quaternion."""
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def quaternion_to_rpy(q: Sequence[float]) -> Tuple[float, float, float]:
    """``(x, y, z, w)`` quaternion (any norm) to ZYX Euler angles."""
    qx, qy, qz, qw = (float(v) for v in q)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("quaternion_to_rpy: zero quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    m00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    m10 = 2.0 * (qx * qy + qz * qw)
    m11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    m20 = 2.0 * (qx * qz - qy * qw)
    m21 = 2.0 * (qy * qz + qx * qw)
    m22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    return rotation_matrix_to_rpy([[m00, 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
                                   [m10, m11, m21],
                                   [m20, m21, m22]])


def marker_poses(cfg: ArenaConfig) -> List[MarkerPose]:
    """World poses of every configured marker board (visual-only in the world)."""
    out: List[MarkerPose] = []
    markers = cfg.markers
    for placement in markers.placements:
        nx, ny = _MOUNT_NORMALS[placement.mount]
        rpy = board_rpy((nx, ny))
        # push the board off the wall face along the normal, i.e. INTO the
        # arena (the normal points away from the wall, into the play area)
        back_offset = markers.stand_off + 0.5 * markers.board_thickness
        position = (
            placement.position[0] + nx * back_offset,
            placement.position[1] + ny * back_offset,
            markers.center_height,
        )
        out.append(
            MarkerPose(
                marker_id=placement.marker_id,
                family=markers.family,
                mount=placement.mount,
                note=placement.note,
                board_size=markers.board_size,
                tag_size=markers.tag_size,
                position=position,
                rpy=rpy,
                normal=(nx, ny, 0.0),
                top_z=markers.center_height + 0.5 * markers.board_size,
            )
        )
    return out


def markers_yaml_document(cfg: ArenaConfig, seed: int) -> Dict[str, object]:
    """Full ``markers.yaml`` payload, including the apriltag_ros-shaped block."""
    poses = marker_poses(cfg)
    standalone = []
    for pose in poses:
        qx, qy, qz, qw = pose.quaternion_xyzw
        standalone.append(
            {
                "id": pose.marker_id,
                "size": round(pose.tag_size, 6),
                "name": f"arena_tag_{pose.marker_id}_{pose.mount}",
                "pose": {
                    "position": {"x": round(pose.position[0], 6), "y": round(pose.position[1], 6), "z": round(pose.position[2], 6)},
                    "orientation": {"x": round(qx, 9), "y": round(qy, 9), "z": round(qz, 9), "w": round(qw, 9)},
                },
            }
        )
    return {
        "schema_version": 1,
        "seed": seed,
        "frame": "arena/map frame (origin = west-south arena corner, +x east, +y north)",
        "family": cfg.markers.family,
        "board_size": cfg.markers.board_size,
        "tag_size": cfg.markers.tag_size,
        "count": len(poses),
        "limits": {"max_count": cfg.markers.max_count, "max_board_size": 0.25, "max_top_z": 0.60},
        "markers": [pose.to_dict() for pose in poses],
        "apriltag_ros": {
            "note": "paste into module 4 tags.yaml; poses are the MARKER CENTRES in the map frame",
            "standalone_tags": standalone,
        },
    }
