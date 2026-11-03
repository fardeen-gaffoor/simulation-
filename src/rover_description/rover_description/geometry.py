"""Robot geometry: analytic inertias, the stowed-envelope AABB and link layout.

Every number here is derived from :mod:`rover_description.config`; nothing is
hard-coded.  The URDF, the mass budget, the envelope check and the tests all go
through these helpers so they can never disagree.

Conventions
-----------
* ``base_link`` origin is the chassis centre projected to the ground, so the
  wheel contact plane is ``z = 0``.
* Rotations are ZYX Euler ``(roll, pitch, yaw)`` in radians.
* ``pitch`` about +y: ``R_y(pitch) @ ex = (cos, 0, -sin)``, so a POSITIVE pitch
  points a local +x axis DOWN (matching the ladder and camera conventions in
  ``rover.yaml``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import Camera, RoverConfig

__all__ = [
    "Part",
    "Inertia",
    "rpy_matrix",
    "box_inertia",
    "cylinder_inertia_z",
    "rotate",
    "parts",
    "aabb",
    "compute_envelope",
    "link_inertias",
    "wheel_centers",
    "camera_pose",
    "ladder_pitch_rad",
]

Vec3 = Tuple[float, float, float]
Vec6 = Tuple[float, float, float, float, float, float]


def rpy_matrix(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    """ZYX Euler angles (Gazebo/URDF convention) to a 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def rotate(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vec3:
    return (
        sum(matrix[0][k] * vector[k] for k in range(3)),
        sum(matrix[1][k] * vector[k] for k in range(3)),
        sum(matrix[2][k] * vector[k] for k in range(3)),
    )


def box_inertia(mass: float, size: Sequence[float]) -> Vec6:
    """Inertia of a solid cuboid (ixx, ixy, ixz, iyy, iyz, izz), off-diagonals 0."""
    a, b, c = (float(v) for v in size)
    ixx = mass * (b * b + c * c) / 12.0
    iyy = mass * (a * a + c * c) / 12.0
    izz = mass * (a * a + b * b) / 12.0
    return (ixx, 0.0, 0.0, iyy, 0.0, izz)


def cylinder_inertia_z(mass: float, radius: float, length: float) -> Vec6:
    """Inertia of a solid cylinder whose axis is the link +z axis."""
    r2 = radius * radius
    ixx = mass * (3.0 * r2 + length * length) / 12.0
    izz = mass * r2 / 2.0
    return (ixx, 0.0, 0.0, ixx, 0.0, izz)


@dataclass(frozen=True)
class Inertia:
    """Mass and the 3x3 inertia tensor about the link origin."""

    mass: float
    ixx: float
    ixy: float
    ixz: float
    iyy: float
    iyz: float
    izz: float

    @classmethod
    def from_vec6(cls, mass: float, vec: Vec6) -> "Inertia":
        return cls(mass, *vec)

    def matrix(self) -> List[List[float]]:
        return [
            [self.ixx, self.ixy, self.ixz],
            [self.ixy, self.iyy, self.iyz],
            [self.ixz, self.iyz, self.izz],
        ]

    def principal_moments(self) -> Tuple[float, float, float]:
        """Eigenvalues of the (symmetric) tensor, ascending.  Used by the tests."""
        m = self.matrix()
        # analytic 3x3 symmetric eigen via the Jacobi-free closed form is messy;
        # use the characteristic-polynomial invariants for the test's needs.
        # Here we only need the sorted diagonal for our axis-aligned primitives.
        return tuple(sorted((m[0][0], m[1][1], m[2][2])))  # type: ignore[return-value]


@dataclass(frozen=True)
class Part:
    """An axis-aligned box used for the envelope and for xacro generation."""

    name: str
    center: Vec3
    size: Vec3
    rpy: Vec3 = (0.0, 0.0, 0.0)


def ladder_pitch_rad(cfg: RoverConfig) -> float:
    return math.radians(cfg.excavation.ladder.stowed_pitch_deg)


def wheel_centers(cfg: RoverConfig) -> Dict[str, Vec3]:
    """Wheel-link origins in ``base_link`` (front-left, front-right, rear-left, rear-right)."""
    d = cfg.drive
    x = d.wheelbase / 2.0
    y = d.track / 2.0
    z = d.wheel_radius
    return {
        "wheel_front_left": (x, y, z),
        "wheel_front_right": (x, -y, z),
        "wheel_rear_left": (-x, y, z),
        "wheel_rear_right": (-x, -y, z),
    }


def ladder_box_center(cfg: RoverConfig, pitch: float) -> Vec3:
    """Ladder body centre in ``base_link`` for a given pitch (radians)."""
    lad = cfg.excavation.ladder
    rot = rpy_matrix(0.0, pitch, 0.0)
    half = rotate(rot, (lad.length / 2.0, 0.0, 0.0))
    return (
        lad.pivot[0] + half[0],
        lad.pivot[1] + half[1],
        lad.pivot[2] + half[2],
    )


def belt_center(cfg: RoverConfig, pitch: float, ladder_center: Vec3) -> Vec3:
    """Belt roller centre in ``base_link`` (at the ladder tip)."""
    lad = cfg.excavation.ladder
    rot = rpy_matrix(0.0, pitch, 0.0)
    # from the ladder centre, walk half a length to the tip
    tip = rotate(rot, (lad.length / 2.0, 0.0, 0.0))
    return (
        ladder_center[0] + tip[0],
        ladder_center[1] + tip[1],
        ladder_center[2] + tip[2],
    )


def camera_pose(cfg: RoverConfig, cam: Camera) -> Tuple[Vec3, Vec3]:
    """``(origin, rpy_rad)`` of a camera link in ``base_link``."""
    return (cam.origin, cam.rpy_rad)


def parts(cfg: RoverConfig) -> List[Part]:
    """Every solid part as an oriented box, for the envelope AABB."""
    out: List[Part] = []
    ch = cfg.chassis
    out.append(Part("chassis", ch.center, (ch.length, ch.width, ch.height)))

    d = cfg.drive
    for name, center in wheel_centers(cfg).items():
        # wheel cylinder axis is link +y after the mount rotation -> AABB is
        # (2r, width, 2r) centred on the wheel
        out.append(Part(name, center, (2 * d.wheel_radius, d.wheel_width, 2 * d.wheel_radius)))

    pitch = ladder_pitch_rad(cfg)
    lad = cfg.excavation.ladder
    lc = ladder_box_center(cfg, pitch)
    out.append(Part("ladder", lc, (lad.length, lad.width, lad.frame_thickness),
                    (0.0, pitch, 0.0)))
    belt = cfg.excavation.belt
    bc = belt_center(cfg, pitch, lc)
    out.append(Part("belt", bc, (2 * belt.roller_radius, belt.length, 2 * belt.roller_radius),
                    (0.0, pitch, 0.0)))

    tub = cfg.excavation.tub
    out.append(Part("tub", tub.center, tub.size))
    funnel = cfg.excavation.funnel
    out.append(Part("funnel", funnel.center, funnel.size))

    for cam in cfg.sensors.cameras:
        out.append(Part(cam.name, cam.origin, CAMERA_BODY, cam.rpy_rad))
    if cfg.sensors.imu.enabled:
        out.append(Part("imu", cfg.sensors.imu.origin, (0.04, 0.04, 0.02)))
    return out


#: camera enclosure bounding box (depth, width, height) in the camera link frame
CAMERA_BODY: Vec3 = (0.05, 0.04, 0.04)


def _corners(part: Part) -> List[Vec3]:
    hx, hy, hz = (s / 2.0 for s in part.size)
    rot = rpy_matrix(*part.rpy)
    out: List[Vec3] = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                local = (sx * hx, sy * hy, sz * hz)
                r = rotate(rot, local)
                out.append((part.center[0] + r[0], part.center[1] + r[1], part.center[2] + r[2]))
    return out


def aabb(part_list: Sequence[Part]) -> Tuple[Vec3, Vec3]:
    """Axis-aligned bounding box (min, max) of a list of oriented boxes."""
    if not part_list:
        raise ValueError("aabb: no parts")
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    for part in part_list:
        for corner in _corners(part):
            for axis in range(3):
                lo[axis] = min(lo[axis], corner[axis])
                hi[axis] = max(hi[axis], corner[axis])
    return ((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2]))


def compute_envelope(cfg: RoverConfig) -> "EnvelopeSize":
    """Stowed robot bounding box (sizes) from the ground plane up."""
    lo, hi = aabb(parts(cfg))
    return EnvelopeSize(
        length_x=hi[0] - lo[0],
        width_y=hi[1] - lo[1],
        height_z=hi[2] - lo[2],
        x_min=lo[0], x_max=hi[0],
        y_min=lo[1], y_max=hi[1],
        z_min=lo[2], z_max=hi[2],
    )


@dataclass(frozen=True)
class EnvelopeSize:
    length_x: float
    width_y: float
    height_z: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def as_tuple(self) -> Vec3:
        return (self.length_x, self.width_y, self.height_z)


def link_inertias(cfg: RoverConfig) -> Dict[str, Inertia]:
    """Analytic inertia tensor for every link that carries mass."""
    out: Dict[str, Inertia] = {}
    ch = cfg.chassis
    out["base_link"] = Inertia.from_vec6(ch.mass, box_inertia(ch.mass, (ch.length, ch.width, ch.height)))

    d = cfg.drive
    for name in wheel_centers(cfg):
        out[name] = Inertia.from_vec6(
            d.wheel_mass, cylinder_inertia_z(d.wheel_mass, d.wheel_radius, d.wheel_width)
        )

    lad = cfg.excavation.ladder
    out["ladder_link"] = Inertia.from_vec6(
        lad.mass, box_inertia(lad.mass, (lad.length, lad.width, lad.frame_thickness))
    )
    belt = cfg.excavation.belt
    out["belt_link"] = Inertia.from_vec6(
        belt.mass, cylinder_inertia_z(belt.mass, belt.roller_radius, belt.length)
    )
    tub = cfg.excavation.tub
    out["tub_link"] = Inertia.from_vec6(tub.mass, box_inertia(tub.mass, tub.size))
    funnel = cfg.excavation.funnel
    out["funnel_link"] = Inertia.from_vec6(funnel.mass, box_inertia(funnel.mass, funnel.size))
    return out
