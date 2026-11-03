"""Randomised rover spawn pose inside the start zone.

The competition specifies a random start pose *and heading* inside the start
zone on every run.  The sampling is deterministic in the run seed, so the
generator, the launch file and the ground-truth file all agree on the pose
without exchanging any file: they call :func:`sample_start_pose` with the same
seed.

A pose is only accepted when the rover's *stowed footprint* (1.5 x 0.75 m,
rotated by the sampled heading) lies completely inside the start zone, inset by
a configurable margin.  At full 45 deg the footprint needs
(l + w) / sqrt(2) = 1.591 m per axis, which the 2.0 x 2.3 m start zone satisfies
for every heading - the sampler can therefore never fail with the given ground
truth; if it ever does, the arena configuration is wrong and this module says so
loudly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import ArenaConfig, ConfigError
from .seeding import rng_python

__all__ = ["StartPose", "footprint_corners", "rotation_matrix", "sample_start_pose"]


def rotation_matrix(yaw: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """2-D rotation by ``yaw`` (radians)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return ((c, -s), (s, c))


def footprint_corners(
    x: float, y: float, yaw: float, length: float, width: float
) -> List[Tuple[float, float]]:
    """World-frame corners of the rotated rectangular footprint (CCW)."""
    rot = rotation_matrix(yaw)
    half_l, half_w = 0.5 * length, 0.5 * width
    corners = []
    for dx, dy in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w)):
        corners.append((x + rot[0][0] * dx + rot[0][1] * dy, y + rot[1][0] * dx + rot[1][1] * dy))
    return corners


@dataclass(frozen=True)
class StartPose:
    """A validated spawn pose in the arena frame."""

    x: float
    y: float
    z: float
    yaw: float
    zone: str
    seed: int
    attempts: int
    declared_forward_deg: float
    footprint_length: float
    footprint_width: float
    corners: Tuple[Tuple[float, float], ...]

    @property
    def yaw_deg(self) -> float:
        return math.degrees(self.yaw)

    def to_dict(self) -> Dict[str, object]:
        return {
            "zone": self.zone,
            "seed": self.seed,
            "position": [round(self.x, 6), round(self.y, 6), round(self.z, 6)],
            "yaw_rad": round(self.yaw, 9),
            "yaw_deg": round(self.yaw_deg, 6),
            "declared_forward_deg": self.declared_forward_deg,
            "footprint": [self.footprint_length, self.footprint_width],
            "corners_xy": [[round(cx, 6), round(cy, 6)] for cx, cy in self.corners],
            "attempts": self.attempts,
        }


def sample_start_pose(
    cfg: ArenaConfig,
    seed: int,
    surface: Optional[Callable[[float, float], float]] = None,
    yaw_override_deg: Optional[float] = None,
) -> StartPose:
    """Draw a deterministic random pose inside the start zone.

    :param cfg: validated arena configuration.
    :param seed: master run seed.
    :param surface: optional ``(x, y) -> sand height`` callable, used to place
        the spawn point on the local sand surface instead of the nominal plane.
    :param yaw_override_deg: force a heading (used for reproducible tests).
    :raises ConfigError: if no pose fits within ``start_pose.max_tries``.
    """
    sp = cfg.start_pose
    zone = cfg.zone("start")
    rect = zone.rect
    rng = rng_python(seed, sp.seed_namespace)
    margin = sp.margin
    length, width = sp.footprint_length, sp.footprint_width

    if rect.width <= 2 * margin or rect.height <= 2 * margin:
        raise ConfigError(
            f"start zone {rect.as_list()} is too small for margin {margin}"
        )

    fixed_yaw = yaw_override_deg is not None or sp.yaw_mode == "fixed"
    if yaw_override_deg is not None:
        yaw = math.radians(yaw_override_deg)
    elif sp.yaw_mode == "fixed":
        yaw = math.radians(sp.yaw_deg)
    else:
        yaw = 0.0  # drawn per attempt below

    for attempt in range(1, sp.max_tries + 1):
        if not fixed_yaw:
            yaw = rng.uniform(-math.pi, math.pi)
        x = rng.uniform(rect.x0 + margin, rect.x1 - margin)
        y = rng.uniform(rect.y0 + margin, rect.y1 - margin)
        corners = footprint_corners(x, y, yaw, length, width)
        if all(rect.contains(cx, cy, margin=margin) for cx, cy in corners):
            surface_z = float(surface(x, y)) if surface is not None else 0.0
            return StartPose(
                x=x,
                y=y,
                z=surface_z + cfg.spawn_z,
                yaw=yaw,
                zone=zone.name,
                seed=seed,
                attempts=attempt,
                declared_forward_deg=sp.declared_forward_deg,
                footprint_length=length,
                footprint_width=width,
                corners=tuple(corners),
            )
    raise ConfigError(
        f"failed to sample a start pose in {sp.max_tries} attempts "
        f"(zone {rect.as_list()}, footprint {length} x {width}, margin {margin})"
    )


def pose_acceptance_rate(cfg: ArenaConfig, seed: int, samples: int = 500) -> float:
    """Fraction of raw draws that are accepted (diagnostic / tuning helper)."""
    sp = cfg.start_pose
    rect = cfg.zone("start").rect
    rng = rng_python(seed, sp.seed_namespace + ".diagnostic")
    accepted = 0
    for _ in range(samples):
        yaw = rng.uniform(-math.pi, math.pi)
        x = rng.uniform(rect.x0 + sp.margin, rect.x1 - sp.margin)
        y = rng.uniform(rect.y0 + sp.margin, rect.y1 - sp.margin)
        corners = footprint_corners(x, y, yaw, sp.footprint_length, sp.footprint_width)
        if all(rect.contains(cx, cy, margin=sp.margin) for cx, cy in corners):
            accepted += 1
    return accepted / float(samples)


def yaw_bucket(yaw: float, buckets: int = 4) -> int:
    """Bucket index of a heading (used by the spread test)."""
    span = 2.0 * math.pi / buckets
    return int(((yaw + math.pi) % (2.0 * math.pi)) // span)


def sample_many(cfg: ArenaConfig, seeds: Sequence[int]) -> List[StartPose]:
    """Sample one pose per seed (helper for tests and batch tooling)."""
    return [sample_start_pose(cfg, s) for s in seeds]
