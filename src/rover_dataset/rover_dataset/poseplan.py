"""Capture-pose planning.

The rover is *teleported* between poses instead of being driven: it gives exact,
reproducible coverage of the arena, cannot get stuck on a boulder, and needs no
controller.  Poses are sampled (weighted by zone) from the competition zones,
rejected if they collide with an obstacle, and aimed at an obstacle so that the
detector sees objects at many ranges and bearings -- plus the inevitable
empty-sand frames, which are useful negatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import CaptureCfg
from .groundtruth import GroundTruth
from .seeding import rng_python

__all__ = ["CapturePose", "plan_poses", "footprint_inside"]


@dataclass(frozen=True)
class CapturePose:
    index: int
    x: float
    y: float
    yaw: float
    region: str

    def as_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "x": self.x, "y": self.y, "yaw": self.yaw, "region": self.region}


def footprint_inside(
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    arena_x: float,
    arena_y: float,
    tol: float = 0.02,
) -> bool:
    """True when the (rotated) rover footprint lies inside the arena rectangle."""
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    for dx, dy in (
        (length / 2.0, width / 2.0),
        (length / 2.0, -width / 2.0),
        (-length / 2.0, width / 2.0),
        (-length / 2.0, -width / 2.0),
    ):
        px = x + dx * cos_y - dy * sin_y
        py = y + dx * sin_y + dy * cos_y
        if px < -tol or py < -tol or px > arena_x + tol or py > arena_y + tol:
            return False
    return True


def _shrink(rect: Sequence[float], margin: float) -> Optional[Tuple[float, float, float, float]]:
    x0, y0, x1, y1 = (float(v) for v in rect)
    sx0, sy0, sx1, sy1 = x0 + margin, y0 + margin, x1 - margin, y1 - margin
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    return (sx0, sy0, sx1, sy1)


def _pick_weighted(regions: Sequence[Any], rng: Any) -> Any:
    total = sum(r.weight for r in regions)
    roll = rng.random() * total
    acc = 0.0
    for region in regions:
        acc += region.weight
        if roll <= acc:
            return region
    return regions[-1]


def plan_poses(
    ground_truth: GroundTruth,
    capture: CaptureCfg,
    master_seed: int,
    scene_id: str,
    footprint: Optional[Tuple[float, float]] = None,
    poses_per_scene: Optional[int] = None,
) -> List[CapturePose]:
    """Sample ``capture.poses_per_scene`` poses (deterministic per scene)."""
    plan = capture.pose_plan
    rng = rng_python(master_seed, plan.seed_namespace, scene_id)
    length, width = footprint or (1.5, 0.75)
    count = int(poses_per_scene) if poses_per_scene else int(capture.poses_per_scene)

    boulders = [(b.xy, b.radius) for b in ground_truth.boulders]
    craters = [(c.xy, c.radius) for c in ground_truth.craters]
    obstacles = [xy for xy, _ in boulders] + [xy for xy, _ in craters]
    centre = (ground_truth.length_x / 2.0, ground_truth.width_y / 2.0)

    poses: List[CapturePose] = []
    for index in range(count):
        pose = None
        for _ in range(plan.max_tries):
            region = _pick_weighted(plan.regions, rng)
            shrunk = _shrink(region.rect, plan.margin)
            if shrunk is None:
                continue
            x = rng.uniform(shrunk[0], shrunk[2])
            y = rng.uniform(shrunk[1], shrunk[3])

            too_close = False
            for (bx, by), radius in boulders:
                if math.hypot(x - bx, y - by) < plan.min_obstacle_clearance + radius:
                    too_close = True
                    break
            if not too_close:
                for (cx, cy), radius in craters:
                    if math.hypot(x - cx, y - cy) < plan.min_crater_clearance + radius:
                        too_close = True
                        break
            if too_close:
                continue

            yaw = _choose_yaw(plan, rng, x, y, obstacles, region, centre, boulders, craters)
            if not footprint_inside(x, y, yaw, length, width, ground_truth.length_x, ground_truth.width_y):
                continue
            pose = CapturePose(index, x, y, yaw, region.name)
            break
        if pose is None:
            continue
        poses.append(pose)
    return poses


def _choose_yaw(
    plan: Any,
    rng: Any,
    x: float,
    y: float,
    obstacles: Sequence[Tuple[float, float]],
    region: Any,
    centre: Tuple[float, float],
    boulders: Sequence[Any],
    craters: Sequence[Any],
) -> float:
    if plan.heading == "random":
        base = rng.uniform(-math.pi, math.pi)
    elif plan.heading == "toward_region_centre":
        base = math.atan2(region.rect[1] + region.rect[3] - 2.0 * y,
                          region.rect[0] + region.rect[2] - 2.0 * x)
    else:  # toward_obstacles
        if obstacles and rng.random() < 0.85:
            tx, ty = obstacles[rng.randrange(len(obstacles))]
            base = math.atan2(ty - y, tx - x)
        else:
            base = math.atan2(centre[1] - y, centre[0] - x)
    jitter = math.radians(plan.heading_jitter_deg)
    yaw = base + rng.uniform(-jitter, jitter)
    return math.atan2(math.sin(yaw), math.cos(yaw))
