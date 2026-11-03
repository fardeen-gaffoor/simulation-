"""Seeded placement of competition obstacles (boulders and craters).

Rules encoded here (competition ground truth, never deviated from):

* per run at least 3 boulders and at least 3 craters,
* boulders are 30-40 cm diameter with varying exposed heights,
* craters are up to 40-50 cm wide with varying NEGATIVE depth,
* boulders may also be placed in the excavation zone,
* craters are depressions in the obstacle zone,
* nothing is ever placed in the start zone, and the generator refuses to run if
  a zone marked ``obstacles_allowed: false`` would receive an obstacle.

The layout is written to the ground-truth file for tests and dataset labels
only.  It is never fed to the autonomy stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .config import ArenaConfig, ConfigError, Rect
from .seeding import rng_python
from .terrain import Crater, TerrainField

__all__ = ["Boulder", "ObstacleLayout", "place_obstacles"]


@dataclass(frozen=True)
class Boulder:
    """A static boulder: a sphere partially buried in the sand."""

    boulder_id: str
    x: float
    y: float
    diameter: float
    exposed_height: float
    surface_z: float
    zone: str

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def center_z(self) -> float:
        """Centre height that exposes exactly :attr:`exposed_height` above the sand."""
        return self.surface_z + self.exposed_height - self.radius

    @property
    def top_z(self) -> float:
        return self.surface_z + self.exposed_height

    @property
    def buried_depth(self) -> float:
        return max(0.0, self.diameter - self.exposed_height)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.boulder_id,
            "shape": "sphere",
            "zone": self.zone,
            "center": [round(self.x, 6), round(self.y, 6), round(self.center_z, 6)],
            "diameter": round(self.diameter, 6),
            "radius": round(self.radius, 6),
            "exposed_height": round(self.exposed_height, 6),
            "buried_depth": round(self.buried_depth, 6),
            "surface_z": round(self.surface_z, 6),
            "top_z": round(self.top_z, 6),
            "static": True,
        }


@dataclass(frozen=True)
class ObstacleLayout:
    """Result of a seeded placement draw."""

    boulders: List[Boulder]
    craters: List[Crater]
    attempts: Dict[str, int]
    seed: int

    def boulder_dicts(self) -> List[Dict[str, object]]:
        return [b.to_dict() for b in self.boulders]

    def crater_dicts(self) -> List[Dict[str, object]]:
        return [c.to_dict() for c in self.craters]

    def as_dict(self) -> Dict[str, object]:
        return {
            "boulders": self.boulder_dicts(),
            "craters": self.crater_dicts(),
        }


def _pick_zone(rng, names: Sequence[str], weights: Dict[str, float]) -> str:
    total = sum(max(0.0, float(weights.get(n, 1.0))) for n in names)
    if total <= 0.0:
        return names[0]
    pick = rng.random() * total
    acc = 0.0
    for name in names:
        acc += max(0.0, float(weights.get(name, 1.0)))
        if pick <= acc:
            return name
    return names[-1]


def _min_separation(a_radius: float, b_radius: float, spacing: float, buffer: float = 0.05) -> float:
    return max(spacing, a_radius + b_radius + buffer)


def _sample_counts(rng, minimum: int, maximum: int) -> int:
    return rng.randint(minimum, maximum)


def _distribute(rng, total: int, zones: Sequence[str], weights: Dict[str, float]) -> Dict[str, int]:
    counts = {z: 0 for z in zones}
    for _ in range(total):
        counts[_pick_zone(rng, zones, weights)] += 1
    return counts


def place_obstacles(cfg: ArenaConfig, seed: int, field: TerrainField) -> ObstacleLayout:
    """Draw a deterministic obstacle layout for ``seed``.

    :param cfg: validated arena configuration.
    :param seed: master run seed (see :mod:`cat_arena.seeding`).
    :param field: terrain field, used to sink boulders to the local sand level.
    :raises ConfigError: if a placement rule cannot be satisfied.
    """
    rng = rng_python(seed, "obstacles")
    bcfg, ccfg = cfg.boulders, cfg.craters
    attempts = {"boulders": 0, "craters": 0}

    for zone in set(bcfg.zones) | set(ccfg.zones):
        z = cfg.zone(zone)
        if not z.obstacles_allowed:
            raise ConfigError(
                f"refusing to generate: zone '{zone}' has obstacles_allowed=false "
                "but is listed as an obstacle placement zone"
            )

    placed_boulders: List[Boulder] = []
    placed_craters: List[Crater] = []

    # ---- craters first (they are carved into the terrain) ------------------
    crater_count = _sample_counts(rng, ccfg.count_min, ccfg.count_max)
    per_zone = _distribute(rng, crater_count, ccfg.zones, ccfg.zone_weights)
    crater_index = 0
    for zone_name in ccfg.zones:
        rect = cfg.zone(zone_name).rect
        for _ in range(per_zone.get(zone_name, 0)):
            radius = 0.5 * rng.uniform(ccfg.width_min, ccfg.width_max)
            depth = rng.uniform(ccfg.depth_min, ccfg.depth_max)
            if depth <= 0.0:
                raise ConfigError("craters.depth must describe a positive (downward) depth")
            for _try in range(20000):
                attempts["craters"] += 1
                x = rng.uniform(rect.x0 + ccfg.margin, rect.x1 - ccfg.margin)
                y = rng.uniform(rect.y0 + ccfg.margin, rect.y1 - ccfg.margin)
                ok = True
                for other in placed_craters:
                    if math.hypot(x - other.x, y - other.y) < _min_separation(
                        radius, other.radius, max(ccfg.spacing, bcfg.spacing)
                    ):
                        ok = False
                        break
                if ok:
                    for other_b in placed_boulders:
                        if math.hypot(x - other_b.x, y - other_b.y) < _min_separation(
                            radius, other_b.radius, max(ccfg.spacing, bcfg.spacing)
                        ):
                            ok = False
                            break
                if not ok:
                    continue
                placed_craters.append(
                    Crater(
                        crater_id=f"crater_{crater_index}",
                        x=x,
                        y=y,
                        radius=radius,
                        depth=depth,
                        zone=zone_name,
                    )
                )
                crater_index += 1
                break
            else:
                raise ConfigError(
                    f"could not place crater {crater_index} in zone '{zone_name}' after "
                    "20000 attempts - reduce craters.margin/spacing or enlarge the zone"
                )

    # ---- boulders (spheres partially buried in the sand) -------------------
    boulder_count = _sample_counts(rng, bcfg.count_min, bcfg.count_max)
    per_zone = _distribute(rng, boulder_count, bcfg.zones, bcfg.zone_weights)
    boulder_index = 0
    for zone_name in bcfg.zones:
        rect = cfg.zone(zone_name).rect
        for _ in range(per_zone.get(zone_name, 0)):
            diameter = rng.uniform(bcfg.diameter_min, bcfg.diameter_max)
            radius = 0.5 * diameter
            exposed = rng.uniform(bcfg.exposed_min, bcfg.exposed_max) * diameter
            for _try in range(20000):
                attempts["boulders"] += 1
                x = rng.uniform(rect.x0 + bcfg.margin, rect.x1 - bcfg.margin)
                y = rng.uniform(rect.y0 + bcfg.margin, rect.y1 - bcfg.margin)
                ok = True
                for other in placed_boulders:
                    if math.hypot(x - other.x, y - other.y) < _min_separation(
                        radius, other.radius, bcfg.spacing
                    ):
                        ok = False
                        break
                if ok:
                    for other_c in placed_craters:
                        if math.hypot(x - other_c.x, y - other_c.y) < _min_separation(
                            radius, other_c.radius, max(ccfg.spacing, bcfg.spacing)
                        ):
                            ok = False
                            break
                if not ok:
                    continue
                placed_boulders.append(
                    Boulder(
                        boulder_id=f"boulder_{boulder_index}",
                        x=x,
                        y=y,
                        diameter=diameter,
                        exposed_height=exposed,
                        surface_z=field.height(x, y),
                        zone=zone_name,
                    )
                )
                boulder_index += 1
                break
            else:
                raise ConfigError(
                    f"could not place boulder {boulder_index} in zone '{zone_name}' after "
                    "20000 attempts - reduce boulders.margin/spacing or enlarge the zone"
                )

    if len(placed_boulders) < 3 or len(placed_craters) < 3:
        raise ConfigError(
            f"competition rule violation: produced {len(placed_boulders)} boulders and "
            f"{len(placed_craters)} craters (at least 3 of each are required)"
        )
    return ObstacleLayout(
        boulders=placed_boulders, craters=placed_craters, attempts=attempts, seed=seed
    )


def obstacle_rects(boulders: Sequence[Boulder], craters: Sequence[Crater]) -> List[Rect]:
    """Axis-aligned footprints of all obstacles (used by layout sanity checks)."""
    rects: List[Rect] = []
    for b in boulders:
        r = b.radius
        rects.append(Rect(b.x - r, b.y - r, b.x + r, b.y + r))
    for c in craters:
        rects.append(Rect(c.x - c.radius, c.y - c.radius, c.x + c.radius, c.y + c.radius))
    return rects


def obstacles_in_rect(boulders: Sequence[Boulder], craters: Sequence[Crater], rect: Rect) -> Tuple[List[Boulder], List[Crater]]:
    """All obstacles whose footprint overlaps ``rect``."""
    b_hits = [b for b in boulders if rect.intersects(Rect(b.x - b.radius, b.y - b.radius, b.x + b.radius, b.y + b.radius))]
    c_hits = [c for c in craters if rect.intersects(Rect(c.x - c.radius, c.y - c.radius, c.x + c.radius, c.y + c.radius))]
    return b_hits, c_hits
