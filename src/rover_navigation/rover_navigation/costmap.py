"""The obstacle map module 5 plans on: memory + rasterisation.

Two separate ideas live here, and separating them matters:

:class:`ObstacleMemory`
    Module 4 publishes what its *camera* can see right now, and its tracker
    decays a track when the obstacle leaves the field of view.  That is correct
    for perception and useless for planning - a boulder seen once and then
    driven past would vanish from the map while the rover is on the other side
    of it.  The memory therefore accumulates observations into a map that
    outlives any single look, keyed by proximity rather than by module 4's track
    id (the ids are per-run counters, so they collide after a restart).

:class:`Costmap`
    A rasterised, inflated view of that memory: a boolean ``lethal`` grid of
    cells the rover's *centre* may not occupy, and a floating ``cost`` grid that
    adds a soft penalty near obstacles so the planner prefers to keep its
    distance instead of grazing them.

The one thing the rasterisation must get right is that the robot is *big*.  A
1.5 x 0.75 m rover that can rotate in place sweeps a 0.84 m circle, so an
obstacle's footprint is grown by that radius (plus a configured clearance)
before a single cell is marked.  Planning then treats the rover as a point,
which is what makes plain A* sufficient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import NavigationConfig, Rect

__all__ = [
    "KIND_UNKNOWN",
    "KIND_BOULDER",
    "KIND_CRATER",
    "KIND_NAMES",
    "ObstacleRecord",
    "ObstacleMemory",
    "Costmap",
]

#: Wire values, identical to ``rover_perception_msgs/Obstacle.KIND_*``.
KIND_UNKNOWN = 0
KIND_BOULDER = 1
KIND_CRATER = 2

KIND_NAMES: Dict[int, str] = {KIND_UNKNOWN: "unknown", KIND_BOULDER: "boulder", KIND_CRATER: "crater"}


@dataclass
class ObstacleRecord:
    """One obstacle in the map frame, as module 5 remembers it."""

    identifier: str
    kind: int
    x: float
    y: float
    radius: float
    confidence: float = 0.0
    observations: int = 1
    first_seen: float = 0.0
    last_seen: float = 0.0
    conservative: bool = False

    @property
    def kind_name(self) -> str:
        return KIND_NAMES.get(self.kind, "unknown")

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)

    def age(self, now: float) -> float:
        return max(0.0, now - self.last_seen)

    def copy(self) -> "ObstacleRecord":
        return replace(self)


class ObstacleMemory:
    """Accumulates module 4's obstacle observations into a map that persists.

    Association is by distance, not by id: an observation joins the nearest
    remembered obstacle within ``costmap.memory_associate_m``, and otherwise
    starts a new one.  When two observations meet, the *conservative* value
    wins - the kind escalates to crater, the radius grows to the larger of the
    two, and the confidence keeps the better evidence.  That is the same
    one-way escalation module 4's tracker applies, deliberately repeated here
    because the map must never become *less* afraid than perception was.
    """

    def __init__(self, config: NavigationConfig) -> None:
        self._cfg = config.costmap
        self._records: List[ObstacleRecord] = []
        self._accepted = 0
        self._rejected = 0
        self._merged = 0

    # -- introspection ------------------------------------------------------ #
    @property
    def records(self) -> List[ObstacleRecord]:
        return list(self._records)

    @property
    def accepted(self) -> int:
        """Observations strong enough to be remembered."""
        return self._accepted

    @property
    def rejected(self) -> int:
        """Observations dropped for being too weak to memorise."""
        return self._rejected

    @property
    def merged(self) -> int:
        """Observations folded into an obstacle already in the map."""
        return self._merged

    def __len__(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()

    # -- ingestion ---------------------------------------------------------- #
    def update(self, observations: Iterable[ObstacleRecord], now: float) -> None:
        """Fold one ``/perception/obstacles`` frame into the map, then prune."""
        for observation in observations:
            if observation.radius <= 0.0:
                self._rejected += 1
                continue
            if observation.confidence < self._cfg.min_confidence:
                self._rejected += 1
                continue
            if observation.observations < self._cfg.min_observations:
                self._rejected += 1
                continue
            index = self._nearest(observation.x, observation.y)
            if index is None:
                self._records.append(
                    ObstacleRecord(
                        identifier=observation.identifier,
                        kind=observation.kind,
                        x=observation.x,
                        y=observation.y,
                        radius=observation.radius,
                        confidence=observation.confidence,
                        observations=observation.observations,
                        first_seen=now,
                        last_seen=now,
                        conservative=observation.conservative,
                    )
                )
            else:
                self._records[index] = self._merge(self._records[index], observation, now)
                self._merged += 1
            self._accepted += 1
        self._prune(now)

    def _nearest(self, x: float, y: float) -> Optional[int]:
        best: Optional[int] = None
        best_distance = self._cfg.memory_associate_m
        for index, record in enumerate(self._records):
            distance = record.distance_to(x, y)
            if distance <= best_distance:
                best_distance = distance
                best = index
        return best

    @staticmethod
    def _merge(stored: ObstacleRecord, incoming: ObstacleRecord, now: float) -> ObstacleRecord:
        # Position: blend rather than snap.  Module 4 already smooths a track,
        # so the blend is only there to stop one bad frame from teleporting an
        # obstacle that the planner has already routed around.
        gain = 0.5
        merged = stored.copy()
        merged.x = stored.x + gain * (incoming.x - stored.x)
        merged.y = stored.y + gain * (incoming.y - stored.y)
        # Kind escalates one way only; a crater outranks a boulder forever.
        if incoming.kind == KIND_CRATER or stored.kind == KIND_CRATER:
            merged.kind = KIND_CRATER
        elif incoming.kind != KIND_UNKNOWN:
            merged.kind = incoming.kind
        merged.radius = max(stored.radius, incoming.radius)
        merged.confidence = max(stored.confidence, incoming.confidence)
        merged.observations = max(stored.observations, incoming.observations)
        merged.conservative = stored.conservative or incoming.conservative
        merged.last_seen = now
        return merged

    def _prune(self, now: float) -> None:
        if self._cfg.obstacle_memory_s > 0.0:
            self._records = [
                record for record in self._records if record.age(now) <= self._cfg.obstacle_memory_s
            ]
        limit = max(1, self._cfg.max_obstacles)
        if len(self._records) > limit:
            # Keep the most confident, most recently seen obstacles.
            self._records.sort(key=lambda record: (record.confidence, record.last_seen), reverse=True)
            del self._records[limit:]
        self._records.sort(key=lambda record: record.identifier)

    def stale(self, now: float) -> List[ObstacleRecord]:
        """Remembered obstacles module 4 has not re-confirmed recently."""
        return [record for record in self._records if record.age(now) > self._cfg.stale_obstacle_s]


class Costmap:
    """A rasterised, inflated occupancy grid of the arena.

    The grid covers the *whole* arena at ``costmap.resolution_m``, indexed
    ``[j, i]`` with ``i`` along ``+x`` and ``j`` along ``+y``; cell ``(i, j)``
    has its centre at ``((i + 0.5) * res, (j + 0.5) * res)``.

    ``lethal`` is the planner's hard constraint: a cell is lethal when the
    rover's swept circle would touch a wall, a remembered obstacle or a keep-out
    rectangle.  ``cost`` is what the planner minimises; it is 1 everywhere free,
    rises smoothly in the ``costmap.inflation_m`` band around every lethal cell,
    and is unused (set to ``COST_LETHAL``) inside the lethal region.
    """

    #: Cost of a lethal cell; the planner refuses to expand into one.
    COST_LETHAL = float("inf")

    def __init__(
        self,
        config: NavigationConfig,
        records: Sequence[ObstacleRecord],
        keep_out_rects: Sequence[Rect] = (),
    ) -> None:
        self.config = config
        self.resolution = config.costmap.resolution_m
        self.free_rect = config.free_rect
        self.nx = max(1, int(round(config.arena.length_x / self.resolution)))
        self.ny = max(1, int(round(config.arena.width_y / self.resolution)))
        self.records = list(records)

        # Cell-centre coordinate planes, computed once and reused for every
        # obstacle: 9 x 5 m at 5 cm is 180 x 100, so a full-grid distance field
        # per obstacle is cheaper than writing a per-obstacle bounding box.
        xs = (np.arange(self.nx, dtype=float) + 0.5) * self.resolution
        ys = (np.arange(self.ny, dtype=float) + 0.5) * self.resolution
        self._grid_x, self._grid_y = np.meshgrid(xs, ys)

        self.lethal = np.zeros((self.ny, self.nx), dtype=bool)
        self.cost = np.ones((self.ny, self.nx), dtype=float)
        # Where the danger came from, for the OccupancyGrid preview and tests.
        self.source = np.zeros((self.ny, self.nx), dtype=np.uint8)

        self.obstacle_count = 0
        self.keep_out_count = 0
        self._build(records, keep_out_rects)

    # -- construction ------------------------------------------------------- #
    def _build(self, records: Sequence[ObstacleRecord], keep_out_rects: Sequence[Rect]) -> None:
        self._apply_walls()
        for record in records:
            self._apply_disc(record.x, record.y, self.lethal_radius(record.radius, record.kind), 1)
            self.obstacle_count += 1
        for rect in keep_out_rects:
            self._apply_rect(rect, self.config.robot.obstacle_inflation_m, 2)
            self.keep_out_count += 1
        self._inflate()

    def lethal_radius(self, radius: float, kind: int) -> float:
        """How big a disc the rover's centre must stay out of, for one obstacle.

        A crater is grown by ``costmap.crater_radius_scale`` on top of the
        boulder scaling because module 4 reports craters with a conservative
        bias and because the cost of the mistake is asymmetric: detouring around
        sand that turns out to be flat costs seconds, driving into a hole ends
        the run.
        """
        scale = (
            self.config.costmap.crater_radius_scale
            if kind == KIND_CRATER
            else self.config.costmap.boulder_radius_scale
        )
        return max(0.0, radius) * scale + self.config.robot.obstacle_inflation_m

    def _apply_walls(self) -> None:
        rect = self.free_rect
        inside = (
            (self._grid_x >= rect.x0)
            & (self._grid_x <= rect.x1)
            & (self._grid_y >= rect.y0)
            & (self._grid_y <= rect.y1)
        )
        self.lethal |= ~inside

    def _apply_disc(self, x: float, y: float, radius: float, source: int) -> None:
        if radius <= 0.0:
            return
        distance = np.hypot(self._grid_x - x, self._grid_y - y)
        bite = distance <= radius
        self.lethal |= bite
        self.source[bite] = source

    def _apply_rect(self, rect: Rect, inflation: float, source: int) -> None:
        grown = rect.inset(-inflation)
        bite = (
            (self._grid_x >= grown.x0)
            & (self._grid_x <= grown.x1)
            & (self._grid_y >= grown.y0)
            & (self._grid_y <= grown.y1)
        )
        self.lethal |= bite
        self.source[bite] = source

    def _inflate(self) -> None:
        """Add the soft-cost band around everything lethal."""
        band = self.config.costmap.inflation_m
        soft = self.config.costmap.soft_cost
        if band <= 0.0 or soft <= 0.0:
            self.cost = np.where(self.lethal, self.COST_LETHAL, 1.0)
            return
        # A single distance transform would be nicer, but numpy has none; a
        # 3x3 dilation repeated ceil(band / resolution) times is exact enough at
        # this resolution and stays dependency-free.
        steps = int(math.ceil(band / self.resolution))
        halo = self.lethal.copy()
        for _ in range(steps):
            shifted = halo.copy()
            shifted[1:, :] |= halo[:-1, :]
            shifted[:-1, :] |= halo[1:, :]
            shifted[:, 1:] |= halo[:, :-1]
            shifted[:, :-1] |= halo[:, 1:]
            halo = shifted
        penalty = np.where(halo & ~self.lethal, soft, 0.0)
        self.cost = np.where(self.lethal, self.COST_LETHAL, 1.0 + penalty)

    # -- queries ------------------------------------------------------------ #
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Cell index ``(i, j)`` containing ``(x, y)``; may be outside the grid."""
        return (int(math.floor(x / self.resolution)), int(math.floor(y / self.resolution)))

    def cell_center(self, i: int, j: int) -> Tuple[float, float]:
        return ((i + 0.5) * self.resolution, (j + 0.5) * self.resolution)

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.nx and 0 <= j < self.ny

    def is_lethal_cell(self, i: int, j: int) -> bool:
        if not self.in_bounds(i, j):
            return True
        return bool(self.lethal[j, i])

    def is_free(self, x: float, y: float) -> bool:
        i, j = self.world_to_cell(x, y)
        if not self.in_bounds(i, j):
            return False
        return not bool(self.lethal[j, i])

    def cost_at(self, i: int, j: int) -> float:
        if not self.in_bounds(i, j):
            return self.COST_LETHAL
        return float(self.cost[j, i])

    def segment_clear(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """True when the straight segment stays wholly in free cells.

        Sampled at half a cell, which cannot step over a cell at this
        resolution.
        """
        length = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, int(math.ceil(length / (0.5 * self.resolution))))
        for step in range(steps + 1):
            fraction = step / steps
            x = x0 + (x1 - x0) * fraction
            y = y0 + (y1 - y0) * fraction
            if not self.is_free(x, y):
                return False
        return True

    def nearest_free(self, x: float, y: float, max_radius: float) -> Optional[Tuple[float, float]]:
        """The closest free cell centre to ``(x, y)`` within ``max_radius``.

        Used to rescue a goal that perception has just discovered sits inside an
        obstacle - refusing the goal outright would leave the caller stuck, and
        the caller is the mission sequencer, which has nowhere else to go.

        The radius is a *Euclidean* bound on the answer: the search walks the
        square rings around the point and skips any cell whose centre is further
        away than ``max_radius``, because a Chebyshev ring reaches out to
        ``sqrt(2) * max_radius`` at its corners and a caller asking for half a
        metre must not be handed 0.7 m.
        """
        i, j = self.world_to_cell(x, y)
        if self.is_free(x, y):
            return (x, y)
        if max_radius <= 0.0:
            return None
        max_ring = int(math.ceil(max_radius / self.resolution))
        best: Optional[Tuple[float, float]] = None
        best_distance = float("inf")
        for ring in range(1, max_ring + 1):
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    if max(abs(di), abs(dj)) != ring:
                        continue
                    ci, cj = i + di, j + dj
                    if not self.in_bounds(ci, cj) or self.lethal[cj, ci]:
                        continue
                    cx, cy = self.cell_center(ci, cj)
                    distance = math.hypot(cx - x, cy - y)
                    if distance <= max_radius and distance < best_distance:
                        best_distance = distance
                        best = (cx, cy)
        return best

    def free_cells(self) -> int:
        return int(np.count_nonzero(~self.lethal))

    def lethal_fraction(self) -> float:
        return float(np.count_nonzero(self.lethal)) / float(self.nx * self.ny)

    def to_occupancy_grid(self) -> Tuple[List[int], Tuple[float, float]]:
        """``nav_msgs/OccupancyGrid`` payload: row-major int8 data and the origin.

        The grid is row-major with row 0 at the lowest y, which is exactly the
        ROS convention, so no flip is needed.
        """
        data = np.where(self.lethal, 100, 0).astype(np.int8)
        return (data.reshape(-1).tolist(), (0.0, 0.0))
