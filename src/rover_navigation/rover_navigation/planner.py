"""Path search: A* on the costmap, then string-pulling.

A plain grid A* is enough here because the costmap already contains the robot's
size: every lethal cell is a place the rover's *centre* may not be, so the
search can treat the rover as a point.  Three details make it work in practice:

* the search minimises a *cost* grid, not a binary grid, so a route that grazes
  a boulder is more expensive than one that swings wide - the paths look like
  driving, not like the walls of a maze;
* diagonal moves are refused when they would cut a corner (both orthogonal
  neighbours must be free), which stops the path from clipping the corner of an
  inflated obstacle;
* the raw cell chain is then string-pulled against the same costmap, so what
  comes out is a handful of long straight legs rather than a staircase, and it
  is resampled at a fixed spacing for the controller's lookahead.

The search is bounded by ``planner.max_expansions`` so a pathological map can
never hang the node: it fails the goal instead, and the caller re-plans.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .config import NavigationConfig
from .costmap import Costmap

__all__ = ["PlanResult", "Planner"]

Point = Tuple[float, float]


@dataclass
class PlanResult:
    """The outcome of one search, including why it failed."""

    success: bool
    waypoints: List[Point] = field(default_factory=list)
    message: str = ""
    expanded: int = 0
    length_m: float = 0.0
    start: Point = (0.0, 0.0)
    goal: Point = (0.0, 0.0)
    start_snapped: bool = False
    goal_snapped: bool = False
    straight: bool = False

    @property
    def legs(self) -> int:
        """Number of polyline *segments* (waypoints - 1).

        Note that the waypoints returned by :meth:`Planner.plan` have been
        resampled for the controller's lookahead, so this is not a count of the
        planner's straight legs - a long straight run is dozens of short
        segments by the time the controller sees it.
        """
        return max(0, len(self.waypoints) - 1)


class Planner:
    """A* over a :class:`~rover_navigation.costmap.Costmap`."""

    def __init__(self, config: NavigationConfig) -> None:
        self.config = config

    # -- public API --------------------------------------------------------- #
    def plan(self, costmap: Costmap, start: Point, goal: Point) -> PlanResult:
        result = PlanResult(success=False, start=start, goal=goal)
        resolution = costmap.resolution

        # A straight shot needs no search at all and avoids grid artefacts on
        # the common case of "drive down the corridor".
        if costmap.segment_clear(start[0], start[1], goal[0], goal[1]):
            result.success = True
            result.waypoints = self._densify([start, goal], resolution)
            result.length_m = math.hypot(goal[0] - start[0], goal[1] - start[1])
            result.straight = True
            return result

        start_cell = self._resolve(costmap, start)
        goal_cell = self._resolve(costmap, goal)
        if start_cell is None:
            result.message = "start is inside an obstacle and no free cell is close enough"
            return result
        if goal_cell is None:
            result.message = "goal is inside an obstacle and no free cell is close enough"
            return result
        result.start_snapped = start_cell != costmap.world_to_cell(start[0], start[1])
        result.goal_snapped = goal_cell != costmap.world_to_cell(goal[0], goal[1])

        chain = self._search(costmap, start_cell, goal_cell, result)
        if chain is None:
            if not result.message:
                result.message = "no path exists on the current map"
            return result

        points = [costmap.cell_center(i, j) for (i, j) in chain]
        points[0] = start if not result.start_snapped else costmap.cell_center(*start_cell)
        points[-1] = goal if not result.goal_snapped else costmap.cell_center(*goal_cell)
        if self.config.planner.smoothing:
            points = self._string_pull(costmap, points)
        points = self._densify(points, resolution)
        result.success = True
        result.waypoints = points
        result.length_m = self._length(points)
        return result

    # -- helpers ------------------------------------------------------------ #
    def _resolve(self, costmap: Costmap, point: Point) -> Optional[Tuple[int, int]]:
        i, j = costmap.world_to_cell(point[0], point[1])
        if costmap.in_bounds(i, j) and not costmap.lethal[j, i]:
            return (i, j)
        snapped = costmap.nearest_free(point[0], point[1], self.config.planner.snap_radius_m)
        if snapped is None:
            return None
        return costmap.world_to_cell(snapped[0], snapped[1])

    def _search(
        self,
        costmap: Costmap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        result: PlanResult,
    ) -> Optional[List[Tuple[int, int]]]:
        cfg = self.config.planner
        diagonal = cfg.allow_diagonal
        moves: List[Tuple[int, int, float]] = []
        for di, dj in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            moves.append((di, dj, 1.0))
        if diagonal:
            root = math.sqrt(2.0)
            for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                moves.append((di, dj, root))

        start_cost = 1.0
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (self._heuristic(start, goal, costmap), 0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: start_cost}
        closed = set()
        expansions = 0

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            expansions += 1
            if expansions > cfg.max_expansions:
                result.expanded = expansions
                result.message = f"search abandoned after {expansions} expansions"
                return None
            if current == goal:
                break
            current_cost = g_score[current]
            ci, cj = current
            for di, dj, step in moves:
                ni, nj = ci + di, cj + dj
                if not costmap.in_bounds(ni, nj):
                    continue
                if costmap.lethal[nj, ni]:
                    continue
                if di != 0 and dj != 0:
                    # Refuse to cut a corner: the rover is a big disc, and an
                    # inflated obstacle's corner is still an obstacle.
                    if costmap.lethal[cj, ni] or costmap.lethal[nj, ci]:
                        continue
                neighbour = (ni, nj)
                if neighbour in closed:
                    continue
                tentative = current_cost + step * costmap.cost_at(ni, nj)
                if tentative < g_score.get(neighbour, float("inf")):
                    g_score[neighbour] = tentative
                    came_from[neighbour] = current
                    priority = tentative + self._heuristic(neighbour, goal, costmap)
                    heapq.heappush(open_heap, (priority, expansions, neighbour))

        result.expanded = expansions
        if goal not in came_from and goal != start:
            return None
        chain = [goal]
        while chain[-1] != start:
            chain.append(came_from[chain[-1]])
        chain.reverse()
        return chain

    def _heuristic(self, cell: Tuple[int, int], goal: Tuple[int, int], costmap: Costmap) -> float:
        # Octile distance is admissible for an 8-connected grid with unit costs;
        # inflating it by heuristic_weight makes the search greedy but can make
        # the path slightly longer, which is why the default weight is 1.0.
        di = abs(cell[0] - goal[0])
        dj = abs(cell[1] - goal[1])
        octile = (di + dj) + (math.sqrt(2.0) - 2.0) * min(di, dj)
        return octile * costmap.resolution * self.config.planner.heuristic_weight

    def _string_pull(self, costmap: Costmap, points: Sequence[Point]) -> List[Point]:
        """Greedy shortcutting: keep the farthest reachable point, repeat."""
        if len(points) <= 2:
            return list(points)
        out = [points[0]]
        anchor = 0
        probe = 1
        while probe < len(points):
            if costmap.segment_clear(points[anchor][0], points[anchor][1], points[probe][0], points[probe][1]):
                probe += 1
                continue
            # `probe` is unreachable; keep the last reachable one.
            if probe - 1 == anchor:
                # Even the next point is unreachable (should not happen on a
                # valid chain): take it anyway so the path keeps making progress.
                anchor = probe
                out.append(points[anchor])
                probe = anchor + 1
                continue
            anchor = probe - 1
            out.append(points[anchor])
            probe = anchor + 1
        if out[-1] != points[-1]:
            out.append(points[-1])
        return self._thin(out)

    def _thin(self, points: Sequence[Point]) -> List[Point]:
        """Drop waypoints closer together than the configured spacing."""
        spacing = self.config.planner.waypoint_spacing_m
        out = [points[0]]
        for point in points[1:-1]:
            if math.hypot(point[0] - out[-1][0], point[1] - out[-1][1]) >= spacing:
                out.append(point)
        out.append(points[-1])
        return out

    def _densify(self, points: Sequence[Point], resolution: float) -> List[Point]:
        """Resample the polyline so no two waypoints are further apart than 2 cells.

        The controller does its own arc-length lookahead, so it wants a smooth
        sequence rather than long legs; densifying here keeps that work simple.
        """
        step = max(resolution, 2.0 * resolution)
        out: List[Point] = [points[0]]
        for index in range(1, len(points)):
            ax, ay = points[index - 1]
            bx, by = points[index]
            length = math.hypot(bx - ax, by - ay)
            if length <= 1e-9:
                continue
            pieces = max(1, int(math.ceil(length / step)))
            for piece in range(1, pieces + 1):
                fraction = piece / pieces
                out.append((ax + (bx - ax) * fraction, ay + (by - ay) * fraction))
        return out

    @staticmethod
    def _length(points: Sequence[Point]) -> float:
        total = 0.0
        for index in range(1, len(points)):
            total += math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
        return total
