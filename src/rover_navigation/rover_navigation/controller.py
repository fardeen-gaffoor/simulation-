"""Path following: pure pursuit with an in-place turn phase.

The rover is a skid-steer, so "point the rover at the path, then drive" is a
legal and cheap primitive: it can rotate on the spot without translating.  The
controller exploits that:

1. if the heading error to the lookahead point is larger than
   ``controller.in_place_turn_rad``, it stops the wheels and rotates in place;
2. otherwise it drives with the standard pure-pursuit curvature
   ``kappa = 2 sin(alpha) / Ld``, with the speed scaled down by the heading
   error and again by the distance still to run, so it arrives rather than
   overshoots;
3. once inside ``robot.goal_tolerance_m`` it stops translating and rotates to
   the goal's heading, then reports arrival.

Everything is clamped to ``robot.max_speed_mps`` and
``robot.max_yaw_rate_radps``.  There is deliberately no reverse: the cameras
only look forward (``cam_nav`` has a 100 degree horizontal field of view mounted
18 degrees down), so reversing would drive blind, and no scoring objective in
the competition needs it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .config import NavigationConfig
from .geometry import Pose3, angle_difference

__all__ = ["ControlCommand", "PurePursuitController"]

Point = Tuple[float, float]


@dataclass
class ControlCommand:
    """One tick of the controller, before the safety gate sees it."""

    linear: float
    angular: float
    state: str
    distance_remaining_m: float
    yaw_error_rad: float
    heading_error_rad: float
    lookahead: Point
    goal_reached: bool

    @property
    def moving(self) -> bool:
        return abs(self.linear) > 1e-6 or abs(self.angular) > 1e-6


class PurePursuitController:
    """Stateless pure pursuit: the same pose and path always give the same tick."""

    def __init__(self, config: NavigationConfig) -> None:
        self.config = config

    # -- public API --------------------------------------------------------- #
    def compute(
        self,
        pose: Pose3,
        path: Sequence[Point],
        goal_yaw: float,
        max_speed: float = 0.0,
        position_tolerance_m: float = 0.0,
        yaw_tolerance_rad: float = 0.0,
        force_drive: bool = False,
    ) -> ControlCommand:
        """One control tick.

        ``force_drive`` skips the in-place turn phase.  The callers that set it
        are the ones that have already asked :func:`rover_navigation.footprint`
        whether the rover has room to spin and been told it has not: arcing along
        a path the planner certified is collision-free is then the better of two
        imperfect options, and the alternative - holding forever - ends the run.
        """
        cfg = self.config.controller
        robot = self.config.robot
        speed_limit = max_speed if max_speed > 0.0 else robot.max_speed_mps
        speed_limit = min(speed_limit, robot.max_speed_mps)
        pos_tol = position_tolerance_m if position_tolerance_m > 0.0 else robot.goal_tolerance_m
        yaw_tol = yaw_tolerance_rad if yaw_tolerance_rad > 0.0 else robot.goal_tolerance_yaw_rad

        points: List[Point] = list(path) if path else []
        if not points:
            points = [(pose.x, pose.y)]

        goal = points[-1]
        distance_to_goal = math.hypot(goal[0] - pose.x, goal[1] - pose.y)
        yaw_error_goal = angle_difference(goal_yaw, pose.yaw)

        # -- arrival ------------------------------------------------------- #
        if distance_to_goal <= pos_tol:
            if abs(yaw_error_goal) <= yaw_tol or force_drive:
                return ControlCommand(0.0, 0.0, "arrived", distance_to_goal, yaw_error_goal, 0.0, goal, True)
            # Final heading: rotate on the spot, no translation.
            yaw_rate = math.copysign(min(cfg.in_place_yaw_rate_radps, robot.max_yaw_rate_radps), yaw_error_goal)
            yaw_rate = self._shape_turn(yaw_error_goal, yaw_rate)
            return ControlCommand(0.0, yaw_rate, "turning", distance_to_goal, yaw_error_goal, yaw_error_goal, goal, False)

        # -- projection onto the path -------------------------------------- #
        arc_remaining, lookahead_default = self._project(points, pose.x, pose.y)
        if arc_remaining <= 0.0:
            arc_remaining = distance_to_goal

        lookahead_distance = self._lookahead(speed_limit, arc_remaining)
        if force_drive:
            # Keep the arc tight: the plan was certified for the body corridor,
            # and a long lookahead would cut the corner out of it.
            lookahead_distance = min(lookahead_distance, cfg.min_lookahead_m)
        target = self._point_at(points, self._project_arc(points, pose.x, pose.y) + lookahead_distance)
        if target is None:  # pragma: no cover - defensive
            target = goal

        heading = math.atan2(target[1] - pose.y, target[0] - pose.x)
        alpha = angle_difference(heading, pose.yaw)
        heading_error = angle_difference(heading, pose.yaw)

        # -- turn first when badly misaligned ------------------------------ #
        if abs(alpha) > cfg.in_place_turn_rad and not force_drive:
            yaw_rate = math.copysign(cfg.in_place_yaw_rate_radps, alpha)
            yaw_rate = min(yaw_rate, robot.max_yaw_rate_radps) if yaw_rate >= 0 else max(yaw_rate, -robot.max_yaw_rate_radps)
            yaw_rate = self._shape_turn(alpha, yaw_rate)
            return ControlCommand(0.0, yaw_rate, "turning", arc_remaining, yaw_error_goal, heading_error, target, False)

        # -- pure pursuit --------------------------------------------------- #
        curvature = 2.0 * math.sin(alpha) / max(lookahead_distance, 1e-3)
        # Speed: full when straight and far, tapered by the heading error and
        # then by the distance still to run.
        speed = speed_limit * max(0.0, 1.0 - 0.6 * abs(alpha) / math.pi)
        if arc_remaining < cfg.slowdown_distance_m:
            speed *= max(0.25, arc_remaining / cfg.slowdown_distance_m)
        speed = max(cfg.min_speed_mps, min(speed, speed_limit))
        yaw_rate = curvature * speed
        yaw_rate = max(-robot.max_yaw_rate_radps, min(robot.max_yaw_rate_radps, yaw_rate))
        if abs(alpha) > cfg.in_place_turn_rad * 0.5 and arc_remaining < cfg.slowdown_distance_m and not force_drive:
            # Close to the goal and still crooked: turn rather than arc.
            return ControlCommand(0.0, math.copysign(cfg.in_place_yaw_rate_radps, alpha), "turning",
                                  arc_remaining, yaw_error_goal, heading_error, target, False)
        return ControlCommand(speed, yaw_rate, "driving", arc_remaining, yaw_error_goal, heading_error, target, False)

    def stop(self) -> ControlCommand:
        """A zero command with no state change (used while holding)."""
        return ControlCommand(0.0, 0.0, "idle", 0.0, 0.0, 0.0, (0.0, 0.0), False)

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _shape_turn(error: float, yaw_rate: float) -> float:
        """Ease the last few degrees so the rover does not oscillate at the goal.

        The taper is proportional and reaches zero with the error, so the rover
        converges onto the heading instead of ringing around it, and a heading
        that is already correct produces no command at all.
        """
        if abs(error) >= 0.25:
            return yaw_rate
        return yaw_rate * min(1.0, abs(error) / 0.25)

    def _lookahead(self, speed: float, arc_remaining: float) -> float:
        cfg = self.config.controller
        wanted = cfg.lookahead_m + cfg.lookahead_speed_gain * speed
        # Never look past the goal, or the pursuit point sails past it.
        wanted = min(wanted, max(cfg.min_lookahead_m, arc_remaining))
        return max(cfg.min_lookahead_m, min(cfg.max_lookahead_m, wanted))

    @staticmethod
    def _project(points: Sequence[Point], x: float, y: float) -> Tuple[float, Point]:
        """Return ``(arc length from the projection point to the end, closest point)``."""
        total = 0.0
        for index in range(1, len(points)):
            total += math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
        arc = PurePursuitController._project_arc(points, x, y)
        closest = PurePursuitController._point_at(points, arc)
        return (max(0.0, total - arc), closest or points[-1])

    @staticmethod
    def _project_arc(points: Sequence[Point], x: float, y: float) -> float:
        """Arc length along the polyline of the point closest to ``(x, y)``."""
        best_arc = 0.0
        best_distance = float("inf")
        arc = 0.0
        for index in range(1, len(points)):
            ax, ay = points[index - 1]
            bx, by = points[index]
            dx, dy = bx - ax, by - ay
            length_squared = dx * dx + dy * dy
            if length_squared <= 1e-12:
                continue
            t = ((x - ax) * dx + (y - ay) * dy) / length_squared
            t = max(0.0, min(1.0, t))
            px, py = ax + t * dx, ay + t * dy
            distance = math.hypot(x - px, y - py)
            if distance < best_distance:
                best_distance = distance
                best_arc = arc + t * math.sqrt(length_squared)
            arc += math.sqrt(length_squared)
        return best_arc

    @staticmethod
    def _point_at(points: Sequence[Point], arc: float) -> Point:
        """The point at arc length ``arc`` along the polyline (clamped)."""
        if not points:
            return (0.0, 0.0)
        if len(points) == 1 or arc <= 0.0:
            return points[0]
        remaining = arc
        for index in range(1, len(points)):
            ax, ay = points[index - 1]
            bx, by = points[index]
            length = math.hypot(bx - ax, by - ay)
            if length <= 1e-12:
                continue
            if remaining <= length:
                t = remaining / length
                return (ax + (bx - ax) * t, ay + (by - ay) * t)
            remaining -= length
        return points[-1]
