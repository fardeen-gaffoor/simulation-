"""The safety layer: the one place that decides whether a command may move sand.

Module 2's ``stub_topics`` already latches ``/estop`` and zeroes ``/cmd_vel``
while it is set; this module is the second half of that contract, and the
authoritative one, because it is the only node that *publishes* ``/cmd_vel``.
Having a single writer is the whole point: "why is the rover stopped" can only
have one answer, and it is in :class:`GateDecision.reason`.

Four independent reasons can stop the rover, checked in this order:

1. **e-stop.**  The hardware chain has the last word.
2. **stale command.**  If the navigator stops talking (it crashed, it is
   re-planning, the link dropped), the wheels stop.  Gazebo's diff-drive holds
   the last velocity command forever, so silence must mean *stop*, never
   "keep going at the last speed you heard".
3. **localisation.**  Every obstacle is in the map frame, and the map frame is
   only meaningful while module 4's AprilTags anchor it.  ``sigma_*`` grow with
   dead reckoning and collapse on a fix, so the gate scales the speed down
   smoothly and stops before the map can be badly wrong.  A rover that does not
   know where it is must not drive towards a hole it also does not know the
   position of.
4. **reactive brake.**  Perception can see a boulder closer than the plan
   assumed - most often just after the plan was made.  A cheap forward-bumper
   test stops the rover without waiting for a re-plan.

Only after all four does the gate clamp the command to the robot's limits and
rate-limit it, so the wheels never see a step change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .config import NavigationConfig
from .costmap import ObstacleRecord
from .footprint import rect_corners
from .geometry import Pose3

__all__ = ["LocalizationSnapshot", "GateDecision", "VelocityGate"]


@dataclass
class LocalizationSnapshot:
    """What module 4's ``LocalizationState`` currently says."""

    has_state: bool = False
    localized: bool = False
    sigma_xy_m: float = 0.0
    sigma_yaw_rad: float = 0.0
    time_since_fix_s: float = 0.0

    @property
    def summary(self) -> str:
        if not self.has_state:
            return "no localization state received"
        return (
            f"localized={self.localized} sigma_xy={self.sigma_xy_m:.3f} "
            f"sigma_yaw={self.sigma_yaw_rad:.3f} since_fix={self.time_since_fix_s:.1f}s"
        )


@dataclass
class GateDecision:
    """What the gate published on ``/cmd_vel``, and why."""

    allowed: bool
    linear: float
    angular: float
    reason: str = ""
    speed_scale: float = 1.0
    emergency: bool = False
    stale: bool = False
    estop: bool = False
    max_speed: float = 0.0
    max_yaw_rate: float = 0.0

    @property
    def stopped(self) -> bool:
        return abs(self.linear) <= 1e-9 and abs(self.angular) <= 1e-9


class VelocityGate:
    """Pure logic for one gate tick; the node owns the clock and the topics."""

    def __init__(self, config: NavigationConfig) -> None:
        self.config = config
        self._linear = 0.0
        self._angular = 0.0

    # -- pieces ------------------------------------------------------------- #
    def localization_scale(self, snapshot: LocalizationSnapshot) -> Tuple[float, bool, str]:
        """``(scale, allowed, reason)`` from the current anchoring quality."""
        cfg = self.config.safety
        if not snapshot.has_state:
            if cfg.require_localized:
                return (0.0, False, "no localization state received")
            return (1.0, True, "")
        if cfg.require_localized and not snapshot.localized:
            return (0.0, False, "map not anchored by any tag fix yet")
        if snapshot.time_since_fix_s > cfg.max_time_since_fix_s:
            return (
                0.0,
                False,
                f"no tag fix for {snapshot.time_since_fix_s:.1f}s "
                f"(limit {cfg.max_time_since_fix_s:.1f}s)",
            )
        scale_xy = self._ramp(snapshot.sigma_xy_m, cfg.sigma_xy_slow_m, cfg.sigma_xy_stop_m)
        scale_yaw = self._ramp(snapshot.sigma_yaw_rad, cfg.sigma_yaw_slow_rad, cfg.sigma_yaw_stop_rad)
        scale = min(scale_xy, scale_yaw)
        if scale <= 0.0:
            return (
                0.0,
                False,
                f"localization uncertainty too large (sigma_xy={snapshot.sigma_xy_m:.2f} m, "
                f"sigma_yaw={snapshot.sigma_yaw_rad:.2f} rad)",
            )
        reason = "" if scale >= 1.0 else f"localization uncertainty: speed scaled to {scale:.0%}"
        return (scale, True, reason)

    @staticmethod
    def _ramp(value: float, slow: float, stop: float) -> float:
        """1 below ``slow``, 0 at or above ``stop``, linear between."""
        if value <= slow:
            return 1.0
        if value >= stop:
            return 0.0
        return (stop - value) / (stop - slow)

    def emergency_brake_reason(
        self, pose: Pose3, records: Sequence[ObstacleRecord], forward: bool
    ) -> Optional[str]:
        """A reason to stop now, or ``None``.

        The test is made at the rover's *front bumper*, so an obstacle beside or
        behind the rover does not stall a legal manoeuvre, and it uses the
        obstacle's own radius, so the threshold is a real clearance and not a
        centre-to-centre distance.
        """
        if not forward:
            return None
        cfg = self.config.safety
        half_length = 0.5 * self.config.robot.length_m
        front_x = pose.x + math.cos(pose.yaw) * half_length
        front_y = pose.y + math.sin(pose.yaw) * half_length
        for record in records:
            clearance = math.hypot(record.x - front_x, record.y - front_y) - record.radius
            if clearance < cfg.emergency_stop_m:
                return (
                    f"{record.kind_name} '{record.identifier}' is {clearance:.2f} m ahead "
                    f"(limit {cfg.emergency_stop_m:.2f} m)"
                )
        return None

    def arena_boundary_reason(self, pose: Pose3, forward: bool) -> Optional[str]:
        """Stop the rover if its body is already outside the arena's free rect,
        or if driving forward would carry its front bumper outside it.

        This is the hard safety net that keeps the rover from driving off the
        arena edge even if the planner or controller lets a path graze it.
        """
        cfg = self.config.safety
        # The rover's *corners* must not cross the wall margin line, so check
        # against the arena inset by wall_margin only (not wall_margin +
        # radius - free_rect is where the centre may go, and we're checking
        # corners here).  This matches footprint_block.
        wall = self.config.arena.rect.inset(self.config.arena.wall_margin_m)
        half_length = 0.5 * self.config.robot.length_m
        half_width = 0.5 * self.config.robot.width_m

        # If the rover's own body is already outside, stop no matter what.
        for cx, cy in rect_corners(pose.x, pose.y, pose.yaw, half_length, half_width):
            if not (wall.x0 <= cx <= wall.x1 and wall.y0 <= cy <= wall.y1):
                return f"rover body outside the arena at ({cx:.2f}, {cy:.2f})"

        # If driving forward, project the front bumper one emergency-stop
        # distance ahead and stop if it would leave the arena.
        if forward:
            front_x = pose.x + math.cos(pose.yaw) * (half_length + cfg.emergency_stop_m)
            front_y = pose.y + math.sin(pose.yaw) * (half_length + cfg.emergency_stop_m)
            if not (wall.x0 <= front_x <= wall.x1 and wall.y0 <= front_y <= wall.y1):
                return f"front bumper would leave the arena at ({front_x:.2f}, {front_y:.2f})"

        return None

    # -- the tick ----------------------------------------------------------- #
    def decide(
        self,
        *,
        command: Optional[tuple],
        now: float,
        last_command_time: Optional[float],
        estop: bool,
        snapshot: LocalizationSnapshot,
        pose: Optional[Pose3],
        records: Sequence[ObstacleRecord],
        dt: float,
    ) -> GateDecision:
        """Fold every safety input into one :class:`GateDecision`."""
        cfg = self.config.safety
        max_speed = self.config.robot.max_speed_mps
        max_yaw = self.config.robot.max_yaw_rate_radps

        def hold(reason: str, *, scale: float, emergency: bool = False, stale: bool = False) -> GateDecision:
            self._linear, self._angular = 0.0, 0.0
            return GateDecision(
                allowed=False,
                linear=0.0,
                angular=0.0,
                reason=reason,
                speed_scale=scale,
                emergency=emergency,
                stale=stale,
                estop=estop,
                max_speed=0.0,
                max_yaw_rate=0.0,
            )

        if estop:
            return hold("e-stop latched", scale=0.0)

        if command is None or last_command_time is None:
            return hold("no command received yet", scale=0.0, stale=True)
        if now - last_command_time > cfg.command_timeout_s:
            return hold(
                f"no fresh command for {now - last_command_time:.2f}s "
                f"(limit {cfg.command_timeout_s:.2f}s)",
                scale=0.0,
                stale=True,
            )

        scale, allowed, reason = self.localization_scale(snapshot)
        if not allowed:
            return hold(reason, scale=0.0)

        linear, angular = command
        linear *= scale
        angular *= scale

        if pose is not None:
            brake = self.emergency_brake_reason(pose, records, forward=linear > 0.0)
            if brake is not None:
                return hold(brake, scale=scale, emergency=True)
            boundary = self.arena_boundary_reason(pose, linear > 0.0)
            if boundary is not None:
                return hold(boundary, scale=scale, emergency=True)

        linear = max(-max_speed, min(max_speed, linear))
        angular = max(-max_yaw, min(max_yaw, angular))

        # Rate-limit so the wheels never see a step; a stopped command is
        # already zero, so no ramp is applied when holding.
        if dt > 0.0:
            max_dv = self.config.robot.max_accel_mps2 * dt
            max_dw = self.config.robot.max_yaw_accel_radps2 * dt
            linear = self._linear + max(-max_dv, min(max_dv, linear - self._linear))
            angular = self._angular + max(-max_dw, min(max_dw, angular - self._angular))
        self._linear, self._angular = linear, angular

        return GateDecision(
            allowed=True,
            linear=linear,
            angular=angular,
            reason=reason,
            speed_scale=scale,
            estop=False,
            max_speed=max_speed * scale,
            max_yaw_rate=max_yaw * scale,
        )

    def reset(self) -> None:
        """Forget the last published command (used when a goal is cancelled)."""
        self._linear, self._angular = 0.0, 0.0
