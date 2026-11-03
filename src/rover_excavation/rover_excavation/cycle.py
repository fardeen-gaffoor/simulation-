"""One work cycle: start, move the mechanism, work, retract, done.

A :class:`WorkCycle` is the whole of what module 6 does on behalf of one
``WorkCycle`` action goal.  It is deliberately ROS-free: the node owns the
action server, the timers and the topics, and this class owns the sequence.

The stages mirror the action's ``STAGE_*`` constants exactly, because they are
what the caller watches:

``STARTING``
    latch the clock, clear any previous fault, aim the mechanism.
``MOVING``
    wait for the ladder (and the tub gate, when dumping) to get where they are
    going.  The ladder slews at a finite rate, so this is real time.
``WORKING``
    the belt runs: sand goes into the tub (dig) or out of it onto the berm
    (dump).  A dig ends when the tub is full, a dump when it is empty.
``RETRACTING``
    belt stopped and gate shut, then the ladder slews back to the travel
    attitude.  ``settle_s`` lets the last sand fall before the ladder moves.
``DONE``
    ``travel_ready`` is true again and the rover may be driven.

The cycle never decides *where* the rover stands - the caller puts it there -
and never publishes anything.  It reports ``stage``, ``progress``, ``detail``,
``amount_moved`` and ``success``, and the node turns those into action feedback
and the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .config import WORK_PHASES, ExcavationConfig, dig_rate_m3_s
from .mechanism import (
    BELT_TOL_RADPS,
    GATE_TOL,
    STATE_DEPLOYING,
    STATE_DIGGING,
    STATE_DUMPING,
    STATE_RETRACTING,
    Mechanism,
)
from .sand import BermField, Tub

__all__ = ["WorkCycle", "STAGE_STARTING", "STAGE_MOVING", "STAGE_WORKING", "STAGE_RETRACTING", "STAGE_DONE", "STAGE_NAMES"]

#: Work-cycle stages, indexed by the ``WorkCycle`` action's constants.
STAGE_STARTING = 0
STAGE_MOVING = 1
STAGE_WORKING = 2
STAGE_RETRACTING = 3
STAGE_DONE = 4

STAGE_NAMES = ("STARTING", "MOVING", "WORKING", "RETRACTING", "DONE")

#: Progress budget per stage: it has to be monotone, because the caller shows it.
_PROGRESS_MOVE = 0.15
_PROGRESS_WORK = 0.75
_PROGRESS_RETRACT = 0.10


class WorkCycle:
    """A single dig or dump, played out against the shared mechanism.

    The node owns the mechanism's clock: it calls :meth:`Mechanism.step` once
    per tick and then :meth:`step`, so a cycle only *reads* and *commands* the
    actuators and never integrates them itself.  That keeps one clock in the
    process even when the rover is between cycles.
    """

    def __init__(
        self,
        config: ExcavationConfig,
        phase: str,
        work_pose: Tuple[float, float, float],
        mechanism: Mechanism,
        tub: Tub,
        berm: BermField,
        *,
        cycle_id: int = 0,
        timeout_s: float = 0.0,
        target_amount: float = 0.0,
        now: float = 0.0,
    ) -> None:
        if phase not in WORK_PHASES:
            raise ValueError(f"WorkCycle: phase must be one of {list(WORK_PHASES)}, got {phase!r}")
        self.config = config
        self.phase = phase
        self.work_pose = (float(work_pose[0]), float(work_pose[1]), float(work_pose[2]))
        self.mechanism = mechanism
        self.tub = tub
        self.berm = berm
        self.cycle_id = int(cycle_id)
        self.target_amount = float(target_amount)

        phase_cfg = config.cycle.dig if phase == "excavate" else config.cycle.dump
        self.timeout_s = float(timeout_s) if timeout_s > 0.0 else float(phase_cfg.timeout_s)

        self.stage = STAGE_STARTING
        self.progress = 0.0
        self.amount_moved = 0.0
        self.success = False
        self.done = False
        self.message = ""

        self._start = float(now)
        self._last = float(now)
        self._stage_since = float(now)
        self._move_total_s = 1.0
        self._start_fraction = tub.fraction
        self._landing: Optional[Tuple[float, float]] = None

    # -- public view -------------------------------------------------------- #
    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage]

    @property
    def elapsed_s(self) -> float:
        return self._last - self._start

    @property
    def detail(self) -> str:
        if self.message:
            return self.message
        if self.stage == STAGE_STARTING:
            return "starting the work cycle"
        if self.stage == STAGE_MOVING:
            if self.phase == "excavate":
                return (
                    f"deploying the ladder to {self.config.mechanism.ladder.dig_pitch_rad:.2f} rad "
                    f"({self.mechanism.pitch_rad:.2f} now)"
                )
            return f"opening the tub gate ({self.mechanism.tub_gate * 100.0:.0f}%)"
        if self.stage == STAGE_WORKING:
            if self.phase == "excavate":
                return f"digging: tub {self.tub.fraction * 100.0:.0f}% full"
            return f"dumping: tub {self.tub.fraction * 100.0:.0f}% full"
        if self.stage == STAGE_RETRACTING:
            return f"stowing the ladder ({self.mechanism.pitch_rad:.2f} rad)"
        return f"done: moved {self.amount_moved:.4f} m3"

    # -- the step ----------------------------------------------------------- #
    def step(self, now: float) -> None:
        """Advance the cycle to wall-clock ``now`` (simulation seconds)."""
        now = float(now)
        dt = max(0.0, now - self._last)
        self._last = now
        if self.done:
            return

        if self.mechanism.fault is not None:
            self._fail(f"mechanism fault: {self.mechanism.fault}")
            return
        if now - self._start > self.timeout_s:
            self._fail(
                f"timed out after {now - self._start:.1f}s (limit {self.timeout_s:.1f}s) during {self.stage_name.lower()}"
            )
            return

        if self.stage == STAGE_STARTING:
            self._tick_starting(now)
        elif self.stage == STAGE_MOVING:
            self._tick_moving(now, dt)
        elif self.stage == STAGE_WORKING:
            self._tick_working(now, dt)
        elif self.stage == STAGE_RETRACTING:
            self._tick_retracting(now, dt)
        self.progress = max(self.progress, min(1.0, self._progress_estimate()))

    # -- stages ------------------------------------------------------------- #
    def _tick_starting(self, now: float) -> None:
        self.mechanism.clear_fault()
        if self.phase == "excavate":
            self.mechanism.command(self.mechanism.dig_pitch_rad, 0.0, 0.0)
            self._move_total_s = max(1e-6, self.mechanism.time_to_pitch(self.mechanism.dig_pitch_rad))
        else:
            self.mechanism.command(self.mechanism.stowed_pitch_rad, 0.0, 1.0)
            self._move_total_s = max(1e-6, self.mechanism.time_to_pitch(self.mechanism.stowed_pitch_rad))
        self.mechanism.state = STATE_DEPLOYING
        self.mechanism.message = f"work cycle {self.cycle_id}: {self.phase}"
        self._enter(STAGE_MOVING, now)

    def _tick_moving(self, now: float, dt: float) -> None:
        if self.phase == "excavate":
            if self.mechanism.at_pitch(self.config.mechanism.ladder.dig_pitch_rad):
                self.mechanism.command(
                    self.mechanism.dig_pitch_rad, self.config.mechanism.belt.dig_velocity_radps, 0.0
                )
                self._enter(STAGE_WORKING, now)
        else:
            gate_open = self.mechanism.tub_gate >= 1.0 - GATE_TOL
            if self.mechanism.at_pitch(self.mechanism.stowed_pitch_rad) and gate_open:
                self.mechanism.command(
                    self.mechanism.stowed_pitch_rad, self.config.mechanism.belt.dump_velocity_radps, 1.0
                )
                self._enter(STAGE_WORKING, now)

    def _tick_working(self, now: float, dt: float) -> None:
        if self.phase == "excavate":
            self._work_dig(dt, now)
        else:
            self._work_dump(dt, now)

    def _work_dig(self, dt: float, now: float) -> None:
        self.mechanism.state = STATE_DIGGING
        rate = dig_rate_m3_s(
            self.config.sand.dig,
            self.config.mechanism,
            self.mechanism.pitch_rad,
            self.mechanism.belt_velocity_radps,
        )
        accepted = self.tub.add(rate * dt)
        self.amount_moved += accepted
        dig_cfg = self.config.cycle.dig
        full = self.tub.fraction >= dig_cfg.target_fill
        enough = self.target_amount > 0.0 and self.amount_moved >= self.target_amount
        long_enough = now - self._stage_since >= dig_cfg.min_s
        if (full or enough) and long_enough:
            self.message = "tub full; stowing"
            self._begin_retract(now)

    def _work_dump(self, dt: float, now: float) -> None:
        self.mechanism.state = STATE_DUMPING
        dump_cfg = self.config.cycle.dump
        gate_open = self.mechanism.tub_gate >= 1.0 - GATE_TOL
        reversing = self.mechanism.belt_velocity_radps <= -BELT_TOL_RADPS
        if gate_open and reversing:
            removed = self.tub.remove(self.config.sand.dump.rate_m3_s * dt)
            if removed > 0.0:
                if self._landing is None:
                    self._landing = self.config.landing_point(*self.work_pose)
                    self.berm.begin_dump()
                self.berm.deposit(self._landing[0], self._landing[1], removed)
                self.amount_moved += removed
        empty = self.tub.fraction <= dump_cfg.target_empty
        long_enough = now - self._stage_since >= dump_cfg.min_s
        if empty and long_enough:
            if self.amount_moved <= 0.0:
                self.message = "tub was already empty; nothing to place"
            else:
                self.message = f"placed {self.amount_moved:.4f} m3; stowing"
            self._begin_retract(now)

    def _tick_retracting(self, now: float, dt: float) -> None:
        self.mechanism.state = STATE_RETRACTING
        settled = now - self._stage_since >= self.config.cycle.settle_s
        if settled and not self.mechanism.travel_ready:
            self.mechanism.command(self.mechanism.stowed_pitch_rad, 0.0, 0.0)
            self.mechanism.state = STATE_RETRACTING
        if settled and self.mechanism.travel_ready:
            self._finish(now)

    # -- transitions -------------------------------------------------------- #
    def _begin_retract(self, now: float) -> None:
        """Stop the belt and shut the gate, holding the ladder for ``settle_s``."""
        self.mechanism.command(self.mechanism.pitch_rad, 0.0, 0.0)
        self._enter(STAGE_RETRACTING, now)

    def _enter(self, stage: int, now: float) -> None:
        self.stage = stage
        self._stage_since = now

    def _finish(self, now: float) -> None:
        self.stage = STAGE_DONE
        self._stage_since = now
        self.progress = 1.0
        self.success = True
        self.done = True
        if not self.message:
            self.message = f"moved {self.amount_moved:.4f} m3"

    def _fail(self, reason: str) -> None:
        self.stage = STAGE_DONE
        self.done = True
        self.success = False
        self.message = reason
        self.progress = 1.0
        if self.mechanism.fault is None:
            # A fault already asked the mechanism to stow; anything else (a
            # timeout, an e-stop) still has to leave it in the travel attitude.
            self.mechanism.stow()

    # -- progress ----------------------------------------------------------- #
    def _progress_estimate(self) -> float:
        if self.stage == STAGE_STARTING:
            return 0.0
        if self.stage == STAGE_MOVING:
            remaining = self.mechanism.time_to_pitch(
                self.mechanism.dig_pitch_rad if self.phase == "excavate" else self.mechanism.stowed_pitch_rad
            )
            done = 1.0 - min(1.0, remaining / self._move_total_s)
            return _PROGRESS_MOVE * done
        if self.stage == STAGE_WORKING:
            if self.phase == "excavate":
                target = max(1e-6, self.config.cycle.dig.target_fill)
                frac = self.tub.fraction / target
            else:
                span = max(1e-6, self._start_fraction - self.config.cycle.dump.target_empty)
                frac = (self._start_fraction - self.tub.fraction) / span
            return _PROGRESS_MOVE + _PROGRESS_WORK * min(1.0, max(0.0, frac))
        if self.stage == STAGE_RETRACTING:
            remaining = self.mechanism.time_to_pitch(self.mechanism.stowed_pitch_rad)
            done = 1.0 - min(1.0, remaining / self._move_total_s)
            return _PROGRESS_MOVE + _PROGRESS_WORK + _PROGRESS_RETRACT * done
        return 1.0
