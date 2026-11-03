"""The excavation mechanism: ladder pitch, belt speed, tub gate - and its limits.

This is the model of the hardware, written so it can be unit-tested without ROS
and without Gazebo.  It is the *dynamics* half of module 6:

* the ladder slews at a finite rate, so deploying and stowing take time and a
  cycle has to wait for them (``MOVING`` and ``RETRACTING`` are real stages);
* the belt spins up against an acceleration limit, signed: forward digs,
  reverse dumps;
* the tub gate opens and shuts over ``gate_open_time_s``;
* and two interlocks stop it: the ladder may never be driven deeper than
  ``max_depth_m`` below the sand, and the mechanism is only ``travel_ready``
  once it is back in the stowed attitude with the belt stopped and the gate
  shut.  ``travel_ready`` is what module 5 has to be told, because module 5
  cannot know it by itself.

Nothing here knows about Gazebo or serial ports: :mod:`rover_excavation.backend`
is the only module that sends the commands anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import MechanismCfg

__all__ = [
    "MechanismCommand",
    "MechanismSnapshot",
    "Mechanism",
    "STATE_STOWED",
    "STATE_DEPLOYING",
    "STATE_DIGGING",
    "STATE_DUMPING",
    "STATE_RETRACTING",
    "STATE_FAULT",
    "STATE_NAMES",
]

#: Mechanism attitudes, indexed by the ``MechanismStatus`` message constants.
STATE_STOWED = 0
STATE_DEPLOYING = 1
STATE_DIGGING = 2
STATE_DUMPING = 3
STATE_RETRACTING = 4
STATE_FAULT = 5

STATE_NAMES = ("STOWED", "DEPLOYING", "DIGGING", "DUMPING", "RETRACTING", "FAULT")

#: Tolerances for "the actuator has got there".
PITCH_TOL_RAD = 1.0e-3
BELT_TOL_RADPS = 0.05
GATE_TOL = 0.02


def _slew(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if delta > max_delta:
        return current + max_delta
    if delta < -max_delta:
        return current - max_delta
    return target


@dataclass(frozen=True)
class MechanismCommand:
    """One set of actuator targets.  This is what crosses the process boundary."""

    ladder_pitch_rad: float
    belt_velocity_radps: float
    tub_gate: float

    def as_tuple(self) -> tuple:
        return (self.ladder_pitch_rad, self.belt_velocity_radps, self.tub_gate)


@dataclass(frozen=True)
class MechanismSnapshot:
    """Everything ``MechanismStatus`` needs, in one immutable record."""

    state: int
    state_name: str
    travel_ready: bool
    ladder_pitch_rad: float
    ladder_depth_m: float
    belt_velocity_radps: float
    belt_linear_mps: float
    tub_gate: float
    message: str


class Mechanism:
    """The commanded mechanism, integrating its own slew-limited response."""

    def __init__(self, config: MechanismCfg, *, pitch_rad: Optional[float] = None) -> None:
        self.config = config
        self.pitch_rad = config.ladder.stowed_pitch_rad if pitch_rad is None else float(pitch_rad)
        self.belt_velocity_radps = 0.0
        self.tub_gate = 0.0
        self.state = STATE_STOWED
        self.fault: Optional[str] = None
        self.message = "stowed"
        self._command = MechanismCommand(self.pitch_rad, 0.0, 0.0)
        self._clamp_state()

    # -- commands ----------------------------------------------------------- #
    def command(self, pitch_rad: float, belt_velocity_radps: float, gate: float) -> None:
        """Set the actuator targets (they are reached over the following steps)."""
        self._command = MechanismCommand(
            self.clamp_pitch(pitch_rad),
            self.clamp_belt(belt_velocity_radps),
            min(1.0, max(0.0, float(gate))),
        )

    def stow(self) -> None:
        """The travel attitude: ladder raised, belt stopped, gate shut."""
        self.command(self.config.ladder.stowed_pitch_rad, 0.0, 0.0)

    def hold(self) -> None:
        """Freeze every actuator where it is (used on a fault or an e-stop)."""
        self._command = MechanismCommand(self.pitch_rad, self.belt_velocity_radps, self.tub_gate)

    def raise_fault(self, reason: str) -> None:
        if self.fault is None:
            self.fault = reason
            self.message = reason
            self.state = STATE_FAULT
            self.stow()

    def clear_fault(self) -> None:
        self.fault = None

    # -- limits ------------------------------------------------------------- #
    def clamp_pitch(self, pitch_rad: float) -> float:
        ladder = self.config.ladder
        return min(ladder.max_pitch_rad, max(ladder.min_pitch_rad, float(pitch_rad)))

    def clamp_belt(self, velocity_radps: float) -> float:
        limit = self.config.belt.max_velocity_radps
        return min(limit, max(-limit, float(velocity_radps)))

    def _clamp_state(self) -> None:
        self.pitch_rad = self.clamp_pitch(self.pitch_rad)
        self.belt_velocity_radps = self.clamp_belt(self.belt_velocity_radps)
        self.tub_gate = min(1.0, max(0.0, self.tub_gate))

    # -- integration -------------------------------------------------------- #
    def step(self, dt: float) -> None:
        """Advance the actuators toward their targets by ``dt`` seconds."""
        if dt <= 0.0:
            return
        ladder = self.config.ladder
        belt = self.config.belt
        self.pitch_rad = self.clamp_pitch(
            _slew(self.pitch_rad, self._command.ladder_pitch_rad, ladder.max_rate_radps * dt)
        )
        self.belt_velocity_radps = self.clamp_belt(
            _slew(self.belt_velocity_radps, self._command.belt_velocity_radps, belt.max_accel_radps2 * dt)
        )
        gate_rate = 1.0 / max(1e-6, self.config.tub.gate_open_time_s)
        self.tub_gate = min(1.0, max(0.0, _slew(self.tub_gate, self._command.tub_gate, gate_rate * dt)))
        self._check_depth()

    def _check_depth(self) -> None:
        if self.fault is not None:
            return
        depth = self.depth_m
        limit = self.config.interlocks.max_depth_m
        if depth > limit + 1e-6:
            self.raise_fault(
                f"ladder {depth:.3f} m below the sand surface exceeds the {limit:.3f} m interlock"
            )

    # -- queries ------------------------------------------------------------ #
    @property
    def target(self) -> MechanismCommand:
        return self._command

    @property
    def depth_m(self) -> float:
        """How far the ladder tip is below the sand surface (0 if it is above)."""
        return max(0.0, self.config.ladder.tip_depth_m(self.pitch_rad))

    @property
    def belt_linear_mps(self) -> float:
        return self.config.belt.surface_speed_mps(self.belt_velocity_radps)

    @property
    def travel_ready(self) -> bool:
        """True only in the stowed attitude: safe for module 5 to drive."""
        return (
            self.fault is None
            and self.pitch_rad <= self.config.interlocks.travel_pitch_max_rad + 1e-6
            and abs(self.belt_velocity_radps) <= BELT_TOL_RADPS
            and self.tub_gate <= GATE_TOL
        )

    def at_pitch(self, target_rad: float, tol: float = PITCH_TOL_RAD) -> bool:
        return abs(self.pitch_rad - target_rad) <= tol

    def at_belt(self, target_radps: float, tol: float = BELT_TOL_RADPS) -> bool:
        return abs(self.belt_velocity_radps - target_radps) <= tol

    def snapshot(self) -> MechanismSnapshot:
        return MechanismSnapshot(
            state=self.state,
            state_name=STATE_NAMES[self.state],
            travel_ready=self.travel_ready,
            ladder_pitch_rad=self.pitch_rad,
            ladder_depth_m=self.depth_m,
            belt_velocity_radps=self.belt_velocity_radps,
            belt_linear_mps=self.belt_linear_mps,
            tub_gate=self.tub_gate,
            message=self.message,
        )

    def settle_idle(self) -> int:
        """Set ``state`` for a mechanism that is not in a work cycle.

        What the state *means* when nothing is commanded: a fault, the travel
        attitude, or still on its way back to it.  The node calls this when no
        cycle is running; keeping it here means the offline tools report the same
        thing the node would, instead of leaving ``RETRACTING`` on the last tick
        of the last cycle forever.
        """
        if self.fault is not None:
            self.state = STATE_FAULT
        elif self.travel_ready:
            self.state = STATE_STOWED
            if not self.message or self.message.startswith("work cycle"):
                self.message = "stowed, ready to work"
        else:
            self.state = STATE_RETRACTING
        return self.state

    # -- convenience -------------------------------------------------------- #
    @property
    def dig_pitch_rad(self) -> float:
        return self.config.ladder.dig_pitch_rad

    @property
    def stowed_pitch_rad(self) -> float:
        return self.config.ladder.stowed_pitch_rad

    def time_to_pitch(self, target_rad: float) -> float:
        """How long the ladder needs to slew from where it is to ``target_rad``."""
        rate = max(1e-9, self.config.ladder.max_rate_radps)
        return abs(self.clamp_pitch(target_rad) - self.pitch_rad) / rate


def state_name(state: int) -> str:
    if 0 <= state < len(STATE_NAMES):
        return STATE_NAMES[state]
    return f"UNKNOWN({state})"
