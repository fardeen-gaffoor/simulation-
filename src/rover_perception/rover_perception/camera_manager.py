"""(a) The camera manager's brain: phases, the bandwidth ladder, thermal de-rate.

This module is pure logic - it owns no ROS and no timers - so the whole policy is
unit-testable, including the parts that are hard to provoke on real hardware
(an over-budget link, a hot SoC).  ``scripts/camera_manager.py`` is the thin
rclpy wrapper that feeds it measurements and publishes
``rover_perception_msgs/CameraPhase``.

Two policies are worth spelling out, because they are the reason the numbers in
``perception.yaml`` look the way they do:

**Phase rates.**  One camera is *active* at ``active_rate_hz`` (the camera the
current perception phase needs: ``nav`` -> cam_nav, ``tag`` -> cam_tag,
``excavate`` -> cam_excavation, ``dump`` -> cam_dump); every other camera runs at
``idle_rate_hz``, enough to notice motion but cheap on the link.  In ``idle``
and ``degraded`` there is no active camera.

**Derating priority comes free from the rate ladder.**  Every camera is scaled
by the same factor from ``bandwidth.derate_factors``, and a camera whose scaled
rate lands below ``min_rate_hz`` is switched off rather than streamed at an
unusable rate.  Because the idle rate starts low, idle cameras cross that
threshold first: the rover keeps its one important camera as long as possible
and sheds the rest, which is exactly the priority wanted when the link or the
SoC is the bottleneck.  A hard override kicks in above ``hard_limit_kbps``
(immediately) and at ``critical_c`` (everything off).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .config import MODES, CameraManagerCfg, THERMAL_STATES
from .thermal import (
    THERMAL_CRITICAL,
    THERMAL_NOMINAL,
    THERMAL_THROTTLING,
    THERMAL_WARNING,
)

__all__ = ["CameraCommand", "PhaseCommand", "PhaseController", "ThermalState"]


class ThermalState:
    """Re-exported constants so callers need not import :mod:`thermal`."""

    NOMINAL = THERMAL_NOMINAL
    WARNING = THERMAL_WARNING
    THROTTLING = THERMAL_THROTTLING
    CRITICAL = THERMAL_CRITICAL
    NAMES = THERMAL_STATES


@dataclass(frozen=True)
class CameraCommand:
    """What one camera is being asked to do."""

    name: str
    rate_hz: float
    base_rate_hz: float
    enabled: bool
    active: bool

    @property
    def kbps_estimate(self) -> float:
        """Nominal cost of this command, for logging (not a measurement)."""
        return 0.0


@dataclass(frozen=True)
class PhaseCommand:
    """The full phase command, ready to be copied onto a ``CameraPhase``."""

    mode: str
    active_camera: str
    active_rate_hz: float
    idle_rate_hz: float
    commands: Tuple[CameraCommand, ...]
    derate_index: int
    derate_factor: float
    bandwidth_ok: bool
    throttled: bool
    measured_kbps: float
    limit_kbps: float
    thermal_state: int
    reasons: Tuple[str, ...] = ()

    def command_for(self, name: str) -> CameraCommand:
        for command in self.commands:
            if command.name == name:
                return command
        raise KeyError(f"camera '{name}' is not managed")

    @property
    def enabled_cameras(self) -> Tuple[str, ...]:
        return tuple(command.name for command in self.commands if command.enabled)

    @property
    def thermal_state_name(self) -> str:
        return THERMAL_STATES[self.thermal_state]


class PhaseController:
    """Phase + bandwidth + thermal state machine for the managed cameras."""

    def __init__(
        self,
        cfg: CameraManagerCfg,
        camera_names: Sequence[str],
        active_camera_for_mode: Callable[[str], Optional[str]],
        initial_thermal_state: int = THERMAL_NOMINAL,
    ):
        if not camera_names:
            raise ValueError("PhaseController needs at least one camera")
        self.cfg = cfg
        self.camera_names = tuple(camera_names)
        self._active_for_mode = active_camera_for_mode
        self._mode = cfg.initial_mode
        self._derate_index = 0
        self._thermal_state = initial_thermal_state
        self._measured_kbps = 0.0
        self._reasons: List[str] = []

    # -- state -------------------------------------------------------------- #
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def derate_index(self) -> int:
        return self._derate_index

    @property
    def derate_factor(self) -> float:
        return float(self.cfg.bandwidth.derate_factors[self._derate_index])

    @property
    def thermal_state(self) -> int:
        return self._thermal_state

    @property
    def throttled(self) -> bool:
        return self._derate_index > 0

    def set_mode(self, mode: str) -> None:
        """Switch perception phase (driven by ``/perception/mode``)."""
        if mode not in MODES:
            raise ValueError(f"unknown perception mode '{mode}' (known: {list(MODES)})")
        if mode != self._mode:
            self._reasons.append(f"mode {self._mode} -> {mode}")
        self._mode = mode

    # -- the bandwidth ladder ----------------------------------------------- #
    def _step_up(self, reason: str) -> bool:
        """Derate one rung, recording ``reason`` only when the rung changes.

        At the floor there is nowhere left to go, and the sim keeps reporting the
        same over-budget measurement forever (Gazebo's sensors cannot actually be
        re-rated).  Recording the reason on every update would put the identical
        line in the log at camera rate for the whole run, which is how a real
        warning gets ignored.
        """
        limit = len(self.cfg.bandwidth.derate_factors) - 1
        if self._derate_index >= limit:
            return False
        self._derate_index += 1
        self._reasons.append(f"{reason} -> derate step {self._derate_index} (x{self.derate_factor:.2f})")
        return True

    def _step_down(self) -> None:
        if self._derate_index > 0:
            self._derate_index -= 1
            self._reasons.append(
                f"bandwidth {self._measured_kbps:.0f} kbps is comfortably under the limit "
                f"-> derate step {self._derate_index} (x{self.derate_factor:.2f})"
            )

    def _apply_bandwidth(self, measured_kbps: float) -> None:
        self._measured_kbps = max(0.0, float(measured_kbps))
        bandwidth = self.cfg.bandwidth
        if self._measured_kbps > bandwidth.hard_limit_kbps:
            self._step_up(
                f"HARD limit exceeded ({self._measured_kbps:.0f} > "
                f"{bandwidth.hard_limit_kbps:.0f} kbps)"
            )
            return
        if self._measured_kbps > bandwidth.limit_kbps:
            self._step_up(
                f"bandwidth {self._measured_kbps:.0f}/{bandwidth.limit_kbps:.0f} kbps"
            )
            return
        if self._measured_kbps < bandwidth.limit_kbps - bandwidth.hysteresis_kbps:
            self._step_down()

    def _thermal_floor(self) -> int:
        if self._thermal_state >= THERMAL_CRITICAL:
            return len(self.cfg.bandwidth.derate_factors) - 1  # everything off
        if self._thermal_state >= THERMAL_THROTTLING:
            return min(
                self.cfg.thermal.throttle_steps,
                len(self.cfg.bandwidth.derate_factors) - 1,
            )
        return 0

    # -- the command -------------------------------------------------------- #
    def _rates(self) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """``(active_camera, base_rate_per_camera, effective_rate_per_camera)``."""
        if self._mode in ("idle", "degraded"):
            active = ""
            base = {name: 0.0 for name in self.camera_names}
        else:
            active = self._active_for_mode(self._mode) or ""
            base = {
                name: (self.cfg.active_rate_hz if name == active else self.cfg.idle_rate_hz)
                for name in self.camera_names
            }
        if self._mode == "degraded":
            base = {name: self.cfg.min_rate_hz for name in self.camera_names}
        factor = self.derate_factor
        effective: Dict[str, float] = {}
        for name, rate in base.items():
            scaled = rate * factor
            if rate <= 0.0:
                effective[name] = 0.0
            elif scaled < self.cfg.min_rate_hz:
                effective[name] = 0.0  # below the usable rate: switch it off
            else:
                effective[name] = scaled
        return active, base, effective

    def update(
        self,
        measured_kbps: float,
        thermal_state: int = THERMAL_NOMINAL,
    ) -> PhaseCommand:
        """Fold in one round of measurements and produce the next command."""
        self._reasons = []
        self._thermal_state = int(thermal_state)
        self._apply_bandwidth(measured_kbps)
        floor = self._thermal_floor()
        if floor > self._derate_index:
            self._derate_index = floor
            self._reasons.append(
                f"thermal {THERMAL_STATES[self._thermal_state]} -> derate step {floor}"
            )
        active, base, effective = self._rates()
        commands = tuple(
            CameraCommand(
                name=name,
                rate_hz=effective[name],
                base_rate_hz=base[name],
                enabled=effective[name] > 0.0,
                active=(name == active and effective[name] > 0.0),
            )
            for name in self.camera_names
        )
        bandwidth = self.cfg.bandwidth
        return PhaseCommand(
            mode=self._mode,
            active_camera=active if effective.get(active, 0.0) > 0.0 else "",
            active_rate_hz=effective.get(active, 0.0),
            idle_rate_hz=self.cfg.idle_rate_hz * self.derate_factor,
            commands=commands,
            derate_index=self._derate_index,
            derate_factor=self.derate_factor,
            bandwidth_ok=self._measured_kbps <= bandwidth.limit_kbps,
            throttled=self._derate_index > 0,
            measured_kbps=self._measured_kbps,
            limit_kbps=bandwidth.limit_kbps,
            thermal_state=self._thermal_state,
            reasons=tuple(self._reasons),
        )

    def snapshot(self) -> Dict[str, object]:
        return {
            "mode": self._mode,
            "derate_index": self._derate_index,
            "derate_factor": self.derate_factor,
            "measured_kbps": round(self._measured_kbps, 3),
            "thermal_state": THERMAL_STATES[self._thermal_state],
            "active_camera": self._active_for_mode(self._mode) or "",
        }
