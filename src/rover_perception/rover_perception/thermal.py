"""Thermal-zone monitoring: the Raspberry Pi 5 gets hot and throttles.

The Pi exposes ``/sys/class/thermal/thermal_zone*/temp``; the Pi 5 in particular
has zones for the CPU cluster and for the board.  The hottest readable zone is
the one that matters, so that is what is reported and what the camera manager
de-rates against.

Everything is injectable (``paths`` and ``reader``) so the tests drive real
files in a temporary directory instead of mocking the world: the same code path
is exercised on the laptop, where there may be no thermal zones at all - in
which case the state is ``nominal`` with zone ``"none"`` and nothing is
de-rated.  That is a legitimate, reported condition, not a failure.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .config import ThermalCfg

__all__ = [
    "THERMAL_NOMINAL",
    "THERMAL_WARNING",
    "THERMAL_THROTTLING",
    "THERMAL_CRITICAL",
    "ThermalReading",
    "ThermalMonitor",
    "read_zone_temperature",
    "discover_zones",
]

THERMAL_NOMINAL = 0
THERMAL_WARNING = 1
THERMAL_THROTTLING = 2
THERMAL_CRITICAL = 3

#: Auto-discovery pattern used when ``zone_paths`` is not configured.
DEFAULT_ZONE_GLOB = "/sys/class/thermal/thermal_zone*/temp"


def read_zone_temperature(path: str) -> Optional[float]:
    """Read one sysfs thermal zone, in degrees Celsius (None if unreadable).

    sysfs reports millidegrees on the Pi; some drivers report whole degrees, so
    a large value is treated as milli-degrees.  Anything outside a physically
    plausible range is rejected rather than de-rated against.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
    except OSError:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value > 1000.0:
        value /= 1000.0
    if not -50.0 < value < 200.0:
        return None
    return value


def discover_zones(pattern: str = DEFAULT_ZONE_GLOB) -> List[str]:
    """Every readable thermal zone, sorted for determinism."""
    return sorted(path for path in glob.glob(pattern) if os.path.isfile(path))


@dataclass(frozen=True)
class ThermalReading:
    """The hottest monitored zone and the state it implies."""

    zone: str
    temperature_c: Optional[float]
    state: int
    zones: Dict[str, float]

    @property
    def state_name(self) -> str:
        from .config import THERMAL_STATES

        return THERMAL_STATES[self.state]

    @property
    def available(self) -> bool:
        return self.temperature_c is not None


class ThermalMonitor:
    """Polls the thermal zones and classifies the hottest one."""

    def __init__(
        self,
        cfg: ThermalCfg,
        paths: Optional[Sequence[str]] = None,
        reader: Callable[[str], Optional[float]] = read_zone_temperature,
    ):
        self.cfg = cfg
        self._paths = list(paths) if paths is not None else None
        self._reader = reader

    def paths(self) -> List[str]:
        if self._paths is not None:
            return list(self._paths)
        if self.cfg.zone_paths:
            return [os.path.expanduser(path) for path in self.cfg.zone_paths]
        return discover_zones()

    def temperatures(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for path in self.paths():
            value = self._reader(path)
            if value is not None:
                out[path] = value
        return out

    def classify(self, temperature_c: Optional[float]) -> int:
        if temperature_c is None:
            return THERMAL_NOMINAL
        if temperature_c >= self.cfg.critical_c:
            return THERMAL_CRITICAL
        if temperature_c >= self.cfg.throttle_c:
            return THERMAL_THROTTLING
        if temperature_c >= self.cfg.warn_c:
            return THERMAL_WARNING
        return THERMAL_NOMINAL

    def read(self) -> ThermalReading:
        values = self.temperatures() if self.cfg.enabled else {}
        if not values:
            return ThermalReading(zone="none", temperature_c=None, state=THERMAL_NOMINAL, zones={})
        zone = max(sorted(values), key=lambda path: values[path])
        temperature = values[zone]
        return ThermalReading(
            zone=os.path.basename(os.path.dirname(zone)) or zone,
            temperature_c=temperature,
            state=self.classify(temperature),
            zones=values,
        )
