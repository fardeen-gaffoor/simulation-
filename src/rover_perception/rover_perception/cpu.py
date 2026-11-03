"""fps, CPU and latency measurement without psutil or a profiler thread.

Everything is derived from ``/proc`` and from timestamps the caller supplies, so
the same code runs on the laptop and on the Pi, and the tests can drive it with
a synthetic clock and a synthetic ``/proc`` file.  Reading ``/proc/self/stat``
is cheap enough for a 1 Hz health report and adds no dependency to the robot
image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

__all__ = [
    "clock_ticks",
    "read_proc_stat",
    "CpuMeter",
    "FpsMeter",
    "LatencyMeter",
]


def clock_ticks() -> float:
    """The kernel's USER_HZ (usually 100)."""
    try:
        return float(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError):  # pragma: no cover - non-POSIX
        return 100.0


def read_proc_stat(path: str = "/proc/self/stat") -> Optional[Tuple[float, float]]:
    """``(utime, stime)`` in seconds for a process, or ``None``.

    The second field of ``/proc/<pid>/stat`` is the executable name in
    parentheses and may itself contain spaces or parentheses, so the line is
    split on the *last* ``)``.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            line = handle.read()
    except OSError:
        return None
    close = line.rfind(")")
    if close < 0:
        return None
    fields = line[close + 1 :].split()
    # after ")" the fields start at state (field 3), so utime (14) is index 11
    if len(fields) < 13:
        return None
    try:
        utime, stime = float(fields[11]), float(fields[12])
    except ValueError:
        return None
    ticks = clock_ticks()
    return (utime / ticks, stime / ticks)


class CpuMeter:
    """Process CPU use as a percentage of one core (can exceed 100 on several)."""

    def __init__(self, reader: Callable[[str], Optional[Tuple[float, float]]] = read_proc_stat, path: str = "/proc/self/stat"):
        self._reader = reader
        self._path = path
        self._last_cpu: Optional[float] = None
        self._last_at: Optional[float] = None
        self._percent = 0.0

    @property
    def percent(self) -> float:
        return self._percent

    def sample(self, now: float) -> float:
        reading = self._reader(self._path)
        if reading is None:
            return self._percent
        cpu_total = reading[0] + reading[1]
        if self._last_cpu is not None and self._last_at is not None:
            elapsed = now - self._last_at
            if elapsed > 1e-9:
                used = max(0.0, cpu_total - self._last_cpu)
                self._percent = 100.0 * used / elapsed
        self._last_cpu = cpu_total
        self._last_at = now
        return self._percent


class FpsMeter:
    """Frames per second over a sliding wall-clock window."""

    def __init__(self, window_s: float = 2.0):
        if window_s <= 0.0:
            raise ValueError("window_s must be > 0")
        self.window_s = float(window_s)
        self._times: List[float] = []

    def tick(self, now: float) -> None:
        self._times.append(float(now))
        if len(self._times) > 1:
            cutoff = now - self.window_s
            if self._times[0] < cutoff:
                self._times = [stamp for stamp in self._times if stamp >= cutoff]

    def fps(self, now: float) -> float:
        if len(self._times) < 2:
            return 0.0
        span = max(1e-6, now - self._times[0])
        return (len(self._times) - 1) / span

    @property
    def frames(self) -> int:
        return len(self._times)

    def reset(self) -> None:
        self._times.clear()


class LatencyMeter:
    """Mean (and worst) latency of an operation over a sliding window."""

    def __init__(self, window_s: float = 5.0):
        self.window_s = float(window_s)
        self._samples: List[Tuple[float, float]] = []

    def add(self, seconds: float, now: float) -> None:
        self._samples.append((float(now), float(seconds)))
        if self._samples[0][0] < now - self.window_s:
            self._samples = [sample for sample in self._samples if sample[0] >= now - self.window_s]

    def mean_ms(self) -> float:
        if not self._samples:
            return 0.0
        return 1000.0 * sum(sample[1] for sample in self._samples) / len(self._samples)

    def max_ms(self) -> float:
        if not self._samples:
            return 0.0
        return 1000.0 * max(sample[1] for sample in self._samples)

    @property
    def samples(self) -> int:
        return len(self._samples)
