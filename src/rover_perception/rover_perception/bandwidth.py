"""The 4000 kbps rule: measure what the cameras really cost on the link.

Competition constraint: **average total comms <= 4000 kbps**.  That is a
*measured* quantity, so the manager counts the actual payload bytes arriving on
each image topic over a sliding window rather than trusting the configured frame
rates (which the sim cannot change anyway and the real driver may not honour).

Two details matter and are explicit here:

* **startup settling.**  In the first moments the window is nearly empty and
  ``bytes / elapsed`` explodes.  The denominator is therefore floored at
  ``settle_s`` (1 s), so a burst of first frames cannot trigger a spurious
  de-rate.
* **the window is a wall-clock window.**  Samples older than ``window_s`` are
  pruned on every tick, so the number always means "the average over the last N
  seconds", which is exactly what the rules measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["CameraTraffic", "ByteRateMeter"]


@dataclass
class CameraTraffic:
    """Cumulative traffic counters for one camera."""

    name: str
    frames: int = 0
    bytes: int = 0
    last_frame_at: Optional[float] = None
    samples: List[Tuple[float, int]] = field(default_factory=list)


class ByteRateMeter:
    """Sliding-window byte-rate meter for a fixed set of cameras."""

    def __init__(
        self,
        names: Sequence[str],
        window_s: float = 10.0,
        limit_kbps: float = 4000.0,
        settle_s: float = 1.0,
    ):
        if window_s <= 0.0:
            raise ValueError("window_s must be > 0")
        if settle_s <= 0.0:
            raise ValueError("settle_s must be > 0")
        self.names = tuple(names)
        self.window_s = float(window_s)
        self.limit_kbps = float(limit_kbps)
        self.settle_s = float(settle_s)
        self._traffic: Dict[str, CameraTraffic] = {name: CameraTraffic(name) for name in self.names}
        self._started_at: Optional[float] = None

    # -- input -------------------------------------------------------------- #
    def observe(self, name: str, n_bytes: int, now: float) -> None:
        """Record one payload of ``n_bytes`` bytes arriving at time ``now``."""
        if name not in self._traffic:
            raise KeyError(f"unknown camera '{name}' (managed: {list(self.names)})")
        traffic = self._traffic[name]
        traffic.frames += 1
        traffic.bytes += int(n_bytes)
        traffic.last_frame_at = float(now)
        traffic.samples.append((float(now), int(n_bytes)))
        if self._started_at is None:
            self._started_at = float(now)
        self._prune(now)

    def tick(self, now: float) -> None:
        """Prune stale samples without recording a frame."""
        self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        for traffic in self._traffic.values():
            if traffic.samples and traffic.samples[0][0] < cutoff:
                traffic.samples = [sample for sample in traffic.samples if sample[0] >= cutoff]

    # -- queries ------------------------------------------------------------ #
    def elapsed(self, now: float) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, now - self._started_at)

    def rate_kbps(self, name: str, now: float) -> float:
        """Measured average for one camera over the window (kbit/s)."""
        traffic = self._traffic[name]
        if not traffic.samples:
            return 0.0
        bits = sum(sample[1] for sample in traffic.samples) * 8.0
        return bits / 1000.0 / max(self.settle_s, min(self.window_s, self.elapsed(now)))

    def total_kbps(self, now: float) -> float:
        return float(sum(self.rate_kbps(name, now) for name in self.names))

    def rate_hz(self, name: str, now: float) -> float:
        traffic = self._traffic[name]
        if not traffic.samples:
            return 0.0
        return len(traffic.samples) / max(self.settle_s, min(self.window_s, self.elapsed(now)))

    def frames(self, name: str) -> int:
        return self._traffic[name].frames

    def bytes(self, name: str) -> int:
        return self._traffic[name].bytes

    def last_frame_at(self, name: str) -> Optional[float]:
        return self._traffic[name].last_frame_at

    def is_streaming(self, name: str, now: float, stale_after_s: float) -> bool:
        last = self.last_frame_at(name)
        return last is not None and (now - last) <= stale_after_s

    def traffic(self, name: str) -> CameraTraffic:
        return self._traffic[name]

    def reset(self) -> None:
        self._traffic = {name: CameraTraffic(name) for name in self.names}
        self._started_at = None
