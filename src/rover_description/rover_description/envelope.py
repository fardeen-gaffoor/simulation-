"""The stowed-envelope check.

The competition fixes a stowed bounding box of 1.5 m (x) x 0.75 m (y) x 0.75 m
(z) from the ground plane up.  ``check_envelope`` measures the generated robot
(sum of every part's oriented bounding box, ladder in its stowed attitude) and
compares it to the limit in ``rover.yaml``.  ``scripts/check_envelope.py`` is the
executable wrapper required by module 2 and the tests import this directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .config import Envelope, RoverConfig
from .geometry import EnvelopeSize, compute_envelope

__all__ = ["EnvelopeReport", "check_envelope"]


@dataclass(frozen=True)
class EnvelopeReport:
    measured: EnvelopeSize
    limit: Envelope

    @property
    def measured_envelope(self) -> Envelope:
        return Envelope(*self.measured.as_tuple())

    @property
    def ok(self) -> bool:
        return self.limit.contains(self.measured_envelope)

    def margins(self) -> Dict[str, float]:
        return {
            "length_x": self.limit.length_x - self.measured.length_x,
            "width_y": self.limit.width_y - self.measured.width_y,
            "height_z": self.limit.height_z - self.measured.height_z,
        }

    def to_dict(self) -> Dict[str, object]:
        return {
            "measured": {
                "length_x": round(self.measured.length_x, 6),
                "width_y": round(self.measured.width_y, 6),
                "height_z": round(self.measured.height_z, 6),
                "x_min": round(self.measured.x_min, 6),
                "x_max": round(self.measured.x_max, 6),
                "y_min": round(self.measured.y_min, 6),
                "y_max": round(self.measured.y_max, 6),
                "z_min": round(self.measured.z_min, 6),
                "z_max": round(self.measured.z_max, 6),
            },
            "limit": {
                "length_x": self.limit.length_x,
                "width_y": self.limit.width_y,
                "height_z": self.limit.height_z,
            },
            "margin": {k: round(v, 6) for k, v in self.margins().items()},
            "ok": self.ok,
        }

    def table(self) -> str:
        lines = [
            f"{'axis'.ljust(10)}{'measured'.rjust(11)}{'limit'.rjust(11)}{'margin'.rjust(11)}",
            "-" * 43,
        ]
        for axis in ("length_x", "width_y", "height_z"):
            measured = getattr(self.measured, axis)
            limit = getattr(self.limit, axis)
            lines.append(f"{axis.ljust(10)}{measured:11.3f}{limit:11.3f}{(limit - measured):11.3f}")
        lines.append("-" * 43)
        lines.append(f"{'result'.ljust(10)}{'PASS' if self.ok else 'FAIL':>11}")
        return "\n".join(lines)


def check_envelope(cfg: RoverConfig) -> EnvelopeReport:
    return EnvelopeReport(measured=compute_envelope(cfg), limit=cfg.envelope_limit)
