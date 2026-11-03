"""The mass budget table.

The competition caps the rover at 80 kg INCLUDING beacons, markers and battery.
This module turns ``rover.yaml`` into an explicit, auditable table (structural
links + payload) and checks the ceiling.  It is what ``mass_budget.py`` prints
and what ``check_envelope.py`` reports alongside the envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .config import RoverConfig

__all__ = ["MassRow", "mass_budget", "MassBudget"]


@dataclass(frozen=True)
class MassRow:
    name: str
    mass: float
    category: str  # "structural" | "payload"


@dataclass(frozen=True)
class MassBudget:
    rows: List[MassRow]
    mass_limit: float

    @property
    def structural_mass(self) -> float:
        return sum(r.mass for r in self.rows if r.category == "structural")

    @property
    def payload_mass(self) -> float:
        return sum(r.mass for r in self.rows if r.category == "payload")

    @property
    def total_mass(self) -> float:
        return self.structural_mass + self.payload_mass

    @property
    def margin(self) -> float:
        return self.mass_limit - self.total_mass

    @property
    def ok(self) -> bool:
        return self.total_mass <= self.mass_limit + 1e-9

    def table(self) -> str:
        width = max(len(r.name) for r in self.rows + [MassRow("total", 0.0, "")])
        lines: List[str] = []
        lines.append(f"{'component'.ljust(width)}  {'mass (kg)':>9}  category")
        lines.append("-" * (width + 22))
        for category in ("structural", "payload"):
            for row in self.rows:
                if row.category == category:
                    lines.append(f"{row.name.ljust(width)}  {row.mass:9.2f}  {category}")
            subtotal = (
                self.structural_mass if category == "structural" else self.payload_mass
            )
            lines.append(f"{('  subtotal ' + category).ljust(width)}  {subtotal:9.2f}")
        lines.append("-" * (width + 22))
        lines.append(f"{'TOTAL'.ljust(width)}  {self.total_mass:9.2f}")
        lines.append(f"{'limit'.ljust(width)}  {self.mass_limit:9.2f}")
        lines.append(f"{'margin'.ljust(width)}  {self.margin:9.2f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": [{"name": r.name, "mass": round(r.mass, 4), "category": r.category} for r in self.rows],
            "structural_mass": round(self.structural_mass, 4),
            "payload_mass": round(self.payload_mass, 4),
            "total_mass": round(self.total_mass, 4),
            "mass_limit": self.mass_limit,
            "margin": round(self.margin, 4),
            "ok": self.ok,
        }


def mass_budget(cfg: RoverConfig) -> MassBudget:
    rows: List[MassRow] = [
        MassRow("chassis", cfg.chassis.mass, "structural"),
        MassRow(f"wheels + motors (4 x {cfg.drive.wheel_mass:.2f})",
                4.0 * cfg.drive.wheel_mass, "structural"),
        MassRow("ladder assembly", cfg.excavation.ladder.mass, "structural"),
        MassRow("belt roller", cfg.excavation.belt.mass, "structural"),
        MassRow("tub", cfg.excavation.tub.mass, "structural"),
        MassRow("funnel", cfg.excavation.funnel.mass, "structural"),
    ]
    for item in cfg.payload:
        rows.append(MassRow(item.name, item.mass, "payload"))
    return MassBudget(rows=rows, mass_limit=cfg.mass_limit)
