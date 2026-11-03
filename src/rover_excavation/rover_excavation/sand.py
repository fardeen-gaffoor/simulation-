"""The sand: what is in the tub, and what has been built into the berm.

Gazebo cannot simulate dry sand, and the arena's terrain is a static mesh, so
module 6 *models* the sand instead of moving it.  Two objects do that:

``Tub``
    the collection bin.  Sand enters while digging and leaves while dumping;
    its fill fraction is what ends a dig cycle and what is reported on
    ``MechanismStatus``.

``BermField``
    a height field over ``arena.zones.berm_target``.  This object *is* the berm:
    the simulator has no other record of it, ``BermStatus`` is a view of it, and
    the rviz marker / Gazebo box are pictures of it.

The deposition model is deliberately simple and physical.  A tub of sand
emptied behind a stationary rover cannot land under the rover, so the pile
starts at the aft face and its centre ends up one basal radius further back
(see :func:`rover_excavation.config.repose_basal_radius_m`).  From there the
material spreads as a Gaussian of width ``sand.dump.spread_m``.  How much of
that Gaussian falls inside the target rectangle is computed **analytically**
(two normal CDFs), so the spilled fraction is exact rather than a pixel count,
and the deposited volume is distributed over the cells in proportion to the
Gaussian.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import ExcavationConfig, TubCfg

__all__ = ["Tub", "BermField", "DepositResult"]

#: How many standard deviations of the deposit kernel are drawn on the field.
KERNEL_SIGMAS = 3.0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class Tub:
    """The collection bin: a volume with a capacity, nothing else."""

    def __init__(self, config: TubCfg, volume_m3: float = 0.0) -> None:
        self.capacity_m3 = float(config.capacity_m3)
        self.volume_m3 = max(0.0, min(float(volume_m3), self.capacity_m3))

    @property
    def fraction(self) -> float:
        if self.capacity_m3 <= 0.0:  # pragma: no cover - guarded by the loader
            return 0.0
        return self.volume_m3 / self.capacity_m3

    @property
    def is_full(self) -> bool:
        return self.volume_m3 >= self.capacity_m3 - 1e-9

    @property
    def is_empty(self) -> bool:
        return self.volume_m3 <= 1e-9

    def add(self, volume_m3: float) -> float:
        """Pour sand in; return how much actually fit (the rest spills over)."""
        if volume_m3 <= 0.0:
            return 0.0
        room = self.capacity_m3 - self.volume_m3
        accepted = min(room, volume_m3)
        self.volume_m3 += accepted
        return accepted

    def remove(self, volume_m3: float) -> float:
        """Take sand out; return how much actually left."""
        if volume_m3 <= 0.0:
            return 0.0
        removed = min(self.volume_m3, volume_m3)
        self.volume_m3 -= removed
        return removed

    def empty(self) -> None:
        self.volume_m3 = 0.0

    def fill(self) -> None:
        self.volume_m3 = self.capacity_m3


@dataclass(frozen=True)
class DepositResult:
    """What one tub emptied at one place did to the berm."""

    placed_m3: float
    spilled_m3: float
    inside_fraction: float
    center_x: float
    center_y: float
    peak_height_m: float


class BermField:
    """A height field over the berm target: the berm itself."""

    def __init__(self, config: ExcavationConfig) -> None:
        self.config = config
        self.rect = config.berm_target
        self.cell_m = float(config.sand.berm.cell_m)
        self.nx = max(1, int(math.ceil(self.rect.width / self.cell_m - 1e-9)))
        self.ny = max(1, int(math.ceil(self.rect.height / self.cell_m - 1e-9)))
        self.dx = self.rect.width / self.nx
        self.dy = self.rect.height / self.ny
        self.cell_area = self.dx * self.dy
        self.height = np.zeros((self.ny, self.nx), dtype=float)
        self._spilled_m3 = 0.0
        self.dumps = 0
        xs = self.rect.x0 + (np.arange(self.nx, dtype=float) + 0.5) * self.dx
        ys = self.rect.y0 + (np.arange(self.ny, dtype=float) + 0.5) * self.dy
        self._xs, self._ys = np.meshgrid(xs, ys)

    # -- geometry ----------------------------------------------------------- #
    @property
    def cells_total(self) -> int:
        return self.nx * self.ny

    @property
    def target_height_m(self) -> float:
        return float(self.config.sand.berm.target_height_m)

    @property
    def target_volume_m3(self) -> float:
        return self.config.berm_target_volume_m3

    # -- deposition --------------------------------------------------------- #
    def begin_dump(self) -> None:
        """Latch that a new pile is starting (one per work cycle, not per tick).

        A dump takes several seconds and the sand is added to the field as it
        leaves the tub, so the berm grows smoothly; but the *count* of dumps has
        to be the number of work cycles, not the number of control ticks.
        """
        self.dumps += 1

    def deposit(self, x: float, y: float, volume_m3: float) -> DepositResult:
        """Empty ``volume_m3`` of sand at map point ``(x, y)``.

        Returns how much landed inside the target and how much missed it.
        """
        if volume_m3 <= 0.0:
            return DepositResult(0.0, 0.0, 1.0, x, y, self.peak_height_m)
        sigma = float(self.config.sand.dump.spread_m)
        fx = _normal_cdf((self.rect.x1 - x) / sigma) - _normal_cdf((self.rect.x0 - x) / sigma)
        fy = _normal_cdf((self.rect.y1 - y) / sigma) - _normal_cdf((self.rect.y0 - y) / sigma)
        inside = max(0.0, min(1.0, fx * fy))
        placed = volume_m3 * inside
        spilled = volume_m3 - placed

        radius = KERNEL_SIGMAS * sigma
        r2 = (self._xs - x) ** 2 + (self._ys - y) ** 2
        weights = np.exp(-r2 / (2.0 * sigma * sigma))
        weights[r2 > radius * radius] = 0.0
        total = float(weights.sum())
        if total > 0.0 and placed > 0.0:
            self.height += (placed * weights / total) / self.cell_area
        else:
            # Nothing of the kernel reached the field: it is all spillage.
            spilled = volume_m3
            placed = 0.0
        self._spilled_m3 += spilled
        return DepositResult(
            placed_m3=placed,
            spilled_m3=spilled,
            inside_fraction=inside,
            center_x=float(x),
            center_y=float(y),
            peak_height_m=self.peak_height_m,
        )

    def reset(self) -> None:
        self.height.fill(0.0)
        self._spilled_m3 = 0.0
        self.dumps = 0

    # -- reports ------------------------------------------------------------ #
    @property
    def placed_volume_m3(self) -> float:
        return float(self.height.sum()) * self.cell_area

    @property
    def spilled_volume_m3(self) -> float:
        return self._spilled_m3

    @property
    def mean_height_m(self) -> float:
        return float(self.height.mean()) if self.height.size else 0.0

    @property
    def peak_height_m(self) -> float:
        return float(self.height.max()) if self.height.size else 0.0

    @property
    def cells_filled(self) -> int:
        return int(np.count_nonzero(self.height >= self.target_height_m - 1e-9))

    @property
    def coverage(self) -> float:
        if self.cells_total == 0:  # pragma: no cover - nx, ny >= 1
            return 0.0
        return self.cells_filled / float(self.cells_total)

    @property
    def volume_fraction(self) -> float:
        if self.target_volume_m3 <= 0.0:  # pragma: no cover - guarded by the loader
            return 0.0
        return self.placed_volume_m3 / self.target_volume_m3

    @property
    def completion(self) -> float:
        """0..1: the fraction of the berm's *volume* that has been placed.

        The judge's yardstick is the berm's size, so the score is the volume
        placed inside the footprint over the target volume.  ``coverage`` is
        reported alongside it because a tall pile on one spot has the volume but
        not the shape; a *flat* prism over the whole target needs roughly 1.4 x
        the target volume (a real pile is not flat), so coverage saturates below
        one for a physical deposition - that is a statement about the dump
        pattern, not a defect of the model.
        """
        return max(0.0, min(1.0, self.volume_fraction))

    # -- views -------------------------------------------------------------- #
    def cells(self) -> List[Tuple[float, float, float]]:
        """``(x, y, height)`` for every cell, for markers and diagnostics."""
        out: List[Tuple[float, float, float]] = []
        for j in range(self.ny):
            for i in range(self.nx):
                out.append((float(self._xs[j, i]), float(self._ys[j, i]), float(self.height[j, i])))
        return out

    def summary(self) -> str:
        return (
            f"{self.placed_volume_m3:.4f}/{self.target_volume_m3:.4f} m3 "
            f"({100.0 * self.completion:.0f}%), coverage {100.0 * self.coverage:.0f}%, "
            f"peak {self.peak_height_m:.3f} m, spilled {self.spilled_volume_m3:.4f} m3"
        )
