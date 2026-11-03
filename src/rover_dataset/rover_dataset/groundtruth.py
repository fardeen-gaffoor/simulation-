"""Read module 1's ``ground_truth.yaml``.

This file is the *source* of the automatic labels, and it is exactly the file
the competition rules forbid the flight software from reading.  Module 3 reads
it offline, at dataset-build time, and nothing it writes is ever consumed by
``rover_perception``/``rover_navigation`` at run time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = ["GroundTruthError", "Boulder", "Crater", "GroundTruth", "load_ground_truth"]


class GroundTruthError(ValueError):
    """Raised when ground_truth.yaml is missing or malformed."""


@dataclass(frozen=True)
class Boulder:
    id: str
    center: Tuple[float, float, float]
    radius: float
    diameter: float
    exposed_height: float
    top_z: float
    zone: str

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.center[0], self.center[1])


@dataclass(frozen=True)
class Crater:
    id: str
    center: Tuple[float, float]
    radius: float
    width: float
    depth: float
    zone: str

    @property
    def xy(self) -> Tuple[float, float]:
        return self.center


@dataclass(frozen=True)
class GroundTruth:
    path: str
    seed: int
    world_name: str
    length_x: float
    width_y: float
    spawn_z_offset: float
    boulders: List[Boulder]
    craters: List[Crater]
    zones: Dict[str, Any] = field(default_factory=dict)
    start_pose: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    raw: Dict[str, Any] = field(default_factory=dict)

    def objects(self) -> List[Tuple[str, Any]]:
        """Every labellable object as ``(kind, object)`` in a stable order."""
        out: List[Tuple[str, Any]] = [("boulder", b) for b in self.boulders]
        out.extend(("crater", c) for c in self.craters)
        return out

    @property
    def obstacle_xy(self) -> List[Tuple[float, float]]:
        return [b.xy for b in self.boulders] + [c.xy for c in self.craters]


def _f(value: Any, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise GroundTruthError(f"{where}: expected a number, got {value!r}") from exc


def _vec(value: Any, where: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise GroundTruthError(f"{where}: expected {length} numbers, got {value!r}")
    return tuple(_f(v, where) for v in value)


def load_ground_truth(path: str) -> GroundTruth:
    """Parse the ground truth written by ``cat_arena.generate``."""
    import yaml

    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise GroundTruthError(f"ground truth not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise GroundTruthError(f"{resolved}: expected a mapping at the top level")
    if int(raw.get("schema_version", 0)) != 1:
        raise GroundTruthError(f"{resolved}: unsupported schema_version {raw.get('schema_version')}")

    arena = raw.get("arena") or {}
    boulders: List[Boulder] = []
    for i, entry in enumerate(raw.get("boulders") or []):
        where = f"boulders[{i}]"
        boulders.append(
            Boulder(
                id=str(entry.get("id", f"boulder_{i}")),
                center=_vec(entry["center"], f"{where}.center", 3),  # type: ignore[arg-type]
                radius=_f(entry["radius"], f"{where}.radius"),
                diameter=_f(entry.get("diameter", 2.0 * float(entry["radius"])), f"{where}.diameter"),
                exposed_height=_f(entry.get("exposed_height", 0.0), f"{where}.exposed_height"),
                top_z=_f(entry.get("top_z", 0.0), f"{where}.top_z"),
                zone=str(entry.get("zone", "")),
            )
        )
    craters: List[Crater] = []
    for i, entry in enumerate(raw.get("craters") or []):
        where = f"craters[{i}]"
        craters.append(
            Crater(
                id=str(entry.get("id", f"crater_{i}")),
                center=_vec(entry["center"], f"{where}.center", 2),  # type: ignore[arg-type]
                radius=_f(entry["radius"], f"{where}.radius"),
                width=_f(entry.get("width", 2.0 * float(entry["radius"])), f"{where}.width"),
                depth=_f(entry.get("depth", 0.0), f"{where}.depth"),
                zone=str(entry.get("zone", "")),
            )
        )

    sp = raw.get("start_pose") or {}
    pos = sp.get("position") or [0.0, 0.0, 0.0]
    start_pose = (_f(pos[0], "start_pose.position"), _f(pos[1], "start_pose.position"),
                  _f(pos[2], "start_pose.position"), _f(sp.get("yaw_rad", 0.0), "start_pose.yaw_rad"))

    if not boulders and not craters:
        raise GroundTruthError(f"{resolved}: no boulders or craters; is this a real ground truth?")

    return GroundTruth(
        path=resolved,
        seed=int(raw.get("seed", 0)),
        world_name=str(raw.get("world_name", "cat_arena")),
        length_x=_f(arena.get("length_x", 9.0), "arena.length_x"),
        width_y=_f(arena.get("width_y", 5.0), "arena.width_y"),
        spawn_z_offset=_f(arena.get("spawn_z_offset", 0.08), "arena.spawn_z_offset"),
        boulders=boulders,
        craters=craters,
        zones=dict(raw.get("zones") or {}),
        start_pose=start_pose,  # type: ignore[arg-type]
        raw=raw,
    )
