"""Small builders shared by the ``rover_navigation`` tests.

Everything here produces *inputs* - a patched config file, an obstacle record, a
grid of them - so a test can state its scenario in one line and still be working
with real, validated objects.  Nothing reaches into the private state of the
modules under test.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from rover_navigation.costmap import KIND_BOULDER, KIND_CRATER, ObstacleRecord
from rover_navigation.geometry import Pose3

__all__ = [
    "boulder",
    "crater",
    "records_from",
    "point_obstacle",
    "config_with",
    "obstacle_line",
    "pose",
]


# --------------------------------------------------------------------------- #
# obstacles
# --------------------------------------------------------------------------- #
def boulder(
    x: float,
    y: float,
    radius: float = 0.17,
    *,
    identifier: str = "",
    confidence: float = 0.9,
    observations: int = 10,
    conservative: bool = False,
) -> ObstacleRecord:
    """A confirmed boulder, as module 4 would report it."""
    return ObstacleRecord(
        identifier=identifier or f"boulder_{abs(hash((x, y))) % 10000}",
        kind=KIND_BOULDER,
        x=float(x),
        y=float(y),
        radius=float(radius),
        confidence=float(confidence),
        observations=int(observations),
        conservative=conservative,
    )


def crater(
    x: float,
    y: float,
    radius: float = 0.22,
    *,
    identifier: str = "",
    confidence: float = 0.9,
    observations: int = 10,
) -> ObstacleRecord:
    """A confirmed crater (lethal: module 4 publishes craters conservatively)."""
    return ObstacleRecord(
        identifier=identifier or f"crater_{abs(hash((x, y))) % 10000}",
        kind=KIND_CRATER,
        x=float(x),
        y=float(y),
        radius=float(radius),
        confidence=float(confidence),
        observations=int(observations),
        conservative=True,
    )


def records_from(pairs: Iterable[Tuple[float, float, float]], kind: int = KIND_BOULDER) -> List[ObstacleRecord]:
    """``[(x, y, radius), ...]`` -> obstacle records of one kind."""
    out: List[ObstacleRecord] = []
    for index, (x, y, radius) in enumerate(pairs):
        out.append(
            ObstacleRecord(
                identifier=f"obstacle_{index}",
                kind=kind,
                x=float(x),
                y=float(y),
                radius=float(radius),
                confidence=0.9,
                observations=10,
            )
        )
    return out


def obstacle_line(
    start: Tuple[float, float],
    end: Tuple[float, float],
    count: int,
    radius: float = 0.17,
    kind: int = KIND_BOULDER,
) -> List[ObstacleRecord]:
    """``count`` obstacles evenly spaced between two points (a wall of them)."""
    if count <= 1:
        return [
            ObstacleRecord("obstacle_0", kind, float(start[0]), float(start[1]), radius, 0.9, 10)
        ]
    out: List[ObstacleRecord] = []
    for index in range(count):
        fraction = index / (count - 1)
        out.append(
            ObstacleRecord(
                identifier=f"obstacle_{index}",
                kind=kind,
                x=start[0] + (end[0] - start[0]) * fraction,
                y=start[1] + (end[1] - start[1]) * fraction,
                radius=radius,
                confidence=0.9,
                observations=10,
            )
        )
    return out


def point_obstacle(x: float, y: float, radius: float = 0.2, kind: int = KIND_BOULDER) -> ObstacleRecord:
    return ObstacleRecord("one", kind, float(x), float(y), float(radius), 0.9, 10)


# --------------------------------------------------------------------------- #
# config files
# --------------------------------------------------------------------------- #
def _apply(document: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = document
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def config_with(base_path: str, tmp_path: Any, overrides: Optional[Dict[str, Any]] = None) -> Any:
    """Load the shipped config with dotted-path overrides applied.

    ``config_with(path, tmp, {"robot.max_speed_mps": 0.2})`` writes a patched copy
    of ``navigation.yaml`` into ``tmp_path`` and returns the validated config.
    The shipped file is never modified, so the override cannot leak into another
    test.
    """
    from rover_navigation.config import load_config

    with open(base_path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document = copy.deepcopy(document)
    for key, value in (overrides or {}).items():
        _apply(document, key, value)
    target = os.path.join(str(tmp_path), "navigation_override.yaml")
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)
    return load_config(target)


def pose(x: float, y: float, yaw: float = 0.0) -> Pose3:
    return Pose3.from_xy_yaw(x, y, yaw)
