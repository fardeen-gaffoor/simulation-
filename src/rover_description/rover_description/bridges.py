"""The ``ros_gz_bridge`` topic table (sim <-> ROS interface for the rover).

Kept as data (``config/bridges.yaml``) plus a tiny loader so the launch files,
the tests and the documentation all agree on exactly which ROS topics exist in
simulation.  On the real robot the same ROS topics are produced by
``rover_camera_driver`` / the drive firmware, so nothing downstream changes.

``/clock`` is deliberately NOT listed: ``cat_arena``'s ``arena.launch.py`` owns
the clock bridge (``bridge_clock:=true``) for the whole simulation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .config import RoverConfigError, package_share_dir

__all__ = [
    "GZ_TO_ROS",
    "ROS_TO_GZ",
    "default_bridges_path",
    "load_bridges",
    "bridge_arguments",
]

GZ_TO_ROS = "gz_to_ros"
ROS_TO_GZ = "ros_to_gz"

_SEP = {GZ_TO_ROS: "[", ROS_TO_GZ: "]"}

_REQUIRED = ("ros_topic", "gz_topic", "ros_type", "gz_type", "direction")


def default_bridges_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    relative = os.path.join("config", "bridges.yaml")
    candidates = []
    share = package_share_dir()
    if share:
        candidates.append(os.path.join(share, relative))
    candidates.append(os.path.join(here, "..", relative))
    candidates.append(os.path.join(os.path.dirname(here), "share", "rover_description", relative))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    raise RoverConfigError("cannot locate config/bridges.yaml")


def load_bridges(path: Optional[str] = None) -> List[Dict[str, str]]:
    import yaml

    resolved = os.path.abspath(os.path.expanduser(path)) if path else default_bridges_path()
    if not os.path.isfile(resolved):
        raise RoverConfigError(f"bridge config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    entries = doc.get("bridges") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RoverConfigError(f"{resolved}: expected a non-empty 'bridges' list")

    out: List[Dict[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise RoverConfigError(f"{resolved}: bridges[{index}] must be a mapping")
        for key in _REQUIRED:
            if key not in entry:
                raise RoverConfigError(f"{resolved}: bridges[{index}] missing '{key}'")
        direction = str(entry["direction"])
        if direction not in _SEP:
            raise RoverConfigError(
                f"{resolved}: bridges[{index}].direction must be {GZ_TO_ROS!r} or {ROS_TO_GZ!r}"
            )
        if str(entry["ros_topic"]) != str(entry["gz_topic"]):
            raise RoverConfigError(
                f"{resolved}: bridges[{index}] ros_topic and gz_topic must match "
                "(the rover uses identical names on both sides)"
            )
        out.append({key: str(entry[key]) for key in _REQUIRED})
    return out


def bridge_arguments(bridges: List[Dict[str, str]]) -> List[str]:
    """``parameter_bridge`` command-line arguments for a bridge table."""
    return [
        f"{b['ros_topic']}@{b['ros_type']}{_SEP[b['direction']]}{b['gz_type']}"
        for b in bridges
    ]
