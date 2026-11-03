"""Read module 1's generated ``markers.yaml``: the wall fiducial catalogue.

``cat_arena`` owns the physical marker boards and their exact arena poses, and
its ``markers.yaml`` carries an ``apriltag_ros``-shaped handoff block.  Module 4
consumes that file only -- it never imports ``cat_arena`` -- and uses it for two
things:

* :mod:`rover_perception.tags` turns it into the runtime ``tags.yaml`` catalogue
  (tag id -> pose in the map frame + physical size) that ``tag_localizer`` needs,
* :mod:`rover_perception.tagboard` turns it into the textured tag boards that
  are spawned into a running simulation, so the tags on the walls are real
  36h11 tags and the detector can genuinely see them.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .geometry import Pose3, quaternion_to_matrix

__all__ = [
    "MarkersError",
    "MarkerBoard",
    "MarkersDoc",
    "resolve_markers_path",
    "load_markers",
]


class MarkersError(ValueError):
    """Raised when ``markers.yaml`` is missing or malformed."""


@dataclass(frozen=True)
class MarkerBoard:
    """One physical marker board, exactly as module 1 placed it."""

    marker_id: int
    family: str
    mount: str
    note: str
    board_size: float
    tag_size: float
    position: Tuple[float, float, float]
    rpy: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    normal: Tuple[float, float, float]
    top_z: float

    @property
    def pose(self) -> Pose3:
        """Board pose in the map frame (its front face looks along ``normal``)."""
        return Pose3.from_rpy(
            self.position[0], self.position[1], self.position[2],
            self.rpy[0], self.rpy[1], self.rpy[2],
        )

    @property
    def standoff_pose(self) -> Pose3:
        """Board pose from the *tag plane* normal, using the quaternion, if given.

        The quaternion and rpy in ``markers.yaml`` describe the same rotation, so
        this is a cross-check rather than a different pose; it exists so a
        mismatch in the emitted file is caught instead of silently trusted.
        """
        return Pose3(quaternion_to_matrix(self.quaternion_xyzw), np.asarray(self.position, dtype=float))


@dataclass(frozen=True)
class MarkersDoc:
    path: str
    seed: int
    family: str
    board_size: float
    tag_size: float
    markers: List[MarkerBoard] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def by_id(self, marker_id: int) -> MarkerBoard:
        for board in self.markers:
            if board.marker_id == marker_id:
                return board
        raise MarkersError(f"marker id {marker_id} not in {self.path}")

    @property
    def ids(self) -> List[int]:
        return [board.marker_id for board in self.markers]


def _default_markers_candidates(explicit: str) -> List[str]:
    candidates: List[str] = []
    if explicit:
        candidates.append(os.path.expanduser(explicit))
    # module 1 generates <cwd>/worlds/seed_<N>/markers.yaml; prefer the newest.
    pattern = os.path.join(os.getcwd(), "worlds", "seed_*", "markers.yaml")
    found = glob.glob(pattern)
    found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    candidates.extend(found)
    home = os.path.expanduser("~")
    candidates.extend(sorted(glob.glob(os.path.join(home, "cat", "worlds", "seed_*", "markers.yaml")),
                             key=lambda p: os.path.getmtime(p), reverse=True))
    return candidates


def resolve_markers_path(explicit: str = "") -> str:
    """Find ``markers.yaml`` (explicit > newest generated world > error).

    An *explicit* path that does not exist is an error rather than a fallback: a
    typo must never silently load another arena's marker layout.
    """
    if explicit:
        requested = os.path.expanduser(explicit)
        if not os.path.isfile(requested):
            raise MarkersError(f"markers.yaml not found at the requested path: {requested}")
        return os.path.normpath(requested)
    candidates = _default_markers_candidates("")
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.normpath(candidate)
    raise MarkersError(
        "markers.yaml not found; generate an arena first "
        "('ros2 run cat_arena generate_arena.py') or pass the path explicitly "
        "(searched: " + ", ".join(c for c in candidates if c) + ")"
    )


def _vec(value: Any, length: int, where: str) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise MarkersError(f"{where}: expected {length} numbers, got {value!r}")
    return tuple(float(v) for v in value)


def load_markers(path: Optional[str] = None) -> MarkersDoc:
    """Load and validate module 1's ``markers.yaml``."""
    import yaml

    resolved = resolve_markers_path(path or "")
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise MarkersError(f"{resolved}: expected a mapping at the top level")
    if int(raw.get("schema_version", 0)) != 1:
        raise MarkersError(f"{resolved}: unsupported schema_version {raw.get('schema_version')!r}")

    boards: List[MarkerBoard] = []
    for index, entry in enumerate(raw.get("markers") or []):
        where = f"{resolved}:markers[{index}]"
        if not isinstance(entry, dict):
            raise MarkersError(f"{where}: expected a mapping")
        quaternion = entry.get("quaternion_xyzw")
        if quaternion is None:
            # accept the apriltag_ros-shaped block when the flat block is absent
            quaternion = [0.0, 0.0, 0.0, 1.0]
        boards.append(
            MarkerBoard(
                marker_id=int(entry["id"]),
                family=str(entry.get("family", raw.get("family", "tag36h11"))),
                mount=str(entry.get("mount", "wall")),
                note=str(entry.get("note", "")),
                board_size=float(entry.get("board_size", raw.get("board_size", 0.25))),
                tag_size=float(entry.get("tag_size", raw.get("tag_size", 0.18))),
                position=_vec(entry["position"], 3, f"{where}.position"),
                rpy=_vec(entry.get("rpy", [0.0, 0.0, 0.0]), 3, f"{where}.rpy"),
                quaternion_xyzw=_vec(quaternion, 4, f"{where}.quaternion_xyzw"),
                normal=_vec(entry.get("normal", [0.0, 0.0, 0.0]), 3, f"{where}.normal"),
                top_z=float(entry.get("top_z", 0.0)),
            )
        )
    if not boards:
        raise MarkersError(f"{resolved}: no markers - is markers.enabled false in arena.yaml?")
    ids = [board.marker_id for board in boards]
    if len(set(ids)) != len(ids):
        raise MarkersError(f"{resolved}: duplicate marker ids {ids}")
    for board in boards:
        if board.tag_size <= 0.0 or board.board_size <= 0.0:
            raise MarkersError(f"{resolved}: marker {board.marker_id} has a non-positive size")
        if board.tag_size > board.board_size + 1e-9:
            raise MarkersError(
                f"{resolved}: marker {board.marker_id} tag_size {board.tag_size} exceeds "
                f"board_size {board.board_size}"
            )
    return MarkersDoc(
        path=resolved,
        seed=int(raw.get("seed", 0)),
        family=str(raw.get("family", "tag36h11")),
        board_size=float(raw.get("board_size", 0.25)),
        tag_size=float(raw.get("tag_size", 0.18)),
        markers=boards,
        raw=raw,
    )
