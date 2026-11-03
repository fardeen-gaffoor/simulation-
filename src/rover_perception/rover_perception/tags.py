"""``tags.yaml``: the runtime tag catalogue (tag id -> pose in the map frame).

``cat_arena`` emits ``markers.yaml`` (the *boards*).  Module 4 emits this file
(the *tags*), because the tag a detector sees is not the board: it is the
textured plane module 4 draws on it, offset by
``tags.board.standoff_m`` in front of the board's centre plane.  Recording that
exact pose is what keeps localisation unbiased - the localizer solves for the
rover from *this* pose, so the standoff is accounted for instead of assumed
away.

The catalogue is a generated artefact, like module 2's ``calibration.yaml``: the
nodes read it at runtime (``scripts/generate_tags.py`` writes it, and
``perception.launch.py`` points the nodes at it), and its provenance
(``markers.yaml`` path + sha256) is recorded so a run can be audited.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import TagBoardCfg
from .geometry import Pose3, matrix_to_quaternion, rotation_matrix_to_rpy, rpy_matrix
from .markers_ref import MarkerBoard, MarkersDoc
from .tagimages import drawing_size_m, png_name, render

__all__ = [
    "TagsError",
    "TagSpec",
    "TagCatalogue",
    "tag_frame_rotation",
    "build_document",
    "write_catalogue",
    "load_tags",
    "default_tags_path",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1

#: World "up", used to orient a tag frame around the axis it faces along.
WORLD_UP = (0.0, 0.0, 1.0)


def tag_frame_rotation(normal: np.ndarray, up: Tuple[float, float, float] = WORLD_UP) -> np.ndarray:
    """The tag frame a detector reports, given the direction the face points.

    **This is the convention that matters and it was measured, not assumed.**
    AprilTag's ``T_cam_tag`` has the tag's ``+z`` pointing *into* the tag (away
    from the viewer), ``+x`` to the viewer's right and ``+y`` down - which is
    exactly the optical frame of an observer standing in front of the board.  A
    head-on tag gives ``R == I`` from ``dt_apriltags`` (see
    ``test_tagdetect.py::test_head_on_tag_faces_back_at_the_camera``).

    So for a board whose visible face points along ``normal`` (module 1's
    ``normal``, which points into the arena):

    * ``z = -normal`` (into the board, away from the viewer),
    * ``x = view_forward x up`` (the viewer's right, where ``view_forward = -normal``),
    * ``y = z x x`` (the viewer's down; equals ``-up`` for a vertical wall).

    Getting this wrong rotates every fix by ~180 degrees, so it is derived here
    once and asserted against the reference boards by ``test_tags.py``.
    """
    into_board = -np.asarray(normal, dtype=float).reshape(3)
    length = float(np.linalg.norm(into_board))
    if length < 1e-9:
        raise TagsError("tag_frame_rotation: the board normal is a zero vector")
    z_axis = into_board / length
    up_vec = np.asarray(up, dtype=float).reshape(3)
    x_axis = np.cross(z_axis, up_vec)
    if float(np.linalg.norm(x_axis)) < 1e-9:
        # a tag facing straight up or down has no wall to hang on; pick any
        # stable perpendicular so the frame is still a valid rotation
        fallback = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, fallback)
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if not np.all(np.isfinite(rotation)):
        raise TagsError("tag_frame_rotation: produced a non-finite rotation")
    return rotation


class TagsError(ValueError):
    """Raised when ``tags.yaml`` is missing or malformed."""


@dataclass(frozen=True)
class TagSpec:
    """One tag, exactly as it is rendered and as the localizer must treat it."""

    id: int
    family: str
    #: edge length of the black square (what the detector's ``tag_size`` means)
    size_m: float
    #: edge length of the whole bitmap including its quiet zone
    drawing_size_m: float
    #: origin of the rendered tag plane in the map frame
    position: Tuple[float, float, float]
    #: the tag frame a detector reports (``+z`` points *into* the board)
    rpy: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]
    #: the direction the tag's visible face points (into the arena, = ``-pose z``)
    normal: Tuple[float, float, float]
    mount: str
    note: str
    #: where module 1 centred the *board* (before the standoff was applied)
    board_position: Tuple[float, float, float]
    standoff_m: float
    texture: str = ""

    @property
    def pose(self) -> Pose3:
        """The tag frame in the map frame (what a detected pose is relative to)."""
        return Pose3(rpy_matrix(*self.rpy), np.asarray(self.position, dtype=float))

    @property
    def face_normal(self) -> np.ndarray:
        """The direction the tag's visible face points (``-pose`` ``+z``)."""
        return -self.pose.rotation[:, 2]

    @property
    def half_size_m(self) -> float:
        return 0.5 * self.size_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "size_m": round(self.size_m, 6),
            "drawing_size_m": round(self.drawing_size_m, 6),
            "position": [round(v, 6) for v in self.position],
            "rpy": [round(v, 9) for v in self.rpy],
            "quaternion_xyzw": [round(v, 9) for v in self.quaternion_xyzw],
            "normal": [round(float(v), 9) for v in self.face_normal],
            "frame_z": [round(float(v), 9) for v in self.pose.rotation[:, 2]],
            "board_position": [round(v, 6) for v in self.board_position],
            "standoff_m": round(self.standoff_m, 6),
            "mount": self.mount,
            "note": self.note,
            "texture": self.texture,
        }


@dataclass(frozen=True)
class TagCatalogue:
    """The loaded catalogue: what the tag detector and the localizer read."""

    path: str
    frame_id: str
    family: str
    tags: Dict[int, TagSpec] = field(default_factory=dict)
    source_markers: str = ""
    source_sha256: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def tag(self, tag_id: int) -> TagSpec:
        if tag_id not in self.tags:
            raise TagsError(
                f"tag id {tag_id} is not in {self.path} (known: {sorted(self.tags)})"
            )
        return self.tags[tag_id]

    def has(self, tag_id: int) -> bool:
        return tag_id in self.tags

    @property
    def ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self.tags))

    @property
    def count(self) -> int:
        return len(self.tags)

    def size_m(self, tag_id: int) -> float:
        return self.tag(tag_id).size_m

    def summary(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "frame_id": self.frame_id,
            "family": self.family,
            "count": self.count,
            "ids": list(self.ids),
            "source_markers": self.source_markers,
        }


def _sha256_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tag_spec(
    board: MarkerBoard,
    cfg: TagBoardCfg,
    family: str,
    texture: str = "",
) -> TagSpec:
    """The rendered tag plane: board pose pushed out along its normal.

    ``position`` is where the printed surface ends up (board centre + standoff
    along the board normal), and ``rpy`` is the *tag frame* from
    :func:`tag_frame_rotation` - not the board's own rpy.  They differ: module 1
    orients the board so its ``+z`` is the visible normal, while AprilTag's frame
    has ``+z`` pointing into the board.  Using the board rpy here would rotate
    every localisation fix by ~180 degrees.

    The normal is taken from ``markers.yaml`` when present (and normalised), and
    derived from ``rpy`` otherwise, because a hand-edited file could omit the
    key; ``test_markers_ref.py`` checks the two agree whenever both are present.
    """
    normal = np.asarray(board.normal, dtype=float)
    if float(np.linalg.norm(normal)) < 1e-9:
        normal = rpy_matrix(*board.rpy)[:, 2]
    normal = normal / float(np.linalg.norm(normal))
    board_position = np.asarray(board.position, dtype=float)
    position = board_position + normal * cfg.standoff_m
    rotation = tag_frame_rotation(normal)
    rpy = rotation_matrix_to_rpy(rotation)
    return TagSpec(
        id=int(board.marker_id),
        family=family or board.family,
        size_m=float(board.tag_size),
        drawing_size_m=drawing_size_m(float(board.tag_size), family or board.family, cfg.quiet_zone_modules),
        position=(float(position[0]), float(position[1]), float(position[2])),
        rpy=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
        quaternion_xyzw=matrix_to_quaternion(rotation),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        mount=board.mount,
        note=board.note,
        board_position=(float(board_position[0]), float(board_position[1]), float(board_position[2])),
        standoff_m=float(cfg.standoff_m),
        texture=texture,
    )


def build_document(
    markers: MarkersDoc,
    cfg: TagBoardCfg,
    frame_id: str = "map",
    family: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the ``tags.yaml`` document from module 1's ``markers.yaml``."""
    resolved_family = family or markers.family or "tag36h11"
    tags: List[Dict[str, Any]] = []
    for board in markers.markers:
        spec = _tag_spec(board, cfg, resolved_family, texture=png_name(board.marker_id, resolved_family))
        tags.append(spec.to_dict())
    return {
        "schema_version": SCHEMA_VERSION,
        "frame": frame_id,
        "family": resolved_family,
        "source": {
            "markers": markers.path,
            "markers_sha256": _sha256_file(markers.path),
            "markers_seed": markers.seed,
            "standoff_m": round(cfg.standoff_m, 6),
            "protrusion_m": round(cfg.protrusion_m, 6),
            "lift_m": round(cfg.lift_m, 6),
            "quiet_zone_modules": cfg.quiet_zone_modules,
        },
        "count": len(tags),
        "tags": tags,
    }


def write_catalogue(path: str, document: Dict[str, Any]) -> str:
    """Write ``tags.yaml`` (deterministic key order, so diffs are meaningful)."""
    import yaml

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# rover_perception tag catalogue - generated, do not edit\n")
        handle.write("# Source of truth: cat_arena's markers.yaml + tags.board.* in perception.yaml\n")
        yaml.safe_dump(document, handle, sort_keys=False, default_flow_style=False)
    return path


def default_tags_path(output_dir: str, filename: str = "tags.yaml") -> str:
    """Where the catalogue lives when nothing else is configured."""
    if output_dir:
        return os.path.join(os.path.expanduser(output_dir), filename)
    return os.path.join(os.getcwd(), filename)


def _vec(value: Any, length: int, where: str) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise TagsError(f"{where}: expected {length} numbers, got {value!r}")
    return tuple(float(v) for v in value)


def load_tags(path: str) -> TagCatalogue:
    """Load and validate the generated ``tags.yaml``."""
    import yaml

    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise TagsError(
            f"tag catalogue not found: {resolved} "
            "(generate it with 'ros2 run rover_perception generate_tags.py')"
        )
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TagsError(f"{resolved}: expected a mapping at the top level")
    if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
        raise TagsError(f"{resolved}: unsupported schema_version {raw.get('schema_version')!r}")
    family = str(raw.get("family", "tag36h11"))
    tags: Dict[int, TagSpec] = {}
    for index, entry in enumerate(raw.get("tags") or []):
        where = f"{resolved}:tags[{index}]"
        if not isinstance(entry, dict):
            raise TagsError(f"{where}: expected a mapping")
        spec = TagSpec(
            id=int(entry["id"]),
            family=str(entry.get("family", family)),
            size_m=float(entry.get("size_m", 0.0)),
            drawing_size_m=float(entry.get("drawing_size_m", 0.0)),
            position=_vec(entry["position"], 3, f"{where}.position"),
            rpy=_vec(entry.get("rpy", [0.0, 0.0, 0.0]), 3, f"{where}.rpy"),
            quaternion_xyzw=_vec(
                entry.get("quaternion_xyzw", matrix_to_quaternion(rpy_matrix(*entry.get("rpy", [0.0, 0.0, 0.0])))),
                4,
                f"{where}.quaternion_xyzw",
            ),
            normal=_vec(entry.get("normal", [0.0, 0.0, 0.0]), 3, f"{where}.normal"),
            mount=str(entry.get("mount", "")),
            note=str(entry.get("note", "")),
            board_position=_vec(entry.get("board_position", entry["position"]), 3, f"{where}.board_position"),
            standoff_m=float(entry.get("standoff_m", 0.0)),
            texture=str(entry.get("texture", "")),
        )
        if spec.size_m <= 0.0:
            raise TagsError(f"{where}: size_m must be > 0")
        if spec.id in tags:
            raise TagsError(f"{where}: duplicate tag id {spec.id}")
        tags[spec.id] = spec
    if not tags:
        raise TagsError(f"{resolved}: no tags")
    return TagCatalogue(
        path=resolved,
        frame_id=str(raw.get("frame", "map")),
        family=family,
        tags=tags,
        source_markers=str((raw.get("source") or {}).get("markers", "")),
        source_sha256=str((raw.get("source") or {}).get("markers_sha256", "")),
        raw=raw,
    )
