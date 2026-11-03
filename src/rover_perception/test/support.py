"""Small builders shared by the ``rover_perception`` tests.

Everything here produces *inputs* (a config file, a marker file, a calibration
file, a detection) so a test can state its scenario in one line and still be
reading real, validated artefacts rather than mocks.  No test helper reaches into
private state of the modules under test.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from rover_perception.detector import Detection
from rover_perception.geometry import matrix_to_quaternion, rpy_matrix
from rover_perception.groundproj import GroundObservation
from rover_perception.tracking import Track

# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def write_markers(
    path: str,
    boards: Sequence[Dict[str, Any]],
    *,
    family: str = "tag36h11",
    board_size: float = 0.25,
    tag_size: float = 0.18,
    include_normal: bool = True,
    include_quaternion: bool = True,
    seed: int = 0,
) -> str:
    """Write a module-1-shaped ``markers.yaml`` and return its path.

    ``quaternion_xyzw`` and ``normal`` are derived from ``rpy`` exactly the way
    module 1 derives them (``normal = R[:, 2]``), unless the caller turns them
    off to test the reader's fallbacks.
    """
    entries: List[Dict[str, Any]] = []
    for board in boards:
        rpy = tuple(float(v) for v in board.get("rpy", (0.0, 0.0, 0.0)))
        rotation = rpy_matrix(*rpy)
        entry: Dict[str, Any] = {
            "id": int(board["id"]),
            "family": str(board.get("family", family)),
            "mount": str(board.get("mount", "wall")),
            "note": str(board.get("note", "")),
            "board_size": float(board.get("board_size", board_size)),
            "tag_size": float(board.get("tag_size", tag_size)),
            "position": [float(v) for v in board["position"]],
            "rpy": list(rpy),
            "top_z": float(board.get("top_z", board["position"][2] + board_size / 2.0)),
        }
        if include_quaternion:
            entry["quaternion_xyzw"] = [
                float(v) for v in board.get("quaternion_xyzw", matrix_to_quaternion(rotation))
            ]
        if include_normal:
            entry["normal"] = [float(v) for v in board.get("normal", rotation[:, 2])]
        entries.append(entry)

    document = {
        "schema_version": 1,
        "seed": int(seed),
        "frame": "arena/map frame (origin = west-south arena corner, +x east, +y north)",
        "family": family,
        "board_size": float(board_size),
        "tag_size": float(tag_size),
        "count": len(entries),
        "markers": entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)
    return path


def write_config(path: str, source_path: str, overrides: Optional[Dict[str, Any]] = None) -> str:
    """Copy a YAML config and apply dotted-key overrides (``a.b.c``)."""
    with open(source_path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    for dotted, value in (overrides or {}).items():
        node = document
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)
    return path


def calibration_document(cameras: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A module-2-shaped calibration document from a camera list."""
    out: Dict[str, Any] = {}
    for camera in cameras:
        rpy = tuple(float(v) for v in camera["rpy"])
        rotation_base_link = rpy_matrix(*rpy)
        rotation_link_optical = rpy_matrix(-1.570796327, 0.0, -1.570796327)
        rotation_base_optical = rotation_base_link @ rotation_link_optical

        def rows(matrix) -> List[List[float]]:
            return [[float(value) for value in row] for row in matrix]

        name = str(camera["name"])
        out[name] = {
            "name": name,
            "topic": str(camera.get("topic", f"/{name}")),
            "camera_info_topic": str(camera.get("camera_info_topic", f"/{name}/camera_info")),
            "role": str(camera.get("role", name)),
            "frame_id": str(camera.get("frame_id", f"{name}_optical_frame")),
            "parent_frame": str(camera.get("parent_frame", f"{name}_link")),
            "model": "pinhole",
            "distortion_model": "plumb_bob",
            "image": {
                "width": int(camera["width"]),
                "height": int(camera["height"]),
                "encoding": "rgb8",
            },
            "intrinsics": {
                "fx": float(camera["fx"]),
                "fy": float(camera["fy"]),
                "cx": float(camera["cx"]),
                "cy": float(camera["cy"]),
                "skew": 0.0,
            },
            "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
            "extrinsics": {
                "translation_xyz": [float(v) for v in camera["translation_xyz"]],
                "rotation_rpy_rad": list(rpy),
                "rotation_matrix_base_camera": rows(rotation_base_link),
                "rotation_matrix_base_optical": rows(rotation_base_optical),
            },
        }
    return {
        "schema_version": 1,
        "source": "rover_perception test support",
        "frame": "base_link",
        "declared_forward": "+x",
        "cameras": out,
    }


#: The reference camera set (module 2's shipped ``rover.yaml``), so the geometry
#: tests have fixed, known numbers even when module 2's generator is absent.
REFERENCE_CAMERAS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "cam_nav",
        "role": "nav",
        "width": 640,
        "height": 480,
        "fx": 268.511881977,
        "fy": 268.511881977,
        "cx": 320.0,
        "cy": 240.0,
        "translation_xyz": (0.36, 0.0, 0.58),
        "rpy": (0.0, 0.314159265, 0.0),
    },
    {
        "name": "cam_tag",
        "role": "tag",
        "width": 1280,
        "height": 720,
        "fx": 1004.598769355,
        "fy": 1004.598769355,
        "cx": 640.0,
        "cy": 360.0,
        "translation_xyz": (0.37, 0.0, 0.5),
        "rpy": (0.0, 0.0, 0.0),
    },
    {
        "name": "cam_excavation",
        "role": "excavation",
        "width": 640,
        "height": 480,
        "fx": 320.0,
        "fy": 320.0,
        "cx": 320.0,
        "cy": 240.0,
        "translation_xyz": (0.3, 0.0, 0.42),
        "rpy": (0.0, 0.872664626, 0.0),
    },
    {
        "name": "cam_dump",
        "role": "dump",
        "width": 640,
        "height": 480,
        "fx": 320.0,
        "fy": 320.0,
        "cx": 320.0,
        "cy": 240.0,
        "translation_xyz": (-0.3, 0.0, 0.46),
        "rpy": (0.0, 0.698131701, 3.141592654),
    },
)


def synthetic_calibration(path: str) -> str:
    """Write :data:`REFERENCE_CAMERAS` as a ``calibration.yaml``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(calibration_document(REFERENCE_CAMERAS), handle, sort_keys=False)
    return path


# --------------------------------------------------------------------------- #
# algorithm inputs
# --------------------------------------------------------------------------- #
def detection(
    kind: str = "boulder",
    x0: float = 300.0,
    y0: float = 300.0,
    x1: float = 340.0,
    y1: float = 340.0,
    score: float = 0.9,
    conservative: bool = False,
    **features: float,
) -> Detection:
    """One image-space detection (class id derived from the module-3 contract)."""
    return Detection(
        kind=kind,
        class_id={"boulder": 0, "crater": 1}.get(kind, 255),
        x0=float(x0),
        y0=float(y0),
        x1=float(x1),
        y1=float(y1),
        score=float(score),
        conservative=bool(conservative),
        features={key: float(value) for key, value in features.items()},
    )


def observation(
    x: float,
    y: float,
    kind: str = "boulder",
    radius: float = 0.18,
    score: float = 0.9,
    range_m: float = 3.0,
    height: float = 0.0,
    conservative: bool = False,
) -> GroundObservation:
    """One map-frame ground observation, ready for the tracker."""
    class_id = {"boulder": 0, "crater": 1}.get(kind, 255)
    return GroundObservation(
        kind=kind,
        class_id=class_id,
        x=float(x),
        y=float(y),
        radius=float(radius),
        height=float(height),
        depth=0.0,
        score=float(score),
        range_m=float(range_m),
        bearing_rad=0.0,
        optical_depth_m=float(range_m),
        pixel=(320.0, 400.0),
        bearing_sigma_rad=0.01,
        range_sigma_m=0.1,
        conservative=bool(conservative),
    )


def track(
    x: float,
    y: float,
    kind: str = "boulder",
    radius: float = 0.18,
    identifier: Optional[str] = None,
    confidence: float = 0.9,
    observations: int = 3,
    now: float = 0.0,
    sigma_xy_m: float = 0.12,
    height: float = 0.0,
    depth: float = 0.0,
) -> Track:
    """One tracker track, for the mask/payload/serialisation tests."""
    suffix = str(int(abs(x) * 100)) + "_" + str(int(abs(y) * 100)) if identifier is None else identifier
    return Track(
        id=f"obstacle_{suffix}",
        kind=kind,
        class_id={"boulder": 0, "crater": 1}.get(kind, 255),
        x=float(x),
        y=float(y),
        radius=float(radius),
        height=float(height),
        depth=float(depth),
        score=0.9,
        confidence=float(confidence),
        observations=int(observations),
        created_at=float(now) - 1.0,
        last_seen_at=float(now),
        sigma_xy_m=float(sigma_xy_m),
    )


def tag_canvas(
    tag_ids: Iterable[int],
    *,
    family: str = "tag36h11",
    pixels_per_module: int = 40,
    background: int = 200,
    canvas_size: int = 800,
    columns: int = 2,
) -> np.ndarray:
    """A grey image with rendered tags pasted in, white-quiet-zoned.

    The tags are laid out on a grid with a wide white margin, which is what the
    real boards look like to ``cam_tag``: a decodable bitmap on a light board.
    """
    from rover_perception.tagimages import render_tag

    canvas = np.full((canvas_size, canvas_size), int(background), dtype=np.uint8)
    ids = list(tag_ids)
    if not ids:
        return canvas
    tile = max(8, pixels_per_module * 12)
    rows = int(np.ceil(len(ids) / columns))
    for index, tag_id in enumerate(ids):
        bitmap = render_tag(int(tag_id), family, int(pixels_per_module))
        row, column = divmod(index, columns)
        y0 = 20 + row * (tile + 20)
        x0 = 20 + column * (tile + 20)
        height, width = bitmap.shape
        if y0 + height > canvas_size or x0 + width > canvas_size:
            continue
        # a white quiet zone, then the bitmap
        canvas[y0 - 10 : y0 + height + 10, x0 - 10 : x0 + width + 10] = 255
        canvas[y0 : y0 + height, x0 : x0 + width] = bitmap
        _ = rows
    return canvas
