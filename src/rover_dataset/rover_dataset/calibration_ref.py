"""Read module 2's ``calibration.yaml`` -- the only file the projection needs.

Module 3 never *infers* camera geometry: it reads the calibration that
``rover_description`` generated alongside the URDF, so the images and the
ground-truth projection can never disagree.  For a randomized scene the
calibration of *that* scene is used (see :mod:`rover_dataset.scene`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "CalibrationError",
    "CameraModel",
    "CalibrationDoc",
    "resolve_calibration_path",
    "load_calibration",
    "camera_by_name",
]

#: Optical-frame rotation relative to the camera link (REP-103/105).
OPTICAL_FRAME_RPY: Tuple[float, float, float] = (-1.5707963267948966, 0.0, -1.5707963267948966)


class CalibrationError(ValueError):
    """Raised when calibration.yaml is missing or malformed."""


@dataclass(frozen=True)
class CameraModel:
    """One camera, in exactly the terms the projection needs."""

    name: str
    topic: str
    camera_info_topic: str
    frame_id: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    #: camera link origin expressed in base_link (metres)
    t_base_cam: Tuple[float, float, float]
    #: R_base_optical, row-major 9 floats: optical axes expressed in base_link
    r_base_optical: Tuple[float, ...]

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass(frozen=True)
class CalibrationDoc:
    path: str
    schema_version: int
    frame: str
    declared_forward: str
    cameras: Dict[str, CameraModel] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def camera(self, name: str) -> CameraModel:
        if name not in self.cameras:
            raise CalibrationError(
                f"camera '{name}' not in calibration ({sorted(self.cameras)})"
            )
        return self.cameras[name]


def _rover_share_dir() -> Optional[str]:
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory("rover_description")
    except Exception:
        return None


def resolve_calibration_path(explicit: str = "") -> str:
    """Find ``calibration.yaml`` (explicit path > install space > source tree)."""
    candidates: List[str] = []
    if explicit:
        candidates.append(os.path.expanduser(explicit))
    share = _rover_share_dir()
    if share:
        candidates.append(os.path.join(share, "config", "calibration.yaml"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "..", "..", "rover_description", "config", "calibration.yaml"))
    candidates.append(
        os.path.join(here, "..", "..", "rover_description", "rover_final", "calibration.yaml")
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    raise CalibrationError(
        "calibration.yaml not found; run 'ros2 run rover_description generate_robot' first "
        "(searched: " + ", ".join(candidates) + ")"
    )


def _flatten_matrix(rows: Any, where: str) -> Tuple[float, ...]:
    if not isinstance(rows, (list, tuple)) or len(rows) != 3:
        raise CalibrationError(f"{where}: expected a 3x3 matrix")
    out: List[float] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise CalibrationError(f"{where}: expected a 3x3 matrix")
        out.extend(float(v) for v in row)
    return tuple(out)


def load_calibration(path: Optional[str] = None) -> CalibrationDoc:
    import yaml

    resolved = resolve_calibration_path(path or "")
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise CalibrationError(f"{resolved}: expected a mapping at the top level")
    if int(raw.get("schema_version", 0)) != 1:
        raise CalibrationError(f"{resolved}: unsupported schema_version {raw.get('schema_version')}")

    cameras: Dict[str, CameraModel] = {}
    for name, entry in (raw.get("cameras") or {}).items():
        where = f"cameras.{name}"
        intr = entry.get("intrinsics") or {}
        extr = entry.get("extrinsics") or {}
        image = entry.get("image") or {}
        if not intr.get("fx") or not intr.get("fy"):
            raise CalibrationError(f"{where}: intrinsics.fx/fy missing")
        cameras[str(name)] = CameraModel(
            name=str(entry.get("name", name)),
            topic=str(entry.get("topic", f"/{name}")),
            camera_info_topic=str(entry.get("camera_info_topic", f"/{name}/camera_info")),
            frame_id=str(entry.get("frame_id", f"{name}_optical_frame")),
            width=int(image.get("width", 0)),
            height=int(image.get("height", 0)),
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            cx=float(intr["cx"]),
            cy=float(intr["cy"]),
            t_base_cam=tuple(float(v) for v in extr["translation_xyz"]),  # type: ignore[arg-type]
            r_base_optical=_flatten_matrix(
                extr["rotation_matrix_base_optical"], f"{where}.extrinsics.rotation_matrix_base_optical"
            ),
        )
    if not cameras:
        raise CalibrationError(f"{resolved}: no cameras")
    return CalibrationDoc(
        path=resolved,
        schema_version=int(raw["schema_version"]),
        frame=str(raw.get("frame", "base_link")),
        declared_forward=str(raw.get("declared_forward", "+x")),
        cameras=cameras,
        raw=raw,
    )


def camera_by_name(doc: CalibrationDoc, name: str) -> CameraModel:
    return doc.camera(name)
