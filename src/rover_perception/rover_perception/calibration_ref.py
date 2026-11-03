"""Read module 2's generated ``calibration.yaml``: the ground projection's input.

Module 4 never infers camera geometry.  It reads the calibration that
``rover_description`` generated next to the URDF, so the camera the simulation
renders with and the camera perception thinks it is looking through can never
disagree.  The same file shape is used for the real CSI cameras once they are
measured, so nothing downstream changes when the hardware arrives.

Only numpy is imported here; ROS is not needed to validate a calibration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "CalibrationError",
    "CameraModel",
    "CalibrationDoc",
    "resolve_calibration_path",
    "load_calibration",
]

#: Optical-frame rotation relative to the camera link (REP-103/105).
OPTICAL_FRAME_RPY: Tuple[float, float, float] = (-1.5707963267948966, 0.0, -1.5707963267948966)

_REQUIRED_CAMERAS = ("cam_nav", "cam_tag")


class CalibrationError(ValueError):
    """Raised when ``calibration.yaml`` is missing or malformed."""


@dataclass(frozen=True)
class CameraModel:
    """One camera, exactly in the terms the projection needs."""

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
    #: camera link origin in base_link (metres)
    t_base_cam: Tuple[float, float, float]
    #: R_base_optical, row-major 9 floats: the optical axes in base_link
    r_base_optical: Tuple[float, ...]

    @property
    def image_size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def r_base_optical_matrix(self) -> np.ndarray:
        return np.asarray(self.r_base_optical, dtype=float).reshape(3, 3)

    @property
    def t_base_cam_vector(self) -> np.ndarray:
        return np.asarray(self.t_base_cam, dtype=float)

    @property
    def k(self) -> np.ndarray:
        """The 3x3 pinhole matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float
        )

    @property
    def hfov_rad(self) -> float:
        import math

        return 2.0 * math.atan(0.5 * self.width / self.fx)

    @property
    def vfov_rad(self) -> float:
        import math

        return 2.0 * math.atan(0.5 * self.height / self.fy)

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Unit-less optical-frame ray ``(x/z, y/z, 1)`` through pixel ``(u, v)``."""
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise CalibrationError(f"{self.name}: non-positive focal length")
        return np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0], dtype=float)


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

    def has(self, name: str) -> bool:
        return name in self.cameras


def _default_generated_dir() -> Optional[str]:
    """``rover_description``'s artefact directory, if module 2 is importable."""
    try:
        from rover_description.generate import default_output_dir

        return default_output_dir()
    except Exception:
        return None


def _share_dir(package: str) -> Optional[str]:
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory(package)
    except Exception:
        return None


def resolve_calibration_path(explicit: str = "") -> str:
    """Find ``calibration.yaml``: explicit > module 2's artefact dir > install > source.

    An *explicit* path that does not exist is an error rather than a fallback: a
    typo must never silently load a different machine's calibration.
    """
    candidates: List[str] = []
    if explicit:
        requested = os.path.expanduser(explicit)
        if not os.path.isfile(requested):
            raise CalibrationError(f"calibration.yaml not found at the requested path: {requested}")
        return os.path.normpath(requested)
    generated = _default_generated_dir()
    if generated:
        candidates.append(os.path.join(generated, "calibration.yaml"))
    share = _share_dir("rover_description")
    if share:
        candidates.append(os.path.join(share, "config", "calibration.yaml"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.extend(
        [
            os.path.abspath(os.path.join(here, "..", "..", "rover_description", "config", "calibration.yaml")),
            os.path.join(os.getcwd(), "calibration.yaml"),
        ]
    )
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return os.path.normpath(candidate)
    raise CalibrationError(
        "calibration.yaml not found; run 'ros2 run rover_description generate_robot' first "
        "(searched: " + ", ".join(c for c in candidates if c) + ")"
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
    """Load and validate ``calibration.yaml``."""
    import yaml

    resolved = resolve_calibration_path(path or "")
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise CalibrationError(f"{resolved}: expected a mapping at the top level")
    if int(raw.get("schema_version", 0)) != 1:
        raise CalibrationError(f"{resolved}: unsupported schema_version {raw.get('schema_version')!r}")

    cameras: Dict[str, CameraModel] = {}
    for name, entry in (raw.get("cameras") or {}).items():
        where = f"cameras.{name}"
        intr = entry.get("intrinsics") or {}
        extr = entry.get("extrinsics") or {}
        image = entry.get("image") or {}
        for key in ("fx", "fy", "cx", "cy"):
            if intr.get(key) is None:
                raise CalibrationError(f"{where}: intrinsics.{key} missing")
        if not extr.get("translation_xyz") or not extr.get("rotation_matrix_base_optical"):
            raise CalibrationError(f"{where}: extrinsics incomplete")
        model = CameraModel(
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
                extr["rotation_matrix_base_optical"],
                f"{where}.extrinsics.rotation_matrix_base_optical",
            ),
        )
        if model.width <= 0 or model.height <= 0:
            raise CalibrationError(f"{where}: image width/height missing")
        if model.fx <= 0.0 or model.fy <= 0.0:
            raise CalibrationError(f"{where}: focal length must be positive")
        cameras[model.name] = model
    if not cameras:
        raise CalibrationError(f"{resolved}: no cameras")
    missing = [name for name in _REQUIRED_CAMERAS if name not in cameras]
    if missing:
        raise CalibrationError(
            f"{resolved}: module 4 needs {list(_REQUIRED_CAMERAS)}, missing {missing}"
        )
    return CalibrationDoc(
        path=resolved,
        schema_version=int(raw["schema_version"]),
        frame=str(raw.get("frame", "base_link")),
        declared_forward=str(raw.get("declared_forward", "+x")),
        cameras=cameras,
        raw=raw,
    )
