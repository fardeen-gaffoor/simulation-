"""The detector interface: what a backend produces, and how it is chosen.

Both backends return the same :class:`Detection` list, so everything downstream
(ground projection, tracking, the wall mask, the obstacle message) is identical
whether the rover is running the bootstrap classical detector or the YOLOv8n
model fine-tuned in module 3.  Switching is one line in ``perception.yaml``.

A ``Detection`` is deliberately *image-space only*: the backend says "there is a
boulder here in these pixels, with this score" and never guesses a map position.
That conversion is :mod:`rover_perception.groundproj`'s job and it is the same
code path for every backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration_ref import CameraModel
from .config import DetectorCfg, OBSTACLE_CLASS_IDS

__all__ = [
    "DetectorError",
    "Detection",
    "Backend",
    "create_backend",
    "annotate",
    "KIND_COLOURS",
]

#: RGB colours used by the debug/preview rendering.
KIND_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "boulder": (0, 255, 0),
    "crater": (255, 0, 255),
    "unknown": (255, 255, 0),
}


class DetectorError(RuntimeError):
    """Raised when a backend cannot be constructed or run."""


@dataclass(frozen=True)
class Detection:
    """One image-space detection."""

    kind: str  # "boulder" | "crater"
    class_id: int  # module 3's YOLO class index (0 = boulder, 1 = crater)
    x0: float
    y0: float
    x1: float
    y1: float
    score: float
    #: True when the backend had to guess (the tracker inflates safety for these)
    conservative: bool = False
    #: backend-specific numbers kept for logging/tuning (never used for geometry)
    features: Dict[str, float] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    @property
    def contact_point(self) -> Tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), self.y1)

    def clamped(self, width: int, height: int) -> "Detection":
        return Detection(
            kind=self.kind,
            class_id=self.class_id,
            x0=float(min(max(self.x0, 0.0), width - 1.0)),
            y0=float(min(max(self.y0, 0.0), height - 1.0)),
            x1=float(min(max(self.x1, 0.0), width - 1.0)),
            y1=float(min(max(self.y1, 0.0), height - 1.0)),
            score=self.score,
            conservative=self.conservative,
            features=dict(self.features),
        )

    def is_degenerate(self) -> bool:
        return self.width <= 1.0 or self.height <= 1.0


class Backend:
    """Backend protocol.  Subclasses implement :meth:`detect` and :meth:`close`."""

    name: str = "base"

    def detect(self, rgb: np.ndarray) -> List[Detection]:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name}

    def close(self) -> None:
        return None


def create_backend(cfg: DetectorCfg, camera: CameraModel) -> Backend:
    """Build the backend named in the config, failing loudly if it cannot run."""
    if not cfg.enabled:
        raise DetectorError("detector.enabled is false; nothing to create")
    if cfg.backend == "classical":
        from .backends.classical import ClassicalDetector

        return ClassicalDetector(cfg.classical, camera)
    if cfg.backend == "onnx":
        from .backends.onnx_yolo import OnnxYoloDetector

        return OnnxYoloDetector(cfg.onnx, camera)
    raise DetectorError(f"unknown detector backend '{cfg.backend}'")


def _class_id_for(kind: str) -> int:
    if kind not in OBSTACLE_CLASS_IDS:
        raise DetectorError(f"unknown obstacle kind '{kind}'")
    return OBSTACLE_CLASS_IDS[kind]


def annotate(rgb: np.ndarray, detections: Sequence[Detection], thickness: int = 2) -> np.ndarray:
    """Draw detections on a copy of an RGB image (debug/preview only)."""
    from .imagecodec import draw_cross, draw_rect

    canvas = np.array(rgb, copy=True)
    if canvas.ndim == 2:
        canvas = np.repeat(canvas[:, :, None], 3, axis=2)
    for detection in detections:
        colour = KIND_COLOURS.get(detection.kind, KIND_COLOURS["unknown"])
        draw_rect(canvas, detection.x0, detection.y0, detection.x1, detection.y1, colour, thickness)
        u, v = detection.contact_point
        draw_cross(canvas, u, v, colour, size=3)
    return canvas
