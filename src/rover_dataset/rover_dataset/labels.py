"""YOLO label I/O -- the wire format between this module and training.

The format is the standard YOLO text format, one ``.txt`` per image::

    <class_id> <cx> <cy> <w> <h>

with all coordinates normalised to ``[0, 1]`` and measured from the top-left
corner.  ``class_id`` is the index into ``classes`` in ``dataset.yaml``; index 0
is a boulder.  The index order is a contract with module 4's detector.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

__all__ = [
    "YoloBox",
    "LabelError",
    "write_labels",
    "read_labels",
    "write_classes",
    "read_classes",
    "write_data_yaml",
    "boxes_to_sidecar",
]


class LabelError(ValueError):
    """Raised when a label file is malformed."""


@dataclass(frozen=True)
class YoloBox:
    """One normalised detection box plus the provenance the sidecar keeps."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    object_id: str = ""
    kind: str = ""
    visible_fraction: float = 1.0
    truncated: bool = False

    # -- pixel helpers ----------------------------------------------------- #
    def pixels(self, image_width: int, image_height: int) -> Tuple[float, float, float, float]:
        x0 = (self.cx - self.w / 2.0) * image_width
        y0 = (self.cy - self.h / 2.0) * image_height
        x1 = (self.cx + self.w / 2.0) * image_width
        y1 = (self.cy + self.h / 2.0) * image_height
        return (x0, y0, x1, y1)

    def area_px(self, image_width: int, image_height: int) -> float:
        return self.w * image_width * self.h * image_height

    @classmethod
    def from_pixels(
        cls,
        class_id: int,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        image_width: int,
        image_height: int,
        **meta: Any,
    ) -> "YoloBox":
        if image_width <= 0 or image_height <= 0:
            raise LabelError("image size must be positive")
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        cx = (x0 + x1) / 2.0 / image_width
        cy = (y0 + y1) / 2.0 / image_height
        w = (x1 - x0) / image_width
        h = (y1 - y0) / image_height
        return cls(class_id, cx, cy, w, h, **meta)

    # -- serialisation ----------------------------------------------------- #
    def to_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}\n"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def validate_box(box: YoloBox, image_width: int, image_height: int, tol: float = 1e-6) -> None:
    """Raise ``LabelError`` if the box is not a legal YOLO box.

    ``tol`` is a *normalised* tolerance; it is scaled to pixels before the
    image-bounds check, because a label is stored with six decimals and a box
    that exactly touches an edge can round to ``-0.0`` after the round trip.
    """
    if box.class_id < 0:
        raise LabelError(f"class_id must be >= 0 (got {box.class_id})")
    for name in ("cx", "cy", "w", "h"):
        value = getattr(box, name)
        if not -tol <= value <= 1.0 + tol:
            raise LabelError(f"{name}={value} outside [0, 1]")
    if box.w <= 0 or box.h <= 0:
        raise LabelError(f"box has non-positive size (w={box.w}, h={box.h})")
    half_w, half_h = box.w / 2.0, box.h / 2.0
    if (
        box.cx - half_w < -tol
        or box.cx + half_w > 1.0 + tol
        or box.cy - half_h < -tol
        or box.cy + half_h > 1.0 + tol
    ):
        x0, y0, x1, y1 = box.pixels(image_width, image_height)
        raise LabelError(f"box ({x0:.1f},{y0:.1f})-({x1:.1f},{y1:.1f}) outside the image")


def write_labels(path: str, boxes: Sequence[YoloBox]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for box in boxes:
            handle.write(box.to_line())


def read_labels(path: str) -> List[YoloBox]:
    if not os.path.isfile(path):
        raise LabelError(f"label file not found: {path}")
    boxes: List[YoloBox] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            parts = text.split()
            if len(parts) != 5:
                raise LabelError(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
            try:
                cls = int(float(parts[0]))
                values = [float(v) for v in parts[1:]]
            except ValueError as exc:
                raise LabelError(f"{path}:{lineno}: non-numeric field") from exc
            boxes.append(YoloBox(cls, *values))
    return boxes


def write_classes(path: str, names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for name in names:
            handle.write(f"{name}\n")


def read_classes(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def write_data_yaml(
    path: str,
    dataset_root: str,
    class_names: Sequence[str],
    *,
    train: str = "images/train",
    val: str = "images/val",
) -> None:
    """Write the ultralytics dataset descriptor."""
    lines = [
        "# generated by rover_dataset -- TRAINING ARTEFACT, never read at run time",
        f"path: {os.path.abspath(dataset_root)}",
        f"train: {train}",
        f"val: {val}",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(class_names))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def boxes_to_sidecar(
    path: str,
    boxes: Iterable[YoloBox],
    dropped: Sequence[Dict[str, Any]],
    meta: Dict[str, Any],
) -> None:
    """Write the per-frame debug/statistics sidecar (not used by training)."""
    payload = {
        "boxes": [box.as_dict() for box in boxes],
        "dropped": list(dropped),
    }
    payload.update(meta)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")


def clamp_normalised(box: YoloBox) -> YoloBox:
    """Clamp a box that only *slightly* exceeds the image (numerical safety)."""
    return YoloBox(
        class_id=box.class_id,
        cx=_clamp01(box.cx),
        cy=_clamp01(box.cy),
        w=_clamp01(box.w),
        h=_clamp01(box.h),
        object_id=box.object_id,
        kind=box.kind,
        visible_fraction=box.visible_fraction,
        truncated=box.truncated,
    )
