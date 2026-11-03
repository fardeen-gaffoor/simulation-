"""The classical bootstrap detector: what is not sand, and which way it is lit.

This backend exists so that module 5 can be developed and tested *today*, before
NVIDIA training hardware and a fine-tuned model are in the loop, and it stays
useful afterwards as a cross-check on the learned detector.  It is honest about
what it is: a hand-tuned segmenter, not a classifier with a published mAP.  See
``docs/detector.md`` for its measured behaviour and its known failure modes.

Algorithm, all numpy, no OpenCV:

1. **Sand reference.**  Within the ground region (everything below the horizon
   row that the camera geometry puts at ``cy - fy * R[2,2] / R[2,1]``) take the
   per-channel *median* so a minority of dark rocks cannot drag the reference,
   and the median absolute deviation as a robust noise scale.
2. **Anomaly mask.**  A pixel is an anomaly when it is darker than the sand by
   ``max(delta_luma, mad_k * MAD)`` or when its chroma (r-g, g-b) is far from the
   sand's.  Boulders in the arena are grey rock on light tilled sand; craters are
   shading, which is exactly a luminance anomaly.
3. **Blobs.**  One pass of run-length connected components (8-connectivity, union
   find) gives every connected anomaly with its area, box and fill ratio.
4. **Boulder or crater?**  A boulder is convex and lit from above, so its top is
   *brighter* than its bottom; a crater is concave, so its far interior is
   shadowed and its near rim is lit, i.e. the top is *darker*.  Blobs whose
   shading is too weak to decide are ambiguous and are classified as
   ``ambiguous_kind`` (craters by default) with ``conservative=True``: a needless
   detour around sand is far cheaper than driving into a hole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..calibration_ref import CameraModel
from ..config import ClassicalCfg
from ..detector import Backend, Detection
from ..groundproj import horizon_row
from ..imagecodec import to_gray

__all__ = ["ClassicalDetector", "row_runs", "label_components", "Region", "Blob", "extract_blobs"]


def row_runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """Inclusive ``(start, end)`` runs of True in a boolean row."""
    flat = np.asarray(row, dtype=bool).reshape(-1)
    padded = np.concatenate(([False], flat, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(changes[0::2].tolist(), (changes[1::2] - 1).tolist()))


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """8-connected component labels of a boolean mask (0 = background).

    Run-length + union-find: one Python loop over the rows, vectorised inside
    each row, which is fast enough for 640x480 at 15 Hz and has no dependencies.
    """
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"label_components: expected a 2-D mask, got {array.shape}")
    height, width = array.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent: List[int] = []

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:  # path compression
            parent[index], index = root, parent[index]
        return root

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    runs: List[Tuple[int, int, int, int]] = []  # y, x0, x1, label
    previous: List[Tuple[int, int, int]] = []
    for y in range(height):
        current: List[Tuple[int, int, int]] = []
        for x0, x1 in row_runs(array[y]):
            index = len(parent)
            parent.append(index)
            runs.append((y, x0, x1, index))
            # 8-connectivity: runs of adjacent rows overlap when expanded by 1
            for px0, px1, pindex in previous:
                if px0 <= x1 + 1 and x0 <= px1 + 1:
                    union(index, pindex)
            current.append((x0, x1, index))
        previous = current

    roots: Dict[int, int] = {}
    for y, x0, x1, index in runs:
        root = find(index)
        label = roots.setdefault(root, len(roots) + 1)
        labels[y, x0 : x1 + 1] = label
    return labels, len(roots)


@dataclass(frozen=True)
class Region:
    """The part of the image the detector is allowed to look at."""

    row0: int
    row1: int
    col0: int
    col1: int

    @property
    def empty(self) -> bool:
        return self.row1 <= self.row0 or self.col1 <= self.col0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.row0, self.row1, self.col0, self.col1)


@dataclass(frozen=True)
class Blob:
    """One connected anomaly region with the features used to classify it."""

    label: int
    area: int
    x0: int
    y0: int
    x1: int
    y1: int
    fill: float
    aspect: float
    bbox_fraction: float
    mean_grey: float
    grey_top: float
    grey_bottom: float
    touches_edge: bool

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def shading_delta(self) -> float:
        """``grey_top - grey_bottom``: positive = lit from above (convex)."""
        return self.grey_top - self.grey_bottom

    def as_features(self) -> Dict[str, float]:
        return {
            "area_px": float(self.area),
            "fill": round(self.fill, 4),
            "aspect": round(self.aspect, 4),
            "bbox_fraction": round(self.bbox_fraction, 4),
            "mean_grey": round(self.mean_grey, 2),
            "grey_top": round(self.grey_top, 2),
            "grey_bottom": round(self.grey_bottom, 2),
            "shading_delta": round(self.shading_delta, 2),
        }


def region_for(cfg: ClassicalCfg, camera: CameraModel) -> Region:
    """The ground region: below the horizon, clipped by the configured fractions."""
    height, width = camera.height, camera.width
    horizon = horizon_row(camera)
    top = 0 if horizon is None else int(np.ceil(horizon + cfg.horizon_margin_px))
    row0 = max(0, min(height, top))
    row1 = max(row0, min(height, int(round(cfg.bottom_fraction * height))))
    col0 = max(0, min(width, int(round(cfg.left_fraction * width))))
    col1 = max(col0, min(width, int(round(cfg.right_fraction * width))))
    return Region(row0, row1, col0, col1)


def _channels(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    image = np.asarray(rgb)
    if image.ndim == 2:
        grey = image.astype(np.float32)
        chroma = np.zeros(grey.shape + (2,), dtype=np.float32)
        return grey, chroma
    if image.shape[2] < 3:
        grey = image[:, :, 0].astype(np.float32)
        chroma = np.zeros(grey.shape + (2,), dtype=np.float32)
        return grey, chroma
    red = image[:, :, 0].astype(np.float32)
    green = image[:, :, 1].astype(np.float32)
    blue = image[:, :, 2].astype(np.float32)
    grey = 0.299 * red + 0.587 * green + 0.114 * blue
    chroma = np.stack([red - green, green - blue], axis=2)
    return grey, chroma


def anomaly_mask(
    grey: np.ndarray,
    chroma: np.ndarray,
    region: Region,
    cfg: ClassicalCfg,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Boolean mask of "not sand" inside the region, plus the reference stats."""
    window_grey = grey[region.row0 : region.row1, region.col0 : region.col1]
    window_chroma = chroma[region.row0 : region.row1, region.col0 : region.col1]
    if window_grey.size == 0:
        return np.zeros(grey.shape, dtype=bool), {"reference_grey": 0.0, "mad": 0.0}

    reference = float(np.median(window_grey))
    mad = float(np.median(np.abs(window_grey - reference))) * 1.4826
    if window_chroma.size:
        ref_chroma = np.median(window_chroma.reshape(-1, window_chroma.shape[2]), axis=0)
    else:  # pragma: no cover - mono images only
        ref_chroma = np.zeros(2, dtype=np.float32)

    luma_limit = max(cfg.delta_luma, cfg.mad_k * mad)
    darker = grey < (reference - luma_limit)
    if window_chroma.size:
        chroma_delta = np.sqrt(np.sum((chroma - ref_chroma) ** 2, axis=2))
        off_colour = chroma_delta > cfg.delta_chroma
    else:  # pragma: no cover - mono images only
        off_colour = np.zeros(grey.shape, dtype=bool)

    mask = np.zeros(grey.shape, dtype=bool)
    mask[region.row0 : region.row1, region.col0 : region.col1] = (
        darker | off_colour
    )[region.row0 : region.row1, region.col0 : region.col1]
    stats = {
        "reference_grey": round(reference, 2),
        "mad": round(mad, 2),
        "luma_limit": round(luma_limit, 2),
        "anomaly_px": float(np.count_nonzero(mask)),
    }
    return mask, stats


def extract_blobs(
    labels: np.ndarray,
    count: int,
    grey: np.ndarray,
    region: Region,
    cfg: ClassicalCfg,
) -> List[Blob]:
    """Features for every component that passes the shape gates."""
    height, width = grey.shape
    blobs: List[Blob] = []
    if count == 0:
        return blobs
    for label in range(1, count + 1):
        rows, cols = np.nonzero(labels == label)
        area = int(rows.size)
        if area < cfg.min_area_px or area > cfg.max_area_px:
            continue
        y0, y1 = int(rows.min()), int(rows.max())
        x0, x1 = int(cols.min()), int(cols.max())
        box_w = x1 - x0 + 1
        box_h = y1 - y0 + 1
        fill = area / float(box_w * box_h)
        aspect = max(box_w / box_h, box_h / box_w)
        bbox_fraction = (box_w * box_h) / float(width * height)
        if fill < cfg.min_fill or aspect > cfg.max_aspect or bbox_fraction > cfg.max_bbox_fraction:
            continue
        values = grey[rows, cols]
        midline = y0 + 0.5 * (y1 - y0)
        top_mask = rows <= midline
        grey_top = float(values[top_mask].mean()) if np.any(top_mask) else float(values.mean())
        grey_bottom = float(values[~top_mask].mean()) if np.any(~top_mask) else float(values.mean())
        touches_edge = bool(
            y0 <= region.row0
            or y1 >= region.row1 - 1
            or x0 <= region.col0
            or x1 >= region.col1 - 1
        )
        blobs.append(
            Blob(
                label=label,
                area=area,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fill=fill,
                aspect=float(aspect),
                bbox_fraction=float(bbox_fraction),
                mean_grey=float(values.mean()),
                grey_top=grey_top,
                grey_bottom=grey_bottom,
                touches_edge=touches_edge,
            )
        )
    blobs.sort(key=lambda blob: blob.area, reverse=True)
    return blobs


def classify_blob(
    blob: Blob, reference_grey: float, cfg: ClassicalCfg
) -> Tuple[str, float, bool]:
    """``(kind, score, conservative)`` for one blob.

    The score is a *ranking* heuristic in [0.30, 1.0], not a calibrated
    probability; the tracker's accumulated confidence is what consumers see.
    """
    delta = blob.shading_delta
    if delta >= cfg.shadow_delta:
        kind, conservative = "boulder", False
    elif delta <= -cfg.shadow_delta:
        kind, conservative = "crater", False
    else:
        if cfg.ambiguous_kind == "drop":
            return ("", 0.0, True)
        kind, conservative = cfg.ambiguous_kind, True

    darkness = float(np.clip((reference_grey - blob.mean_grey) / max(cfg.delta_luma, 1e-6), 0.0, 2.0)) / 2.0
    fill_term = float(np.clip((blob.fill - cfg.min_fill) / max(1.0 - cfg.min_fill, 1e-6), 0.0, 1.0))
    shade_term = float(np.clip(abs(delta) / max(2.0 * cfg.shadow_delta, 1e-6), 0.0, 1.0))
    score = 0.30 + 0.35 * darkness + 0.20 * fill_term + 0.15 * shade_term
    if blob.touches_edge:
        score *= 0.8
        conservative = True
    return kind, float(min(1.0, score)), conservative


class ClassicalDetector(Backend):
    """Numpy-only boulder/crater segmenter (see the module docstring)."""

    name = "classical"

    def __init__(self, cfg: ClassicalCfg, camera: CameraModel):
        self.cfg = cfg
        self.camera = camera
        self.region = region_for(cfg, camera)
        self.last_stats: Dict[str, float] = {}

    def describe(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "region": self.region.as_tuple(),
            "horizon_row": horizon_row(self.camera),
            "min_area_px": self.cfg.min_area_px,
            "ambiguous_kind": self.cfg.ambiguous_kind,
        }

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        image = np.asarray(rgb)
        if image.ndim == 3 and image.shape[2] == 3:
            grey, chroma = _channels(image)
        else:
            grey = to_gray(image) if image.dtype != np.uint8 else image.astype(np.float32)
            chroma = np.zeros(grey.shape + (2,), dtype=np.float32)
        if self.region.empty:
            self.last_stats = {"region_empty": 1.0}
            return []
        mask, stats = anomaly_mask(grey, chroma, self.region, self.cfg)
        labels, count = label_components(mask)
        blobs = extract_blobs(labels, count, grey, self.region, self.cfg)
        stats["blobs"] = float(len(blobs))

        detections: List[Detection] = []
        for blob in blobs:
            kind, score, conservative = classify_blob(blob, stats["reference_grey"], self.cfg)
            if not kind:
                continue
            detections.append(
                Detection(
                    kind=kind,
                    class_id=0 if kind == "boulder" else 1,
                    x0=float(blob.x0),
                    y0=float(blob.y0),
                    x1=float(blob.x1) + 1.0,
                    y1=float(blob.y1) + 1.0,
                    score=score,
                    conservative=conservative,
                    features=blob.as_features(),
                )
            )
        detections.sort(key=lambda detection: detection.score, reverse=True)
        detections = detections[: self.cfg.max_detections]
        stats["detections"] = float(len(detections))
        self.last_stats = stats
        return detections
