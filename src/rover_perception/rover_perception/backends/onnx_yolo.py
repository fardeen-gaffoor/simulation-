"""The learned detector: a YOLOv8n ONNX model on the Pi's CPU.

Module 3 produced the dataset (boulder = 0, crater = 1) and fine-tunes YOLOv8n;
this backend consumes the exported ONNX graph.  The pre-processing (letterbox
with the YOLO grey pad), the output decoding and the NMS are implemented here in
numpy rather than delegated to ``ultralytics``, for three reasons:

* ``ultralytics`` pulls in torch and OpenCV, and OpenCV is exactly the package
  that cannot load against this machine's numpy (see ``docs/detector.md``);
* the Pi has no GPU, so the graph runs under ``onnxruntime``'s CPU provider -
  the deployment target - not under torch;
* the whole decode path stays unit-testable with a *stub* session, which is how
  ``test/test_onnx_backend.py`` verifies it without a trained model present.

Nothing here runs unless ``detector.backend: onnx`` and
``detector.onnx.model`` point at a real file.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..calibration_ref import CameraModel
from ..config import OnnxCfg
from ..detector import Backend, Detection, DetectorError

__all__ = [
    "resize_bilinear",
    "letterbox",
    "nms",
    "decode_predictions",
    "OnnxYoloDetector",
    "is_available",
    "install_hint",
]

_INSTALL_HINT = "pip install --user onnxruntime   (see docs/detector.md)"


def is_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def install_hint() -> str:
    return _INSTALL_HINT


def resize_bilinear(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of an ``(h, w)`` or ``(h, w, c)`` array to float32."""
    src = np.asarray(image, dtype=np.float32)
    if src.ndim not in (2, 3):
        raise ValueError(f"resize_bilinear: expected 2-D or 3-D input, got {src.shape}")
    if src.shape[0] == out_h and src.shape[1] == out_w:
        return src.copy()
    src_h, src_w = src.shape[0], src.shape[1]
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (src_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / out_w) - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, src_h - 1)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = np.clip(ys - y0, 0.0, 1.0)[:, None]
    wx = np.clip(xs - x0, 0.0, 1.0)[None, :]
    if src.ndim == 3:
        wy = wy[:, :, None]
        wx = wx[:, :, None]
    top = src[y0][:, x0] * (1.0 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1.0 - wx) + src[y1][:, x1] * wx
    return (top * (1.0 - wy) + bottom * wy).astype(np.float32)


def letterbox(
    image: np.ndarray, out_size: int, pad: int = 114
) -> Tuple[np.ndarray, float, int, int]:
    """Resize ``image`` to fit ``out_size`` and centre-pad it to a square.

    Returns ``(padded, scale, dx, dy)`` where ``padded`` is
    ``(out_size, out_size, 3)`` uint8 and ``(dx, dy)`` is the top-left offset of
    the resized image inside it - the inverse map used to put boxes back.
    """
    src = np.asarray(image)
    if src.ndim == 2:
        src = np.repeat(src[:, :, None], 3, axis=2)
    if src.shape[2] > 3:
        src = src[:, :, :3]
    src_h, src_w = src.shape[0], src.shape[1]
    if src_h <= 0 or src_w <= 0:
        raise ValueError("letterbox: empty image")
    scale = min(out_size / float(src_h), out_size / float(src_w))
    new_h = max(1, int(round(src_h * scale)))
    new_w = max(1, int(round(src_w * scale)))
    resized = resize_bilinear(src, new_h, new_w)
    canvas = np.full((out_size, out_size, 3), float(pad), dtype=np.float32)
    dx = (out_size - new_w) // 2
    dy = (out_size - new_h) // 2
    canvas[dy : dy + new_h, dx : dx + new_w, :] = resized
    return canvas.astype(np.uint8), float(scale), int(dx), int(dy)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression; returns the kept indices."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        best = int(order[0])
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[best, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[best, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[best, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0.0, None) * np.clip(yy1 - yy0, 0.0, None)
        area_best = (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_best + area_rest - inter
        iou = np.where(union > 0.0, inter / np.maximum(union, 1e-12), 0.0)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_predictions(
    output: np.ndarray,
    confidence: float,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8 head into ``(boxes_xyxy, scores, class_ids)``.

    Accepts both ``(1, 4 + nc, N)`` and ``(1, N, 4 + nc)`` layouts, because the
    export shape depends on how the graph was written.  Boxes come back in
    *letterbox pixels*; the caller maps them back to image pixels.
    """
    array = np.asarray(output, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] != 1:
            raise ValueError(f"decode_predictions: expected batch 1, got {array.shape}")
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"decode_predictions: expected a 2-D head, got {array.shape}")
    expected = 4 + num_classes
    if array.shape[0] == expected:
        array = array.T
    elif array.shape[1] != expected:
        raise ValueError(
            f"decode_predictions: head has shape {array.shape}, expected {expected} channels"
        )
    boxes_cxcywh = array[:, :4]
    class_scores = array[:, 4 : 4 + num_classes]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    keep = scores >= confidence
    boxes_cxcywh, scores, class_ids = boxes_cxcywh[keep], scores[keep], class_ids[keep]
    if boxes_cxcywh.shape[0] == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=np.int64)
    cx, cy, width, height = (
        boxes_cxcywh[:, 0],
        boxes_cxcywh[:, 1],
        boxes_cxcywh[:, 2],
        boxes_cxcywh[:, 3],
    )
    boxes = np.stack(
        [cx - 0.5 * width, cy - 0.5 * height, cx + 0.5 * width, cy + 0.5 * height], axis=1
    )
    return boxes, scores, class_ids.astype(np.int64)


class OnnxYoloDetector(Backend):
    """YOLOv8n/YOLO11-style ONNX detector with a numpy decode path."""

    name = "onnx"

    def __init__(self, cfg: OnnxCfg, camera: CameraModel, session: Any = None):
        self.cfg = cfg
        self.camera = camera
        self.num_classes = len(cfg.classes)
        self._session = session
        self._input_name: Optional[str] = None
        if self._session is None:
            self._session = self._open_session()

    # -- setup ------------------------------------------------------------- #
    def _open_session(self) -> Any:
        model = os.path.expanduser(self.cfg.model)
        if not model:
            raise DetectorError(
                "detector.onnx.model is empty: export a model first (module 3's "
                "finetune.py) or set detector.backend: classical"
            )
        if not os.path.isfile(model):
            raise DetectorError(f"ONNX model not found: {model}")
        try:
            import onnxruntime
        except Exception as exc:
            raise DetectorError(
                f"onnxruntime is not installed ({exc}); {_INSTALL_HINT}"
            ) from exc
        session = onnxruntime.InferenceSession(model, providers=list(self.cfg.providers))
        inputs = session.get_inputs()
        if not inputs:
            raise DetectorError(f"{model}: the ONNX graph has no inputs")
        self._input_name = inputs[0].name
        return session

    def describe(self) -> Dict[str, object]:
        return {
            "backend": self.name,
            "model": os.path.expanduser(self.cfg.model),
            "input_size": self.cfg.input_size,
            "providers": list(self.cfg.providers),
            "confidence": self.cfg.confidence,
            "iou": self.cfg.iou,
            "classes": list(self.cfg.classes),
            "available": is_available(),
        }

    # -- inference --------------------------------------------------------- #
    def _infer(self, blob: np.ndarray) -> np.ndarray:
        if self._input_name is None:
            inputs = self._session.get_inputs()
            self._input_name = inputs[0].name
        outputs = self._session.run(None, {self._input_name: blob})
        if not outputs:
            raise DetectorError("the ONNX session returned no outputs")
        return np.asarray(outputs[0])

    def detect(self, rgb: np.ndarray) -> List[Detection]:
        padded, scale, dx, dy = letterbox(rgb, self.cfg.input_size, self.cfg.letterbox_pad)
        blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None, :, :, :].astype(np.float32) / 255.0)
        raw = self._infer(blob)
        boxes, scores, class_ids = decode_predictions(raw, self.cfg.confidence, self.num_classes)
        if boxes.shape[0] == 0:
            return []
        # per-class NMS: a boulder and a crater may legitimately overlap
        keep_indices: List[int] = []
        for class_id in np.unique(class_ids):
            same = np.flatnonzero(class_ids == class_id)
            kept = nms(boxes[same], scores[same], self.cfg.iou)
            keep_indices.extend(same[kept].tolist())
        keep_indices.sort(key=lambda index: -float(scores[index]))
        keep_indices = keep_indices[: self.cfg.max_detections]

        width, height = self.camera.width, self.camera.height
        detections: List[Detection] = []
        for index in keep_indices:
            x0 = (boxes[index, 0] - dx) / scale
            y0 = (boxes[index, 1] - dy) / scale
            x1 = (boxes[index, 2] - dx) / scale
            y1 = (boxes[index, 3] - dy) / scale
            kind = self.cfg.classes[int(class_ids[index])]
            detection = Detection(
                kind=kind,
                class_id=int(class_ids[index]),
                x0=float(x0),
                y0=float(y0),
                x1=float(x1),
                y1=float(y1),
                score=float(scores[index]),
                conservative=False,
                features={"letterbox_scale": scale, "pad_x": float(dx), "pad_y": float(dy)},
            )
            detection = detection.clamped(width, height)
            if detection.is_degenerate():
                continue
            detections.append(detection)
        return detections

    def close(self) -> None:
        """Drop the session (onnxruntime frees the graph on the last reference)."""
        self._session = None
