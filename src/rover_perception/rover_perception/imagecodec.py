"""``sensor_msgs/Image`` <-> numpy, without cv_bridge and without OpenCV.

``cv_bridge`` imports ``cv2``, and on this machine apt's OpenCV 4.5.4 cannot
load against the numpy 2.x that pip installed (``_ARRAY_API not found``), so
module 4 does its own conversions - exactly the same trade module 3 made when it
chose numpy + Pillow for the dataset.  Nothing here imports ROS either: the
message is handled duck-typed, so the unit tests drive it with a stub.

Supported encodings are the ones this stack actually produces: the four RGB
cameras (``rgb8``) plus the usual mono/16-bit/float variants so a future sensor
does not need a new codec.  Row padding (``step``) is honoured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "ImageCodecError",
    "ENCODINGS",
    "encoding_info",
    "payload_bytes",
    "to_array",
    "to_gray",
    "to_rgb",
    "array_to_bytes",
    "build_image",
    "draw_rect",
    "draw_filled_rect",
    "draw_cross",
]

#: encoding -> (numpy dtype, channels)
ENCODINGS: Dict[str, Tuple[str, int]] = {
    "rgb8": ("uint8", 3),
    "bgr8": ("uint8", 3),
    "rgba8": ("uint8", 4),
    "bgra8": ("uint8", 4),
    "mono8": ("uint8", 1),
    "mono16": ("uint16", 1),
    "16UC1": ("uint16", 1),
    "32FC1": ("float32", 1),
}

#: ITU-R BT.601 luma weights, in the fixed-point form used for uint8 RGB.
_LUMA_R = np.array([0.299, 0.587, 0.114], dtype=np.float64)


class ImageCodecError(ValueError):
    """Raised for an unsupported encoding or a truncated/inconsistent message."""


def encoding_info(encoding: str) -> Tuple[str, int]:
    if encoding not in ENCODINGS:
        raise ImageCodecError(f"unsupported encoding '{encoding}' (known: {sorted(ENCODINGS)})")
    return ENCODINGS[encoding]


def payload_bytes(width: int, height: int, encoding: str) -> int:
    """Exact byte count of an un-padded image with this encoding."""
    dtype_name, channels = encoding_info(encoding)
    return int(width) * int(height) * channels * np.dtype(dtype_name).itemsize


def _shape_and_dtype(width: int, height: int, encoding: str) -> Tuple[Tuple[int, ...], np.dtype]:
    name, channels = encoding_info(encoding)
    shape: Tuple[int, ...] = (height, width) if channels == 1 else (height, width, channels)
    return shape, np.dtype(name)


def to_array(image: Any) -> np.ndarray:
    """Decode a ``sensor_msgs/Image``-shaped object into a numpy array.

    The result is ``(height, width)`` for one-channel encodings and
    ``(height, width, channels)`` otherwise, always in row-major order with the
    image's own top-left origin.
    """
    height = int(image.height)
    width = int(image.width)
    encoding = str(image.encoding)
    shape, dtype = _shape_and_dtype(width, height, encoding)
    channels = 1 if len(shape) == 2 else shape[2]
    itemsize = dtype.itemsize
    row_bytes = width * channels * itemsize
    step = int(getattr(image, "step", row_bytes)) or row_bytes
    data = image.data
    raw = np.frombuffer(data, dtype=np.uint8)
    if height <= 0 or width <= 0:
        raise ImageCodecError(f"degenerate image {width}x{height}")
    if step < row_bytes:
        raise ImageCodecError(f"step {step} < row bytes {row_bytes} for {width}x{height} {encoding}")
    if raw.size < height * step:
        raise ImageCodecError(
            f"image payload {raw.size} bytes is smaller than {height} rows of {step} bytes"
        )
    if step == row_bytes:
        array = np.frombuffer(raw.data[: height * row_bytes], dtype=dtype)
    else:
        padded = raw[: height * step].reshape(height, step)
        array = padded[:, :row_bytes].copy().view(dtype)
    return array.reshape(shape)


def _to_uint8_luma(array: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "mono8":
        return array
    if encoding in ("mono16", "16UC1"):
        return (array.astype(np.uint32) * 255 // 65535).astype(np.uint8)
    if encoding == "32FC1":
        # No camera in this stack produces float images; normalise 0..1 so a
        # future sensor at least gives a usable image instead of an overflow.
        return np.clip(np.nan_to_num(array, nan=0.0) * 255.0, 0.0, 255.0).astype(np.uint8)
    if encoding in ("rgb8", "rgba8"):
        rgb = array[:, :, :3].astype(np.float64)
    elif encoding in ("bgr8", "bgra8"):
        rgb = array[:, :, 2::-1].astype(np.float64)
    else:  # pragma: no cover - encoding_info already raised
        raise ImageCodecError(f"cannot convert '{encoding}' to grey")
    return np.clip(rgb @ _LUMA_R + 0.5, 0, 255).astype(np.uint8)


def to_gray(image: Any) -> np.ndarray:
    """Decode any supported image straight to a ``uint8`` (h, w) grey array."""
    return _to_uint8_luma(to_array(image), str(image.encoding))


def to_rgb(image: Any) -> np.ndarray:
    """Decode any supported image to ``uint8`` (h, w, 3) RGB."""
    array = to_array(image)
    encoding = str(image.encoding)
    if encoding == "rgb8":
        return array
    if encoding in ("bgr8", "bgra8"):
        return np.ascontiguousarray(array[:, :, 2::-1])
    if encoding == "rgba8":
        return np.ascontiguousarray(array[:, :, :3])
    grey = _to_uint8_luma(array, encoding)
    return np.repeat(grey[:, :, None], 3, axis=2)


def array_to_bytes(array: np.ndarray, encoding: str) -> bytes:
    """Serialise a numpy array to the byte layout of ``encoding``."""
    dtype, channels = ENCODINGS[encoding]
    expected = (array.shape[0], array.shape[1]) if channels == 1 else (array.shape[0], array.shape[1], channels)
    if tuple(array.shape) != expected:
        raise ImageCodecError(
            f"array shape {array.shape} does not match encoding '{encoding}' (expected {expected})"
        )
    return np.ascontiguousarray(array, dtype=np.dtype(dtype)).tobytes()


def build_image(
    array: np.ndarray,
    encoding: str,
    frame_id: str,
    stamp: Any,
    image_cls: Any,
) -> Any:
    """Build a ``sensor_msgs/Image`` (``image_cls``) from a numpy array.

    ``image_cls`` is passed in rather than imported so this module stays
    ROS-free and testable with a stub class.
    """
    dtype, channels = ENCODINGS[encoding]
    message = image_cls()
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.height = int(array.shape[0])
    message.width = int(array.shape[1])
    message.encoding = encoding
    message.is_bigendian = False
    message.step = int(array.shape[1]) * channels * np.dtype(dtype).itemsize
    message.data = array_to_bytes(array, encoding)
    return message


# --------------------------------------------------------------------------- #
# drawing helpers (numpy only - the tag debug image and the preview tools)
# --------------------------------------------------------------------------- #
def _clip_index(value: float, limit: int) -> int:
    return int(max(0, min(limit, value)))


def draw_rect(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    colour: Sequence[int],
    thickness: int = 1,
) -> None:
    """Draw an axis-aligned rectangle in place (no clipping artefacts)."""
    height, width = canvas.shape[:2]
    ix0, ix1 = _clip_index(min(x0, x1), width - 1), _clip_index(max(x0, x1), width - 1)
    iy0, iy1 = _clip_index(min(y0, y1), height - 1), _clip_index(max(y0, y1), height - 1)
    thickness = max(1, int(thickness))
    canvas[iy0 : iy0 + thickness, ix0 : ix1 + 1] = colour
    canvas[max(iy0, iy1 - thickness + 1) : iy1 + 1, ix0 : ix1 + 1] = colour
    canvas[iy0 : iy1 + 1, ix0 : ix0 + thickness] = colour
    canvas[iy0 : iy1 + 1, max(ix0, ix1 - thickness + 1) : ix1 + 1] = colour


def draw_filled_rect(
    canvas: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    colour: Sequence[int],
) -> None:
    height, width = canvas.shape[:2]
    ix0, ix1 = _clip_index(min(x0, x1), width - 1), _clip_index(max(x0, x1), width - 1)
    iy0, iy1 = _clip_index(min(y0, y1), height - 1), _clip_index(max(y0, y1), height - 1)
    canvas[iy0 : iy1 + 1, ix0 : ix1 + 1] = colour


def draw_cross(
    canvas: np.ndarray, x: float, y: float, colour: Sequence[int], size: int = 4
) -> None:
    height, width = canvas.shape[:2]
    ix, iy = _clip_index(x, width - 1), _clip_index(y, height - 1)
    size = max(1, int(size))
    canvas[max(0, iy - size) : iy + size + 1, ix] = colour
    canvas[iy, max(0, ix - size) : ix + size + 1] = colour
