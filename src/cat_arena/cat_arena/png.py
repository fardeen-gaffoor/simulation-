"""Minimal, dependency-free PNG reader/writer.

Only what the arena tooling needs:

* :func:`write_gray8`   - 8-bit greyscale (heightfield artifact)
* :func:`write_gray16`  - 16-bit greyscale (high precision heightfield)
* :func:`write_rgb8`    - 8-bit RGB (sand albedo texture, overview capture)
* :func:`read_png`      - 8/16-bit greyscale and 8-bit RGB, non-interlaced

We deliberately avoid Pillow/OpenCV here: these files are produced by the world
generator and by an acceptance-test capture tool, and the only thing worse than
a broken image is a broken image from an undeclared dependency.  Everything is
built on :mod:`zlib` and :mod:`struct` from the standard library.

The PNGs written here are non-interlaced, filter type 0, colour types 0 (grey)
and 2 (RGB), which is exactly what Gazebo's ``ImageHeightmap`` and OGRE's
texture loader handle without surprises.
"""

from __future__ import annotations

import struct
import zlib
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["write_gray8", "write_gray16", "write_rgb8", "read_png", "PngImage"]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngImage:
    """Decoded PNG image."""

    __slots__ = ("width", "height", "bit_depth", "color_type", "pixels")

    def __init__(self, width: int, height: int, bit_depth: int, color_type: int, pixels: np.ndarray):
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.color_type = color_type
        self.pixels = pixels

    @property
    def mode(self) -> str:
        return {0: "gray", 2: "rgb"}.get(self.color_type, f"color_type_{self.color_type}")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"PngImage(width={self.width}, height={self.height}, "
            f"bit_depth={self.bit_depth}, mode={self.mode})"
        )


def _chunk(tag: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _write(path: str, width: int, height: int, bit_depth: int, color_type: int, raw: bytes) -> str:
    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    blob = (
        _PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )
    with open(path, "wb") as handle:
        handle.write(blob)
    return path


def _as_uint8(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D array, got shape {arr.shape}")
    return arr


def write_gray8(path: str, image: np.ndarray) -> str:
    """Write a 2-D uint8 greyscale PNG (row 0 = top row)."""
    arr = _as_uint8(image, "write_gray8")
    height, width = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    return _write(path, width, height, 8, 0, raw)


def write_gray16(path: str, image: np.ndarray) -> str:
    """Write a 2-D uint16 greyscale PNG (big-endian samples)."""
    arr = np.asarray(image)
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    if arr.ndim != 2:
        raise ValueError(f"write_gray16: expected a 2-D array, got shape {arr.shape}")
    height, width = arr.shape
    big = arr.astype(">u2")
    raw = b"".join(b"\x00" + big[row].tobytes() for row in range(height))
    return _write(path, width, height, 16, 0, raw)


def write_rgb8(path: str, image: np.ndarray) -> str:
    """Write a 3-D uint8 RGB PNG of shape ``(height, width, 3)``."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"write_rgb8: expected shape (h, w, 3), got {arr.shape}")
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    height, width, _ = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(height))
    return _write(path, width, height, 8, 2, raw)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter(raw: bytes, width: int, height: int, bytes_per_pixel: int, bytes_per_row: int) -> bytearray:
    out = bytearray(height * bytes_per_row)
    prev = bytearray(bytes_per_row)
    pos = 0
    for row in range(height):
        filter_type = raw[pos]
        pos += 1
        line = bytearray(raw[pos:pos + bytes_per_row])
        pos += bytes_per_row
        if filter_type == 0:
            pass
        elif filter_type == 1:  # Sub
            for i in range(bytes_per_pixel, bytes_per_row):
                line[i] = (line[i] + line[i - bytes_per_pixel]) & 0xFF
        elif filter_type == 2:  # Up
            for i in range(bytes_per_row):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif filter_type == 3:  # Average
            for i in range(bytes_per_row):
                left = line[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif filter_type == 4:  # Paeth
            for i in range(bytes_per_row):
                left = line[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                upleft = prev[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                line[i] = (line[i] + _paeth(left, prev[i], upleft)) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter type {filter_type} on row {row}")
        out[row * bytes_per_row:(row + 1) * bytes_per_row] = line
        prev = line
    return out


def read_png(path: str) -> PngImage:
    """Decode a non-interlaced 8/16-bit greyscale or 8-bit RGB PNG."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:8] != _PNG_SIGNATURE:
        raise ValueError(f"{path}: not a PNG file")
    pos = 8
    width = height = bit_depth = color_type = interlace = 0
    idat = bytearray()
    while pos + 8 <= len(blob):
        length = struct.unpack(">I", blob[pos:pos + 4])[0]
        tag = blob[pos + 4:pos + 8]
        payload = blob[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
    if width == 0 or height == 0:
        raise ValueError(f"{path}: missing IHDR")
    if interlace != 0:
        raise ValueError(f"{path}: interlaced PNG is not supported")
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if channels is None or bit_depth not in (8, 16):
        raise ValueError(f"{path}: unsupported color_type {color_type} / bit_depth {bit_depth}")
    bytes_per_pixel = channels * (bit_depth // 8)
    bytes_per_row = width * bytes_per_pixel
    raw = zlib.decompress(bytes(idat))
    expected = height * (bytes_per_row + 1)
    if len(raw) != expected:
        raise ValueError(f"{path}: decompressed size {len(raw)} != expected {expected}")
    flat = _unfilter(raw, width, height, bytes_per_pixel, bytes_per_row)
    if bit_depth == 8:
        arr = np.frombuffer(bytes(flat), dtype=np.uint8).reshape(height, width, channels)
    else:
        arr = np.frombuffer(bytes(flat), dtype=">u2").astype(np.uint16).reshape(height, width, channels)
    if channels == 1:
        arr = arr[:, :, 0]
    return PngImage(width, height, bit_depth, color_type, arr)


def rgb_from_channels(channels: Sequence[np.ndarray]) -> np.ndarray:
    """Stack three equal-shaped 2-D arrays into an ``(h, w, 3)`` uint8 image."""
    if len(channels) != 3:
        raise ValueError("rgb_from_channels expects exactly 3 channels")
    stacked = np.stack([np.asarray(c, dtype=np.uint8) for c in channels], axis=-1)
    return stacked
