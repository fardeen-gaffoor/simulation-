"""Per-frame image domain randomization.

Photometric ops (blur, noise, colour, compression, vignette) never change a
label.  The two geometric ops (``flip_180``, ``horizontal_flip``) do, and
:func:`transform_boxes` rewrites the boxes to match.

Ops are applied in one fixed order -- geometric, optics, sensor, colour,
display, compression -- and each is applied when its own seeded coin comes up
heads.  That makes a frame's augmentation reproducible from its seed alone,
independent of which other ops fired.

Only :mod:`numpy` is used for the pixel maths; Pillow is used for JPEG and for
the HSV colour pass (both are declared dependencies).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AugmentCfg
from .labels import YoloBox

__all__ = ["AugmentOutcome", "augment_image", "transform_boxes", "OP_ORDER"]

#: The fixed application order.  Changing it changes every generated dataset.
OP_ORDER: Tuple[str, ...] = (
    "flip_180",
    "horizontal_flip",
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "salt_pepper",
    "brightness",
    "contrast",
    "gamma",
    "hue_shift_deg",
    "saturation",
    "vignette",
    "jpeg",
)


@dataclass
class AugmentOutcome:
    image: np.ndarray
    flip_h: bool = False
    flip_v: bool = False
    applied: List[str] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)

    @property
    def geometry_changed(self) -> bool:
        return self.flip_h or self.flip_v


# --------------------------------------------------------------------------- #
# low level pixel ops
# --------------------------------------------------------------------------- #
def _as_float(image: np.ndarray) -> np.ndarray:
    return np.asarray(image, dtype=np.float64)


def _as_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image + 0.5, 0.0, 255.0).astype(np.uint8)


def _gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _convolve1d(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pad = [(0, 0)] * image.ndim
    pad[axis] = (radius, radius)
    padded = np.pad(image, pad, mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    length = image.shape[axis]
    for i, weight in enumerate(kernel):
        if weight == 0.0:
            continue
        sl = [slice(None)] * image.ndim
        sl[axis] = slice(i, i + length)
        out += weight * padded[tuple(sl)]
    return out


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ry, rx = kh // 2, kw // 2
    padded = np.pad(image, ((ry, ry), (rx, rx), (0, 0)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    h, w = image.shape[0], image.shape[1]
    for iy in range(kh):
        for ix in range(kw):
            weight = float(kernel[iy, ix])
            if weight == 0.0:
                continue
            out += weight * padded[iy:iy + h, ix:ix + w, :]
    return out


def _line_kernel(size: int, angle_deg: float) -> np.ndarray:
    """A normalised line kernel of odd ``size`` at ``angle_deg`` (motion blur)."""
    size = int(size)
    if size % 2 == 0:
        size += 1
    size = max(3, size)
    centre = (size - 1) / 2.0
    theta = math.radians(angle_deg)
    weights = np.zeros((size, size), dtype=np.float64)
    for t in np.linspace(-centre, centre, max(16, 4 * size)):
        x = int(round(centre + t * math.cos(theta)))
        y = int(round(centre + t * math.sin(theta)))
        if 0 <= x < size and 0 <= y < size:
            weights[y, x] += 1.0
    if weights.sum() <= 0.0:
        weights[int(centre), int(centre)] = 1.0
    return weights / weights.sum()


def _vignette_mask(shape: Tuple[int, int], strength: float) -> np.ndarray:
    h, w = shape
    yy = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) / ((h - 1) / 2.0)
    xx = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) / ((w - 1) / 2.0)
    rr = np.sqrt(xx[None, :] ** 2 + yy[:, None] ** 2) / math.sqrt(2.0)
    return (1.0 - strength * (rr ** 2))[:, :, None]


def _hsv_adjust(image: np.ndarray, hue_shift_deg: float, sat_factor: float) -> np.ndarray:
    """Hue rotation and saturation scaling in one pass (Pillow's HSV mode)."""
    from PIL import Image

    im = Image.fromarray(_as_uint8(image), "RGB").convert("HSV")
    h, s, v = im.split()
    if abs(hue_shift_deg) > 1e-9:
        shift = int(round(hue_shift_deg / 360.0 * 256.0)) % 256
        h = h.point([(i + shift) % 256 for i in range(256)])
    if abs(sat_factor - 1.0) > 1e-9:
        s = s.point([int(round(min(255.0, max(0.0, i * sat_factor)))) for i in range(256)])
    return np.asarray(Image.merge("HSV", (h, s, v)).convert("RGB"), dtype=np.float64)


def _jpeg_roundtrip(image: np.ndarray, quality: float) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(_as_uint8(image), "RGB")
    buffer = io.BytesIO()
    im.save(buffer, format="JPEG", quality=int(round(min(100.0, max(1.0, quality)))))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float64)


def _apply_numeric(name: str, image: np.ndarray, params: Dict[str, float],
                   rng: np.random.Generator) -> np.ndarray:
    """Every photometric op except the HSV pair."""
    if name == "gaussian_blur":
        sigma = params["sigma"]
        if sigma <= 1e-3:
            return image
        kernel = _gaussian_kernel(sigma)
        return _convolve1d(_convolve1d(image, kernel, 0), kernel, 1)
    if name == "motion_blur":
        angle = float(rng.uniform(0.0, 180.0))
        params["angle_deg"] = angle
        return _convolve2d(image, _line_kernel(int(round(params["kernel"])), angle))
    if name == "gaussian_noise":
        sigma = params["sigma"]
        if sigma <= 1e-6:
            return image
        return image + rng.normal(0.0, sigma, image.shape)
    if name == "salt_pepper":
        amount = params["amount"]
        if amount <= 1e-9:
            return image
        out = image.copy()
        flat = out.reshape(-1, out.shape[2])
        total = flat.shape[0]
        n = min(int(amount * total), total)
        if n > 0:
            idx = rng.choice(total, size=n, replace=False)
            salt = rng.random(idx.shape[0]) < 0.5
            flat[idx[salt]] = 255.0
            flat[idx[~salt]] = 0.0
        return out
    if name == "brightness":
        return image + params["delta"] * 255.0
    if name == "contrast":
        mean = image.reshape(-1, image.shape[2]).mean(axis=0)
        return (image - mean) * params["factor"] + mean
    if name == "gamma":
        value = max(1e-3, params["value"])
        return 255.0 * np.power(np.clip(image / 255.0, 0.0, 1.0), value)
    if name == "vignette":
        return image * _vignette_mask(image.shape[:2], params["strength"])
    if name == "jpeg":
        return _jpeg_roundtrip(image, params["quality"])
    raise AssertionError(f"unhandled op {name}")


def _maybe(cfg: AugmentCfg, name: str, rng: np.random.Generator) -> Optional[Dict[str, float]]:
    """Draw this op's coin and, if it fired, its parameters (in a fixed order)."""
    op = cfg.ops.get(name)
    if op is None:
        return None
    if rng.random() >= op.prob:
        return None
    return op.draws(rng)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def augment_image(image: np.ndarray, cfg: AugmentCfg, rng: np.random.Generator) -> AugmentOutcome:
    """Apply the configured randomization to one RGB uint8 image.

    The output has the same shape as the input (both geometric ops are
    shape-preserving), so labels stay valid after :func:`transform_boxes`.
    """
    src = np.asarray(image)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {src.shape}")
    outcome = AugmentOutcome(image=src)
    if not cfg.enabled:
        return outcome
    working = _as_float(src)

    params = _maybe(cfg, "flip_180", rng)
    if params is not None:
        working = working[::-1, ::-1, :]
        outcome.flip_h = True
        outcome.flip_v = True
        outcome.applied.append("flip_180")
    params = _maybe(cfg, "horizontal_flip", rng)
    if params is not None:
        working = working[:, ::-1, :]
        outcome.flip_h = not outcome.flip_h
        outcome.applied.append("horizontal_flip")

    for name in ("gaussian_blur", "motion_blur", "gaussian_noise", "salt_pepper",
                 "brightness", "contrast", "gamma"):
        params = _maybe(cfg, name, rng)
        if params is None:
            continue
        working = _apply_numeric(name, working, params, rng)
        outcome.params.update({f"{name}_{k}": v for k, v in params.items()})
        outcome.applied.append(name)

    hue = _maybe(cfg, "hue_shift_deg", rng)
    sat = _maybe(cfg, "saturation", rng)
    if hue is not None or sat is not None:
        hue_delta = float(hue["delta"]) if hue else 0.0
        sat_factor = float(sat["factor"]) if sat else 1.0
        working = _hsv_adjust(working, hue_delta, sat_factor)
        if hue is not None:
            outcome.params["hue_shift_deg"] = hue_delta
            outcome.applied.append("hue_shift_deg")
        if sat is not None:
            outcome.params["saturation_factor"] = sat_factor
            outcome.applied.append("saturation")

    for name in ("vignette", "jpeg"):
        params = _maybe(cfg, name, rng)
        if params is None:
            continue
        working = _apply_numeric(name, working, params, rng)
        outcome.params.update({f"{name}_{k}": v for k, v in params.items()})
        outcome.applied.append(name)

    outcome.image = _as_uint8(working)
    return outcome


# --------------------------------------------------------------------------- #
# label geometry
# --------------------------------------------------------------------------- #
def transform_boxes(boxes: Sequence[YoloBox], flip_h: bool, flip_v: bool) -> List[YoloBox]:
    """Mirror boxes to match a flipped image (normalised coordinates)."""
    if not flip_h and not flip_v:
        return list(boxes)
    out: List[YoloBox] = []
    for box in boxes:
        cx = (1.0 - box.cx) if flip_h else box.cx
        cy = (1.0 - box.cy) if flip_v else box.cy
        out.append(
            YoloBox(
                class_id=box.class_id,
                cx=cx,
                cy=cy,
                w=box.w,
                h=box.h,
                object_id=box.object_id,
                kind=box.kind,
                visible_fraction=box.visible_fraction,
                truncated=box.truncated,
            )
        )
    return out
