#!/usr/bin/env python3
"""``ros2 run cat_arena postprocess_image.py`` - camera post-processing for training images.

Stage D of ``PROMPT_SAND.md``: apply realistic camera imperfections to a rendered
arena image so that training data matches the look of the real CCTV reference:

* **exposure jitter**   - random gain / brightness drift
* **white-balance jitter** - per-channel colour temperature drift
* **sensor noise**      - Poisson shot noise + Gaussian read noise
* **lens blur**         - Gaussian blur simulating defocus / lens softness

The same seed gives the same post-processed image (deterministic), while
different seeds vary the noise realisation and exposure within the documented
ranges.  Input and output are PNG files.

Usage::

    ros2 run cat_arena postprocess_image.py \
        --input arena_view.png --output arena_view_noisy.png --seed 7 \
        --exposure 0.1 --wb 0.05 --noise 0.02 --blur 1.2

All perturbation parameters are optional with sensible defaults; use ``--help``
for the full list.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

from cat_arena.png import read_png, write_rgb8


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian lens blur via Pillow (fast, dependency-light)."""
    if sigma <= 0.0:
        return image
    pil = Image.fromarray(image.astype(np.uint8), mode="RGB")
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(pil, dtype=np.float32)


def _sensor_noise(image: np.ndarray, shot: float, read: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise + Gaussian read noise on a 0-1 image."""
    img = image.astype(np.float32)
    # shot noise: variance proportional to signal
    shot_var = shot * img
    img += rng.normal(0.0, 1.0, img.shape) * np.sqrt(np.maximum(shot_var, 0.0))
    # read noise: fixed variance
    img += rng.normal(0.0, read, img.shape)
    return img


def _exposure(image: np.ndarray, gain: float, offset: float) -> np.ndarray:
    """Linear exposure: img = gain * img + offset, clamped."""
    return gain * image + offset


def _white_balance(image: np.ndarray, gains: Sequence[float]) -> np.ndarray:
    """Per-channel white-balance gains (RGB)."""
    return image * np.asarray(gains, dtype=np.float32)[None, None, :]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="postprocess_image",
        description="Apply camera post-processing (exposure, white balance, noise, blur) to a rendered PNG.",
    )
    parser.add_argument("--input", required=True, help="input rendered PNG")
    parser.add_argument("--output", required=True, help="output post-processed PNG")
    parser.add_argument("--seed", type=int, default=0, help="noise / jitter seed (deterministic)")
    parser.add_argument(
        "--exposure", type=float, default=0.08, help="exposure gain jitter std (fraction of mean gain)"
    )
    parser.add_argument(
        "--wb", type=float, default=0.06, help="white-balance per-channel jitter std (fraction)"
    )
    parser.add_argument(
        "--noise", type=float, default=0.015, help="combined sensor noise std (fraction of full scale)"
    )
    parser.add_argument(
        "--blur", type=float, default=1.0, help="lens blur radius in pixels (0 disables)"
    )
    parser.add_argument(
        "--base-gain", type=float, default=1.0, help="baseline exposure gain before jitter"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"[postprocess_image] input not found: {args.input}", file=sys.stderr)
        return 2

    img = read_png(args.input)
    arr = np.asarray(img.pixels, dtype=np.float32) / 255.0

    rng = np.random.default_rng(args.seed)

    # --- exposure jitter -------------------------------------------------
    gain = args.base_gain + rng.normal(0.0, args.exposure)
    gain = max(0.1, gain)
    offset = rng.normal(0.0, 0.01)
    arr = _exposure(arr, gain, offset)

    # --- white-balance jitter -------------------------------------------
    wb_gains = 1.0 + rng.normal(0.0, args.wb, size=3)
    wb_gains = np.clip(wb_gains, 0.5, 1.5)
    arr = _white_balance(arr, wb_gains)

    # --- sensor noise ----------------------------------------------------
    shot = args.noise * 0.6
    read = args.noise * 0.4
    arr = _sensor_noise(arr, shot, read, rng)

    # --- lens blur -------------------------------------------------------
    arr = np.clip(arr, 0.0, 1.0)
    arr = _gaussian_blur((arr * 255.0).astype(np.uint8), args.blur) / 255.0
    arr = np.clip(arr, 0.0, 1.0)

    out = np.clip(arr, 0.0, 1.0)
    write_rgb8(args.output, (out * 255.0).astype(np.uint8))
    print(
        f"[postprocess_image] {args.input} -> {args.output} "
        f"(seed={args.seed}, gain={gain:.3f}, wb={wb_gains.round(3).tolist()}, "
        f"noise={args.noise:.3f}, blur={args.blur})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
