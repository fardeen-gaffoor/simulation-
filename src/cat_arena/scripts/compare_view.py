#!/usr/bin/env python3
"""``ros2 run cat_arena compare_view.py`` - CCTV-style arena render + comparison.

Stage A validation tool of ``PROMPT_SAND.md``:

* place a camera at a corner of the arena, about 3 m high, looking diagonally
  across the arena like the CCTV reference shot;
* render the seeded arena (headless ``gz sim`` + camera capture);
* write the render next to the reference image;
* print HSV colour histograms of the sand region for the reference and the
  render, and a simple mean-difference distance score.

Usage::

    ros2 run cat_arena compare_view.py --seed 7 --quality light
    python3 scripts/compare_view.py --seed 7 --quality full --out /tmp/opencode/view.png
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cat_arena.generate import generate
from cat_arena.png import read_png, write_rgb8

DEFAULT_REFERENCE = "/home/fewest/cat/reference/arena_ref.png"
_FALLBACK_REFERENCE = "/home/fewest/cat/reference/preview.webp"
_PIXEL_FORMATS: Tuple[str, ...] = (
    "RGB_INT8",
    "RGBA_INT8",
    "BGRA_INT8",
    "L_INT8",
    "L_INT16",
    "R8G8B8",
)


# --------------------------------------------------------------------------- #
# gz process handling (same protocol as scripts/render_overview.py)
# --------------------------------------------------------------------------- #
def _start_sim(world_path: str, verbose: int, log_path: str):
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        ["gz", "sim", "-v", str(verbose), "-r", world_path],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log


def _stop_sim(process) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass
    try:
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            process.kill()
    try:
        process.stdout.close()
    except Exception:
        pass


def _wait_for_world(world_name: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if subprocess.run(["gz", "model", "--list"], capture_output=True, text=True, check=False, timeout=20).returncode == 0:
            out = subprocess.run(["gz", "model", "--list"], capture_output=True, text=True, check=False, timeout=20)
            if world_name in (out.stdout + out.stderr):
                return True
        time.sleep(1.0)
    return False


def _pixel_format_name(message) -> str:
    try:
        from gz.msgs10.image_pb2 import PixelFormatType

        return PixelFormatType.Name(message.pixel_format_type)
    except Exception:
        return str(message.pixel_format_type)


def _image_to_rgb(message) -> np.ndarray:
    name = _pixel_format_name(message)
    height, width = int(message.height), int(message.width)
    step = int(message.step) or None
    if name == "RGB_INT8":
        buffer = np.frombuffer(message.data, dtype=np.uint8)
        if step and step != width * 3:
            buffer = buffer.reshape(height, step)[:, : width * 3]
        return buffer.reshape(height, width, 3).copy()
    if name in ("RGBA_INT8", "BGRA_INT8"):
        buffer = np.frombuffer(message.data, dtype=np.uint8)
        if step and step != width * 4:
            buffer = buffer.reshape(height, step)[:, : width * 4]
        rgba = buffer.reshape(height, width, 4)
        if name == "RGBA_INT8":
            return rgba[:, :, :3].copy()
        return rgba[:, :, [2, 1, 0]].copy()
    if name in ("L_INT8", "L_INT16"):
        dtype = np.uint8 if name == "L_INT8" else np.uint16
        buffer = np.frombuffer(message.data, dtype=dtype)
        if step and step != width * np.dtype(dtype).itemsize:
            buffer = buffer.reshape(height, step // np.dtype(dtype).itemsize)[:, :width]
        grey = buffer.reshape(height, width)
        if name == "L_INT16":
            grey = (grey.astype(np.float32) / 257.0).astype(np.uint8)
        return np.repeat(grey[:, :, None], 3, axis=2)
    raise RuntimeError(f"unsupported camera pixel format '{name}'; supported: {', '.join(_PIXEL_FORMATS)}")


def _capture(topic: str, timeout: float) -> np.ndarray:
    import gz.transport13 as gz_transport
    from gz.msgs10.image_pb2 import Image

    node = gz_transport.Node()
    state: Dict[str, Any] = {"image": None}

    def _callback(message) -> None:
        state["image"] = message

    normalized = "/" + topic.lstrip("/")
    if not node.subscribe(Image, normalized, _callback):
        raise RuntimeError(f"could not subscribe to gz topic '{normalized}'")
    deadline = time.time() + timeout
    while time.time() < deadline and state["image"] is None:
        time.sleep(0.15)
    if state["image"] is None:
        raise RuntimeError(f"no image on gz topic '{normalized}' within {timeout:.0f}s")
    return _image_to_rgb(state["image"])


# --------------------------------------------------------------------------- #
# camera placement
# --------------------------------------------------------------------------- #
def _patch_camera(world_path: str, position: Sequence[float], look_at: Sequence[float], hfov_deg: float) -> str:
    """Rewrite the arena overview camera pose / fov inside a generated world."""
    import re

    with open(world_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    pattern = re.compile(
        r"(    <model name=\"arena_overview_camera\">.*?<pose>)([^<]*)(</pose>)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return world_path
    rpy = ""  # placeholder, replaced below
    from cat_arena.sdf import look_at_rpy

    rpy = look_at_rpy((float(position[0]), float(position[1]), float(position[2])), (float(look_at[0]), float(look_at[1]), float(look_at[2])))
    new_pose = f"{float(position[0]):.6f} {float(position[1]):.6f} {float(position[2]):.6f} {rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"
    patched = text[: match.start(2)] + new_pose + text[match.end(2) :]
    patched = patched.replace(
        f"<horizontal_fov>{__import__('math').radians(62.0):.9f}</horizontal_fov>",
        f"<horizontal_fov>{__import__('math').radians(hfov_deg):.9f}</horizontal_fov>",
    )
    out_path = os.path.join(os.path.dirname(world_path), "arena_compare.sdf")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(patched)
    return out_path


# --------------------------------------------------------------------------- #
# HSV histograms + distance score
# --------------------------------------------------------------------------- #
def _hsv(image: np.ndarray) -> np.ndarray:
    rgb = image.astype(float) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.maximum(r, np.maximum(g, b))
    mn = np.minimum(r, np.minimum(g, b))
    diff = mx - mn
    v = mx.copy()
    s = np.divide(diff, np.maximum(mx, 1e-9), out=np.zeros_like(mx), where=mx > 0)
    h = np.zeros_like(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        sector = np.where(
            mx == r,
            (g - b) / np.maximum(diff, 1e-9),
            np.where(mx == g, (b - r) / np.maximum(diff, 1e-9) + 2.0, (r - g) / np.maximum(diff, 1e-9) + 4.0),
        )
        h = np.where(diff > 0, 60.0 * sector, 0.0)
    return np.stack([h, s, v], axis=-1)


def _sand_mask(image: np.ndarray) -> np.ndarray:
    hsv = _hsv(image)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat < 0.35) & (val > 0.02) & (val < 0.98)
    if not mask.any():
        mask = np.ones(mask.shape, dtype=bool)
    return mask


def _histograms(image: np.ndarray) -> Dict[str, Any]:
    hsv = _hsv(image)
    mask = _sand_mask(image)
    out: Dict[str, Any] = {}
    ranges = {0: (0.0, 360.0), 1: (0.0, 1.0), 2: (0.0, 1.0)}
    for channel_name, channel in (("h", 0), ("s", 1), ("v", 2)):
        lo, hi = ranges[channel]
        hist, edges = np.histogram(hsv[:, :, channel][mask].ravel(), bins=32, range=(lo, hi))
        out[channel_name] = {"hist": hist.tolist(), "edges": edges.tolist(), "count": int(mask.sum())}
    return out


def _distance_score(ref: np.ndarray, rend: np.ndarray) -> float:
    mask = _sand_mask(ref) & _sand_mask(rend)
    if not mask.any():
        mask = _sand_mask(ref)
    diff = np.abs(ref[:, :, :3].astype(float) - rend[:, :, :3].astype(float))[mask]
    return float(diff.mean()) if diff.size else 0.0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compare_view",
        description="Render a corner CCTV-style view of a seeded arena and compare it with the reference.",
    )
    parser.add_argument("--seed", type=int, default=1, help="arena seed")
    parser.add_argument("--quality", choices=("light", "full"), default="light", help="texture quality preset")
    parser.add_argument("--world", type=str, default=None, help="existing world instead of generating one")
    parser.add_argument("--out", type=str, default=None, help="render output path (default: next to the reference)")
    parser.add_argument("--config", type=str, default=None, help="arena.yaml override")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for the frame")
    parser.add_argument("--verbose", type=int, default=1, help="gz verbosity")
    parser.add_argument("--hfov", type=float, default=62.0, help="camera horizontal fov in degrees")
    return parser


def _load_image(path: str) -> Optional[np.ndarray]:
    if path.endswith(".webp"):
        from PIL import Image

        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if not os.path.isfile(path):
        fallback = "/home/fewest/cat/reference/preview.webp"
        if os.path.isfile(fallback):
            return _load_image(fallback)
        return None
    return np.asarray(read_png(path).pixels, dtype=np.uint8)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workdir = tempfile.mkdtemp(prefix="cat_arena_compare_")
    os.makedirs(workdir, exist_ok=True)

    if args.world:
        world = os.path.abspath(args.world)
        world_name = "cat_arena"
    else:
        result = generate(seed=args.seed, config_path=args.config, output_dir=workdir, overview_camera=True, validate=True, verbose=False)
        world = result.world_path
        world_name = result.config.world_name
        print(f"[compare_view] world: {world}")

    world = _patch_camera(world, (7.5, 0.0, 3.0), (1.5, 4.5, 0.0), args.hfov)
    log_path = os.path.join(workdir, "compare_gz.log")
    process, log = _start_sim(world, args.verbose, log_path)
    image: Optional[np.ndarray] = None
    try:
        if not _wait_for_world(world_name, timeout=min(90.0, args.timeout)):
            print(f"[compare_view] gz sim did not come up; log tail:", file=sys.stderr)
            return 2
        try:
            text = open(world, "r", encoding="utf-8").read()
            topic = "/arena_overview/image"
            marker = "<topic>"
            if marker in text:
                topic = "/" + text.split(marker, 1)[1].split("</topic>", 1)[0].strip().lstrip("/")
            image = _capture(topic, timeout=args.timeout)
        except Exception as exc:
            print(f"[compare_view] capture failed: {exc}", file=sys.stderr)
            return 3
    finally:
        _stop_sim(process)

    out = args.out or os.path.join(os.path.dirname(DEFAULT_REFERENCE), f"arena_view_{args.seed}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    write_rgb8(out, image)
    print(f"[compare_view] wrote {out} ({image.shape[1]}x{image.shape[0]})")

    ref_image = _load_image(DEFAULT_REFERENCE)
    if ref_image is None:
        print(f"[compare_view] reference image not found: {DEFAULT_REFERENCE}", file=sys.stderr)
        return 4
    ref = np.asarray(ref_image, dtype=np.uint8)
    if ref.shape[0] != image.shape[0] or ref.shape[1] != image.shape[1]:
        # resize the reference onto the render's canvas for a fair comparison
        from PIL import Image

        ref = np.asarray(Image.fromarray(ref).resize((image.shape[1], image.shape[0])).convert("RGB"), dtype=np.uint8)

    ref_hist = _histograms(ref)
    rend_hist = _histograms(image)
    score = _distance_score(ref, image)
    print(f"[compare_view] sand region pixels: ref={ref_hist['h']['count']} rend={rend_hist['h']['count']}")
    for channel_name in ("h", "s", "v"):
        print(f"[compare_view] HSV {channel_name}: ref={ref_hist[channel_name]['hist']} rend={rend_hist[channel_name]['hist']}")
    print(f"[compare_view] distance score (mean |delta| over sand region): {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
