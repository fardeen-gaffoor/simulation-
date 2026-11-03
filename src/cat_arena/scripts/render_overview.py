#!/usr/bin/env python3
"""Render a top-down verification image of a generated arena.

This is the visual acceptance test for module 1: it generates a seeded arena
with the verification camera enabled, boots ``gz sim`` headless, captures one
frame straight off the Gazebo transport (no ROS bridge, so it also works when
only the simulator is installed) and writes a PNG.

    ros2 run cat_arena render_overview.py --seed 7 --out /tmp/arena_7.png

The camera is *not* part of the robot and is off by default in generated worlds:
an always-on camera sensor costs render time in every simulation run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Optional, Sequence, Tuple

import numpy as np

from cat_arena.generate import generate
from cat_arena.png import write_rgb8

_PIXEL_FORMATS: Tuple[str, ...] = (
    "RGB_INT8",
    "RGBA_INT8",
    "BGRA_INT8",
    "L_INT8",
    "L_INT16",
    "RGB_INT16",
)


def _start_sim(world: str, verbose: int, log_path: str) -> Tuple[subprocess.Popen, object]:
    handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        ["gz", "sim", "-s", "-r", "-v", str(verbose), world],
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process, handle


def _stop_sim(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass


def _wait_for_world(world_name: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if shutil.which("gz"):
            proc = subprocess.run(
                ["gz", "model", "--list"], capture_output=True, text=True, check=False, timeout=20
            )
            if proc.returncode == 0 and world_name in (proc.stdout + proc.stderr):
                return True
        time.sleep(1.0)
    return False


def _capture(topic: str, timeout: float) -> np.ndarray:
    """Subscribe to a gz image topic and return one frame as ``(h, w, 3)`` uint8."""
    import gz.transport13 as gz_transport
    from gz.msgs10.image_pb2 import Image

    node = gz_transport.Node()
    state: dict = {"image": None}

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


def _pixel_format_name(message) -> str:
    try:
        from gz.msgs10.image_pb2 import PixelFormatType

        return PixelFormatType.Name(message.pixel_format_type)
    except Exception:  # pragma: no cover - enum always available in practice
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
    raise RuntimeError(
        f"unsupported camera pixel format '{name}'; supported: {', '.join(_PIXEL_FORMATS)}"
    )


def _patch_resolution(world_path: str, width: Optional[int], height: Optional[int]) -> str:
    """Write a copy of the world with the camera ``<image>`` size overridden."""
    import re

    with open(world_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    def _replace(match: "re.Match") -> str:
        new_w = width if width else int(match.group(1))
        new_h = height if height else int(match.group(2))
        return f"<width>{new_w}</width>\n              <height>{new_h}</height>"

    patched, count = re.subn(
        r"<width>(\d+)</width>\s*<height>(\d+)</height>", _replace, text, count=1
    )
    if count == 0:
        return world_path
    out_path = os.path.join(os.path.dirname(world_path), "arena_render.sdf")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(patched)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="render_overview",
        description="Capture a top-down PNG of a seeded arena (visual acceptance test).",
    )
    parser.add_argument("--seed", "-s", type=int, default=1, help="run seed")
    parser.add_argument("--config", "-c", type=str, default=None, help="arena.yaml override")
    parser.add_argument("--world", "-w", type=str, default=None, help="use an existing world instead of generating")
    parser.add_argument("--out", "-o", type=str, default=None, help="output PNG (default: <workdir>/overview_seed_N.png)")
    parser.add_argument("--workdir", type=str, default=None, help="where to generate the world (default: a temp dir)")
    parser.add_argument("--width", type=int, default=None, help="override camera width")
    parser.add_argument("--height", type=int, default=None, help="override camera height")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait for the image")
    parser.add_argument("--keep", action="store_true", help="keep the generated world and the gz log")
    parser.add_argument("--verbose", "-v", type=int, default=1, help="gz verbosity")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    workdir = args.workdir or os.path.join(tempfile.gettempdir(), "cat_arena_overview")
    os.makedirs(workdir, exist_ok=True)

    if args.world:
        world = os.path.abspath(args.world)
        world_name = "cat_arena"
    else:
        result = generate(
            seed=args.seed,
            config_path=args.config,
            output_dir=workdir,
            overview_camera=True,
            validate=True,
            verbose=False,
        )
        world = result.world_path
        world_name = result.config.world_name
        print(f"[render_overview] world: {world}")

    if args.width or args.height:
        world = _patch_resolution(world, args.width, args.height)

    log_path = os.path.join(workdir, "render_gz.log")
    process, _handle = _start_sim(world, args.verbose, log_path)
    try:
        if not _wait_for_world(world_name, timeout=min(90.0, args.timeout)):
            tail = ""
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
                    tail = "".join(handle.readlines()[-5:]).strip()
            except OSError:
                pass
            print(f"[render_overview] gz sim did not come up. log tail:\n{tail}", file=sys.stderr)
            return 2
        topic = "/arena_overview/image"
        try:
            text = open(world, "r", encoding="utf-8").read()
            marker = "<topic>"
            if marker in text:
                topic = "/" + text.split(marker, 1)[1].split("</topic>", 1)[0].strip().lstrip("/")
        except OSError:
            pass
        image = _capture(topic, timeout=args.timeout)
    except Exception as exc:
        print(f"[render_overview] capture failed: {exc}", file=sys.stderr)
        return 3
    finally:
        _stop_sim(process)

    out = args.out or os.path.join(workdir, f"overview_seed_{args.seed}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_rgb8(out, image)
    print(f"[render_overview] wrote {out} ({image.shape[1]}x{image.shape[0]})")
    if not args.keep and not args.workdir:
        pass  # generated world stays in the temp workdir on purpose
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
