#!/usr/bin/env python3
"""Spawn a model into a running Gazebo world, retrying until the world is ready.

``ros2 run ros_gz_sim create`` sends one request with a fixed (5 s) timeout and
then exits, so spawning during a slow world load (large terrain, many markers)
silently fails and the arena ends up with no robot.  This wrapper uses the same
``/world/<world>/create`` service but keeps retrying until the entity exists.

The command line mirrors the interesting ``ros_gz_sim create`` flags:

    spawn_robot.py -world cat_arena -file rover.urdf -name rover \
        -x 1.29 -y 3.75 -z 0.08 -Y 1.385 -timeout 120
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time


def _run(cmd, timeout):
    """Run a command, returning None on timeout/missing executable."""
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _entity_exists(name: str) -> bool:
    result = _run(["gz", "model", "--list"], timeout=10.0)
    if result is None:
        return False
    wanted = f"- {name}"
    return any(line.strip() == wanted for line in result.stdout.splitlines())


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def _request_create(args, qx, qy, qz, qw) -> bool:
    request = (
        f'sdf_filename: "{args.file}" '
        f'name: "{args.name}" '
        "allow_renaming: false "
        f"pose: {{position: {{x: {args.x:.9f}, y: {args.y:.9f}, z: {args.z:.9f}}}, "
        f"orientation: {{x: {qx:.9f}, y: {qy:.9f}, z: {qz:.9f}, w: {qw:.9f}}}}}"
    )
    cmd = [
        "gz",
        "service",
        "-s",
        f"/world/{args.world}/create",
        "--reqtype",
        "gz.msgs.EntityFactory",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "10000",
        "--req",
        request,
    ]
    result = _run(cmd, timeout=20.0)
    if result is None:
        return False
    return "data: true" in result.stdout


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-world", default="cat_arena")
    parser.add_argument("-file", required=True)
    parser.add_argument("-name", default="rover")
    parser.add_argument("-x", type=float, default=0.0)
    parser.add_argument("-y", type=float, default=0.0)
    parser.add_argument("-z", type=float, default=0.0)
    parser.add_argument("-R", type=float, default=0.0)
    parser.add_argument("-P", type=float, default=0.0)
    parser.add_argument("-Y", type=float, default=0.0)
    parser.add_argument(
        "-timeout",
        type=float,
        default=120.0,
        help="seconds to keep retrying while gz sim finishes loading",
    )
    args = parser.parse_args(argv)

    qx, qy, qz, qw = _quaternion_from_rpy(args.R, args.P, args.Y)
    deadline = time.monotonic() + max(args.timeout, 1.0)
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if _entity_exists(args.name):
            print(f"[spawn_robot] '{args.name}' is already in world '{args.world}'")
            return 0
        if _request_create(args, qx, qy, qz, qw):
            print(
                f"[spawn_robot] spawned '{args.name}' in world '{args.world}' "
                f"at ({args.x:.3f}, {args.y:.3f}, {args.z:.3f}) yaw {args.Y:.3f} rad"
            )
            return 0
        print(
            f"[spawn_robot] world '{args.world}' not ready for '{args.name}' "
            f"(attempt {attempt}); retrying...",
            flush=True,
        )
        time.sleep(2.0)

    print(
        f"[spawn_robot] gave up spawning '{args.name}' after {args.timeout:.0f}s",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
