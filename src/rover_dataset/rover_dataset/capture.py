"""gz control backends used by the collector.

Two operations are needed to place the rover for a capture:

* **teleport** -- ``/world/<world>/set_pose`` (a gz service), and
* **read back** -- ``gz model -m <name> -p``, which returns the *actual* pose
  after the rover has settled onto the (uneven, cratered) sand.

The pose is always *read back* rather than assumed: the terrain is not flat, so
the true roll/pitch/z differ slightly from the commanded value, and the labels
must use the pose the images were actually rendered from.

Both are behind :class:`GzBackend` so the collector can be tested without gz.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .projection import RoverPose

__all__ = ["GzError", "EntityPose", "GzBackend", "GzCliBackend", "FakeGzBackend"]


class GzError(RuntimeError):
    """Raised when a gz service call or pose query fails."""


@dataclass(frozen=True)
class EntityPose:
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def as_rover_pose(self) -> RoverPose:
        return RoverPose(self.x, self.y, self.z, self.roll, self.pitch, self.yaw)


class GzBackend:
    """Interface: teleport an entity and read its pose back."""

    def set_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> bool:
        raise NotImplementedError

    def get_pose(self, name: str) -> EntityPose:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        return None


_TRIPLE = re.compile(r"\[\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\]")


def parse_model_pose(text: str) -> EntityPose:
    """Parse ``gz model -m <name> -p`` output.

    Tolerant on purpose: gz prints a header, a model id and then the two
    bracketed triples (XYZ then RPY in radians).  Any bracketed triple groups
    are collected and the last two are used.
    """
    groups: List[Tuple[float, float, float]] = []
    for match in _TRIPLE.finditer(text):
        try:
            groups.append((float(match.group(1)), float(match.group(2)), float(match.group(3))))
        except ValueError:  # pragma: no cover - regex guarantees floats
            continue
    if len(groups) < 2:
        raise GzError(f"could not parse a pose from: {text!r}")
    xyz, rpy = groups[-2], groups[-1]
    return EntityPose(xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2])


class GzCliBackend(GzBackend):
    """Shells out to the ``gz`` CLI (the only stable gz-Harmonic interface)."""

    def __init__(
        self,
        world: str = "cat_arena",
        gz_binary: str = "gz",
        timeout_ms: int = 3000,
        subprocess_timeout: float = 20.0,
    ) -> None:
        self.world = world
        self.gz_binary = gz_binary
        self.timeout_ms = int(timeout_ms)
        self.subprocess_timeout = float(subprocess_timeout)
        self.calls: List[Sequence[str]] = []  # audit trail (tests + logs)

    # -- internals --------------------------------------------------------- #
    def _run(self, args: Sequence[str]) -> Tuple[int, str]:
        self.calls.append(list(args))
        try:
            proc = subprocess.run(
                list(args), capture_output=True, text=True, timeout=self.subprocess_timeout
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GzError(f"failed to run {args[0]}: {exc}") from exc
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")

    # -- interface --------------------------------------------------------- #
    def set_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> bool:
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        request = (
            f'name: "{name}" position: {{x: {x:.6f} y: {y:.6f} z: {z:.6f}}} '
            f"orientation: {{x: 0 y: 0 z: {qz:.9f} w: {qw:.9f}}}"
        )
        code, out = self._run(
            [
                self.gz_binary, "service",
                "-s", f"/world/{self.world}/set_pose",
                "--reqtype", "gz.msgs.Pose",
                "--reptype", "gz.msgs.Boolean",
                "--timeout", str(self.timeout_ms),
                "--req", request,
            ]
        )
        return code == 0 and "true" in out

    def get_pose(self, name: str) -> EntityPose:
        code, out = self._run([self.gz_binary, "model", "-m", name, "-p"])
        if code != 0:
            raise GzError(f"gz model -m {name} -p failed: {out.strip()}")
        return parse_model_pose(out)


class FakeGzBackend(GzBackend):
    """In-memory backend for tests / dry runs.

    ``error_every`` makes every n-th :meth:`set_pose` fail, exercising the
    collector's retry path.
    """

    def __init__(self, *, error_every: int = 0, settle_offset: float = 0.0) -> None:
        self.commanded: List[Tuple[str, float, float, float, float]] = []
        self.poses: dict = {}
        self.error_every = int(error_every)
        self.settle_offset = float(settle_offset)
        self._calls = 0

    def set_pose(self, name: str, x: float, y: float, z: float, yaw: float) -> bool:
        self._calls += 1
        if self.error_every and self._calls % self.error_every == 0:
            return False
        self.commanded.append((name, x, y, z, yaw))
        # the rover "settles" onto the sand: z drops, everything else is exact
        self.poses[name] = EntityPose(x, y, max(0.0, z - self.settle_offset), 0.0, 0.0, yaw)
        return True

    def get_pose(self, name: str) -> EntityPose:
        if name not in self.poses:
            raise GzError(f"no pose for {name}")
        return self.poses[name]
