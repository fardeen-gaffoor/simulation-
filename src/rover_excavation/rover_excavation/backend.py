"""How a mechanism command gets to the actuators - and how the berm gets drawn.

This is module 6's version of module 4's camera-driver split: everything above
this file works in terms of :class:`~rover_excavation.mechanism.MechanismCommand`
and never learns whether the actuators are behind Gazebo, behind a serial link,
or not there at all.  The choice is ``backend.kind`` in ``excavation.yaml``:

``sim``
    publish the two joint targets as ``std_msgs/Float64`` on the topics
    ``ros_gz_bridge`` carries into the Gazebo ``JointPositionController`` /
    ``JointController`` plugins on the rover model.  (``ros_to_gz`` bridges for
    ``/excavation/ladder_cmd`` and ``/excavation/belt_cmd``.)

``hardware``
    speak the digger's serial protocol.  The protocol is implemented here as
    pure encode/decode functions so it is testable without a serial port; the
    port itself is opened lazily and a missing ``pyserial`` is an error, not a
    silent no-op, because a digging machine that silently stops digging is
    worse than one that refuses to start.

``null``
    drive nothing.  Used for dry runs and unit tests.

Two things are deliberately *not* here: the actuator dynamics (that is
:mod:`rover_excavation.mechanism`, and they are the same on every backend) and
anything that imports ``rclpy``.  ``SimIO`` is handed a ``publish`` callable by
the node, which is what keeps this module importable and testable anywhere.

The same file holds :class:`GzBermVisual`, because it is the same kind of thing:
process-boundary plumbing with a pure, testable core and an injected runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from .config import BackendCfg, VisualCfg
from .mechanism import MechanismCommand

__all__ = [
    "MechanismIOError",
    "MechanismIO",
    "NullIO",
    "SimIO",
    "HardwareIO",
    "build_io",
    "crc8",
    "encode_command",
    "decode_status",
    "GzBermVisual",
    "GzParticles",
    "berm_sdf",
    "particle_emitter_topic",
    "particle_command_request",
]


class MechanismIOError(RuntimeError):
    """Raised when a backend cannot be started (no port, no pyserial, ...)."""


# --------------------------------------------------------------------------- #
# mechanism backends
# --------------------------------------------------------------------------- #
class MechanismIO:
    """Sends actuator targets somewhere.  One instance per process."""

    name = "none"

    def start(self) -> None:
        """Open whatever the backend needs.  May raise :class:`MechanismIOError`."""

    def send(self, command: MechanismCommand) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release anything :meth:`start` acquired."""

    def describe(self) -> str:
        return self.name


class NullIO(MechanismIO):
    """Drives nothing; keeps the last command so a dry run can still be checked."""

    name = "null"

    def __init__(self) -> None:
        self.sent: int = 0
        self.last: Optional[MechanismCommand] = None

    def send(self, command: MechanismCommand) -> None:
        self.sent += 1
        self.last = command


class SimIO(MechanismIO):
    """Publishes the two joint targets to the bridged Gazebo controllers."""

    name = "sim"

    def __init__(self, config: BackendCfg, publish: Callable[[str, float], None]) -> None:
        self.config = config
        self._publish = publish
        self.sent: int = 0
        self.last: Optional[MechanismCommand] = None

    def send(self, command: MechanismCommand) -> None:
        self._publish(self.config.ladder_cmd_topic, float(command.ladder_pitch_rad))
        self._publish(self.config.belt_cmd_topic, float(command.belt_velocity_radps))
        self.sent += 1
        self.last = command

    def describe(self) -> str:
        return f"sim ({self.config.ladder_cmd_topic}, {self.config.belt_cmd_topic})"


# -- the real digger's serial protocol -------------------------------------- #
# One command frame per control tick, ASCII, newline terminated:
#
#     $C,<ladder_pitch_rad>,<belt_velocity_radps>,<tub_gate>*<crc8>\n
#
# and one status frame back from the controller, same shape:
#
#     $S,<ladder_pitch_rad>,<belt_velocity_radps>,<tub_gate>,<fault>*<crc8>\n
#
# crc8 is CRC-8/ATM (polynomial 0x07, init 0x00) over everything between the
# '$' and the '*'.  The firmware is a thin servo/ESC loop: it clamps nothing and
# decides nothing, because every limit already lives in
# rover_excavation/config/excavation.yaml and must not be duplicated there.
def crc8(data: bytes) -> int:
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if (crc & 0x80) else ((crc << 1) & 0xFF)
    return crc


def _frame(kind: str, fields: Tuple[float, ...], fault: Optional[int] = None) -> bytes:
    parts = [kind] + [f"{value:.4f}" for value in fields]
    if fault is not None:
        parts.append(str(int(fault)))
    body = ",".join(parts).encode("ascii")
    return b"$" + body + b"*" + f"{crc8(body):02X}".encode("ascii") + b"\n"


def encode_command(command: MechanismCommand) -> bytes:
    """Encode one :class:`MechanismCommand` as a command frame."""
    return _frame(
        "C",
        (
            float(command.ladder_pitch_rad),
            float(command.belt_velocity_radps),
            float(command.tub_gate),
        ),
    )


def decode_status(line: bytes) -> Optional[Tuple[float, float, float, int]]:
    """Decode a status frame, or return ``None`` if it is malformed.

    Returning ``None`` rather than raising is deliberate: a serial link at 20 Hz
    sees the occasional torn line, and the caller has to be able to skip one
    without treating it as a mechanism failure.
    """
    if isinstance(line, bytes):
        try:
            text = line.decode("ascii").strip()
        except UnicodeDecodeError:
            return None
    else:  # pragma: no cover - convenience for tests
        text = str(line).strip()
    if not text.startswith("$") or "*" not in text:
        return None
    body, _, checksum = text[1:].partition("*")
    if len(checksum) != 2:
        return None
    try:
        if crc8(body.encode("ascii")) != int(checksum, 16):
            return None
    except ValueError:
        return None
    parts = body.split(",")
    if len(parts) != 5 or parts[0] != "S":
        return None
    try:
        pitch, belt, gate, fault = (float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]))
    except ValueError:
        return None
    return (pitch, belt, gate, fault)


class HardwareIO(MechanismIO):
    """The real digger over serial.  Opens lazily; refuses clearly if it cannot."""

    name = "hardware"

    def __init__(self, config: BackendCfg) -> None:
        self.config = config
        self._port = None
        self.sent = 0
        self.last: Optional[MechanismCommand] = None
        self.last_status: Optional[Tuple[float, float, float, int]] = None

    def start(self) -> None:
        try:
            import serial  # type: ignore import-not-found
        except ImportError as exc:  # pragma: no cover - hardware only
            raise MechanismIOError(
                "backend.kind is 'hardware' but pyserial is not installed; "
                "install python3-serial on the rover computer (or use backend.kind: null)"
            ) from exc
        try:  # pragma: no cover - hardware only
            self._port = serial.Serial(
                self.config.serial.port,
                self.config.serial.baud,
                timeout=self.config.serial.timeout_s,
            )
        except Exception as exc:  # pragma: no cover - hardware only
            raise MechanismIOError(f"cannot open {self.config.serial.port}: {exc}") from exc

    def send(self, command: MechanismCommand) -> None:
        if self._port is None:  # pragma: no cover - guarded by start()
            raise MechanismIOError("HardwareIO.send before start()")
        self._port.write(encode_command(command))
        self.sent += 1
        self.last = command
        line = self._port.readline()
        if line:
            self.last_status = decode_status(line)

    def close(self) -> None:
        if self._port is not None:  # pragma: no cover - hardware only
            try:
                self._port.close()
            finally:
                self._port = None

    def describe(self) -> str:
        return f"hardware ({self.config.serial.port}@{self.config.serial.baud})"


def build_io(config: BackendCfg, publish: Optional[Callable[[str, float], None]] = None) -> MechanismIO:
    """Construct the backend the config asks for.  ``start()`` is not called."""
    kind = config.kind
    if kind == "sim":
        if publish is None:
            raise MechanismIOError("backend.kind is 'sim' but no publish callable was given")
        return SimIO(config, publish)
    if kind == "hardware":
        return HardwareIO(config)
    if kind == "null":
        return NullIO()
    raise MechanismIOError(f"unknown backend.kind {kind!r}")  # pragma: no cover - guarded by the loader


# --------------------------------------------------------------------------- #
# the berm in Gazebo
# --------------------------------------------------------------------------- #
def berm_sdf(name: str, size_x: float, size_y: float, height: float, visual_only: bool) -> str:
    """The SDF of the berm box.

    A single box that rises out of the ground as the berm grows, so a Gazebo
    screenshot shows the build.  It is visual-only by default (``visual.gazebo.
    visual_only``): the authoritative obstacle is the navigator's keep-out, and
    giving modelled sand a collision shape would make the simulator assert
    something it does not simulate.
    """
    collision = ""
    if not visual_only:
        collision = (
            '<collision name="collision"><geometry><box>'
            f"<size>{size_x:.4f} {size_y:.4f} {height:.4f}</size>"
            "</box></geometry></collision>"
        )
    return (
        '<?xml version="1.0"?><sdf version="1.10"><model name="' + name + '"><static>true</static>'
        f'<link name="link">{collision}<visual name="visual"><geometry><box>'
        f"<size>{size_x:.4f} {size_y:.4f} {height:.4f}</size></box></geometry>"
        '<material><ambient>0.84 0.78 0.58 1</ambient><diffuse>0.84 0.78 0.58 1</diffuse></material>'
        "</visual></link></model></sdf>"
    )


class GzBermVisual:
    """A rising box in the Gazebo world that shows how the berm is coming along.

    Talks to Gazebo the same way the documented teleport trick does - a
    ``gz service`` call - so it needs no extra ROS package and no gz Python
    bindings.  ``run`` is injected: production passes a subprocess runner, tests
    pass a recorder, and if ``gz`` is missing the visual disables itself once
    instead of failing a run.

    One box of the berm target's footprint and the target height is created
    once, fully buried, and then only its ``z`` is moved: the box rises out of
    the ground exactly as fast as the berm's mean height grows.  That is one
    create and a handful of cheap ``set_pose`` calls per run, and a screenshot
    at any moment shows the berm's real height.
    """

    def __init__(
        self,
        config: VisualCfg,
        target: Tuple[float, float, float, float],
        target_height_m: float,
        run: Callable[[list], int],
    ) -> None:
        self.config = config
        self.target = target  # (x0, y0, x1, y1)
        self.target_height_m = float(target_height_m)
        self._run = run
        self.spawned = False
        self.enabled = bool(config.gz_berm)
        self.height_m = 0.0
        self.commands = 0
        self.note = ""

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.target[0] + self.target[2]), 0.5 * (self.target[1] + self.target[3]))

    @property
    def size_x(self) -> float:
        return self.target[2] - self.target[0]

    @property
    def size_y(self) -> float:
        return self.target[3] - self.target[1]

    def _service(self, service: str, reqtype: str, reptype: str, req: str) -> list:
        return [
            "gz",
            "service",
            "-s",
            f"/world/{self.config.gz_world}/{service}",
            "--reqtype",
            reqtype,
            "--reptype",
            reptype,
            "--timeout",
            "2000",
            "--req",
            req,
        ]

    def _pose_request(self, height_m: float) -> str:
        cx, cy = self.center
        z = height_m - 0.5 * self.target_height_m
        return (
            f'name: "{self.config.gz_model}" position: {{x: {cx:.4f} y: {cy:.4f} z: {z:.4f}}} '
            "orientation: {x: 0 y: 0 z: 0 w: 1}"
        )

    def update(self, height_m: float) -> bool:
        """Move the box so its top sits at ``height_m``.  Returns True if it spoke."""
        if not self.enabled:
            return False
        height_m = max(0.0, min(float(height_m), self.target_height_m))
        if self.spawned and abs(height_m - self.height_m) < self.config.gz_height_step_m:
            return False
        if not self.spawned:
            sdf = berm_sdf(
                self.config.gz_model,
                self.size_x,
                self.size_y,
                self.target_height_m,
                self.config.gz_visual_only,
            )
            request = f"sdf: {sdf!r}, name: \"{self.config.gz_model}\", pose: {{position: {{z: {-0.5 * self.target_height_m:.4f}}}}}"
            code = self._run(
                self._service("create", "gz.msgs.EntityFactory", "gz.msgs.Boolean", request)
            )
            if code != 0:
                self.enabled = False
                self.note = "gz service create failed; the berm will not be drawn in Gazebo"
                return False
            self.spawned = True
        self._run(
            self._service("set_pose", "gz.msgs.Pose", "gz.msgs.Boolean", self._pose_request(height_m))
        )
        self.commands += 1
        self.height_m = height_m
        return True

    def close(self) -> None:
        if self.spawned:
            self._run(self._service("remove", "gz.msgs.Entity", "gz.msgs.Boolean", f'name: "{self.config.gz_model}"'))
            self.spawned = False


# --------------------------------------------------------------------------- #
# particle emitters (digging dust / falling sand)
# --------------------------------------------------------------------------- #
def particle_emitter_topic(model: str, link: str, emitter: str) -> str:
    """The gz cmd topic of one particle emitter inside a spawned model."""
    return f"/model/{model}/link/{link}/particle_emitter/{emitter}/cmd"


def particle_command_request(emitting: bool) -> str:
    """A ``gz.msgs.ParticleEmitter`` text-format request that only sets the flag."""
    return "emitting: {data: %s}" % ("true" if emitting else "false")


class GzParticles:
    """Toggle the rover's dig/dump particle emitters over their gz cmd topics.

    The emitters themselves are SDF in the robot (see ``rover.urdf.xacro``), so
    the only run-time state is on/off.  Commands are sent only when the desired
    state changes, and a failure disables the effect instead of failing the
    mechanism.  The runner is injected for the same reason as
    :class:`GzBermVisual`'s: it keeps this testable without Gazebo.
    """

    def __init__(
        self,
        model: str,
        emitters: Tuple[Tuple[str, str], ...],
        run: Callable[[list], int],
    ) -> None:
        self.model = str(model)
        self.emitters = tuple(emitters)
        self._run = run
        self.enabled = True
        self.note = ""
        self._state = [False for _ in self.emitters]

    def _command(self, index: int, emitting: bool) -> list:
        link, emitter = self.emitters[index]
        return [
            "gz",
            "topic",
            "-t",
            particle_emitter_topic(self.model, link, emitter),
            "-m",
            "gz.msgs.ParticleEmitter",
            "-p",
            particle_command_request(emitting),
        ]

    def set_all(self, wanted) -> bool:
        """Set every emitter from ``wanted`` (one bool per emitter)."""
        if not self.enabled:
            return False
        spoke = False
        for index, emitting in enumerate(wanted):
            emitting = bool(emitting)
            if self._state[index] == emitting:
                continue
            if self._run(self._command(index, emitting)) != 0:
                self.enabled = False
                self.note = "gz topic publish failed; particle effects are off"
                return False
            self._state[index] = emitting
            spoke = True
        return spoke

    def stop(self) -> None:
        if self.enabled:
            self.set_all([False for _ in self.emitters])
