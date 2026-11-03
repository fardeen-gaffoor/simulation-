"""Typed, validated ``rover.yaml`` for the rover_description package.

Like ``cat_arena``, the robot has exactly one source of truth.  This module
loads it, validates the competition rules that must always hold (envelope and
mass ceilings, camera count/topics, the sensor ban) and exposes immutable
dataclasses that the xacro model, the calibration writer, the mass budget and
the tests all consume.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "RoverConfigError",
    "load_config",
    "default_config_path",
    "package_share_dir",
    "RoverConfig",
    "Camera",
    "Envelope",
]

#: Topics every rover camera MUST publish on (competition/stack contract).
CAMERA_TOPICS: Tuple[str, ...] = ("/cam_nav", "/cam_tag", "/cam_excavation", "/cam_dump")

#: Sensors that are forbidden by the competition.  Checked against the model.
FORBIDDEN_SENSOR_HINTS: Tuple[str, ...] = (
    "gps",
    "navsat",
    "magnetometer",
    "magnetic",
    "compass",
    "ultrasonic",
    "sonar",
    "touch",
    "force_torque",  # no touch sensing
    "hydraulic",
)


class RoverConfigError(ValueError):
    """Raised when ``rover.yaml`` is missing something or breaks a rule."""


def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise RoverConfigError(f"{where}: missing required key '{key}'")
    return section[key]


def _as_float(value: Any, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RoverConfigError(f"{where}: expected a number, got {value!r}") from exc


def _as_vec(value: Any, where: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise RoverConfigError(f"{where}: expected {length} numbers, got {value!r}")
    return tuple(_as_float(v, where) for v in value)


@dataclass(frozen=True)
class Envelope:
    length_x: float
    width_y: float
    height_z: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.length_x, self.width_y, self.height_z)

    def contains(self, other: "Envelope", tol: float = 1e-9) -> bool:
        return (
            other.length_x <= self.length_x + tol
            and other.width_y <= self.width_y + tol
            and other.height_z <= self.height_z + tol
        )


@dataclass(frozen=True)
class Chassis:
    length: float
    width: float
    height: float
    center: Tuple[float, float, float]
    mass: float
    color: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Drive:
    wheel_radius: float
    wheel_width: float
    track: float
    wheelbase: float
    wheel_mass: float
    joint_damping: float
    joint_friction: float
    max_wheel_torque: float
    max_wheel_velocity: float
    wheel_color: Tuple[float, float, float, float]
    hub_color: Tuple[float, float, float, float]
    wheel_mu: float
    wheel_mu2: float


@dataclass(frozen=True)
class Ladder:
    length: float
    width: float
    frame_thickness: float
    plate_thickness: float
    mass: float
    pivot: Tuple[float, float, float]
    stowed_pitch_deg: float
    min_pitch_deg: float
    max_pitch_deg: float
    color: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Belt:
    roller_radius: float
    length: float
    mass: float
    offset_from_ladder_tip: float
    color: Tuple[float, float, float, float]
    max_velocity: float


@dataclass(frozen=True)
class Box:
    size: Tuple[float, float, float]
    center: Tuple[float, float, float]
    mass: float
    color: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Excavation:
    ladder: Ladder
    belt: Belt
    tub: Box
    funnel: Box


@dataclass(frozen=True)
class Imu:
    enabled: bool
    link: str
    origin: Tuple[float, float, float]
    update_rate: float


@dataclass(frozen=True)
class Camera:
    name: str
    topic: str
    frame: str
    origin: Tuple[float, float, float]
    rpy_deg: Tuple[float, float, float]
    width: int
    height: int
    hfov_deg: float
    update_rate: float
    role: str

    @property
    def optical_frame(self) -> str:
        return f"{self.frame.replace('_link', '')}_optical_frame"

    @property
    def camera_info_topic(self) -> str:
        return f"{self.topic}/camera_info"

    @property
    def rpy_rad(self) -> Tuple[float, float, float]:
        return tuple(math.radians(v) for v in self.rpy_deg)  # type: ignore[return-value]

    @property
    def hfov_rad(self) -> float:
        return math.radians(self.hfov_deg)

    # -- pinhole intrinsics (square pixels, centred principal point) --------- #
    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(self.hfov_rad / 2.0)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan((self.height / 2.0) / self.fy))


@dataclass(frozen=True)
class Sensors:
    imu: Imu
    cameras: List[Camera]


@dataclass(frozen=True)
class PayloadItem:
    name: str
    mass: float


@dataclass(frozen=True)
class Battery:
    capacity_wh: float
    initial_energy_wh: float
    idle_draw_w: float


@dataclass(frozen=True)
class RoverConfig:
    schema_version: int
    path: str
    raw: Dict[str, Any]
    name: str
    declared_forward: str
    limits: Dict[str, Any]
    envelope_limit: Envelope
    mass_limit: float
    chassis: Chassis
    drive: Drive
    excavation: Excavation
    sensors: Sensors
    payload: List[PayloadItem]
    topics: Dict[str, str]
    battery: Battery
    assumptions: List[str] = field(default_factory=list)

    # -- structural (geometry) mass, excluding payload ---------------------- #
    @property
    def structural_mass(self) -> float:
        return (
            self.chassis.mass
            + 4.0 * self.drive.wheel_mass
            + self.excavation.ladder.mass
            + self.excavation.belt.mass
            + self.excavation.tub.mass
            + self.excavation.funnel.mass
        )

    @property
    def payload_mass(self) -> float:
        return sum(item.mass for item in self.payload)

    @property
    def total_mass(self) -> float:
        return self.structural_mass + self.payload_mass

    def camera(self, name: str) -> Camera:
        for cam in self.sensors.cameras:
            if cam.name == name:
                return cam
        raise RoverConfigError(f"unknown camera '{name}'")

    def config_sha256(self) -> str:
        with open(self.path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    def assumptions_report(self) -> List[str]:
        return list(self.assumptions)


def package_share_dir() -> Optional[str]:
    """The installed ``share/rover_description`` directory, or ``None``.

    Needed because ``colcon test`` and installed runs import the package from the
    install space, where ``__file__`` no longer sits next to ``config/`` and
    ``urdf/``.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory("rover_description")
    except Exception:
        return None


def _asset_path(relative: str) -> str:
    """Resolve an asset (``config/...``, ``urdf/...``) in install or source tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    share = package_share_dir()
    if share:
        candidates.append(os.path.join(share, relative))
    candidates.append(os.path.join(here, "..", relative))
    candidates.append(os.path.join(os.path.dirname(here), "share", "rover_description", relative))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    raise RoverConfigError(f"cannot locate {relative}")


def default_config_path() -> str:
    """Installed ``config/rover.yaml`` (source tree fallback for pytest)."""
    return _asset_path(os.path.join("config", "rover.yaml"))


def _parse_camera(entry: Dict[str, Any], index: int) -> Camera:
    where = f"sensors.cameras[{index}]"
    name = str(_require(entry, "name", where))
    topic = str(_require(entry, "topic", where))
    frame = str(_require(entry, "frame", where))
    origin = _as_vec(_require(entry, "origin", where), f"{where}.origin", 3)
    rpy = _as_vec(_require(entry, "rpy_deg", where), f"{where}.rpy_deg", 3)
    width = int(_require(entry, "width", where))
    height = int(_require(entry, "height", where))
    hfov = _as_float(_require(entry, "hfov_deg", where), f"{where}.hfov_deg")
    rate = _as_float(entry.get("update_rate", 15.0), f"{where}.update_rate")
    role = str(entry.get("role", ""))
    if width <= 0 or height <= 0:
        raise RoverConfigError(f"{where}: image size must be positive")
    if not (0.0 < hfov < 180.0):
        raise RoverConfigError(f"{where}: hfov_deg must be in (0,180)")
    if topic not in CAMERA_TOPICS:
        raise RoverConfigError(
            f"{where}: camera topic '{topic}' must be one of {list(CAMERA_TOPICS)}"
        )
    if not frame.endswith("_link"):
        raise RoverConfigError(f"{where}: camera frame '{frame}' must end with '_link'")
    return Camera(name, topic, frame, origin, rpy, width, height, hfov, rate, role)


def load_config(path: str | None = None) -> RoverConfig:
    """Load and fully validate ``rover.yaml``."""
    import yaml  # local import keeps the module import-light for --help paths

    resolved = os.path.abspath(os.path.expanduser(path)) if path else default_config_path()
    if not os.path.isfile(resolved):
        raise RoverConfigError(f"rover config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise RoverConfigError("rover.yaml must contain a mapping at the top level")

    meta = _require(raw, "meta", "rover.yaml")
    name = str(meta.get("name", "rover"))
    declared_forward = str(meta.get("declared_forward", "+x"))

    limits_raw = _require(raw, "limits", "rover.yaml")
    env_raw = _require(limits_raw, "envelope", "limits")
    envelope_limit = Envelope(
        _as_float(_require(env_raw, "length_x", "limits.envelope"), "limits.envelope.length_x"),
        _as_float(_require(env_raw, "width_y", "limits.envelope"), "limits.envelope.width_y"),
        _as_float(_require(env_raw, "height_z", "limits.envelope"), "limits.envelope.height_z"),
    )
    mass_limit = _as_float(_require(limits_raw, "mass_kg", "limits"), "limits.mass_kg")

    c = _require(raw, "chassis", "rover.yaml")
    chassis = Chassis(
        _as_float(_require(c, "length", "chassis"), "chassis.length"),
        _as_float(_require(c, "width", "chassis"), "chassis.width"),
        _as_float(_require(c, "height", "chassis"), "chassis.height"),
        _as_vec(_require(c, "center", "chassis"), "chassis.center", 3),
        _as_float(_require(c, "mass", "chassis"), "chassis.mass"),
        _as_vec(c.get("color", [0.9, 0.72, 0.05, 1.0]), "chassis.color", 4),
    )

    d = _require(raw, "drive", "rover.yaml")
    drive = Drive(
        _as_float(_require(d, "wheel_radius", "drive"), "drive.wheel_radius"),
        _as_float(_require(d, "wheel_width", "drive"), "drive.wheel_width"),
        _as_float(_require(d, "track", "drive"), "drive.track"),
        _as_float(_require(d, "wheelbase", "drive"), "drive.wheelbase"),
        _as_float(_require(d, "wheel_mass", "drive"), "drive.wheel_mass"),
        _as_float(d.get("joint_damping", 0.05), "drive.joint_damping"),
        _as_float(d.get("joint_friction", 0.02), "drive.joint_friction"),
        _as_float(d.get("max_wheel_torque", 25.0), "drive.max_wheel_torque"),
        _as_float(d.get("max_wheel_velocity", 12.0), "drive.max_wheel_velocity"),
        _as_vec(d.get("wheel_color", [0.12, 0.12, 0.13, 1.0]), "drive.wheel_color", 4),
        _as_vec(d.get("hub_color", [0.65, 0.65, 0.68, 1.0]), "drive.hub_color", 4),
        _as_float(d.get("wheel_mu", 1.0), "drive.wheel_mu"),
        _as_float(d.get("wheel_mu2", 1.0), "drive.wheel_mu2"),
    )
    if drive.wheel_radius <= 0 or drive.track <= 0 or drive.wheelbase <= 0:
        raise RoverConfigError("drive: wheel_radius, track and wheelbase must be > 0")
    if drive.track <= drive.wheel_width:
        raise RoverConfigError("drive: track must exceed wheel_width")

    ex = _require(raw, "excavation", "rover.yaml")
    lr = _require(ex, "ladder", "excavation")
    ladder = Ladder(
        _as_float(_require(lr, "length", "excavation.ladder"), "ladder.length"),
        _as_float(_require(lr, "width", "excavation.ladder"), "ladder.width"),
        _as_float(lr.get("frame_thickness", 0.05), "ladder.frame_thickness"),
        _as_float(lr.get("plate_thickness", 0.02), "ladder.plate_thickness"),
        _as_float(_require(lr, "mass", "excavation.ladder"), "ladder.mass"),
        _as_vec(_require(lr, "pivot", "excavation.ladder"), "ladder.pivot", 3),
        _as_float(lr.get("stowed_pitch_deg", 25.0), "ladder.stowed_pitch_deg"),
        _as_float(lr.get("min_pitch_deg", -12.0), "ladder.min_pitch_deg"),
        _as_float(lr.get("max_pitch_deg", 58.0), "ladder.max_pitch_deg"),
        _as_vec(lr.get("color", [0.25, 0.27, 0.30, 1.0]), "ladder.color", 4),
    )
    if not (ladder.min_pitch_deg < ladder.stowed_pitch_deg < ladder.max_pitch_deg):
        raise RoverConfigError(
            "excavation.ladder: stowed_pitch_deg must lie strictly inside [min,max]"
        )
    bl = _require(ex, "belt", "excavation")
    belt = Belt(
        _as_float(_require(bl, "roller_radius", "excavation.belt"), "belt.roller_radius"),
        _as_float(_require(bl, "length", "excavation.belt"), "belt.length"),
        _as_float(_require(bl, "mass", "excavation.belt"), "belt.mass"),
        _as_float(bl.get("offset_from_ladder_tip", 0.06), "belt.offset_from_ladder_tip"),
        _as_vec(bl.get("color", [0.1, 0.1, 0.1, 1.0]), "belt.color", 4),
        _as_float(bl.get("max_velocity", 6.0), "belt.max_velocity"),
    )

    def _box(section: Dict[str, Any], where: str) -> Box:
        return Box(
            _as_vec(_require(section, "size", where), f"{where}.size", 3),
            _as_vec(_require(section, "center", where), f"{where}.center", 3),
            _as_float(_require(section, "mass", where), f"{where}.mass"),
            _as_vec(section.get("color", [0.35, 0.35, 0.38, 1.0]), f"{where}.color", 4),
        )

    excavation = Excavation(
        ladder=ladder,
        belt=belt,
        tub=_box(_require(ex, "tub", "excavation"), "excavation.tub"),
        funnel=_box(_require(ex, "funnel", "excavation"), "excavation.funnel"),
    )

    s = _require(raw, "sensors", "rover.yaml")
    imu_raw = s.get("imu", {})
    imu = Imu(
        enabled=bool(imu_raw.get("enabled", True)),
        link=str(imu_raw.get("link", "imu_link")),
        origin=_as_vec(imu_raw.get("origin", [0.0, 0.0, 0.15]), "sensors.imu.origin", 3),
        update_rate=_as_float(imu_raw.get("update_rate", 100.0), "sensors.imu.update_rate"),
    )
    cameras_raw = _require(s, "cameras", "sensors")
    if not isinstance(cameras_raw, list) or not cameras_raw:
        raise RoverConfigError("sensors.cameras must be a non-empty list")
    cameras = [_parse_camera(entry, i) for i, entry in enumerate(cameras_raw)]
    names = [cam.name for cam in cameras]
    if len(set(names)) != len(names):
        raise RoverConfigError("sensors.cameras: duplicate camera names")
    topics = [cam.topic for cam in cameras]
    if sorted(topics) != sorted(CAMERA_TOPICS):
        raise RoverConfigError(
            f"sensors.cameras: topics must be exactly {list(CAMERA_TOPICS)}, got {topics}"
        )
    sensors = Sensors(imu=imu, cameras=cameras)

    payload = []
    for i, item in enumerate(raw.get("payload", []) or []):
        payload.append(
            PayloadItem(str(_require(item, "name", f"payload[{i}]")),
                        _as_float(_require(item, "mass", f"payload[{i}]"), f"payload[{i}].mass"))
        )

    topics_map = {str(k): str(v) for k, v in (raw.get("topics") or {}).items()}
    b = raw.get("battery") or {}
    battery = Battery(
        _as_float(b.get("capacity_wh", 480.0), "battery.capacity_wh"),
        _as_float(b.get("initial_energy_wh", b.get("capacity_wh", 480.0)), "battery.initial_energy_wh"),
        _as_float(b.get("idle_draw_w", 6.0), "battery.idle_draw_w"),
    )

    assumptions = [
        "A1: internal robot layout (chassis size, wheel position, ladder/tub geometry) "
        "is an engineering choice - the competition fixes only the envelope, mass and sensor ban.",
        "A2: body +x is the DECLARED COMPETITION FORWARD (change meta.declared_forward and "
        "start_pose.declared_forward_deg together if the real rover differs).",
        "A3: total mass 80 kg ceiling includes beacons/markers/battery; the budget below leaves margin.",
        "A4: sand-facing parts use friction mu=mu2=%.2f for the wheels." % drive.wheel_mu,
        "A5: camera mounts use POSITIVE PITCH = tilt down; cam_tag <= 60 cm per the rules.",
        "A6: the ladder stowed attitude (%.0f deg) defines the height/length used by the envelope."
        % ladder.stowed_pitch_deg,
    ]

    cfg = RoverConfig(
        schema_version=int(raw.get("schema_version", 1)),
        path=resolved,
        raw=raw,
        name=name,
        declared_forward=declared_forward,
        limits=limits_raw,
        envelope_limit=envelope_limit,
        mass_limit=mass_limit,
        chassis=chassis,
        drive=drive,
        excavation=excavation,
        sensors=sensors,
        payload=payload,
        topics=topics_map,
        battery=battery,
        assumptions=assumptions,
    )

    if cfg.total_mass > cfg.mass_limit + 1e-9:
        raise RoverConfigError(
            f"mass budget {cfg.total_mass:.1f} kg exceeds the {cfg.mass_limit:.1f} kg limit"
        )
    return cfg
