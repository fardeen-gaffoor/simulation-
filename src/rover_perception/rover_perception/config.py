"""Typed, validated ``perception.yaml``: module 4's single source of truth.

Everything module 4 does is decided here - which cameras exist, how the phase
manager rates them, what the bandwidth budget is, which detector backend runs,
how a pixel becomes an arena coordinate, how tracks are confirmed and decayed,
how AprilTags are detected and turned into a ``map -> odom`` correction, and
which thermal thresholds de-rate the cameras.

The file is parsed and range-checked here so a bad plan fails immediately at
startup, not halfway through a competition run.  This mirrors
``cat_arena/config/arena.yaml`` (module 1), ``rover_description/config/rover.yaml``
(module 2) and ``rover_dataset/config/dataset.yaml`` (module 3).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "PerceptionConfigError",
    "Rect",
    "CameraCfg",
    "ArenaCfg",
    "ArtifactCfg",
    "BandwidthCfg",
    "ThermalCfg",
    "CameraManagerCfg",
    "GroundCfg",
    "ClassicalCfg",
    "OnnxCfg",
    "DetectorCfg",
    "TrackingCfg",
    "TagDetectionCfg",
    "TagBoardCfg",
    "TagCfg",
    "LocalizationCfg",
    "HealthCfg",
    "PerceptionConfig",
    "load_config",
    "default_config_path",
    "package_share_dir",
    # wire contracts shared with the nodes and the tests
    "MODES",
    "ACTIVE_CAMERA_ROLE",
    "OBSTACLE_CLASS_IDS",
    "THERMAL_STATES",
]

#: The perception phases the camera manager understands.  A phase names the
#: camera that must be at full rate; every other camera drops to the idle rate.
MODES: Tuple[str, ...] = ("idle", "nav", "tag", "excavate", "dump", "degraded")

#: Which camera *role* gets the full rate in each mode.
ACTIVE_CAMERA_ROLE: Dict[str, str] = {
    "idle": "",
    "nav": "nav",
    "tag": "tag",
    "excavate": "excavation",
    "dump": "dump",
    "degraded": "",
}

#: YOLO class indices of module 3's dataset: 0 = boulder, 1 = crater.
OBSTACLE_CLASS_IDS: Dict[str, int] = {"boulder": 0, "crater": 1}

#: Thermal states, as published in PerceptionHealth.thermal_state.
THERMAL_STATES: Tuple[str, ...] = ("nominal", "warning", "throttling", "critical")


class PerceptionConfigError(ValueError):
    """Raised when ``perception.yaml`` is missing something or breaks a rule."""


# --------------------------------------------------------------------------- #
# scalar helpers
# --------------------------------------------------------------------------- #
def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise PerceptionConfigError(f"{where}: missing required key '{key}'")
    return section[key]


def _as_float(value: Any, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionConfigError(f"{where}: expected a number, got {value!r}") from exc


def _as_int(value: Any, where: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PerceptionConfigError(f"{where}: expected an integer, got {value!r}") from exc


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise PerceptionConfigError(f"{where}: expected true/false, got {value!r}")


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise PerceptionConfigError(f"{where}: expected a string, got {value!r}")
    return value


def _as_vec(value: Any, where: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise PerceptionConfigError(f"{where}: expected {length} numbers, got {value!r}")
    return tuple(_as_float(v, where) for v in value)


def _as_str_list(value: Any, where: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise PerceptionConfigError(f"{where}: expected a list of strings, got {value!r}")
    return [_as_str(v, where) for v in value]


def _section(raw: Dict[str, Any], key: str, where: str, required: bool = True) -> Dict[str, Any]:
    value = raw.get(key)
    if value is None:
        if required:
            raise PerceptionConfigError(f"{where}: missing required section '{key}'")
        return {}
    if not isinstance(value, dict):
        raise PerceptionConfigError(f"{where}.{key}: expected a mapping, got {value!r}")
    return value


def _positive(value: float, where: str) -> float:
    if value <= 0.0:
        raise PerceptionConfigError(f"{where}: must be > 0, got {value}")
    return value


def _ordered(section: Dict[str, float], keys: Sequence[str], where: str) -> None:
    previous_key = keys[0]
    for key in keys[1:]:
        if section[key] <= section[previous_key]:
            raise PerceptionConfigError(
                f"{where}: {key} ({section[key]}) must be greater than "
                f"{previous_key} ({section[previous_key]})"
            )
        previous_key = key


# --------------------------------------------------------------------------- #
# value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle in the map frame (metres)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1))

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x0 + margin <= x <= self.x1 - margin
            and self.y0 + margin <= y <= self.y1 - margin
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class ArenaCfg:
    """The arena, as module 4 needs to know it (bounds + zones + wall margin)."""

    frame_id: str
    robot_frame: str
    odom_frame: str
    ground_z: float
    length_x: float
    width_y: float
    origin_xy: Tuple[float, float]
    wall_height: float
    wall_margin_m: float
    zones: Dict[str, Rect]

    @property
    def bounds(self) -> Rect:
        ox, oy = self.origin_xy
        return Rect(ox, oy, ox + self.length_x, oy + self.width_y)

    def inside(self, x: float, y: float, margin: Optional[float] = None) -> bool:
        """True when ``(x, y)`` is inside the arena, shrunk by ``margin``.

        This is the wall mask: anything reported outside (or within the margin
        of) the arena rectangle is a wall, a reflection or a mis-projection.
        """
        return self.bounds.contains(x, y, self.wall_margin_m if margin is None else margin)

    def zone_of(self, x: float, y: float) -> Optional[str]:
        for name, rect in self.zones.items():
            if rect.contains(x, y):
                return name
        return None


@dataclass(frozen=True)
class CameraCfg:
    """One managed camera: name, role, topics and payload size."""

    name: str
    role: str
    topic: str
    camera_info_topic: str
    width: int
    height: int
    encoding: str
    bytes_per_pixel: int

    @property
    def payload_bytes(self) -> int:
        """Exact image payload size for the configured encoding."""
        return self.width * self.height * self.bytes_per_pixel

    @property
    def bits_per_pixel(self) -> float:
        return 8.0 * self.bytes_per_pixel


@dataclass(frozen=True)
class ArtifactCfg:
    """Where module 4 reads module 1/2 artefacts and writes its own."""

    calibration: str
    markers: str
    tags_file: str
    output_dir: str


@dataclass(frozen=True)
class BandwidthCfg:
    """The competition bandwidth rule: <= 4000 kbps average."""

    limit_kbps: float
    window_s: float
    hysteresis_kbps: float
    hard_limit_kbps: float
    derate_factors: Tuple[float, ...]


@dataclass(frozen=True)
class ThermalCfg:
    enabled: bool
    zone_paths: Tuple[str, ...]
    warn_c: float
    throttle_c: float
    critical_c: float
    throttle_steps: int
    poll_hz: float


@dataclass(frozen=True)
class CameraManagerCfg:
    enabled: bool
    mode_topic: str
    initial_mode: str
    phase_topic: str
    status_topic: str
    active_rate_hz: float
    idle_rate_hz: float
    min_rate_hz: float
    stale_after_s: float
    publish_hz: float
    bandwidth: BandwidthCfg
    thermal: ThermalCfg


@dataclass(frozen=True)
class GroundCfg:
    """Flat-ground projection policy (no depth sensor exists)."""

    min_range_m: float
    max_range_m: float
    contact_point: str
    crater_radius_scale: float
    boulder_radius_scale: float
    min_radius_m: float
    max_radius_m: float
    default_sigma_xy_m: float


@dataclass(frozen=True)
class ClassicalCfg:
    horizon_margin_px: int
    left_fraction: float
    right_fraction: float
    bottom_fraction: float
    delta_luma: float
    delta_chroma: float
    mad_k: float
    min_area_px: int
    max_area_px: int
    min_fill: float
    max_aspect: float
    max_bbox_fraction: float
    shadow_delta: float
    ambiguous_kind: str
    max_detections: int
    reference_rows: int


@dataclass(frozen=True)
class OnnxCfg:
    model: str
    providers: Tuple[str, ...]
    input_size: int
    confidence: float
    iou: float
    classes: Tuple[str, ...]
    max_detections: int
    letterbox_pad: int


@dataclass(frozen=True)
class DetectorCfg:
    enabled: bool
    camera: str
    backend: str
    publish_hz: float
    max_frame_age_s: float
    obstacles_topic: str
    stats_topic: str
    cloud_enabled: bool
    cloud_topic: str
    ground: GroundCfg
    classical: ClassicalCfg
    onnx: OnnxCfg


@dataclass(frozen=True)
class TrackingCfg:
    min_hits: int
    max_misses: int
    min_confidence: float
    associate_radius_m: float
    position_gain: float
    hit_gain: float
    decay_half_life_s: float
    crater_bias: float
    max_obstacles: int
    sigma_xy_m: float


@dataclass(frozen=True)
class TagDetectionCfg:
    quad_decimate: float
    quad_sigma: float
    nthreads: int
    refine_edges: bool
    decision_margin_min: float
    hamming_max: int
    max_detections: int
    debug_image: bool
    publish_debug_topic: str


@dataclass(frozen=True)
class TagBoardCfg:
    enabled: bool
    quiet_zone_modules: int
    protrusion_m: float
    lift_m: float
    texture_size_px: int
    spawn_timeout_s: float

    @property
    def standoff_m(self) -> float:
        """Total offset of the rendered tag plane from module 1's board centre.

        Module 1 draws the dark tag square as a box of
        ``1.4 * board_thickness`` centred on the board, so it protrudes
        ``0.7 * board_thickness`` (7 mm for the configured 10 mm board) in front
        of the board's centre plane.  The textured plane must sit in front of
        *that*, and the offset it is placed at is exactly the pose recorded in
        ``tags.yaml``, so the tag's stated position is never guessed.
        """
        return self.protrusion_m + self.lift_m


@dataclass(frozen=True)
class TagCfg:
    enabled: bool
    camera: str
    family: str
    topic: str
    detection: TagDetectionCfg
    board: TagBoardCfg


@dataclass(frozen=True)
class LocalizationCfg:
    enabled: bool
    odometry_topics: Tuple[str, ...]
    state_topic: str
    fix_topic: str
    publish_hz: float
    fix_sigma_xy_m: float
    fix_sigma_yaw_rad: float
    growth_m_per_m: float
    growth_rad_per_rad: float
    max_sigma_xy_m: float
    max_sigma_yaw_rad: float
    min_tag_range_m: float
    max_tag_range_m: float
    max_correction_m: float
    max_correction_yaw_rad: float
    xy_gain: float
    yaw_gain: float
    accept_margin_scale: float


@dataclass(frozen=True)
class HealthCfg:
    enabled: bool
    topic: str
    diagnostics_topic: str
    publish_hz: float
    cpu_warn_pct: float
    thermal_warn_c: float


# --------------------------------------------------------------------------- #
# the whole document
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PerceptionConfig:
    path: str
    sha256: str
    schema_version: int
    arena: ArenaCfg
    artifacts: ArtifactCfg
    cameras: Tuple[CameraCfg, ...]
    camera_manager: CameraManagerCfg
    detector: DetectorCfg
    tracking: TrackingCfg
    tags: TagCfg
    localization: LocalizationCfg
    health: HealthCfg
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    # -- convenience ------------------------------------------------------ #
    def camera(self, name: str) -> CameraCfg:
        for cam in self.cameras:
            if cam.name == name:
                return cam
        raise PerceptionConfigError(f"cameras: no camera named '{name}'")

    def camera_by_role(self, role: str) -> Optional[CameraCfg]:
        for cam in self.cameras:
            if cam.role == role:
                return cam
        return None

    @property
    def camera_names(self) -> Tuple[str, ...]:
        return tuple(cam.name for cam in self.cameras)

    def active_camera_for_mode(self, mode: str) -> Optional[str]:
        role = ACTIVE_CAMERA_ROLE.get(mode)
        if not role:
            return None
        cam = self.camera_by_role(role)
        return cam.name if cam is not None else None


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _parse_rect(value: Any, where: str) -> Rect:
    x0, y0, x1, y1 = _as_vec(value, where, 4)
    if x1 <= x0 or y1 <= y0:
        raise PerceptionConfigError(f"{where}: rectangle must have positive size, got {value!r}")
    return Rect(x0, y0, x1, y1)


def _parse_arena(raw: Dict[str, Any]) -> ArenaCfg:
    where = "arena"
    section = _section(raw, "arena", "config")
    zones_raw = section.get("zones", {})
    if not isinstance(zones_raw, dict):
        raise PerceptionConfigError(f"{where}.zones: expected a mapping, got {zones_raw!r}")
    zones = {str(k): _parse_rect(v, f"{where}.zones.{k}") for k, v in zones_raw.items()}
    length_x = _positive(_as_float(_require(section, "length_x", where), f"{where}.length_x"), f"{where}.length_x")
    width_y = _positive(_as_float(_require(section, "width_y", where), f"{where}.width_y"), f"{where}.width_y")
    origin = _as_vec(section.get("origin_xy", [0.0, 0.0]), f"{where}.origin_xy", 2)
    wall_margin = _as_float(section.get("wall_margin_m", 0.15), f"{where}.wall_margin_m")
    if wall_margin < 0.0:
        raise PerceptionConfigError(f"{where}.wall_margin_m: must be >= 0, got {wall_margin}")
    for name, rect in zones.items():
        ox, oy = origin
        if (
            rect.x0 < ox - 1e-9
            or rect.y0 < oy - 1e-9
            or rect.x1 > ox + length_x + 1e-9
            or rect.y1 > oy + width_y + 1e-9
        ):
            raise PerceptionConfigError(
                f"{where}.zones.{name}: {rect.as_tuple()} is outside the "
                f"{length_x} x {width_y} arena at {origin}"
            )
    return ArenaCfg(
        frame_id=_as_str(section.get("frame_id", "map"), f"{where}.frame_id"),
        robot_frame=_as_str(section.get("robot_frame", "base_link"), f"{where}.robot_frame"),
        odom_frame=_as_str(section.get("odom_frame", "odom"), f"{where}.odom_frame"),
        ground_z=_as_float(section.get("ground_z", 0.0), f"{where}.ground_z"),
        length_x=length_x,
        width_y=width_y,
        origin_xy=(origin[0], origin[1]),
        wall_height=_as_float(section.get("wall_height", 0.5), f"{where}.wall_height"),
        wall_margin_m=wall_margin,
        zones=zones,
    )


def _parse_artifacts(raw: Dict[str, Any]) -> ArtifactCfg:
    section = _section(raw, "artifacts", "config", required=False)
    return ArtifactCfg(
        calibration=_as_str(section.get("calibration", ""), "artifacts.calibration"),
        markers=_as_str(section.get("markers", ""), "artifacts.markers"),
        tags_file=_as_str(section.get("tags_file", "tags.yaml"), "artifacts.tags_file"),
        output_dir=_as_str(section.get("output_dir", ""), "artifacts.output_dir"),
    )


_ENCODING_BYTES = {
    "rgb8": 3,
    "bgr8": 3,
    "rgba8": 4,
    "bgra8": 4,
    "mono8": 1,
    "mono16": 2,
    "16uc1": 2,
    "32fc1": 4,
}


def _parse_cameras(raw: Dict[str, Any]) -> Tuple[CameraCfg, ...]:
    entries = raw.get("cameras")
    if not isinstance(entries, list) or not entries:
        raise PerceptionConfigError("cameras: expected a non-empty list")
    cameras: List[CameraCfg] = []
    seen = set()
    for index, entry in enumerate(entries):
        where = f"cameras[{index}]"
        if not isinstance(entry, dict):
            raise PerceptionConfigError(f"{where}: expected a mapping, got {entry!r}")
        name = _as_str(_require(entry, "name", where), f"{where}.name")
        if name in seen:
            raise PerceptionConfigError(f"{where}.name: duplicate camera '{name}'")
        seen.add(name)
        encoding = _as_str(entry.get("encoding", "rgb8"), f"{where}.encoding")
        if encoding not in _ENCODING_BYTES:
            raise PerceptionConfigError(
                f"{where}.encoding: unsupported encoding '{encoding}' "
                f"(known: {sorted(_ENCODING_BYTES)})"
            )
        topic = _as_str(_require(entry, "topic", where), f"{where}.topic")
        camera_info = _as_str(
            entry.get("camera_info_topic", f"{topic}/camera_info"), f"{where}.camera_info_topic"
        )
        width = _as_int(_require(entry, "width", where), f"{where}.width")
        height = _as_int(_require(entry, "height", where), f"{where}.height")
        if width <= 0 or height <= 0:
            raise PerceptionConfigError(f"{where}: width/height must be > 0, got {width}x{height}")
        cameras.append(
            CameraCfg(
                name=name,
                role=_as_str(entry.get("role", name), f"{where}.role"),
                topic=topic,
                camera_info_topic=camera_info,
                width=width,
                height=height,
                encoding=encoding,
                bytes_per_pixel=_ENCODING_BYTES[encoding],
            )
        )
    roles = [cam.role for cam in cameras]
    if len(set(roles)) != len(roles):
        raise PerceptionConfigError(f"cameras: roles must be unique, got {roles}")
    for role in ACTIVE_CAMERA_ROLE.values():
        if role and role not in roles:
            raise PerceptionConfigError(
                f"cameras: no camera has role '{role}', but a perception mode needs it"
            )
    return tuple(cameras)


def _parse_camera_manager(raw: Dict[str, Any], cameras: Tuple[CameraCfg, ...]) -> CameraManagerCfg:
    where = "camera_manager"
    section = _section(raw, "camera_manager", "config")
    active = _positive(_as_float(_require(section, "active_rate_hz", where), f"{where}.active_rate_hz"), f"{where}.active_rate_hz")
    idle = _as_float(_require(section, "idle_rate_hz", where), f"{where}.idle_rate_hz")
    min_rate = _as_float(section.get("min_rate_hz", 1.0), f"{where}.min_rate_hz")
    if idle < 0.0 or min_rate < 0.0:
        raise PerceptionConfigError(f"{where}: idle_rate_hz and min_rate_hz must be >= 0")
    if idle > active:
        raise PerceptionConfigError(
            f"{where}.idle_rate_hz ({idle}) must not exceed active_rate_hz ({active})"
        )
    if 0.0 < idle < min_rate:
        raise PerceptionConfigError(
            f"{where}.idle_rate_hz ({idle}) is below min_rate_hz ({min_rate}); a streaming "
            "camera must not be commanded below the minimum (use 0 to switch it off)"
        )
    initial = _as_str(section.get("initial_mode", "nav"), f"{where}.initial_mode")
    if initial not in MODES:
        raise PerceptionConfigError(f"{where}.initial_mode: '{initial}' not in {MODES}")

    bw_section = _section(section, "bandwidth", where)
    bw_where = f"{where}.bandwidth"
    limit = _positive(_as_float(_require(bw_section, "limit_kbps", bw_where), f"{bw_where}.limit_kbps"), f"{bw_where}.limit_kbps")
    hard = _as_float(bw_section.get("hard_limit_kbps", 1.5 * limit), f"{bw_where}.hard_limit_kbps")
    if hard < limit:
        raise PerceptionConfigError(
            f"{bw_where}.hard_limit_kbps ({hard}) must be >= limit_kbps ({limit})"
        )
    window = _positive(_as_float(bw_section.get("window_s", 10.0), f"{bw_where}.window_s"), f"{bw_where}.window_s")
    hysteresis = _as_float(bw_section.get("hysteresis_kbps", 0.0), f"{bw_where}.hysteresis_kbps")
    if hysteresis < 0.0:
        raise PerceptionConfigError(f"{bw_where}.hysteresis_kbps: must be >= 0")
    factors = _as_vec_tuple(bw_section.get("derate_factors", [1.0, 0.7, 0.5, 0.35, 0.2, 0.0]), f"{bw_where}.derate_factors")
    if len(factors) < 2:
        raise PerceptionConfigError(f"{bw_where}.derate_factors: need at least two steps")
    for factor in factors:
        if not 0.0 <= factor <= 1.0:
            raise PerceptionConfigError(f"{bw_where}.derate_factors: {factor} is not in [0, 1]")
    for lo, hi in zip(factors, factors[1:]):
        if hi > lo:
            raise PerceptionConfigError(
                f"{bw_where}.derate_factors: must be non-increasing, got {factors}"
            )
    if factors[0] != 1.0:
        raise PerceptionConfigError(f"{bw_where}.derate_factors: the first step must be 1.0")

    th_section = _section(section, "thermal", where, required=False)
    th_where = f"{where}.thermal"
    thresholds = {
        "warn_c": _as_float(th_section.get("warn_c", 75.0), f"{th_where}.warn_c"),
        "throttle_c": _as_float(th_section.get("throttle_c", 85.0), f"{th_where}.throttle_c"),
        "critical_c": _as_float(th_section.get("critical_c", 95.0), f"{th_where}.critical_c"),
    }
    _ordered(thresholds, ("warn_c", "throttle_c", "critical_c"), th_where)
    thermal = ThermalCfg(
        enabled=_as_bool(th_section.get("enabled", True), f"{th_where}.enabled"),
        zone_paths=tuple(_as_str_list(th_section.get("zone_paths", []), f"{th_where}.zone_paths")),
        warn_c=thresholds["warn_c"],
        throttle_c=thresholds["throttle_c"],
        critical_c=thresholds["critical_c"],
        throttle_steps=_as_int(th_section.get("throttle_steps", 2), f"{th_where}.throttle_steps"),
        poll_hz=_positive(_as_float(th_section.get("poll_hz", 1.0), f"{th_where}.poll_hz"), f"{th_where}.poll_hz"),
    )
    if thermal.throttle_steps < 1:
        raise PerceptionConfigError(f"{th_where}.throttle_steps: must be >= 1")
    if thermal.throttle_steps >= len(factors):
        raise PerceptionConfigError(
            f"{th_where}.throttle_steps ({thermal.throttle_steps}) must be smaller than the "
            f"number of derate steps ({len(factors)})"
        )

    return CameraManagerCfg(
        enabled=_as_bool(section.get("enabled", True), f"{where}.enabled"),
        mode_topic=_as_str(section.get("mode_topic", "/perception/mode"), f"{where}.mode_topic"),
        initial_mode=initial,
        phase_topic=_as_str(section.get("phase_topic", "/perception/camera_phase"), f"{where}.phase_topic"),
        status_topic=_as_str(section.get("status_topic", "/perception/camera_status"), f"{where}.status_topic"),
        active_rate_hz=active,
        idle_rate_hz=idle,
        min_rate_hz=min_rate,
        stale_after_s=_positive(_as_float(section.get("stale_after_s", 2.0), f"{where}.stale_after_s"), f"{where}.stale_after_s"),
        publish_hz=_positive(_as_float(section.get("publish_hz", 5.0), f"{where}.publish_hz"), f"{where}.publish_hz"),
        bandwidth=BandwidthCfg(
            limit_kbps=limit,
            window_s=window,
            hysteresis_kbps=hysteresis,
            hard_limit_kbps=hard,
            derate_factors=factors,
        ),
        thermal=thermal,
    )


def _as_vec_tuple(value: Any, where: str) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise PerceptionConfigError(f"{where}: expected a non-empty list of numbers, got {value!r}")
    return tuple(_as_float(v, where) for v in value)


def _parse_detector(raw: Dict[str, Any], cameras: Tuple[CameraCfg, ...]) -> DetectorCfg:
    where = "detector"
    section = _section(raw, "detector", "config")
    names = {cam.name for cam in cameras}
    camera = _as_str(_require(section, "camera", where), f"{where}.camera")
    if camera not in names:
        raise PerceptionConfigError(f"{where}.camera: '{camera}' is not a configured camera")
    backend = _as_str(section.get("backend", "classical"), f"{where}.backend")
    if backend not in ("classical", "onnx"):
        raise PerceptionConfigError(f"{where}.backend: '{backend}' is not classical|onnx")

    ground_section = _section(section, "ground", where, required=False)
    ground_where = f"{where}.ground"
    min_range = _positive(_as_float(ground_section.get("min_range_m", 0.35), f"{ground_where}.min_range_m"), f"{ground_where}.min_range_m")
    max_range = _positive(_as_float(ground_section.get("max_range_m", 6.5), f"{ground_where}.max_range_m"), f"{ground_where}.max_range_m")
    if max_range <= min_range:
        raise PerceptionConfigError(f"{ground_where}: max_range_m must exceed min_range_m")
    contact = _as_str(ground_section.get("contact_point", "bbox_bottom_center"), f"{ground_where}.contact_point")
    if contact != "bbox_bottom_center":
        raise PerceptionConfigError(
            f"{ground_where}.contact_point: only 'bbox_bottom_center' is implemented, got '{contact}'"
        )
    min_radius = _as_float(ground_section.get("min_radius_m", 0.08), f"{ground_where}.min_radius_m")
    max_radius = _positive(_as_float(ground_section.get("max_radius_m", 0.45), f"{ground_where}.max_radius_m"), f"{ground_where}.max_radius_m")
    if max_radius <= min_radius:
        raise PerceptionConfigError(f"{ground_where}: max_radius_m must exceed min_radius_m")
    ground = GroundCfg(
        min_range_m=min_range,
        max_range_m=max_range,
        contact_point=contact,
        crater_radius_scale=_positive(_as_float(ground_section.get("crater_radius_scale", 1.15), f"{ground_where}.crater_radius_scale"), f"{ground_where}.crater_radius_scale"),
        boulder_radius_scale=_positive(_as_float(ground_section.get("boulder_radius_scale", 1.0), f"{ground_where}.boulder_radius_scale"), f"{ground_where}.boulder_radius_scale"),
        min_radius_m=min_radius,
        max_radius_m=max_radius,
        default_sigma_xy_m=_positive(_as_float(ground_section.get("default_sigma_xy_m", 0.12), f"{ground_where}.default_sigma_xy_m"), f"{ground_where}.default_sigma_xy_m"),
    )

    classical_section = _section(section, "classical", where, required=False)
    classical_where = f"{where}.classical"
    ambiguous = _as_str(classical_section.get("ambiguous_kind", "crater"), f"{classical_where}.ambiguous_kind")
    if ambiguous not in ("crater", "boulder", "drop"):
        raise PerceptionConfigError(
            f"{classical_where}.ambiguous_kind: '{ambiguous}' is not crater|boulder|drop"
        )
    left = _as_float(classical_section.get("left_fraction", 0.0), f"{classical_where}.left_fraction")
    right = _as_float(classical_section.get("right_fraction", 1.0), f"{classical_where}.right_fraction")
    bottom = _as_float(classical_section.get("bottom_fraction", 1.0), f"{classical_where}.bottom_fraction")
    for name, value in (("left_fraction", left), ("right_fraction", right), ("bottom_fraction", bottom)):
        if not 0.0 <= value <= 1.0:
            raise PerceptionConfigError(f"{classical_where}.{name}: must be in [0, 1], got {value}")
    if right <= left:
        raise PerceptionConfigError(f"{classical_where}: right_fraction must exceed left_fraction")
    classical = ClassicalCfg(
        horizon_margin_px=_as_int(classical_section.get("horizon_margin_px", 4), f"{classical_where}.horizon_margin_px"),
        left_fraction=left,
        right_fraction=right,
        bottom_fraction=bottom,
        delta_luma=_positive(_as_float(classical_section.get("delta_luma", 18.0), f"{classical_where}.delta_luma"), f"{classical_where}.delta_luma"),
        delta_chroma=_positive(_as_float(classical_section.get("delta_chroma", 14.0), f"{classical_where}.delta_chroma"), f"{classical_where}.delta_chroma"),
        mad_k=_positive(_as_float(classical_section.get("mad_k", 2.5), f"{classical_where}.mad_k"), f"{classical_where}.mad_k"),
        min_area_px=_as_int(classical_section.get("min_area_px", 60), f"{classical_where}.min_area_px"),
        max_area_px=_as_int(classical_section.get("max_area_px", 60000), f"{classical_where}.max_area_px"),
        min_fill=_as_float(classical_section.get("min_fill", 0.45), f"{classical_where}.min_fill"),
        max_aspect=_positive(_as_float(classical_section.get("max_aspect", 3.0), f"{classical_where}.max_aspect"), f"{classical_where}.max_aspect"),
        max_bbox_fraction=_as_float(classical_section.get("max_bbox_fraction", 0.6), f"{classical_where}.max_bbox_fraction"),
        shadow_delta=_as_float(classical_section.get("shadow_delta", 8.0), f"{classical_where}.shadow_delta"),
        ambiguous_kind=ambiguous,
        max_detections=_as_int(classical_section.get("max_detections", 16), f"{classical_where}.max_detections"),
        reference_rows=_as_int(classical_section.get("reference_rows", 24), f"{classical_where}.reference_rows"),
    )
    if classical.min_area_px < 1:
        raise PerceptionConfigError(f"{classical_where}.min_area_px: must be >= 1")
    if classical.max_area_px <= classical.min_area_px:
        raise PerceptionConfigError(f"{classical_where}.max_area_px must exceed min_area_px")
    if not 0.0 < classical.min_fill <= 1.0:
        raise PerceptionConfigError(f"{classical_where}.min_fill: must be in (0, 1]")
    if not 0.0 < classical.max_bbox_fraction <= 1.0:
        raise PerceptionConfigError(f"{classical_where}.max_bbox_fraction: must be in (0, 1]")
    if classical.reference_rows < 1:
        raise PerceptionConfigError(f"{classical_where}.reference_rows: must be >= 1")

    onnx_section = _section(section, "onnx", where, required=False)
    onnx_where = f"{where}.onnx"
    classes = tuple(_as_str_list(onnx_section.get("classes", ["boulder", "crater"]), f"{onnx_where}.classes"))
    if list(classes) != ["boulder", "crater"]:
        raise PerceptionConfigError(
            f"{onnx_where}.classes: must be exactly ['boulder', 'crater'] to match module 3's "
            f"dataset class order (0 = boulder, 1 = crater), got {list(classes)}"
        )
    onnx = OnnxCfg(
        model=_as_str(onnx_section.get("model", ""), f"{onnx_where}.model"),
        providers=tuple(_as_str_list(onnx_section.get("providers", ["CPUExecutionProvider"]), f"{onnx_where}.providers")),
        input_size=_as_int(onnx_section.get("input_size", 640), f"{onnx_where}.input_size"),
        confidence=_as_float(onnx_section.get("confidence", 0.25), f"{onnx_where}.confidence"),
        iou=_as_float(onnx_section.get("iou", 0.45), f"{onnx_where}.iou"),
        classes=classes,
        max_detections=_as_int(onnx_section.get("max_detections", 32), f"{onnx_where}.max_detections"),
        letterbox_pad=_as_int(onnx_section.get("letterbox_pad", 114), f"{onnx_where}.letterbox_pad"),
    )
    if not 0.0 < onnx.confidence < 1.0:
        raise PerceptionConfigError(f"{onnx_where}.confidence: must be in (0, 1), got {onnx.confidence}")
    if not 0.0 < onnx.iou < 1.0:
        raise PerceptionConfigError(f"{onnx_where}.iou: must be in (0, 1), got {onnx.iou}")
    if onnx.input_size % 32 != 0:
        raise PerceptionConfigError(
            f"{onnx_where}.input_size: YOLO backbones need a multiple of 32, got {onnx.input_size}"
        )
    if backend == "onnx" and not onnx.model:
        raise PerceptionConfigError(
            f"{where}.backend is 'onnx' but {onnx_where}.model is empty; run module 3's "
            "finetune.py and export a model first (see docs/detector.md)"
        )

    return DetectorCfg(
        enabled=_as_bool(section.get("enabled", True), f"{where}.enabled"),
        camera=camera,
        backend=backend,
        publish_hz=_positive(_as_float(section.get("publish_hz", 10.0), f"{where}.publish_hz"), f"{where}.publish_hz"),
        max_frame_age_s=_positive(_as_float(section.get("max_frame_age_s", 0.5), f"{where}.max_frame_age_s"), f"{where}.max_frame_age_s"),
        obstacles_topic=_as_str(section.get("obstacles_topic", "/perception/obstacles"), f"{where}.obstacles_topic"),
        stats_topic=_as_str(section.get("stats_topic", "/perception/detector_stats"), f"{where}.stats_topic"),
        cloud_enabled=_as_bool(_section(section, "cloud", where, required=False).get("enabled", True), f"{where}.cloud.enabled"),
        cloud_topic=_as_str(
            _section(section, "cloud", where, required=False).get("topic", "/perception/obstacles_cloud"),
            f"{where}.cloud.topic",
        ),
        ground=ground,
        classical=classical,
        onnx=onnx,
    )


def _parse_tracking(raw: Dict[str, Any]) -> TrackingCfg:
    where = "tracking"
    section = _section(raw, "tracking", "config")
    gain = _as_float(section.get("position_gain", 0.5), f"{where}.position_gain")
    hit_gain = _as_float(section.get("hit_gain", 0.45), f"{where}.hit_gain")
    bias = _as_float(section.get("crater_bias", 0.15), f"{where}.crater_bias")
    for name, value in (("position_gain", gain), ("hit_gain", hit_gain)):
        if not 0.0 < value <= 1.0:
            raise PerceptionConfigError(f"{where}.{name}: must be in (0, 1], got {value}")
    if not 0.0 <= bias <= 1.0:
        raise PerceptionConfigError(f"{where}.crater_bias: must be in [0, 1], got {bias}")
    min_conf = _as_float(section.get("min_confidence", 0.25), f"{where}.min_confidence")
    if not 0.0 < min_conf < 1.0:
        raise PerceptionConfigError(f"{where}.min_confidence: must be in (0, 1), got {min_conf}")
    return TrackingCfg(
        min_hits=_as_int(section.get("min_hits", 2), f"{where}.min_hits"),
        max_misses=_as_int(section.get("max_misses", 3), f"{where}.max_misses"),
        min_confidence=min_conf,
        associate_radius_m=_positive(_as_float(section.get("associate_radius_m", 0.30), f"{where}.associate_radius_m"), f"{where}.associate_radius_m"),
        position_gain=gain,
        hit_gain=hit_gain,
        decay_half_life_s=_positive(_as_float(section.get("decay_half_life_s", 1.5), f"{where}.decay_half_life_s"), f"{where}.decay_half_life_s"),
        crater_bias=bias,
        max_obstacles=_as_int(section.get("max_obstacles", 64), f"{where}.max_obstacles"),
        sigma_xy_m=_positive(_as_float(section.get("sigma_xy_m", 0.12), f"{where}.sigma_xy_m"), f"{where}.sigma_xy_m"),
    )


def _parse_tags(raw: Dict[str, Any], cameras: Tuple[CameraCfg, ...]) -> TagCfg:
    where = "tags"
    section = _section(raw, "tags", "config")
    names = {cam.name for cam in cameras}
    camera = _as_str(_require(section, "camera", where), f"{where}.camera")
    if camera not in names:
        raise PerceptionConfigError(f"{where}.camera: '{camera}' is not a configured camera")
    det_section = _section(section, "detection", where, required=False)
    det_where = f"{where}.detection"
    quad_decimate = _positive(_as_float(det_section.get("quad_decimate", 1.0), f"{det_where}.quad_decimate"), f"{det_where}.quad_decimate")
    detection = TagDetectionCfg(
        quad_decimate=quad_decimate,
        quad_sigma=_as_float(det_section.get("quad_sigma", 0.0), f"{det_where}.quad_sigma"),
        nthreads=_as_int(det_section.get("nthreads", 2), f"{det_where}.nthreads"),
        refine_edges=_as_bool(det_section.get("refine_edges", True), f"{det_where}.refine_edges"),
        decision_margin_min=_as_float(det_section.get("decision_margin_min", 15.0), f"{det_where}.decision_margin_min"),
        hamming_max=_as_int(det_section.get("hamming_max", 1), f"{det_where}.hamming_max"),
        max_detections=_as_int(det_section.get("max_detections", 8), f"{det_where}.max_detections"),
        debug_image=_as_bool(det_section.get("debug_image", False), f"{det_where}.debug_image"),
        publish_debug_topic=_as_str(det_section.get("publish_debug_topic", "/perception/tag_debug"), f"{det_where}.publish_debug_topic"),
    )
    if detection.nthreads < 1:
        raise PerceptionConfigError(f"{det_where}.nthreads: must be >= 1")
    if detection.hamming_max < 0:
        raise PerceptionConfigError(f"{det_where}.hamming_max: must be >= 0")

    board_section = _section(section, "board", where, required=False)
    board_where = f"{where}.board"
    board = TagBoardCfg(
        enabled=_as_bool(board_section.get("enabled", True), f"{board_where}.enabled"),
        quiet_zone_modules=_as_int(board_section.get("quiet_zone_modules", 1), f"{board_where}.quiet_zone_modules"),
        protrusion_m=_as_float(board_section.get("protrusion_m", 0.007), f"{board_where}.protrusion_m"),
        lift_m=_as_float(board_section.get("lift_m", 0.003), f"{board_where}.lift_m"),
        texture_size_px=_as_int(board_section.get("texture_size_px", 512), f"{board_where}.texture_size_px"),
        spawn_timeout_s=_positive(_as_float(board_section.get("spawn_timeout_s", 60.0), f"{board_where}.spawn_timeout_s"), f"{board_where}.spawn_timeout_s"),
    )
    if board.quiet_zone_modules < 1:
        raise PerceptionConfigError(f"{board_where}.quiet_zone_modules: must be >= 1")
    if board.texture_size_px < 64:
        raise PerceptionConfigError(f"{board_where}.texture_size_px: must be >= 64")
    if board.protrusion_m < 0.0:
        raise PerceptionConfigError(
            f"{board_where}.protrusion_m: must be >= 0 (it mirrors module 1's "
            "0.5 * board_thickness * 1.4)"
        )
    if board.lift_m <= 0.0:
        raise PerceptionConfigError(f"{board_where}.lift_m: must be > 0 to clear the arena board")

    return TagCfg(
        enabled=_as_bool(section.get("enabled", True), f"{where}.enabled"),
        camera=camera,
        family=_as_str(section.get("family", "tag36h11"), f"{where}.family"),
        topic=_as_str(section.get("topic", "/perception/tag_detections"), f"{where}.topic"),
        detection=detection,
        board=board,
    )


def _parse_localization(raw: Dict[str, Any]) -> LocalizationCfg:
    where = "localization"
    section = _section(raw, "localization", "config")
    sigma = _section(section, "sigma", where, required=False)
    sigma_where = f"{where}.sigma"
    fix_xy = _positive(_as_float(sigma.get("fix_xy_m", 0.04), f"{sigma_where}.fix_xy_m"), f"{sigma_where}.fix_xy_m")
    fix_yaw = _positive(_as_float(sigma.get("fix_yaw_rad", 0.06), f"{sigma_where}.fix_yaw_rad"), f"{sigma_where}.fix_yaw_rad")
    growth_m = _as_float(sigma.get("growth_m_per_m", 0.05), f"{sigma_where}.growth_m_per_m")
    growth_r = _as_float(sigma.get("growth_rad_per_rad", 0.03), f"{sigma_where}.growth_rad_per_rad")
    if growth_m < 0.0 or growth_r < 0.0:
        raise PerceptionConfigError(f"{sigma_where}: growth rates must be >= 0")
    max_xy = _positive(_as_float(sigma.get("max_xy_m", 3.0), f"{sigma_where}.max_xy_m"), f"{sigma_where}.max_xy_m")
    max_yaw = _positive(_as_float(sigma.get("max_yaw_rad", 1.2), f"{sigma_where}.max_yaw_rad"), f"{sigma_where}.max_yaw_rad")
    if fix_xy > max_xy:
        raise PerceptionConfigError(f"{sigma_where}: fix_xy_m must not exceed max_xy_m")
    if fix_yaw > max_yaw:
        raise PerceptionConfigError(f"{sigma_where}: fix_yaw_rad must not exceed max_yaw_rad")
    xy_gain = _as_float(section.get("xy_gain", 1.0), f"{where}.xy_gain")
    yaw_gain = _as_float(section.get("yaw_gain", 1.0), f"{where}.yaw_gain")
    for name, value in (("xy_gain", xy_gain), ("yaw_gain", yaw_gain)):
        if not 0.0 < value <= 1.0:
            raise PerceptionConfigError(f"{where}.{name}: must be in (0, 1], got {value}")
    min_range = _positive(_as_float(section.get("min_tag_range_m", 0.25), f"{where}.min_tag_range_m"), f"{where}.min_tag_range_m")
    max_range = _positive(_as_float(section.get("max_tag_range_m", 6.0), f"{where}.max_tag_range_m"), f"{where}.max_tag_range_m")
    if max_range <= min_range:
        raise PerceptionConfigError(f"{where}: max_tag_range_m must exceed min_tag_range_m")
    return LocalizationCfg(
        enabled=_as_bool(section.get("enabled", True), f"{where}.enabled"),
        odometry_topics=tuple(_as_str_list(section.get("odometry_topics", ["/odometry/filtered", "/odom"]), f"{where}.odometry_topics")),
        state_topic=_as_str(section.get("state_topic", "/perception/localization_state"), f"{where}.state_topic"),
        fix_topic=_as_str(section.get("fix_topic", "/localization/tag_fix"), f"{where}.fix_topic"),
        publish_hz=_positive(_as_float(section.get("publish_hz", 20.0), f"{where}.publish_hz"), f"{where}.publish_hz"),
        fix_sigma_xy_m=fix_xy,
        fix_sigma_yaw_rad=fix_yaw,
        growth_m_per_m=growth_m,
        growth_rad_per_rad=growth_r,
        max_sigma_xy_m=max_xy,
        max_sigma_yaw_rad=max_yaw,
        min_tag_range_m=min_range,
        max_tag_range_m=max_range,
        max_correction_m=_positive(_as_float(section.get("max_correction_m", 1.5), f"{where}.max_correction_m"), f"{where}.max_correction_m"),
        max_correction_yaw_rad=_positive(_as_float(section.get("max_correction_yaw_rad", 0.6), f"{where}.max_correction_yaw_rad"), f"{where}.max_correction_yaw_rad"),
        xy_gain=xy_gain,
        yaw_gain=yaw_gain,
        accept_margin_scale=_positive(_as_float(section.get("accept_margin_scale", 1.0), f"{where}.accept_margin_scale"), f"{where}.accept_margin_scale"),
    )


def _parse_health(raw: Dict[str, Any]) -> HealthCfg:
    where = "health"
    section = _section(raw, "health", "config", required=False)
    return HealthCfg(
        enabled=_as_bool(section.get("enabled", True), f"{where}.enabled"),
        topic=_as_str(section.get("topic", "/perception/health"), f"{where}.topic"),
        diagnostics_topic=_as_str(section.get("diagnostics_topic", "/diagnostics"), f"{where}.diagnostics_topic"),
        publish_hz=_positive(_as_float(section.get("publish_hz", 1.0), f"{where}.publish_hz"), f"{where}.publish_hz"),
        cpu_warn_pct=_as_float(section.get("cpu_warn_pct", 90.0), f"{where}.cpu_warn_pct"),
        thermal_warn_c=_as_float(section.get("thermal_warn_c", 80.0), f"{where}.thermal_warn_c"),
    )


def config_from_mapping(raw: Dict[str, Any], path: str = "<memory>", sha256: str = "") -> PerceptionConfig:
    """Validate an already-parsed mapping (used by the tests and by ``load_config``)."""
    if not isinstance(raw, dict):
        raise PerceptionConfigError(f"{path}: top level must be a mapping")
    schema_version = _as_int(_require(raw, "schema_version", "config"), "config.schema_version")
    if schema_version != 1:
        raise PerceptionConfigError(f"config.schema_version: expected 1, got {schema_version}")
    cameras = _parse_cameras(raw)
    return PerceptionConfig(
        path=path,
        sha256=sha256,
        schema_version=schema_version,
        arena=_parse_arena(raw),
        artifacts=_parse_artifacts(raw),
        cameras=cameras,
        camera_manager=_parse_camera_manager(raw, cameras),
        detector=_parse_detector(raw, cameras),
        tracking=_parse_tracking(raw),
        tags=_parse_tags(raw, cameras),
        localization=_parse_localization(raw),
        health=_parse_health(raw),
        raw=raw,
    )


def load_config(path: Optional[str] = None) -> PerceptionConfig:
    """Load and validate ``perception.yaml`` (default: this package's copy)."""
    import yaml

    resolved = path or default_config_path()
    if not os.path.isfile(resolved):
        raise PerceptionConfigError(f"config file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        blob = handle.read()
    try:
        raw = yaml.safe_load(blob)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised by a config test
        raise PerceptionConfigError(f"{resolved}: not valid YAML: {exc}") from exc
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return config_from_mapping(raw, path=resolved, sha256=digest)


def package_share_dir() -> str:
    """This package's share directory (install space first, source tree second)."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory("rover_perception")
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def default_config_path() -> str:
    return os.path.join(package_share_dir(), "config", "perception.yaml")
