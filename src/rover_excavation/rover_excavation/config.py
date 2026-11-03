"""Typed, validated ``excavation.yaml``: module 6's single source of truth.

Everything module 6 does is decided here: how the ladder, belt and tub move,
how fast sand enters and leaves the tub, where a dump lands, how big the berm
target is and what counts as a finished berm, and how the actuators are reached
(Gazebo joint controllers, a serial link, or nothing at all).

This mirrors ``cat_arena/config/arena.yaml`` (module 1),
``rover_description/config/rover.yaml`` (module 2),
``rover_dataset/config/dataset.yaml`` (module 3),
``rover_perception/config/perception.yaml`` (module 4) and
``rover_navigation/config/navigation.yaml`` (module 5): one file, parsed and
range-checked once at startup, so a bad plan fails immediately rather than
halfway through a run.

**The arena geometry is deliberately duplicated here**, because on the rover
computer ``cat_arena`` is not installed (there is no Gazebo there) and module 6
still has to know where the berm target is.  A test cross-checks the copy
against ``cat_arena/config/arena.yaml`` - the same arrangement module 5 uses.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "ExcavationConfigError",
    "Rect",
    "FramesCfg",
    "ArenaCfg",
    "RobotCfg",
    "NavDumpCfg",
    "LadderCfg",
    "BeltCfg",
    "TubCfg",
    "InterlocksCfg",
    "MechanismCfg",
    "DigRateCfg",
    "DumpSandCfg",
    "BermCfg",
    "SandCfg",
    "DigCycleCfg",
    "DumpCycleCfg",
    "CycleCfg",
    "SerialCfg",
    "BackendCfg",
    "VisualCfg",
    "TopicsCfg",
    "ExcavationConfig",
    "config_from_mapping",
    "load_config",
    "default_config_path",
    "package_share_dir",
    "repose_basal_radius_m",
    "dig_rate_m3_s",
    "WORK_PHASES",
    "STAGES",
    "STAGE_NAMES",
    "MECHANISM_STATES",
    "BACKEND_KINDS",
]

#: ``WorkCycle`` goal ``phase`` values module 6 answers, and the name of the
#: perception mode module 5 sets while it is running (they are the same word).
WORK_PHASES: Tuple[str, ...] = ("excavate", "dump")

#: ``WorkCycle`` feedback ``stage`` names, indexed by the action's constants.
STAGES: Tuple[str, ...] = ("starting", "moving", "working", "retracting", "done")
STAGE_NAMES: Tuple[str, ...] = ("STARTING", "MOVING", "WORKING", "RETRACTING", "DONE")

#: ``MechanismStatus.state`` names, indexed by the message's constants.
MECHANISM_STATES: Tuple[str, ...] = ("stowed", "deploying", "digging", "dumping", "retracting", "fault")

#: ``backend.kind`` values.
BACKEND_KINDS: Tuple[str, ...] = ("sim", "hardware", "null")


class ExcavationConfigError(ValueError):
    """Raised when ``excavation.yaml`` is missing, malformed or out of range."""


# --------------------------------------------------------------------------- #
# coercion helpers - each one names the offending key in its error
# --------------------------------------------------------------------------- #
def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section or section[key] is None:
        raise ExcavationConfigError(f"{where}: missing required key '{key}'")
    return section[key]


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExcavationConfigError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ExcavationConfigError(f"{where}: expected an integer, got {value!r}")
    return int(value)


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ExcavationConfigError(f"{where}: expected true/false, got {value!r}")


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise ExcavationConfigError(f"{where}: expected a string, got {value!r}")
    return value


def _as_vec(value: Any, where: str, length: int) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ExcavationConfigError(f"{where}: expected {length} numbers, got {value!r}")
    if len(value) != length:
        raise ExcavationConfigError(f"{where}: expected {length} numbers, got {value!r}")
    return tuple(_as_float(v, where) for v in value)


def _as_enum(value: Any, options: Sequence[str], where: str) -> str:
    text = _as_str(value, where)
    if text not in options:
        raise ExcavationConfigError(f"{where}: expected one of {list(options)}, got {text!r}")
    return text


def _section(raw: Dict[str, Any], key: str, where: str = "", required: bool = True) -> Dict[str, Any]:
    value = raw.get(key)
    if value is None:
        if required:
            raise ExcavationConfigError(f"{where or 'excavation.yaml'}: missing required section '{key}'")
        return {}
    if not isinstance(value, dict):
        raise ExcavationConfigError(f"{where or 'excavation.yaml'}.{key}: expected a mapping, got {value!r}")
    return value


def _positive(value: float, where: str) -> float:
    if value <= 0.0:
        raise ExcavationConfigError(f"{where}: must be > 0, got {value}")
    return value


def _non_negative(value: float, where: str) -> float:
    if value < 0.0:
        raise ExcavationConfigError(f"{where}: must be >= 0, got {value}")
    return value


def _in_range(value: float, low: float, high: float, where: str) -> float:
    if not (low <= value <= high):
        raise ExcavationConfigError(f"{where}: must be within [{low}, {high}], got {value}")
    return value


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

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_sequence(cls, value: Any, where: str) -> "Rect":
        x0, y0, x1, y1 = _as_vec(value, where, 4)
        if x1 <= x0 or y1 <= y0:
            raise ExcavationConfigError(f"{where}: degenerate rectangle {value!r}")
        return cls(x0, y0, x1, y1)


@dataclass(frozen=True)
class FramesCfg:
    map: str
    base: str


@dataclass(frozen=True)
class ArenaCfg:
    length_x: float
    width_y: float
    sand_depth: float
    zones: Dict[str, Rect]

    @property
    def rect(self) -> Rect:
        return Rect(0.0, 0.0, self.length_x, self.width_y)

    def zone(self, name: str) -> Rect:
        try:
            return self.zones[name]
        except KeyError as exc:  # pragma: no cover - guarded by the loader
            raise ExcavationConfigError(f"arena.zones: no zone named '{name}'") from exc


@dataclass(frozen=True)
class RobotCfg:
    build_length_m: float
    build_width_m: float
    length_m: float
    width_m: float
    circumscribed_radius_m: float

    @property
    def half_length_m(self) -> float:
        """How far the *body* reaches fore and aft: the aft face is here."""
        return 0.5 * self.length_m

    @property
    def measured_circumscribed_radius_m(self) -> float:
        return 0.5 * math.hypot(self.length_m, self.width_m)


@dataclass(frozen=True)
class NavDumpCfg:
    """A copy of module 5's dump geometry, for the dump-lands-in-target test."""

    heading_rad: float
    standoff_m: float
    points: int
    inset_m: float


@dataclass(frozen=True)
class LadderCfg:
    joint: str
    pivot_x_m: float
    pivot_z_m: float
    ride_height_m: float
    length_m: float
    width_m: float
    frame_thickness_m: float
    stowed_pitch_rad: float
    dig_pitch_rad: float
    min_pitch_rad: float
    max_pitch_rad: float
    max_rate_radps: float

    @property
    def engaged_width_m(self) -> float:
        """The clear cut width: the ladder minus its two side frames."""
        return max(0.0, self.width_m - 2.0 * self.frame_thickness_m)

    @property
    def pivot_z_above_sand_m(self) -> float:
        """The pitch pivot's height above the **sand surface**, not above base_link.

        ``pivot_z_m`` is the pivot's height inside ``base_link`` and is copied
        from the URDF, where the drive wheels put the body one wheel radius
        (``ride_height_m``) above the sand.  Every depth figure has to be measured
        from the sand, so both terms are needed; using ``pivot_z_m`` alone would
        have the ladder start biting 0.10 m too early and overstate the cut depth,
        and therefore the dig rate, by roughly 4x.
        """
        return self.pivot_z_m + self.ride_height_m

    @property
    def min_dig_pitch_rad(self) -> float:
        """The shallowest pitch whose tip reaches the sand surface at all.

        Below this the tip is still above the ground, so the belt has nothing to
        bite.  It follows from the pivot height and the ladder length rather
        than being a separate tuning knob.
        """
        ratio = self.pivot_z_above_sand_m / self.length_m
        if ratio >= 1.0:
            return math.pi / 2.0
        return math.asin(max(-1.0, min(1.0, ratio)))

    def tip_depth_m(self, pitch_rad: float) -> float:
        """How far the ladder tip is below the sand surface at ``pitch_rad``.

        Negative means the tip is in the air.  Positive pitch tips the ladder
        down, matching the URDF joint convention.
        """
        return self.length_m * math.sin(pitch_rad) - self.pivot_z_above_sand_m


@dataclass(frozen=True)
class BeltCfg:
    joint: str
    roller_radius_m: float
    dig_velocity_radps: float
    dump_velocity_radps: float
    max_velocity_radps: float
    max_accel_radps2: float

    def surface_speed_mps(self, velocity_radps: float) -> float:
        """Signed belt surface speed: what the sand actually sees."""
        return velocity_radps * self.roller_radius_m


@dataclass(frozen=True)
class TubCfg:
    box_m: Tuple[float, float, float]
    fill_fraction: float
    capacity_m3: float
    gate_open_time_s: float

    @property
    def box_volume_m3(self) -> float:
        return self.box_m[0] * self.box_m[1] * self.box_m[2]


@dataclass(frozen=True)
class InterlocksCfg:
    travel_pitch_max_rad: float
    max_depth_m: float


@dataclass(frozen=True)
class MechanismCfg:
    rate_hz: float
    ladder: LadderCfg
    belt: BeltCfg
    tub: TubCfg
    interlocks: InterlocksCfg

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz


@dataclass(frozen=True)
class DigRateCfg:
    coefficient: float
    min_rate_m3_s: float
    max_rate_m3_s: float


@dataclass(frozen=True)
class DumpSandCfg:
    rate_m3_s: float
    repose_deg: float
    landing_margin_m: float
    spread_m: float

    @property
    def repose_rad(self) -> float:
        return math.radians(self.repose_deg)


def repose_basal_radius_m(volume_m3: float, repose_rad: float) -> float:
    """The base radius of a cone of ``volume_m3`` at the angle of repose.

    ``V = (pi/3) r^2 h`` with ``h = r tan(theta)``, so ``r = (3V/(pi tan theta))^(1/3)``.
    """
    if volume_m3 <= 0.0:
        return 0.0
    tan = math.tan(max(1e-3, min(repose_rad, math.pi / 2.0 - 1e-3)))
    return (3.0 * volume_m3 / (math.pi * tan)) ** (1.0 / 3.0)


@dataclass(frozen=True)
class BermCfg:
    target_height_m: float
    cell_m: float


@dataclass(frozen=True)
class SandCfg:
    dig: DigRateCfg
    dump: DumpSandCfg
    berm: BermCfg


@dataclass(frozen=True)
class DigCycleCfg:
    min_s: float
    timeout_s: float
    target_fill: float


@dataclass(frozen=True)
class DumpCycleCfg:
    min_s: float
    timeout_s: float
    target_empty: float


@dataclass(frozen=True)
class CycleCfg:
    settle_s: float
    dig: DigCycleCfg
    dump: DumpCycleCfg


@dataclass(frozen=True)
class SerialCfg:
    port: str
    baud: int
    timeout_s: float


@dataclass(frozen=True)
class BackendCfg:
    kind: str
    ladder_cmd_topic: str
    belt_cmd_topic: str
    joint_states_topic: str
    serial: SerialCfg


@dataclass(frozen=True)
class VisualCfg:
    publish_marker: bool
    marker_topic: str
    gz_berm: bool
    gz_world: str
    gz_model: str
    gz_visual_only: bool
    gz_height_step_m: float


@dataclass(frozen=True)
class TopicsCfg:
    work_action: str
    status: str
    berm: str
    estop: str
    perception_mode: str


@dataclass(frozen=True)
class ExcavationConfig:
    frames: FramesCfg
    arena: ArenaCfg
    robot: RobotCfg
    nav_dump: NavDumpCfg
    mechanism: MechanismCfg
    sand: SandCfg
    cycle: CycleCfg
    backend: BackendCfg
    visual: VisualCfg
    topics: TopicsCfg
    path: str = ""
    sha256: str = ""
    schema_version: int = 1
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------- #
    @property
    def berm_target(self) -> Rect:
        return self.arena.zone("berm_target")

    @property
    def berm_target_volume_m3(self) -> float:
        return self.berm_target.area * self.sand.berm.target_height_m

    @property
    def berm_dump_pose_y(self) -> float:
        """The ``y`` of module 5's dump poses: outside the target by the stand-off."""
        return self.berm_target.y1 + self.nav_dump.standoff_m + self.robot.circumscribed_radius_m

    @property
    def pile_basal_radius_m(self) -> float:
        """How far a full tub of sand spreads when it lands and slumps."""
        return repose_basal_radius_m(self.mechanism.tub.capacity_m3, self.sand.dump.repose_rad)

    @property
    def landing_aft_m(self) -> float:
        """Distance behind ``base_link`` at which the centre of a dump forms.

        Sand cannot land under the rover, so the pile starts at the aft face and
        its centre sits one basal radius further back.  See ``sand.dump`` in
        ``excavation.yaml`` for the derivation.
        """
        return self.robot.half_length_m + self.pile_basal_radius_m + self.sand.dump.landing_margin_m

    def landing_point(self, x: float, y: float, yaw: float) -> Tuple[float, float]:
        """Where a tub emptied at map pose ``(x, y, yaw)`` puts the pile centre.

        The pile forms behind the rover (the discharge side is aft), so this is
        ``base_link`` minus ``landing_aft_m`` along the heading.
        """
        distance = self.landing_aft_m
        return (x - distance * math.cos(yaw), y - distance * math.sin(yaw))


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
def _parse_frames(raw: Dict[str, Any]) -> FramesCfg:
    section = _section(raw, "frames")
    frames = FramesCfg(
        map=_as_str(_require(section, "map", "frames"), "frames.map"),
        base=_as_str(_require(section, "base", "frames"), "frames.base"),
    )
    if not frames.map or not frames.base:
        raise ExcavationConfigError("frames: map and base must both be non-empty")
    if frames.map == frames.base:
        raise ExcavationConfigError("frames: map and base must be different frames")
    return frames


def _parse_arena(raw: Dict[str, Any]) -> ArenaCfg:
    section = _section(raw, "arena")
    zones_raw = _section(section, "zones", "arena")
    zones: Dict[str, Rect] = {}
    for name, value in zones_raw.items():
        if isinstance(value, dict):
            value = _require(value, "rect", f"arena.zones.{name}")
        zones[name] = Rect.from_sequence(value, f"arena.zones.{name}")
    for required in ("construction", "berm_target"):
        if required not in zones:
            raise ExcavationConfigError(f"arena.zones: missing required zone '{required}'")
    arena = ArenaCfg(
        length_x=_positive(_as_float(_require(section, "length_x", "arena"), "arena.length_x"), "arena.length_x"),
        width_y=_positive(_as_float(_require(section, "width_y", "arena"), "arena.width_y"), "arena.width_y"),
        sand_depth=_positive(_as_float(_require(section, "sand_depth", "arena"), "arena.sand_depth"), "arena.sand_depth"),
        zones=zones,
    )
    inner = arena.rect
    for name, rect in zones.items():
        if not (inner.x0 - 1e-9 <= rect.x0 and rect.x1 <= inner.x1 + 1e-9):
            raise ExcavationConfigError(f"arena.zones.{name}: x extent {rect.as_tuple()} leaves the arena")
        if not (inner.y0 - 1e-9 <= rect.y0 and rect.y1 <= inner.y1 + 1e-9):
            raise ExcavationConfigError(f"arena.zones.{name}: y extent {rect.as_tuple()} leaves the arena")
    if arena.zone("berm_target").area <= 0.0:
        raise ExcavationConfigError("arena.zones.berm_target: empty target footprint")
    if not arena.zone("construction").contains(*arena.zone("berm_target").center):
        raise ExcavationConfigError("arena.zones.berm_target: must lie inside the construction zone")
    return arena


def _parse_robot(raw: Dict[str, Any]) -> RobotCfg:
    section = _section(raw, "robot")
    length = _positive(_as_float(_require(section, "length_m", "robot"), "robot.length_m"), "robot.length_m")
    width = _positive(_as_float(_require(section, "width_m", "robot"), "robot.width_m"), "robot.width_m")
    build_length = _positive(
        _as_float(section.get("build_length_m", 1.50), "robot.build_length_m"), "robot.build_length_m"
    )
    build_width = _positive(_as_float(section.get("build_width_m", 0.75), "robot.build_width_m"), "robot.build_width_m")
    if length > build_length + 1e-9 or width > build_width + 1e-9:
        raise ExcavationConfigError(
            f"robot: the operational footprint {length} x {width} m does not fit the "
            f"competition envelope {build_length} x {build_width} m"
        )
    circumscribed = _positive(
        _as_float(
            section.get("circumscribed_radius_m", 0.5 * math.hypot(length, width)),
            "robot.circumscribed_radius_m",
        ),
        "robot.circumscribed_radius_m",
    )
    measured = 0.5 * math.hypot(length, width)
    if abs(circumscribed - measured) > 0.02:
        raise ExcavationConfigError(
            f"robot.circumscribed_radius_m ({circumscribed}) does not match the operational "
            f"footprint {length} x {width} m, which gives {measured:.4f}"
        )
    return RobotCfg(
        build_length_m=build_length,
        build_width_m=build_width,
        length_m=length,
        width_m=width,
        circumscribed_radius_m=circumscribed,
    )


def _parse_nav_dump(raw: Dict[str, Any]) -> NavDumpCfg:
    section = _section(raw, "nav_dump")
    return NavDumpCfg(
        heading_rad=_as_float(_require(section, "heading_rad", "nav_dump"), "nav_dump.heading_rad"),
        standoff_m=_non_negative(_as_float(section.get("standoff_m", 0.05), "nav_dump.standoff_m"), "nav_dump.standoff_m"),
        points=max(1, _as_int(section.get("points", 5), "nav_dump.points")),
        inset_m=_non_negative(_as_float(section.get("inset_m", 0.30), "nav_dump.inset_m"), "nav_dump.inset_m"),
    )


def _parse_ladder(raw: Dict[str, Any]) -> LadderCfg:
    section = _section(raw, "ladder")
    ladder = LadderCfg(
        joint=_as_str(_require(section, "joint", "mechanism.ladder"), "mechanism.ladder.joint"),
        pivot_x_m=_as_float(_require(section, "pivot_x_m", "mechanism.ladder"), "mechanism.ladder.pivot_x_m"),
        pivot_z_m=_as_float(_require(section, "pivot_z_m", "mechanism.ladder"), "mechanism.ladder.pivot_z_m"),
        ride_height_m=_non_negative(
            _as_float(_require(section, "ride_height_m", "mechanism.ladder"), "mechanism.ladder.ride_height_m"),
            "mechanism.ladder.ride_height_m",
        ),
        length_m=_positive(
            _as_float(_require(section, "length_m", "mechanism.ladder"), "mechanism.ladder.length_m"),
            "mechanism.ladder.length_m",
        ),
        width_m=_positive(
            _as_float(_require(section, "width_m", "mechanism.ladder"), "mechanism.ladder.width_m"),
            "mechanism.ladder.width_m",
        ),
        frame_thickness_m=_non_negative(
            _as_float(section.get("frame_thickness_m", 0.05), "mechanism.ladder.frame_thickness_m"),
            "mechanism.ladder.frame_thickness_m",
        ),
        stowed_pitch_rad=_as_float(
            _require(section, "stowed_pitch_rad", "mechanism.ladder"), "mechanism.ladder.stowed_pitch_rad"
        ),
        dig_pitch_rad=_as_float(_require(section, "dig_pitch_rad", "mechanism.ladder"), "mechanism.ladder.dig_pitch_rad"),
        min_pitch_rad=_as_float(_require(section, "min_pitch_rad", "mechanism.ladder"), "mechanism.ladder.min_pitch_rad"),
        max_pitch_rad=_as_float(_require(section, "max_pitch_rad", "mechanism.ladder"), "mechanism.ladder.max_pitch_rad"),
        max_rate_radps=_positive(
            _as_float(_require(section, "max_rate_radps", "mechanism.ladder"), "mechanism.ladder.max_rate_radps"),
            "mechanism.ladder.max_rate_radps",
        ),
    )
    if not (ladder.min_pitch_rad <= ladder.stowed_pitch_rad <= ladder.max_pitch_rad):
        raise ExcavationConfigError("mechanism.ladder.stowed_pitch_rad: outside [min_pitch_rad, max_pitch_rad]")
    if not (ladder.min_pitch_rad <= ladder.dig_pitch_rad <= ladder.max_pitch_rad):
        raise ExcavationConfigError("mechanism.ladder.dig_pitch_rad: outside [min_pitch_rad, max_pitch_rad]")
    if ladder.dig_pitch_rad <= ladder.min_dig_pitch_rad:
        raise ExcavationConfigError(
            f"mechanism.ladder.dig_pitch_rad ({ladder.dig_pitch_rad}) must be steeper than the "
            f"pitch at which the tip reaches the sand ({ladder.min_dig_pitch_rad:.4f}); "
            "otherwise the belt has nothing to dig"
        )
    if ladder.engaged_width_m <= 0.0:
        raise ExcavationConfigError("mechanism.ladder: the side frames leave no clear cut width")
    return ladder


def _parse_belt(raw: Dict[str, Any]) -> BeltCfg:
    section = _section(raw, "belt")
    belt = BeltCfg(
        joint=_as_str(_require(section, "joint", "mechanism.belt"), "mechanism.belt.joint"),
        roller_radius_m=_positive(
            _as_float(_require(section, "roller_radius_m", "mechanism.belt"), "mechanism.belt.roller_radius_m"),
            "mechanism.belt.roller_radius_m",
        ),
        dig_velocity_radps=_positive(
            _as_float(_require(section, "dig_velocity_radps", "mechanism.belt"), "mechanism.belt.dig_velocity_radps"),
            "mechanism.belt.dig_velocity_radps",
        ),
        dump_velocity_radps=-_positive(
            abs(_as_float(_require(section, "dump_velocity_radps", "mechanism.belt"), "mechanism.belt.dump_velocity_radps")),
            "mechanism.belt.dump_velocity_radps",
        ),
        max_velocity_radps=_positive(
            _as_float(_require(section, "max_velocity_radps", "mechanism.belt"), "mechanism.belt.max_velocity_radps"),
            "mechanism.belt.max_velocity_radps",
        ),
        max_accel_radps2=_positive(
            _as_float(_require(section, "max_accel_radps2", "mechanism.belt"), "mechanism.belt.max_accel_radps2"),
            "mechanism.belt.max_accel_radps2",
        ),
    )
    if belt.dig_velocity_radps > belt.max_velocity_radps + 1e-9:
        raise ExcavationConfigError("mechanism.belt.dig_velocity_radps: exceeds max_velocity_radps")
    if abs(belt.dump_velocity_radps) > belt.max_velocity_radps + 1e-9:
        raise ExcavationConfigError("mechanism.belt.dump_velocity_radps: exceeds max_velocity_radps")
    return belt


def _parse_tub(raw: Dict[str, Any]) -> TubCfg:
    section = _section(raw, "tub")
    box = _as_vec(_require(section, "box_m", "mechanism.tub"), "mechanism.tub.box_m", 3)
    if min(box) <= 0.0:
        raise ExcavationConfigError("mechanism.tub.box_m: every side must be > 0")
    fill_fraction = _in_range(
        _as_float(section.get("fill_fraction", 0.85), "mechanism.tub.fill_fraction"), 0.0, 1.0, "mechanism.tub.fill_fraction"
    )
    capacity = _positive(
        _as_float(_require(section, "capacity_m3", "mechanism.tub"), "mechanism.tub.capacity_m3"),
        "mechanism.tub.capacity_m3",
    )
    box_volume = box[0] * box[1] * box[2]
    expected = fill_fraction * box_volume
    if abs(capacity - expected) > 0.005 * box_volume + 1e-9:
        raise ExcavationConfigError(
            f"mechanism.tub.capacity_m3 ({capacity}) is not fill_fraction * box volume "
            f"({fill_fraction} * {box_volume:.5f} = {expected:.5f})"
        )
    return TubCfg(
        box_m=box,
        fill_fraction=fill_fraction,
        capacity_m3=capacity,
        gate_open_time_s=_positive(
            _as_float(section.get("gate_open_time_s", 0.8), "mechanism.tub.gate_open_time_s"),
            "mechanism.tub.gate_open_time_s",
        ),
    )


def _parse_interlocks(raw: Dict[str, Any], ladder: LadderCfg, arena: ArenaCfg) -> InterlocksCfg:
    section = _section(raw, "interlocks")
    interlocks = InterlocksCfg(
        travel_pitch_max_rad=_as_float(
            _require(section, "travel_pitch_max_rad", "mechanism.interlocks"),
            "mechanism.interlocks.travel_pitch_max_rad",
        ),
        max_depth_m=_positive(
            _as_float(_require(section, "max_depth_m", "mechanism.interlocks"), "mechanism.interlocks.max_depth_m"),
            "mechanism.interlocks.max_depth_m",
        ),
    )
    if interlocks.travel_pitch_max_rad < ladder.min_pitch_rad:
        raise ExcavationConfigError(
            "mechanism.interlocks.travel_pitch_max_rad: below the ladder's min_pitch_rad, so the "
            "ladder could never be called travel-ready"
        )
    if ladder.stowed_pitch_rad > interlocks.travel_pitch_max_rad + 1e-9:
        raise ExcavationConfigError(
            "mechanism.interlocks.travel_pitch_max_rad: the stowed attitude is not travel-ready"
        )
    if interlocks.max_depth_m > arena.sand_depth:
        raise ExcavationConfigError(
            f"mechanism.interlocks.max_depth_m ({interlocks.max_depth_m}) is deeper than the sand "
            f"({arena.sand_depth}); the ladder would dig through the floor"
        )
    dig_reach = ladder.tip_depth_m(ladder.max_pitch_rad)
    if interlocks.max_depth_m > dig_reach + 1e-9:
        raise ExcavationConfigError(
            f"mechanism.interlocks.max_depth_m ({interlocks.max_depth_m}) is deeper than the ladder "
            f"can reach at max_pitch_rad ({dig_reach:.4f})"
        )
    return interlocks


def _parse_mechanism(raw: Dict[str, Any], arena: ArenaCfg) -> MechanismCfg:
    section = _section(raw, "mechanism")
    ladder = _parse_ladder(section)
    return MechanismCfg(
        rate_hz=_in_range(
            _as_float(section.get("rate_hz", 10.0), "mechanism.rate_hz"), 1.0, 100.0, "mechanism.rate_hz"
        ),
        ladder=ladder,
        belt=_parse_belt(section),
        tub=_parse_tub(section),
        interlocks=_parse_interlocks(section, ladder, arena),
    )


def _parse_sand(raw: Dict[str, Any], mechanism: MechanismCfg) -> SandCfg:
    section = _section(raw, "sand")
    dig_raw = _section(section, "dig", "sand")
    dig = DigRateCfg(
        coefficient=_positive(_as_float(dig_raw.get("coefficient", 0.5), "sand.dig.coefficient"), "sand.dig.coefficient"),
        min_rate_m3_s=_non_negative(
            _as_float(dig_raw.get("min_rate_m3_s", 0.0005), "sand.dig.min_rate_m3_s"), "sand.dig.min_rate_m3_s"
        ),
        max_rate_m3_s=_positive(
            _as_float(dig_raw.get("max_rate_m3_s", 0.012), "sand.dig.max_rate_m3_s"), "sand.dig.max_rate_m3_s"
        ),
    )
    if dig.min_rate_m3_s > dig.max_rate_m3_s:
        raise ExcavationConfigError("sand.dig: min_rate_m3_s must not exceed max_rate_m3_s")

    dump_raw = _section(section, "dump", "sand")
    dump = DumpSandCfg(
        rate_m3_s=_positive(_as_float(dump_raw.get("rate_m3_s", 0.008), "sand.dump.rate_m3_s"), "sand.dump.rate_m3_s"),
        repose_deg=_in_range(
            _as_float(dump_raw.get("repose_deg", 34.0), "sand.dump.repose_deg"), 5.0, 85.0, "sand.dump.repose_deg"
        ),
        landing_margin_m=_non_negative(
            _as_float(dump_raw.get("landing_margin_m", 0.0), "sand.dump.landing_margin_m"),
            "sand.dump.landing_margin_m",
        ),
        spread_m=_positive(_as_float(dump_raw.get("spread_m", 0.20), "sand.dump.spread_m"), "sand.dump.spread_m"),
    )

    berm_raw = _section(section, "berm", "sand")
    berm = BermCfg(
        target_height_m=_positive(
            _as_float(berm_raw.get("target_height_m", 0.15), "sand.berm.target_height_m"), "sand.berm.target_height_m"
        ),
        cell_m=_positive(_as_float(berm_raw.get("cell_m", 0.05), "sand.berm.cell_m"), "sand.berm.cell_m"),
    )
    if berm.cell_m > 0.25 * min(mechanism.ladder.width_m, 1.0):
        raise ExcavationConfigError("sand.berm.cell_m: too coarse to resolve the berm")

    # The deposit's width must be of the same order as the pile it stands for.
    basal = repose_basal_radius_m(mechanism.tub.capacity_m3, dump.repose_rad)
    if not (0.4 * basal <= dump.spread_m <= 2.0 * basal):
        raise ExcavationConfigError(
            f"sand.dump.spread_m ({dump.spread_m}) is not comparable with the basal radius of a "
            f"tub's pile at {dump.repose_deg} deg ({basal:.4f}); one of the two is wrong"
        )

    # A full tub must be reachable in a sensible time at the dig attitude.
    dig_rate_bound = dig_rate_m3_s(dig, mechanism, mechanism.ladder.dig_pitch_rad, mechanism.belt.dig_velocity_radps)
    if dig_rate_bound <= 0.0:
        raise ExcavationConfigError(
            "sand: at the configured dig pitch and belt speed the dig rate is zero; the tub would "
            "never fill"
        )
    return SandCfg(dig=dig, dump=dump, berm=berm)


def dig_rate_m3_s(dig: DigRateCfg, mechanism: MechanismCfg, pitch_rad: float, belt_velocity_radps: float) -> float:
    """The dig model: how fast sand enters the tub, in m3/s.

    ``rate = coefficient * engaged_width * engaged_depth * belt_surface_speed``
    clamped into ``[min_rate, max_rate]`` - and zero whenever the ladder is not
    pointed into the sand or the belt is not running forward.

    Defined here, next to the parameters it uses, and imported by
    :mod:`rover_excavation.sand` and :mod:`rover_excavation.cycle` so the model
    exists in exactly one place.
    """
    if belt_velocity_radps <= 0.0:
        return 0.0
    depth = mechanism.ladder.tip_depth_m(pitch_rad)
    if depth <= 0.0:
        return 0.0
    engaged = min(depth, mechanism.interlocks.max_depth_m)
    raw_rate = (
        dig.coefficient
        * mechanism.ladder.engaged_width_m
        * engaged
        * mechanism.belt.surface_speed_mps(belt_velocity_radps)
    )
    if raw_rate <= 0.0:
        return 0.0
    return max(min(raw_rate, dig.max_rate_m3_s), min(dig.min_rate_m3_s, dig.max_rate_m3_s))


def _parse_cycle(raw: Dict[str, Any]) -> CycleCfg:
    section = _section(raw, "cycle")
    dig_raw = _section(section, "dig", "cycle")
    dump_raw = _section(section, "dump", "cycle")
    dig = DigCycleCfg(
        min_s=_non_negative(_as_float(dig_raw.get("min_s", 2.0), "cycle.dig.min_s"), "cycle.dig.min_s"),
        timeout_s=_positive(_as_float(dig_raw.get("timeout_s", 60.0), "cycle.dig.timeout_s"), "cycle.dig.timeout_s"),
        target_fill=_in_range(
            _as_float(dig_raw.get("target_fill", 0.95), "cycle.dig.target_fill"), 0.05, 1.0, "cycle.dig.target_fill"
        ),
    )
    dump = DumpCycleCfg(
        min_s=_non_negative(_as_float(dump_raw.get("min_s", 1.5), "cycle.dump.min_s"), "cycle.dump.min_s"),
        timeout_s=_positive(
            _as_float(dump_raw.get("timeout_s", 30.0), "cycle.dump.timeout_s"), "cycle.dump.timeout_s"
        ),
        target_empty=_in_range(
            _as_float(dump_raw.get("target_empty", 0.02), "cycle.dump.target_empty"),
            0.0,
            0.5,
            "cycle.dump.target_empty",
        ),
    )
    if dig.timeout_s <= dig.min_s:
        raise ExcavationConfigError("cycle.dig: timeout_s must be longer than min_s")
    if dump.timeout_s <= dump.min_s:
        raise ExcavationConfigError("cycle.dump: timeout_s must be longer than min_s")
    return CycleCfg(
        settle_s=_non_negative(_as_float(section.get("settle_s", 0.4), "cycle.settle_s"), "cycle.settle_s"),
        dig=dig,
        dump=dump,
    )


def _parse_backend(raw: Dict[str, Any]) -> BackendCfg:
    section = _section(raw, "backend")
    serial_raw = _section(section, "serial", "backend", required=False)
    return BackendCfg(
        kind=_as_enum(section.get("kind", "sim"), BACKEND_KINDS, "backend.kind"),
        ladder_cmd_topic=_as_str(
            section.get("ladder_cmd_topic", "/excavation/ladder_cmd"), "backend.ladder_cmd_topic"
        ),
        belt_cmd_topic=_as_str(section.get("belt_cmd_topic", "/excavation/belt_cmd"), "backend.belt_cmd_topic"),
        joint_states_topic=_as_str(
            section.get("joint_states_topic", "/joint_states"), "backend.joint_states_topic"
        ),
        serial=SerialCfg(
            port=_as_str(serial_raw.get("port", "/dev/ttyACM0"), "backend.serial.port"),
            baud=_as_int(serial_raw.get("baud", 115200), "backend.serial.baud"),
            timeout_s=_positive(
                _as_float(serial_raw.get("timeout_s", 0.2), "backend.serial.timeout_s"), "backend.serial.timeout_s"
            ),
        ),
    )


def _parse_visual(raw: Dict[str, Any]) -> VisualCfg:
    section = _section(raw, "visual", required=False)
    return VisualCfg(
        publish_marker=_as_bool(section.get("publish_marker", True), "visual.publish_marker"),
        marker_topic=_as_str(section.get("marker_topic", "/excavation/berm_marker"), "visual.marker_topic"),
        gz_berm=_as_bool(section.get("gz_berm", False), "visual.gz_berm"),
        gz_world=_as_str(section.get("gz_world", "cat_arena"), "visual.gz_world"),
        gz_model=_as_str(section.get("gz_model", "berm"), "visual.gz_model"),
        gz_visual_only=_as_bool(section.get("gz_visual_only", True), "visual.gz_visual_only"),
        gz_height_step_m=_positive(
            _as_float(section.get("gz_height_step_m", 0.01), "visual.gz_height_step_m"), "visual.gz_height_step_m"
        ),
    )


def _parse_topics(raw: Dict[str, Any]) -> TopicsCfg:
    section = _section(raw, "topics")
    topics = TopicsCfg(
        work_action=_as_str(_require(section, "work_action", "topics"), "topics.work_action"),
        status=_as_str(_require(section, "status", "topics"), "topics.status"),
        berm=_as_str(_require(section, "berm", "topics"), "topics.berm"),
        estop=_as_str(_require(section, "estop", "topics"), "topics.estop"),
        perception_mode=_as_str(
            _require(section, "perception_mode", "topics"), "topics.perception_mode"
        ),
    )
    for name, value in (
        ("work_action", topics.work_action),
        ("status", topics.status),
        ("berm", topics.berm),
        ("estop", topics.estop),
        ("perception_mode", topics.perception_mode),
    ):
        if not value.startswith("/"):
            raise ExcavationConfigError(f"topics.{name}: must be an absolute topic name, got {value!r}")
    return topics


def config_from_mapping(
    raw: Dict[str, Any], *, path: str = "", sha256: str = "", schema_version: int = 1
) -> ExcavationConfig:
    """Build and validate a config from an already-parsed mapping."""
    if not isinstance(raw, dict):
        raise ExcavationConfigError("excavation.yaml: expected a mapping at the top level")
    schema = raw.get("schema_version", 1)
    if schema != 1:
        raise ExcavationConfigError(f"schema_version: expected 1, got {schema!r}")
    arena = _parse_arena(raw)
    mechanism = _parse_mechanism(raw, arena)
    robot = _parse_robot(raw)
    config = ExcavationConfig(
        frames=_parse_frames(raw),
        arena=arena,
        robot=robot,
        nav_dump=_parse_nav_dump(raw),
        mechanism=mechanism,
        sand=_parse_sand(raw, mechanism),
        cycle=_parse_cycle(raw),
        backend=_parse_backend(raw),
        visual=_parse_visual(raw),
        topics=_parse_topics(raw),
        path=path,
        sha256=sha256,
        schema_version=schema,
        raw=raw,
    )
    return config


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def default_config_path() -> str:
    """``<share>/rover_excavation/config/excavation.yaml`` (source tree or install)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "config", "excavation.yaml"),  # source tree
        os.path.join(package_share_dir(), "config", "excavation.yaml"),  # installed
    ]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            return candidate
    return os.path.abspath(candidates[0])


def package_share_dir() -> str:
    """This package's share directory (install space first, source tree second).

    ``__file__`` cannot be used directly: ``colcon --symlink-install`` puts the
    python package under ``local/lib/.../dist-packages``, so walking up from it
    lands somewhere that has no ``config`` directory at all.  Asking the ament
    index is what makes the *default* config path work under ``ros2 launch``
    without anybody passing ``--config``.
    """
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory("rover_excavation")
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def load_config(path: Optional[str] = None) -> ExcavationConfig:
    """Load and validate ``excavation.yaml`` (default: this package's copy)."""
    import yaml

    resolved = os.path.abspath(os.path.expanduser(path)) if path else default_config_path()
    if not os.path.isfile(resolved):
        raise ExcavationConfigError(f"config file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        blob = handle.read()
    try:
        raw = yaml.safe_load(blob)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised by a config test
        raise ExcavationConfigError(f"{resolved}: not valid YAML: {exc}") from exc
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return config_from_mapping(raw, path=resolved, sha256=digest)
