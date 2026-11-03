"""Typed, validated ``navigation.yaml``: module 5's single source of truth.

Everything module 5 does is decided here: the arena geometry it is allowed to
know, the rover's size and speed limits, how the obstacle map is rasterised and
inflated, how the planner searches and smooths, how the controller follows a
path, when safety stops the rover, and the order in which the run is executed.

The file is parsed and range-checked at startup so a bad plan fails immediately,
not halfway through a run.  This mirrors ``cat_arena/config/arena.yaml`` (module
1), ``rover_description/config/rover.yaml`` (module 2),
``rover_dataset/config/dataset.yaml`` (module 3) and
``rover_perception/config/perception.yaml`` (module 4).

**The arena geometry is deliberately duplicated here.**  On the rover computer
``cat_arena`` is not installed (there is no Gazebo there), so module 5 must
carry its own copy of the numbers a navigator is *allowed* to know - the arena
size, the zone rectangles and the wall margin.  A test cross-checks the copy
against ``cat_arena/config/arena.yaml`` so the two cannot drift silently.  The
copy contains no seeded obstacle, exactly like module 4's.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "NavigationConfigError",
    "Rect",
    "FramesCfg",
    "ArenaCfg",
    "RobotCfg",
    "CostmapCfg",
    "PlannerCfg",
    "ControllerCfg",
    "SafetyCfg",
    "DigCfg",
    "DumpCfg",
    "ParkCfg",
    "MissionCfg",
    "TopicsCfg",
    "NavigationConfig",
    "config_from_mapping",
    "load_config",
    "with_work_mode",
    "default_config_path",
    "package_share_dir",
    "NAV_STATES",
    "PHASE_NAMES",
    "WORK_MODES",
    "WORK_PHASES",
]

#: ``NavigationStatus.state`` names, indexed by the message constant.
NAV_STATES: Tuple[str, ...] = (
    "idle",
    "planning",
    "turning",
    "driving",
    "arrived",
    "holding",
    "blocked",
    "failed",
)

#: ``MissionStatus.phase`` names, indexed by the message constant.
PHASE_NAMES: Tuple[str, ...] = (
    "idle",
    "acquire",
    "travel_to_dig",
    "excavate",
    "travel_to_dump",
    "dump",
    "return",
    "park",
    "complete",
    "aborted",
)

#: How the sequencer obtains its work cycles.
#: ``delegate``      - ask module 6 over ``WorkCycle`` and count only real cycles;
#: ``navigate_only`` - drive every leg and never claim a cycle happened; the mode
#:                     used to exercise navigation in the simulator before the
#:                     excavation mechanism exists.
WORK_MODES: Tuple[str, ...] = ("delegate", "navigate_only")

#: The work phases a ``WorkCycle`` goal may ask for.
WORK_PHASES: Tuple[str, ...] = ("excavate", "dump")


class NavigationConfigError(ValueError):
    """Raised when ``navigation.yaml`` is missing something or breaks a rule."""


# --------------------------------------------------------------------------- #
# scalar helpers (the same shape module 4 uses)
# --------------------------------------------------------------------------- #
def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise NavigationConfigError(f"{where}: missing required key '{key}'")
    return section[key]


def _as_float(value: Any, where: str) -> float:
    if isinstance(value, bool):
        raise NavigationConfigError(f"{where}: expected a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise NavigationConfigError(f"{where}: expected a number, got {value!r}") from exc


def _as_int(value: Any, where: str) -> int:
    if isinstance(value, bool):
        raise NavigationConfigError(f"{where}: expected an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise NavigationConfigError(f"{where}: expected an integer, got {value!r}") from exc


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    raise NavigationConfigError(f"{where}: expected true/false, got {value!r}")


def _as_str(value: Any, where: str) -> str:
    if not isinstance(value, str):
        raise NavigationConfigError(f"{where}: expected a string, got {value!r}")
    return value


def _as_vec(value: Any, where: str, length: int) -> Tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise NavigationConfigError(f"{where}: expected {length} numbers, got {value!r}")
    if len(value) != length:
        raise NavigationConfigError(f"{where}: expected {length} numbers, got {value!r}")
    return tuple(_as_float(v, where) for v in value)


def _as_str_list(value: Any, where: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise NavigationConfigError(f"{where}: expected a list of strings, got {value!r}")
    return [_as_str(v, where) for v in value]


def _section(raw: Dict[str, Any], key: str, where: str = "", required: bool = True) -> Dict[str, Any]:
    value = raw.get(key)
    if value is None:
        if required:
            raise NavigationConfigError(f"{where or 'navigation.yaml'}: missing required section '{key}'")
        return {}
    if not isinstance(value, dict):
        raise NavigationConfigError(f"{where or 'navigation.yaml'}.{key}: expected a mapping, got {value!r}")
    return value


def _positive(value: float, where: str) -> float:
    if value <= 0.0:
        raise NavigationConfigError(f"{where}: must be > 0, got {value}")
    return value


def _non_negative(value: float, where: str) -> float:
    if value < 0.0:
        raise NavigationConfigError(f"{where}: must be >= 0, got {value}")
    return value


def _in_range(value: float, low: float, high: float, where: str) -> float:
    if not (low <= value <= high):
        raise NavigationConfigError(f"{where}: must be within [{low}, {high}], got {value}")
    return value


def _ordered(section: Dict[str, float], keys: Sequence[str], where: str) -> None:
    previous = keys[0]
    for key in keys[1:]:
        if section[key] <= section[previous]:
            raise NavigationConfigError(
                f"{where}: {key} ({section[key]}) must be greater than {previous} ({section[previous]})"
            )
        previous = key


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

    def inset(self, margin: float) -> "Rect":
        """A rectangle pulled in by ``margin``; empty (and ``None``-safe) if too small."""
        return Rect(self.x0 + margin, self.y0 + margin, self.x1 - margin, self.y1 - margin)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    @classmethod
    def from_sequence(cls, value: Any, where: str) -> "Rect":
        x0, y0, x1, y1 = _as_vec(value, where, 4)
        if x1 <= x0 or y1 <= y0:
            raise NavigationConfigError(f"{where}: degenerate rectangle {value!r}")
        return cls(x0, y0, x1, y1)


@dataclass(frozen=True)
class FramesCfg:
    map: str
    odom: str
    robot: str
    ground_z: float


@dataclass(frozen=True)
class ArenaCfg:
    length_x: float
    width_y: float
    wall_margin_m: float
    zones: Dict[str, Rect]

    @property
    def rect(self) -> Rect:
        return Rect(0.0, 0.0, self.length_x, self.width_y)

    def zone(self, name: str) -> Rect:
        try:
            return self.zones[name]
        except KeyError as exc:  # pragma: no cover - guarded by the loader
            raise NavigationConfigError(f"arena.zones: no zone named '{name}'") from exc

    def zone_names(self) -> Tuple[str, ...]:
        return tuple(self.zones)


@dataclass(frozen=True)
class RobotCfg:
    length_m: float
    width_m: float
    build_length_m: float
    build_width_m: float
    radius_m: float
    footprint_clearance_m: float
    max_speed_mps: float
    max_yaw_rate_radps: float
    max_accel_mps2: float
    max_yaw_accel_radps2: float
    goal_tolerance_m: float
    goal_tolerance_yaw_rad: float

    @property
    def circumscribed_radius_m(self) -> float:
        """Half the diagonal: the circle a rotating-in-place rover sweeps."""
        return 0.5 * math.hypot(self.length_m, self.width_m)

    @property
    def inscribed_radius_m(self) -> float:
        """Half the *shorter* side: what a body translating along a path sweeps."""
        return 0.5 * min(self.length_m, self.width_m)

    @property
    def obstacle_inflation_m(self) -> float:
        """How far an obstacle's footprint must be grown for the planner.

        This is the inscribed radius, not the circumscribed one: the planner
        rasterises what a body *following the path* sweeps, and the awkward part
        of the problem - turning, or being parked next to something - is instead
        handled exactly by :mod:`rover_navigation.footprint`.
        """
        return self.radius_m + self.footprint_clearance_m

    @property
    def wall_inset_m(self) -> float:
        """How far the arena is pulled in before the rover's centre may go."""
        return self.radius_m


@dataclass(frozen=True)
class CostmapCfg:
    resolution_m: float
    boulder_radius_scale: float
    crater_radius_scale: float
    inflation_m: float
    soft_cost: float
    obstacle_memory_s: float
    stale_obstacle_s: float
    memory_associate_m: float
    min_observations: int
    min_confidence: float
    max_obstacles: int


@dataclass(frozen=True)
class PlannerCfg:
    allow_diagonal: bool
    heuristic_weight: float
    unknown_is_free: bool
    max_expansions: int
    smoothing: bool
    smoothing_passes: int
    waypoint_spacing_m: float
    snap_radius_m: float


@dataclass(frozen=True)
class ControllerCfg:
    lookahead_m: float
    min_lookahead_m: float
    max_lookahead_m: float
    lookahead_speed_gain: float
    heading_gain: float
    in_place_turn_rad: float
    in_place_yaw_rate_radps: float
    slowdown_distance_m: float
    min_speed_mps: float
    goal_timeout_s: float
    replan_period_s: float
    pose_timeout_s: float


@dataclass(frozen=True)
class SafetyCfg:
    command_timeout_s: float
    sigma_xy_slow_m: float
    sigma_xy_stop_m: float
    sigma_yaw_slow_rad: float
    sigma_yaw_stop_rad: float
    max_time_since_fix_s: float
    require_localized: bool
    emergency_stop_m: float
    gate_rate_hz: float


@dataclass(frozen=True)
class DigCfg:
    heading_rad: float
    pass_spacing_m: float
    points_per_pass: int
    cross_inset_m: float
    y_inset_m: float


@dataclass(frozen=True)
class DumpCfg:
    heading_rad: float
    points: int
    inset_m: float
    standoff_m: float


@dataclass(frozen=True)
class ParkCfg:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class MissionCfg:
    enabled: bool
    work_mode: str
    cycles_target: int
    excavation_action: str
    excavation_wait_s: float
    work_timeout_s: float
    acquire_timeout_s: float
    acquire_spin_rad: float
    dig: DigCfg
    dump: DumpCfg
    park: ParkCfg


@dataclass(frozen=True)
class TopicsCfg:
    obstacles: str
    localization_state: str
    odometry_topics: List[str]
    cmd_vel: str
    nav_cmd_vel: str
    path: str
    nav_status: str
    gate_status: str
    costmap: str
    mission_status: str
    perception_mode: str
    navigate_action: str
    berm_started: str
    estop: str


@dataclass(frozen=True)
class NavigationConfig:
    frames: FramesCfg
    arena: ArenaCfg
    robot: RobotCfg
    costmap: CostmapCfg
    planner: PlannerCfg
    controller: ControllerCfg
    safety: SafetyCfg
    mission: MissionCfg
    topics: TopicsCfg
    path: str = ""
    sha256: str = ""
    schema_version: int = 1
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------- #
    @property
    def free_rect(self) -> Rect:
        """Where the rover's *centre* may go: the arena inset by the wall margin + radius."""
        return self.arena.rect.inset(self.arena.wall_margin_m + self.robot.wall_inset_m)

    def zone_center(self, name: str) -> Tuple[float, float]:
        return self.arena.zone(name).center


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
def _parse_frames(raw: Dict[str, Any]) -> FramesCfg:
    section = _section(raw, "frames")
    frames = FramesCfg(
        map=_as_str(_require(section, "map", "frames"), "frames.map"),
        odom=_as_str(_require(section, "odom", "frames"), "frames.odom"),
        robot=_as_str(_require(section, "robot", "frames"), "frames.robot"),
        ground_z=_as_float(section.get("ground_z", 0.0), "frames.ground_z"),
    )
    if not frames.map or not frames.odom or not frames.robot:
        raise NavigationConfigError("frames: map/odom/robot must all be non-empty")
    if frames.map == frames.odom:
        raise NavigationConfigError("frames: map and odom must be different frames")
    return frames


def _parse_arena(raw: Dict[str, Any]) -> ArenaCfg:
    section = _section(raw, "arena")
    zones_raw = _section(section, "zones", "arena")
    zones = {name: Rect.from_sequence(value, f"arena.zones.{name}") for name, value in zones_raw.items()}
    for required in ("start", "construction", "obstacle", "excavation", "berm_target"):
        if required not in zones:
            raise NavigationConfigError(f"arena.zones: missing required zone '{required}'")
    arena = ArenaCfg(
        length_x=_positive(_as_float(_require(section, "length_x", "arena"), "arena.length_x"), "arena.length_x"),
        width_y=_positive(_as_float(_require(section, "width_y", "arena"), "arena.width_y"), "arena.width_y"),
        wall_margin_m=_non_negative(_as_float(section.get("wall_margin_m", 0.15), "arena.wall_margin_m"), "arena.wall_margin_m"),
        zones=zones,
    )
    inner = arena.rect
    for name, rect in zones.items():
        if not (inner.x0 - 1e-9 <= rect.x0 and rect.x1 <= inner.x1 + 1e-9):
            raise NavigationConfigError(f"arena.zones.{name}: x extent {rect.as_tuple()} leaves the arena")
        if not (inner.y0 - 1e-9 <= rect.y0 and rect.y1 <= inner.y1 + 1e-9):
            raise NavigationConfigError(f"arena.zones.{name}: y extent {rect.as_tuple()} leaves the arena")
    if not arena.zone("berm_target").area > 0.0:
        raise NavigationConfigError("arena.zones.berm_target: empty target footprint")
    if not arena.zone("construction").contains(*arena.zone("berm_target").center):
        raise NavigationConfigError("arena.zones.berm_target: must lie inside the construction zone")
    return arena


def _parse_robot(raw: Dict[str, Any]) -> RobotCfg:
    section = _section(raw, "robot")
    length = _positive(_as_float(_require(section, "length_m", "robot"), "robot.length_m"), "robot.length_m")
    width = _positive(_as_float(_require(section, "width_m", "robot"), "robot.width_m"), "robot.width_m")
    build_length = _positive(
        _as_float(section.get("build_length_m", 1.50), "robot.build_length_m"), "robot.build_length_m"
    )
    build_width = _positive(
        _as_float(section.get("build_width_m", 0.75), "robot.build_width_m"), "robot.build_width_m"
    )
    if length > build_length + 1e-9 or width > build_width + 1e-9:
        raise NavigationConfigError(
            f"robot: the operational footprint {length} x {width} m does not fit the "
            f"competition envelope {build_length} x {build_width} m"
        )
    circumscribed = 0.5 * math.hypot(length, width)
    inscribed = 0.5 * min(length, width)
    radius = _as_float(section.get("radius_m", 0.0), "robot.radius_m")
    if radius <= 0.0:
        # Default to the inscribed radius: what a body translating along a path
        # sweeps.  Turning is handled exactly by rover_navigation.footprint.
        radius = inscribed
    if radius > circumscribed + 1e-9:
        raise NavigationConfigError(
            f"robot.radius_m ({radius}) must not exceed the circumscribed radius ({circumscribed:.4f}); "
            "the planner cannot inflate for a circle larger than the rover sweeps"
        )
    if radius < inscribed - 1e-9:
        raise NavigationConfigError(
            f"robot.radius_m ({radius}) is below the inscribed radius ({inscribed:.4f}); "
            "the planner would clip the rover's own body"
        )
    robot = RobotCfg(
        length_m=length,
        width_m=width,
        build_length_m=build_length,
        build_width_m=build_width,
        radius_m=radius,
        footprint_clearance_m=_non_negative(
            _as_float(section.get("footprint_clearance_m", 0.08), "robot.footprint_clearance_m"),
            "robot.footprint_clearance_m",
        ),
        max_speed_mps=_positive(_as_float(_require(section, "max_speed_mps", "robot"), "robot.max_speed_mps"), "robot.max_speed_mps"),
        max_yaw_rate_radps=_positive(
            _as_float(_require(section, "max_yaw_rate_radps", "robot"), "robot.max_yaw_rate_radps"),
            "robot.max_yaw_rate_radps",
        ),
        max_accel_mps2=_positive(_as_float(section.get("max_accel_mps2", 0.5), "robot.max_accel_mps2"), "robot.max_accel_mps2"),
        max_yaw_accel_radps2=_positive(
            _as_float(section.get("max_yaw_accel_radps2", 2.0), "robot.max_yaw_accel_radps2"),
            "robot.max_yaw_accel_radps2",
        ),
        goal_tolerance_m=_positive(
            _as_float(section.get("goal_tolerance_m", 0.12), "robot.goal_tolerance_m"), "robot.goal_tolerance_m"
        ),
        goal_tolerance_yaw_rad=_positive(
            _as_float(section.get("goal_tolerance_yaw_rad", 0.20), "robot.goal_tolerance_yaw_rad"),
            "robot.goal_tolerance_yaw_rad",
        ),
    )
    if robot.goal_tolerance_yaw_rad >= math.pi:
        raise NavigationConfigError("robot.goal_tolerance_yaw_rad: must be < pi")
    return robot


def _parse_costmap(raw: Dict[str, Any]) -> CostmapCfg:
    section = _section(raw, "costmap")
    return CostmapCfg(
        resolution_m=_in_range(
            _as_float(section.get("resolution_m", 0.05), "costmap.resolution_m"), 0.01, 0.50, "costmap.resolution_m"
        ),
        boulder_radius_scale=_positive(
            _as_float(section.get("boulder_radius_scale", 1.0), "costmap.boulder_radius_scale"),
            "costmap.boulder_radius_scale",
        ),
        crater_radius_scale=_positive(
            _as_float(section.get("crater_radius_scale", 1.2), "costmap.crater_radius_scale"),
            "costmap.crater_radius_scale",
        ),
        inflation_m=_non_negative(_as_float(section.get("inflation_m", 0.10), "costmap.inflation_m"), "costmap.inflation_m"),
        soft_cost=_non_negative(_as_float(section.get("soft_cost", 6.0), "costmap.soft_cost"), "costmap.soft_cost"),
        obstacle_memory_s=_non_negative(
            _as_float(section.get("obstacle_memory_s", 0.0), "costmap.obstacle_memory_s"), "costmap.obstacle_memory_s"
        ),
        stale_obstacle_s=_positive(
            _as_float(section.get("stale_obstacle_s", 2.5), "costmap.stale_obstacle_s"), "costmap.stale_obstacle_s"
        ),
        memory_associate_m=_positive(
            _as_float(section.get("memory_associate_m", 0.35), "costmap.memory_associate_m"), "costmap.memory_associate_m"
        ),
        min_observations=_as_int(section.get("min_observations", 3), "costmap.min_observations"),
        min_confidence=_in_range(
            _as_float(section.get("min_confidence", 0.30), "costmap.min_confidence"), 0.0, 1.0, "costmap.min_confidence"
        ),
        max_obstacles=_as_int(section.get("max_obstacles", 128), "costmap.max_obstacles"),
    )


def _parse_planner(raw: Dict[str, Any]) -> PlannerCfg:
    section = _section(raw, "planner")
    return PlannerCfg(
        allow_diagonal=_as_bool(section.get("allow_diagonal", True), "planner.allow_diagonal"),
        heuristic_weight=_in_range(
            _as_float(section.get("heuristic_weight", 1.0), "planner.heuristic_weight"), 1.0, 5.0, "planner.heuristic_weight"
        ),
        unknown_is_free=_as_bool(section.get("unknown_is_free", True), "planner.unknown_is_free"),
        max_expansions=_as_int(section.get("max_expansions", 200000), "planner.max_expansions"),
        smoothing=_as_bool(section.get("smoothing", True), "planner.smoothing"),
        smoothing_passes=_as_int(section.get("smoothing_passes", 64), "planner.smoothing_passes"),
        waypoint_spacing_m=_positive(
            _as_float(section.get("waypoint_spacing_m", 0.10), "planner.waypoint_spacing_m"),
            "planner.waypoint_spacing_m",
        ),
        snap_radius_m=_positive(
            _as_float(section.get("snap_radius_m", 0.50), "planner.snap_radius_m"), "planner.snap_radius_m"
        ),
    )


def _parse_controller(raw: Dict[str, Any]) -> ControllerCfg:
    section = _section(raw, "controller")
    controller = ControllerCfg(
        lookahead_m=_positive(_as_float(section.get("lookahead_m", 0.55), "controller.lookahead_m"), "controller.lookahead_m"),
        min_lookahead_m=_positive(
            _as_float(section.get("min_lookahead_m", 0.30), "controller.min_lookahead_m"), "controller.min_lookahead_m"
        ),
        max_lookahead_m=_positive(
            _as_float(section.get("max_lookahead_m", 0.90), "controller.max_lookahead_m"), "controller.max_lookahead_m"
        ),
        lookahead_speed_gain=_non_negative(
            _as_float(section.get("lookahead_speed_gain", 0.8), "controller.lookahead_speed_gain"),
            "controller.lookahead_speed_gain",
        ),
        heading_gain=_positive(_as_float(section.get("heading_gain", 1.4), "controller.heading_gain"), "controller.heading_gain"),
        in_place_turn_rad=_in_range(
            _as_float(section.get("in_place_turn_rad", 0.55), "controller.in_place_turn_rad"),
            0.05,
            math.pi,
            "controller.in_place_turn_rad",
        ),
        in_place_yaw_rate_radps=_positive(
            _as_float(section.get("in_place_yaw_rate_radps", 0.9), "controller.in_place_yaw_rate_radps"),
            "controller.in_place_yaw_rate_radps",
        ),
        slowdown_distance_m=_positive(
            _as_float(section.get("slowdown_distance_m", 0.75), "controller.slowdown_distance_m"),
            "controller.slowdown_distance_m",
        ),
        min_speed_mps=_non_negative(_as_float(section.get("min_speed_mps", 0.08), "controller.min_speed_mps"), "controller.min_speed_mps"),
        goal_timeout_s=_positive(_as_float(section.get("goal_timeout_s", 120.0), "controller.goal_timeout_s"), "controller.goal_timeout_s"),
        replan_period_s=_positive(
            _as_float(section.get("replan_period_s", 1.0), "controller.replan_period_s"), "controller.replan_period_s"
        ),
        pose_timeout_s=_positive(_as_float(section.get("pose_timeout_s", 1.0), "controller.pose_timeout_s"), "controller.pose_timeout_s"),
    )
    _ordered(
        {"min_lookahead_m": controller.min_lookahead_m, "lookahead_m": controller.lookahead_m, "max_lookahead_m": controller.max_lookahead_m},
        ("min_lookahead_m", "lookahead_m", "max_lookahead_m"),
        "controller",
    )
    return controller


def _parse_safety(raw: Dict[str, Any]) -> SafetyCfg:
    section = _section(raw, "safety")
    safety = SafetyCfg(
        command_timeout_s=_positive(
            _as_float(section.get("command_timeout_s", 0.5), "safety.command_timeout_s"), "safety.command_timeout_s"
        ),
        sigma_xy_slow_m=_non_negative(
            _as_float(section.get("sigma_xy_slow_m", 0.15), "safety.sigma_xy_slow_m"), "safety.sigma_xy_slow_m"
        ),
        sigma_xy_stop_m=_positive(_as_float(section.get("sigma_xy_stop_m", 0.60), "safety.sigma_xy_stop_m"), "safety.sigma_xy_stop_m"),
        sigma_yaw_slow_rad=_non_negative(
            _as_float(section.get("sigma_yaw_slow_rad", 0.10), "safety.sigma_yaw_slow_rad"), "safety.sigma_yaw_slow_rad"
        ),
        sigma_yaw_stop_rad=_positive(
            _as_float(section.get("sigma_yaw_stop_rad", 0.50), "safety.sigma_yaw_stop_rad"), "safety.sigma_yaw_stop_rad"
        ),
        max_time_since_fix_s=_positive(
            _as_float(section.get("max_time_since_fix_s", 20.0), "safety.max_time_since_fix_s"), "safety.max_time_since_fix_s"
        ),
        require_localized=_as_bool(section.get("require_localized", True), "safety.require_localized"),
        emergency_stop_m=_positive(
            _as_float(section.get("emergency_stop_m", 0.45), "safety.emergency_stop_m"), "safety.emergency_stop_m"
        ),
        gate_rate_hz=_positive(_as_float(section.get("gate_rate_hz", 20.0), "safety.gate_rate_hz"), "safety.gate_rate_hz"),
    )
    _ordered({"slow": safety.sigma_xy_slow_m, "stop": safety.sigma_xy_stop_m}, ("slow", "stop"), "safety.sigma_xy")
    if safety.sigma_yaw_slow_rad >= safety.sigma_yaw_stop_rad:
        raise NavigationConfigError("safety: sigma_yaw_slow_rad must be < sigma_yaw_stop_rad")
    return safety


def _parse_dig(mission: Dict[str, Any]) -> DigCfg:
    section = _section(mission, "dig", "mission")
    return DigCfg(
        heading_rad=_as_float(section.get("heading_rad", math.pi / 2.0), "mission.dig.heading_rad"),
        pass_spacing_m=_positive(
            _as_float(section.get("pass_spacing_m", 0.55), "mission.dig.pass_spacing_m"), "mission.dig.pass_spacing_m"
        ),
        points_per_pass=_as_int(section.get("points_per_pass", 3), "mission.dig.points_per_pass"),
        cross_inset_m=_non_negative(
            _as_float(section.get("cross_inset_m", 0.25), "mission.dig.cross_inset_m"), "mission.dig.cross_inset_m"
        ),
        y_inset_m=_non_negative(_as_float(section.get("y_inset_m", 0.60), "mission.dig.y_inset_m"), "mission.dig.y_inset_m"),
    )


def _parse_dump(mission: Dict[str, Any]) -> DumpCfg:
    section = _section(mission, "dump", "mission")
    return DumpCfg(
        heading_rad=_as_float(section.get("heading_rad", math.pi / 2.0), "mission.dump.heading_rad"),
        points=_as_int(section.get("points", 5), "mission.dump.points"),
        inset_m=_non_negative(_as_float(section.get("inset_m", 0.30), "mission.dump.inset_m"), "mission.dump.inset_m"),
        standoff_m=_non_negative(_as_float(section.get("standoff_m", 0.35), "mission.dump.standoff_m"), "mission.dump.standoff_m"),
    )


def _parse_park(mission: Dict[str, Any]) -> ParkCfg:
    section = _section(mission, "park", "mission")
    return ParkCfg(
        x=_as_float(_require(section, "x", "mission.park"), "mission.park.x"),
        y=_as_float(_require(section, "y", "mission.park"), "mission.park.y"),
        yaw=_as_float(section.get("yaw", 0.0), "mission.park.yaw"),
    )


def _parse_mission(raw: Dict[str, Any]) -> MissionCfg:
    section = _section(raw, "mission")
    work_mode = _as_str(section.get("work_mode", "delegate"), "mission.work_mode")
    if work_mode not in WORK_MODES:
        raise NavigationConfigError(f"mission.work_mode: must be one of {WORK_MODES}, got {work_mode!r}")
    mission = MissionCfg(
        enabled=_as_bool(section.get("enabled", True), "mission.enabled"),
        work_mode=work_mode,
        cycles_target=_as_int(section.get("cycles_target", 6), "mission.cycles_target"),
        excavation_action=_as_str(
            section.get("excavation_action", "/excavation/work_cycle"), "mission.excavation_action"
        ),
        excavation_wait_s=_non_negative(
            _as_float(section.get("excavation_wait_s", 5.0), "mission.excavation_wait_s"), "mission.excavation_wait_s"
        ),
        work_timeout_s=_positive(_as_float(section.get("work_timeout_s", 90.0), "mission.work_timeout_s"), "mission.work_timeout_s"),
        acquire_timeout_s=_positive(
            _as_float(section.get("acquire_timeout_s", 45.0), "mission.acquire_timeout_s"), "mission.acquire_timeout_s"
        ),
        acquire_spin_rad=_positive(
            _as_float(section.get("acquire_spin_rad", 1.2), "mission.acquire_spin_rad"), "mission.acquire_spin_rad"
        ),
        dig=_parse_dig(section),
        dump=_parse_dump(section),
        park=_parse_park(section),
    )
    if mission.cycles_target < 1:
        raise NavigationConfigError("mission.cycles_target: must be >= 1")
    if mission.dig.points_per_pass < 1:
        raise NavigationConfigError("mission.dig.points_per_pass: must be >= 1")
    if mission.dump.points < 1:
        raise NavigationConfigError("mission.dump.points: must be >= 1")
    return mission


def _parse_topics(raw: Dict[str, Any]) -> TopicsCfg:
    section = _section(raw, "topics")
    odometry = _as_str_list(_require(section, "odometry_topics", "topics"), "topics.odometry_topics")
    if not odometry:
        raise NavigationConfigError("topics.odometry_topics: at least one topic is required")
    return TopicsCfg(
        obstacles=_as_str(_require(section, "obstacles", "topics"), "topics.obstacles"),
        localization_state=_as_str(_require(section, "localization_state", "topics"), "topics.localization_state"),
        odometry_topics=odometry,
        cmd_vel=_as_str(_require(section, "cmd_vel", "topics"), "topics.cmd_vel"),
        nav_cmd_vel=_as_str(_require(section, "nav_cmd_vel", "topics"), "topics.nav_cmd_vel"),
        path=_as_str(_require(section, "path", "topics"), "topics.path"),
        nav_status=_as_str(_require(section, "nav_status", "topics"), "topics.nav_status"),
        gate_status=_as_str(_require(section, "gate_status", "topics"), "topics.gate_status"),
        costmap=_as_str(_require(section, "costmap", "topics"), "topics.costmap"),
        mission_status=_as_str(_require(section, "mission_status", "topics"), "topics.mission_status"),
        perception_mode=_as_str(_require(section, "perception_mode", "topics"), "topics.perception_mode"),
        navigate_action=_as_str(_require(section, "navigate_action", "topics"), "topics.navigate_action"),
        berm_started=_as_str(_require(section, "berm_started", "topics"), "topics.berm_started"),
        estop=_as_str(_require(section, "estop", "topics"), "topics.estop"),
    )


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def config_from_mapping(raw: Dict[str, Any], path: str = "", sha256: str = "") -> NavigationConfig:
    """Validate a parsed ``navigation.yaml`` mapping."""
    if not isinstance(raw, dict):
        raise NavigationConfigError("navigation.yaml: the top level must be a mapping")
    schema_version = _as_int(raw.get("schema_version", 1), "schema_version")
    if schema_version != 1:
        raise NavigationConfigError(f"schema_version: expected 1, got {schema_version}")

    frames = _parse_frames(raw)
    arena = _parse_arena(raw)
    robot = _parse_robot(raw)
    costmap = _parse_costmap(raw)
    planner = _parse_planner(raw)
    controller = _parse_controller(raw)
    safety = _parse_safety(raw)
    mission = _parse_mission(raw)
    topics = _parse_topics(raw)

    config = NavigationConfig(
        frames=frames,
        arena=arena,
        robot=robot,
        costmap=costmap,
        planner=planner,
        controller=controller,
        safety=safety,
        mission=mission,
        topics=topics,
        path=path,
        sha256=sha256,
        schema_version=schema_version,
        raw=raw,
    )

    # Cross-checks that need more than one section.
    if config.free_rect.width <= 0.0 or config.free_rect.height <= 0.0:
        raise NavigationConfigError(
            "arena/robot: the wall margin plus the rover radius leaves no free space in the arena"
        )
    if costmap.resolution_m > config.free_rect.width or costmap.resolution_m > config.free_rect.height:
        raise NavigationConfigError("costmap.resolution_m: coarser than the free arena")
    if controller.min_speed_mps >= robot.max_speed_mps:
        raise NavigationConfigError("controller.min_speed_mps: must be below robot.max_speed_mps")
    if controller.slowdown_distance_m <= robot.goal_tolerance_m:
        raise NavigationConfigError("controller.slowdown_distance_m: must exceed robot.goal_tolerance_m")
    if safety.emergency_stop_m <= robot.goal_tolerance_m:
        raise NavigationConfigError("safety.emergency_stop_m: must exceed robot.goal_tolerance_m")
    if mission.enabled and not arena.zone("excavation").area > 0.0:  # pragma: no cover - guarded above
        raise NavigationConfigError("mission: the excavation zone is empty")
    return config


def package_share_dir() -> str:
    """This package's share directory (install space first, source tree second)."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory("rover_navigation")
    except Exception:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def default_config_path() -> str:
    return os.path.join(package_share_dir(), "config", "navigation.yaml")


def with_work_mode(config: NavigationConfig, work_mode: str) -> NavigationConfig:
    """A copy of ``config`` with ``mission.work_mode`` overridden.

    The launch files need a way to say "drive the legs but do not claim any sand
    moved" without editing the single source of truth.  Rather than adding a
    second code path through the config, the override produces a normal
    :class:`NavigationConfig` and everything downstream is unchanged - including
    the value the status messages report, so a ``navigate_only`` run can never be
    mistaken for a real one.
    """
    import dataclasses

    if work_mode not in WORK_MODES:
        raise NavigationConfigError(f"mission.work_mode: must be one of {WORK_MODES}, got {work_mode!r}")
    if work_mode == config.mission.work_mode:
        return config
    return dataclasses.replace(config, mission=dataclasses.replace(config.mission, work_mode=work_mode))


def load_config(path: Optional[str] = None) -> NavigationConfig:
    """Load and validate ``navigation.yaml`` (default: this package's copy)."""
    import yaml

    resolved = os.path.abspath(os.path.expanduser(path)) if path else default_config_path()
    if not os.path.isfile(resolved):
        raise NavigationConfigError(f"config file not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        blob = handle.read()
    try:
        raw = yaml.safe_load(blob)
    except yaml.YAMLError as exc:  # pragma: no cover - exercised by a config test
        raise NavigationConfigError(f"{resolved}: not valid YAML: {exc}") from exc
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return config_from_mapping(raw, path=resolved, sha256=digest)
