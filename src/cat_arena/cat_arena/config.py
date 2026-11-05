"""Loading and validation of the single-source-of-truth arena description.

``arena.yaml`` is the only place arena geometry is allowed to live.  This
module turns it into typed, validated Python objects (:class:`ArenaConfig`)
and refuses to load a configuration that is internally inconsistent, so that a
typo in a scaled-from-the-diagram dimension fails loudly at generation time
instead of silently producing a wrong world.

No ROS imports: this module is importable from plain Python, from a launch
file and from pytest.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

__all__ = [
    "ConfigError",
    "Rect",
    "Friction",
    "Walls",
    "Physics",
    "TerrainCfg",
    "Zone",
    "Marking",
    "BouldersCfg",
    "CratersCfg",
    "MarkerPlacement",
    "MarkersCfg",
    "StartPoseCfg",
    "OverviewCameraCfg",
    "ArenaConfig",
    "load_config",
    "default_config_path",
]


class ConfigError(ValueError):
    """Raised when arena.yaml is missing, malformed or inconsistent."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{where}: missing required key '{key}'")
    return mapping[key]


def _as_float(value: Any, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: expected a number, got {value!r}") from exc


def _as_float_list(value: Any, where: str, length: Optional[int] = None) -> List[float]:
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{where}: expected a list, got {value!r}")
    out = [_as_float(v, where) for v in value]
    if length is not None and len(out) != length:
        raise ConfigError(f"{where}: expected {length} values, got {len(out)}")
    return out


def _check_range(lo: float, hi: float, where: str, *, minimum: Optional[float] = None) -> None:
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ConfigError(f"{where}: bounds must be finite (got {lo}, {hi})")
    if lo > hi:
        raise ConfigError(f"{where}: min ({lo}) must be <= max ({hi})")
    if minimum is not None and lo < minimum:
        raise ConfigError(f"{where}: min ({lo}) must be >= {minimum}")


# --------------------------------------------------------------------------- #
# value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in the arena frame (metres)."""

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

    def contains_rect(self, other: "Rect", margin: float = 0.0) -> bool:
        """True when ``other`` lies inside this rectangle, inset by ``margin``."""
        return (
            other.x0 >= self.x0 + margin
            and other.y0 >= self.y0 + margin
            and other.x1 <= self.x1 - margin
            and other.y1 <= self.y1 - margin
        )

    def covers(self, other: "Rect", tol: float = 0.0) -> bool:
        """True when ``other`` lies inside this rectangle, allowing ``tol`` overshoot."""
        return (
            other.x0 >= self.x0 - tol
            and other.y0 >= self.y0 - tol
            and other.x1 <= self.x1 + tol
            and other.y1 <= self.y1 + tol
        )

    def intersects(self, other: "Rect") -> bool:
        return not (
            other.x1 <= self.x0 or other.x0 >= self.x1
            or other.y1 <= self.y0 or other.y0 >= self.y1
        )

    def distance_to_border(self, x: float, y: float) -> float:
        """Smallest distance from an interior point to any border (negative if outside)."""
        return min(x - self.x0, self.x1 - x, y - self.y0, self.y1 - y)

    def as_list(self) -> List[float]:
        return [self.x0, self.y0, self.x1, self.y1]

    @staticmethod
    def from_list(values: Sequence[float], where: str) -> "Rect":
        x0, y0, x1, y1 = _as_float_list(values, where, 4)
        if x1 <= x0 or y1 <= y0:
            raise ConfigError(
                f"{where}: degenerate rectangle [{x0}, {y0}, {x1}, {y1}] "
                "(x1 must be > x0 and y1 must be > y0)"
            )
        return Rect(x0, y0, x1, y1)


@dataclass(frozen=True)
class Friction:
    mu: float
    mu2: float


@dataclass(frozen=True)
class Walls:
    present: bool
    height: float
    thickness: float
    color: Tuple[float, float, float, float]


@dataclass(frozen=True)
class Physics:
    engine: str
    collision_detector: str
    max_step_size: float
    real_time_factor: float
    gravity: Tuple[float, float, float]


@dataclass(frozen=True)
class TerrainCfg:
    use_heightmap: bool = True
    heightmap_png: str = "height.png"
    heightmap_resolution: float = 0.05
    max_relief: float = 0.4
    skirt: bool = True
    sand_depth: float = 0.30
    albedo_texture: bool = True
    texture_size: int = 1024
    normal_map: bool = True
    friction: Friction = None  # type: ignore

    # Backward compatibility properties (computed)
    @property
    def cell_size(self) -> float:
        return self.heightmap_resolution

    @property
    def noise_amplitude(self) -> float:
        return 0.0  # No ripple noise in heightmap mode

    @property
    def noise_wavelength(self) -> float:
        return 3.0

    @property
    def heightmap_png_bool(self) -> bool:
        return True


@dataclass(frozen=True)
class Zone:
    name: str
    rect: Rect
    color: Tuple[float, float, float, float]
    fill: bool
    tiling: bool
    obstacles_allowed: bool
    description: str


@dataclass(frozen=True)
class Marking:
    width: float
    thickness: float
    z: float
    berm_height: float


@dataclass(frozen=True)
class BouldersCfg:
    count_min: int
    count_max: int
    diameter_min: float
    diameter_max: float
    exposed_min: float
    exposed_max: float
    spacing: float
    margin: float
    zones: List[str]
    zone_weights: Dict[str, float]
    color: Tuple[float, float, float, float]
    static: bool


@dataclass(frozen=True)
class CratersCfg:
    count_min: int
    count_max: int
    width_min: float
    width_max: float
    depth_min: float
    depth_max: float
    spacing: float
    margin: float
    zones: List[str]
    zone_weights: Dict[str, float]


@dataclass(frozen=True)
class MarkerPlacement:
    marker_id: int
    mount: str
    position: Tuple[float, float]
    note: str = ""


@dataclass(frozen=True)
class MarkersCfg:
    enabled: bool
    family: str
    max_count: int
    board_size: float
    tag_size: float
    board_thickness: float
    center_height: float
    stand_off: float
    board_color: Tuple[float, float, float, float]
    tag_color: Tuple[float, float, float, float]
    placements: List[MarkerPlacement]


@dataclass(frozen=True)
class StartPoseCfg:
    footprint_length: float
    footprint_width: float
    margin: float
    yaw_mode: str
    yaw_deg: float
    declared_forward_deg: float
    max_tries: int
    seed_namespace: str


@dataclass(frozen=True)
class OverviewCameraCfg:
    enabled: bool
    topic: str
    width: int
    height: int
    fmt: str
    update_rate: float
    hfov_deg: float
    position: Tuple[float, float, float]
    look_at: Tuple[float, float, float]


@dataclass(frozen=True)
class LightCfg:
    """The single directional ``sun`` light.

    Parameterized (rather than hard-coded) so that module 3 can randomize
    lighting per dataset scene by writing an arena.yaml override.  The
    defaults reproduce the original hard-coded light exactly.
    """

    name: str
    cast_shadows: bool
    pose: Tuple[float, float, float, float, float, float]
    diffuse: Tuple[float, float, float, float]
    specular: Tuple[float, float, float, float]
    direction: Tuple[float, float, float]
    attenuation: Tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneCfg:
    """Scene-level rendering settings (ambient, background, shadow toggle)."""

    ambient: Tuple[float, float, float, float]
    background: Tuple[float, float, float, float]
    shadows: bool
    grid: bool
    origin_visual: bool


@dataclass(frozen=True)
class ArenaConfig:
    """Fully validated arena description."""

    schema_version: int
    path: str
    raw: Dict[str, Any]
    world_name: str
    length_x: float
    width_y: float
    sand_depth: float
    friction: Friction
    walls: Walls
    sand_color: Tuple[float, float, float, float]
    physics: Physics
    terrain: TerrainCfg
    zones: Dict[str, Zone]
    marking: Marking
    boulders: BouldersCfg
    craters: CratersCfg
    markers: MarkersCfg
    start_pose: StartPoseCfg
    spawn_z: float
    overview_camera: OverviewCameraCfg
    light: LightCfg
    scene: SceneCfg
    assumptions: List[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------- #
    @property
    def footprint(self) -> Rect:
        """Arena footprint as a rectangle."""
        return Rect(0.0, 0.0, self.length_x, self.width_y)

    def zone(self, name: str) -> Zone:
        try:
            return self.zones[name]
        except KeyError as exc:
            raise ConfigError(
                f"unknown zone '{name}' (known: {sorted(self.zones)})"
            ) from exc

    def tiling_zones(self) -> List[Zone]:
        return [z for z in self.zones.values() if z.tiling]

    def obstacle_placement_zones(self) -> List[str]:
        return list(self.boulders.zones)

    def config_sha256(self) -> str:
        """Hash of the normalised configuration (reproducibility manifest)."""
        blob = yaml.safe_dump(self.raw, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def assumptions_report(self) -> List[str]:
        return list(self.assumptions)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _parse_color(value: Any, where: str) -> Tuple[float, float, float, float]:
    parts = _as_float_list(value, where, 4)
    for c in parts:
        if not 0.0 <= c <= 1.0:
            raise ConfigError(f"{where}: colour components must be in [0, 1] (got {c})")
    return (parts[0], parts[1], parts[2], parts[3])


def _parse_markers(section: Dict[str, Any], assumptions: List[str]) -> MarkersCfg:
    placements_raw = section.get("placements", [])
    if not isinstance(placements_raw, list):
        raise ConfigError("markers.placements: expected a list")
    placements: List[MarkerPlacement] = []
    seen_ids = set()
    valid_mounts = {"north_wall", "south_wall", "east_wall", "west_wall"}
    for i, entry in enumerate(placements_raw):
        where = f"markers.placements[{i}]"
        mount = str(_require(entry, "mount", where))
        if mount not in valid_mounts:
            raise ConfigError(f"{where}.mount: '{mount}' not one of {sorted(valid_mounts)}")
        pos = _as_float_list(_require(entry, "position", where), f"{where}.position", 2)
        marker_id = int(_require(entry, "id", where))
        if marker_id in seen_ids:
            raise ConfigError(f"{where}.id: duplicate marker id {marker_id}")
        seen_ids.add(marker_id)
        placements.append(
            MarkerPlacement(
                marker_id=marker_id,
                mount=mount,
                position=(pos[0], pos[1]),
                note=str(entry.get("note", "")),
            )
        )
    cfg = MarkersCfg(
        enabled=bool(section.get("enabled", True)),
        family=str(section.get("family", "tag36h11")),
        max_count=int(section.get("max_count", 5)),
        board_size=_as_float(_require(section, "board_size", "markers"), "markers.board_size"),
        tag_size=_as_float(_require(section, "tag_size", "markers"), "markers.tag_size"),
        board_thickness=_as_float(
            section.get("board_thickness", 0.01), "markers.board_thickness"
        ),
        center_height=_as_float(
            _require(section, "center_height", "markers"), "markers.center_height"
        ),
        stand_off=_as_float(section.get("stand_off", 0.012), "markers.stand_off"),
        board_color=_parse_color(section.get("board_color", [1, 1, 1, 1]), "markers.board_color"),
        tag_color=_parse_color(section.get("tag_color", [0, 0, 0, 1]), "markers.tag_color"),
        placements=placements,
    )
    if cfg.enabled:
        if len(placements) > cfg.max_count:
            raise ConfigError(
                f"markers: {len(placements)} placements > allowed max_count "
                f"{cfg.max_count} (competition rule: <= 5 total)"
            )
        # competition: <= 25 x 25 cm
        if cfg.board_size > 0.25 + 1e-9:
            raise ConfigError(
                f"markers.board_size {cfg.board_size} m exceeds the 0.25 m competition limit"
            )
        if cfg.tag_size > cfg.board_size:
            raise ConfigError("markers.tag_size must be <= markers.board_size")
        # competition: <= 60 cm high (top edge of the board)
        top = cfg.center_height + 0.5 * cfg.board_size
        if top > 0.60 + 1e-9:
            raise ConfigError(
                f"marker board top edge at {top:.3f} m exceeds the 0.60 m competition limit"
            )
        # the start-zone frame must carry 2 sides
        start_frame = [p for p in placements if p.mount in ("north_wall", "west_wall")]
        if len(start_frame) < 2:
            assumptions.append(
                "markers: fewer than 2 marker boards on the start-zone frame walls "
                "(north + west); the rules allow 2 sides - check markers.placements"
            )
    return cfg


def _parse_start_pose(section: Dict[str, Any]) -> StartPoseCfg:
    fp = _require(section, "robot_footprint", "start_pose")
    length = _as_float(_require(fp, "length", "start_pose.robot_footprint"), "start_pose.robot_footprint.length")
    width = _as_float(_require(fp, "width", "start_pose.robot_footprint"), "start_pose.robot_footprint.width")
    cfg = StartPoseCfg(
        footprint_length=length,
        footprint_width=width,
        margin=_as_float(section.get("margin", 0.05), "start_pose.margin"),
        yaw_mode=str(section.get("yaw_mode", "random")),
        yaw_deg=_as_float(section.get("yaw_deg", 0.0), "start_pose.yaw_deg"),
        declared_forward_deg=_as_float(
            section.get("declared_forward_deg", 0.0), "start_pose.declared_forward_deg"
        ),
        max_tries=int(section.get("max_tries", 200000)),
        seed_namespace=str(section.get("seed_namespace", "start_pose")),
    )
    if cfg.yaw_mode not in ("random", "fixed"):
        raise ConfigError(f"start_pose.yaw_mode: '{cfg.yaw_mode}' not one of ['random', 'fixed']")
    if cfg.footprint_length <= 0 or cfg.footprint_width <= 0:
        raise ConfigError("start_pose.robot_footprint: both dimensions must be > 0")
    if cfg.margin < 0:
        raise ConfigError("start_pose.margin must be >= 0")
    if cfg.max_tries < 1:
        raise ConfigError("start_pose.max_tries must be >= 1")
    return cfg


def _parse_boulders(section: Dict[str, Any]) -> BouldersCfg:
    count = _require(section, "count", "boulders")
    dia = _require(section, "diameter", "boulders")
    exposed = _require(section, "exposed_fraction", "boulders")
    zones = list(section.get("zones", ["obstacle"]))
    weights = {str(k): _as_float(v, f"boulders.zone_weights.{k}")
               for k, v in dict(section.get("zone_weights", {})).items()}
    cfg = BouldersCfg(
        count_min=int(_require(count, "min", "boulders.count")),
        count_max=int(_require(count, "max", "boulders.count")),
        diameter_min=_as_float(_require(dia, "min", "boulders.diameter"), "boulders.diameter.min"),
        diameter_max=_as_float(_require(dia, "max", "boulders.diameter"), "boulders.diameter.max"),
        exposed_min=_as_float(_require(exposed, "min", "boulders.exposed_fraction"), "boulders.exposed_fraction.min"),
        exposed_max=_as_float(_require(exposed, "max", "boulders.exposed_fraction"), "boulders.exposed_fraction.max"),
        spacing=_as_float(section.get("spacing", 0.25), "boulders.spacing"),
        margin=_as_float(section.get("margin", 0.30), "boulders.margin"),
        zones=zones,
        zone_weights=weights,
        color=_parse_color(section.get("color", [0.42, 0.40, 0.38, 1.0]), "boulders.color"),
        static=bool(section.get("static", True)),
    )
    _check_range(float(cfg.count_min), float(cfg.count_max), "boulders.count", minimum=3.0)
    _check_range(cfg.diameter_min, cfg.diameter_max, "boulders.diameter", minimum=0.01)
    _check_range(cfg.exposed_min, cfg.exposed_max, "boulders.exposed_fraction", minimum=0.01)
    if cfg.exposed_max > 1.0:
        raise ConfigError("boulders.exposed_fraction.max must be <= 1.0")
    if cfg.margin < 0 or cfg.spacing < 0:
        raise ConfigError("boulders.margin and boulders.spacing must be >= 0")
    if not cfg.zones:
        raise ConfigError("boulders.zones must not be empty")
    return cfg


def _parse_craters(section: Dict[str, Any]) -> CratersCfg:
    count = _require(section, "count", "craters")
    width = _require(section, "width", "craters")
    depth = _require(section, "depth", "craters")
    zones = list(section.get("zones", ["obstacle"]))
    weights = {str(k): _as_float(v, f"craters.zone_weights.{k}")
               for k, v in dict(section.get("zone_weights", {})).items()}
    cfg = CratersCfg(
        count_min=int(_require(count, "min", "craters.count")),
        count_max=int(_require(count, "max", "craters.count")),
        width_min=_as_float(_require(width, "min", "craters.width"), "craters.width.min"),
        width_max=_as_float(_require(width, "max", "craters.width"), "craters.width.max"),
        depth_min=_as_float(_require(depth, "min", "craters.depth"), "craters.depth.min"),
        depth_max=_as_float(_require(depth, "max", "craters.depth"), "craters.depth.max"),
        spacing=_as_float(section.get("spacing", 0.30), "craters.spacing"),
        margin=_as_float(section.get("margin", 0.35), "craters.margin"),
        zones=zones,
        zone_weights=weights,
    )
    _check_range(float(cfg.count_min), float(cfg.count_max), "craters.count", minimum=3.0)
    _check_range(cfg.width_min, cfg.width_max, "craters.width", minimum=0.01)
    _check_range(cfg.depth_min, cfg.depth_max, "craters.depth", minimum=0.001)
    if cfg.margin < 0 or cfg.spacing < 0:
        raise ConfigError("craters.margin and craters.spacing must be >= 0")
    if not cfg.zones:
        raise ConfigError("craters.zones must not be empty")
    return cfg


def _validate_tiling(cfg: ArenaConfig) -> None:
    """The tiling zones must exactly partition the arena rectangle."""
    tiling = cfg.tiling_zones()
    if not tiling:
        raise ConfigError("at least one zone must have tiling: true")
    area = sum(z.rect.width * z.rect.height for z in tiling)
    expected = cfg.length_x * cfg.width_y
    if abs(area - expected) > 1e-6:
        raise ConfigError(
            f"tiling zones cover {area:.6f} m^2 but the arena is {expected:.6f} m^2 "
            "- the zone rectangles do not partition the arena"
        )
    for i, a in enumerate(tiling):
        for b in tiling[i + 1:]:
            if a.rect.intersects(b.rect):
                raise ConfigError(
                    f"tiling zones '{a.name}' and '{b.name}' overlap "
                    f"({a.rect.as_list()} vs {b.rect.as_list()})"
                )
    for z in tiling:
        if not cfg.footprint.covers(z.rect, tol=1e-9):
            raise ConfigError(
                f"zone '{z.name}' {z.rect.as_list()} leaves the arena "
                f"{cfg.footprint.as_list()}"
            )


def _validate_rules(cfg: ArenaConfig, assumptions: List[str]) -> None:
    """Competition rules that must hold for any generated arena."""
    for zname in (cfg.boulders.zones + cfg.craters.zones):
        if zname not in cfg.zones:
            raise ConfigError(f"unknown zone '{zname}' referenced by obstacles/craters")
        if not cfg.zones[zname].obstacles_allowed:
            raise ConfigError(
                f"zone '{zname}' is listed as an obstacle placement zone but has "
                "obstacles_allowed: false"
            )
    # no obstacles may ever spawn in the start zone
    for zname in ("start",):
        z = cfg.zones[zname]
        for other in (cfg.boulders, cfg.craters):
            if zname in other.zones:
                raise ConfigError(
                    f"rule violation: zone '{zname}' has obstacles_allowed=false but "
                    "is listed in an obstacle placement zone"
                )
    # the construction zone must contain the berm target
    if not cfg.zone("construction").rect.contains_rect(cfg.zone("berm_target").rect):
        raise ConfigError(
            "berm_target must lie fully inside the construction zone "
            f"({cfg.zone('berm_target').rect.as_list()} vs "
            f"{cfg.zone('construction').rect.as_list()})"
        )
    # the stowed footprint must be able to fit inside the start zone at any heading
    worst = (cfg.start_pose.footprint_length + cfg.start_pose.footprint_width) / math.sqrt(2.0)
    start = cfg.zone("start").rect
    if start.width < worst + 2 * cfg.start_pose.margin or start.height < worst + 2 * cfg.start_pose.margin:
        raise ConfigError(
            f"start zone {start.as_list()} is too small for a "
            f"{cfg.start_pose.footprint_length} x {cfg.start_pose.footprint_width} m rover "
            f"at arbitrary heading (needs >= {worst:.3f} m per axis + margin)"
        )
    # markers must sit on the arena walls
    if cfg.markers.enabled:
        for p in cfg.markers.placements:
            if p.mount == "north_wall" and abs(p.position[1] - cfg.width_y) > 1e-6:
                raise ConfigError(
                    f"marker {p.marker_id}: north_wall placement must have y == "
                    f"{cfg.width_y} (got {p.position[1]})"
                )
            if p.mount == "south_wall" and abs(p.position[1]) > 1e-6:
                raise ConfigError(f"marker {p.marker_id}: south_wall placement must have y == 0")
            if p.mount == "west_wall" and abs(p.position[0]) > 1e-6:
                raise ConfigError(f"marker {p.marker_id}: west_wall placement must have x == 0")
            if p.mount == "east_wall" and abs(p.position[0] - cfg.length_x) > 1e-6:
                raise ConfigError(
                    f"marker {p.marker_id}: east_wall placement must have x == {cfg.length_x}"
                )
            if p.mount in ("north_wall", "south_wall") and not (0.0 <= p.position[0] <= cfg.length_x):
                raise ConfigError(f"marker {p.marker_id}: x outside the arena")
            if p.mount in ("east_wall", "west_wall") and not (0.0 <= p.position[1] <= cfg.width_y):
                raise ConfigError(f"marker {p.marker_id}: y outside the arena")
    assumptions.extend(
        [
            f"arena wall height {cfg.walls.height} m is an assumption (rules do not state it)",
            f"sand friction mu=mu2={cfg.friction.mu} is an assumed starting value",
            f"zone rectangles are scaled from the diagram and need organizer confirmation",
        ]
    )


def _parse(cfg_raw: Dict[str, Any], path: str) -> ArenaConfig:
    if not isinstance(cfg_raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    schema_version = int(cfg_raw.get("schema_version", 0))
    if schema_version != 1:
        raise ConfigError(f"{path}: unsupported schema_version {schema_version} (expected 1)")

    meta = dict(cfg_raw.get("meta", {}))
    arena = dict(_require(cfg_raw, "arena", path))
    friction_raw = dict(_require(arena, "friction", "arena"))
    friction = Friction(
        mu=_as_float(_require(friction_raw, "mu", "arena.friction"), "arena.friction.mu"),
        mu2=_as_float(_require(friction_raw, "mu2", "arena.friction"), "arena.friction.mu2"),
    )
    if friction.mu <= 0 or friction.mu2 <= 0:
        raise ConfigError("arena.friction mu and mu2 must be > 0")

    walls_raw = dict(arena.get("walls", {}))
    walls = Walls(
        present=bool(walls_raw.get("present", True)),
        height=_as_float(walls_raw.get("height", 0.5), "arena.walls.height"),
        thickness=_as_float(walls_raw.get("thickness", 0.06), "arena.walls.thickness"),
        color=_parse_color(walls_raw.get("color", [0.55, 0.55, 0.57, 1.0]), "arena.walls.color"),
    )
    if walls.height <= 0 or walls.thickness <= 0:
        raise ConfigError("arena.walls.height and thickness must be > 0")

    physics_raw = dict(cfg_raw.get("physics", {}))
    gravity = _as_float_list(physics_raw.get("gravity", [0.0, 0.0, -9.8]), "physics.gravity", 3)
    physics = Physics(
        engine=str(physics_raw.get("engine", "dartsim")),
        collision_detector=str(physics_raw.get("collision_detector", "bullet")),
        max_step_size=_as_float(physics_raw.get("max_step_size", 0.001), "physics.max_step_size"),
        real_time_factor=_as_float(physics_raw.get("real_time_factor", 1.0), "physics.real_time_factor"),
        gravity=(gravity[0], gravity[1], gravity[2]),
    )
    if physics.max_step_size <= 0:
        raise ConfigError("physics.max_step_size must be > 0")
    if physics.collision_detector not in ("bullet", "fcl", "ode"):
        raise ConfigError(
            f"physics.collision_detector '{physics.collision_detector}' is not supported "
            "(expected 'bullet' for mesh terrain)"
        )

    terrain_raw = dict(cfg_raw.get("terrain", {}))
    friction_raw = dict(terrain_raw.get("friction", {"mu": 0.9, "mu2": 0.9}))
    terrain_friction = Friction(
        mu=_as_float(_require(friction_raw, "mu", "terrain.friction"), "terrain.friction.mu"),
        mu2=_as_float(_require(friction_raw, "mu2", "terrain.friction"), "terrain.friction.mu2"),
    )
    terrain = TerrainCfg(
        use_heightmap=bool(terrain_raw.get("use_heightmap", True)),
        heightmap_png=str(terrain_raw.get("heightmap_png", "height.png")),
        heightmap_resolution=_as_float(terrain_raw.get("heightmap_resolution", 0.05), "terrain.heightmap_resolution"),
        max_relief=_as_float(terrain_raw.get("max_relief", 0.4), "terrain.max_relief"),
        skirt=bool(terrain_raw.get("skirt", True)),
        sand_depth=_as_float(terrain_raw.get("sand_depth", 0.30), "terrain.sand_depth"),
        albedo_texture=bool(terrain_raw.get("albedo_texture", True)),
        texture_size=int(terrain_raw.get("texture_size", 1024)),
        normal_map=bool(terrain_raw.get("normal_map", True)),
        friction=terrain_friction,
    )
    if terrain.heightmap_resolution <= 0:
        raise ConfigError("terrain.heightmap_resolution must be > 0")
    if terrain.max_relief < 0:
        raise ConfigError("terrain.max_relief must be >= 0")
    if terrain.sand_depth < 0:
        raise ConfigError("terrain.sand_depth must be >= 0")
    if terrain.texture_size < 32:
        raise ConfigError("terrain.texture_size must be >= 32")
    if terrain.friction.mu <= 0 or terrain.friction.mu2 <= 0:
        raise ConfigError("terrain.friction mu and mu2 must be > 0")

    zones: Dict[str, Zone] = {}
    zones_raw = dict(_require(cfg_raw, "zones", path))
    for name, entry in zones_raw.items():
        where = f"zones.{name}"
        if not isinstance(entry, dict):
            raise ConfigError(f"{where}: expected a mapping")
        zones[name] = Zone(
            name=name,
            rect=Rect.from_list(_require(entry, "rect", where), f"{where}.rect"),
            color=_parse_color(entry.get("color", [0.5, 0.5, 0.5, 1.0]), f"{where}.color"),
            fill=bool(entry.get("fill", False)),
            tiling=bool(entry.get("tiling", False)),
            obstacles_allowed=bool(entry.get("obstacles_allowed", False)),
            description=str(entry.get("description", "")),
        )
    for required in ("start", "construction", "obstacle", "excavation", "berm_target"):
        if required not in zones:
            raise ConfigError(f"zones: required zone '{required}' is missing")

    marking_raw = dict(cfg_raw.get("marking", {}))
    marking = Marking(
        width=_as_float(marking_raw.get("width", 0.04), "marking.width"),
        thickness=_as_float(marking_raw.get("thickness", 0.002), "marking.thickness"),
        z=_as_float(marking_raw.get("z", 0.004), "marking.z"),
        berm_height=_as_float(marking_raw.get("berm_height", 0.15), "marking.berm_height"),
    )
    if marking.width <= 0 or marking.thickness <= 0:
        raise ConfigError("marking.width and marking.thickness must be > 0")

    assumptions: List[str] = []
    boulders = _parse_boulders(dict(_require(cfg_raw, "boulders", path)))
    craters = _parse_craters(dict(_require(cfg_raw, "craters", path)))
    markers = _parse_markers(dict(cfg_raw.get("markers", {})), assumptions)
    start_pose = _parse_start_pose(dict(_require(cfg_raw, "start_pose", path)))

    spawn_raw = dict(cfg_raw.get("spawn", {}))
    spawn_z = _as_float(spawn_raw.get("z", 0.08), "spawn.z")

    ov_raw = dict(cfg_raw.get("overview_camera", {}))
    ov = OverviewCameraCfg(
        enabled=bool(ov_raw.get("enabled", False)),
        topic=str(ov_raw.get("topic", "/arena_overview/image")),
        width=int(ov_raw.get("width", 1280)),
        height=int(ov_raw.get("height", 720)),
        fmt=str(ov_raw.get("format", "R8G8B8")),
        update_rate=_as_float(ov_raw.get("update_rate", 2.0), "overview_camera.update_rate"),
        hfov_deg=_as_float(ov_raw.get("hfov_deg", 62.0), "overview_camera.hfov_deg"),
        position=tuple(_as_float_list(ov_raw.get("position", [4.5, -2.4, 7.2]), "overview_camera.position", 3)),
        look_at=tuple(_as_float_list(ov_raw.get("look_at", [4.5, 2.5, 0.0]), "overview_camera.look_at", 3)),
    )
    if ov.update_rate <= 0:
        raise ConfigError("overview_camera.update_rate must be > 0")

    light_raw = dict(cfg_raw.get("light", {}))
    light = LightCfg(
        name=str(light_raw.get("name", "sun")),
        cast_shadows=bool(light_raw.get("cast_shadows", True)),
        pose=tuple(_as_float_list(light_raw.get("pose", [0.0, 0.0, 12.0, 0.0, 0.0, 0.0]), "light.pose", 6)),
        diffuse=_parse_color(light_raw.get("diffuse", [0.92, 0.90, 0.86, 1.0]), "light.diffuse"),
        specular=_parse_color(light_raw.get("specular", [0.25, 0.25, 0.25, 1.0]), "light.specular"),
        direction=tuple(
            _as_float_list(light_raw.get("direction", [-0.45, 0.35, -0.82]), "light.direction", 3)
        ),
        attenuation=tuple(
            _as_float_list(
                light_raw.get("attenuation", [1000.0, 0.9, 0.01, 0.001]), "light.attenuation", 4
            )
        ),
    )
    if sum(abs(v) for v in light.direction) <= 0.0:
        raise ConfigError("light.direction must be a non-zero vector")

    scene_raw = dict(cfg_raw.get("scene", {}))
    scene = SceneCfg(
        ambient=_parse_color(scene_raw.get("ambient", [0.55, 0.55, 0.55, 1.0]), "scene.ambient"),
        background=_parse_color(
            scene_raw.get("background", [0.72, 0.80, 0.90, 1.0]), "scene.background"
        ),
        shadows=bool(scene_raw.get("shadows", True)),
        grid=bool(scene_raw.get("grid", False)),
        origin_visual=bool(scene_raw.get("origin_visual", False)),
    )

    cfg = ArenaConfig(
        schema_version=schema_version,
        path=path,
        raw=cfg_raw,
        world_name=str(meta.get("world_name", "cat_arena")),
        length_x=_as_float(_require(arena, "length_x", "arena"), "arena.length_x"),
        width_y=_as_float(_require(arena, "width_y", "arena"), "arena.width_y"),
        sand_depth=_as_float(_require(arena, "sand_depth", "arena"), "arena.sand_depth"),
        friction=friction,
        walls=walls,
        sand_color=_parse_color(arena.get("sand_color", [0.84, 0.78, 0.58, 1.0]), "arena.sand_color"),
        physics=physics,
        terrain=terrain,
        zones=zones,
        marking=marking,
        boulders=boulders,
        craters=craters,
        markers=markers,
        start_pose=start_pose,
        spawn_z=spawn_z,
        overview_camera=ov,
        light=light,
        scene=scene,
        assumptions=assumptions,
    )
    if cfg.length_x <= 0 or cfg.width_y <= 0:
        raise ConfigError("arena.length_x and arena.width_y must be > 0")
    if cfg.sand_depth <= 0:
        raise ConfigError("arena.sand_depth must be > 0")
    if not cfg.world_name:
        raise ConfigError("meta.world_name must not be empty")

    _validate_tiling(cfg)
    _validate_rules(cfg, cfg.assumptions)
    return cfg


def default_config_path() -> str:
    """Path of ``arena.yaml`` inside the installed ``cat_arena`` share dir."""
    override = os.environ.get("CAT_ARENA_CONFIG")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(get_package_share_directory("cat_arena"), "config", "arena.yaml")
    except Exception:  # pragma: no cover - only when ament index is unavailable
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(here, "..", "config", "arena.yaml"))
        return candidate


def load_config(path: Optional[str] = None) -> ArenaConfig:
    """Load and validate the arena configuration.

    :param path: explicit YAML path; when ``None`` the installed
        ``cat_arena/config/arena.yaml`` (or ``$CAT_ARENA_CONFIG``) is used.
    :raises ConfigError: if the file is missing, malformed or inconsistent.
    """
    resolved = os.path.abspath(os.path.expanduser(path)) if path else default_config_path()
    if not os.path.isfile(resolved):
        raise ConfigError(f"arena configuration not found: {resolved}")
    try:
        with open(resolved, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{resolved}: invalid YAML: {exc}") from exc
    return _parse(raw, resolved)
