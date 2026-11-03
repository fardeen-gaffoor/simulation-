"""Typed, validated ``dataset.yaml`` for the rover_dataset package.

This is module 3's single source of truth, in the same spirit as
``cat_arena/config/arena.yaml`` and ``rover_description/config/rover.yaml``: the
collection plan, the labelling rules, the augmentation ranges and the split
policy all live in one file that is parsed and range-checked here.  A malformed
plan must fail *here*, not halfway through an eight-scene collection run.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "DatasetConfigError",
    "Range",
    "RandomOp",
    "ClassDef",
    "Region",
    "PosePlan",
    "CaptureCfg",
    "LabelingCfg",
    "AugmentCfg",
    "SceneRandomization",
    "SplitCfg",
    "SamplingCfg",
    "PathsCfg",
    "FinetuneCfg",
    "DatasetConfig",
    "load_config",
    "default_config_path",
    "package_share_dir",
    # known augmentation op names (a wire contract with augment.py)
    "PHOTOMETRIC_OPS",
    "GEOMETRIC_OPS",
]

#: Augmentation ops that never change a label.
PHOTOMETRIC_OPS: Tuple[str, ...] = (
    "gaussian_blur",
    "motion_blur",
    "gaussian_noise",
    "salt_pepper",
    "brightness",
    "contrast",
    "gamma",
    "saturation",
    "hue_shift_deg",
    "jpeg",
    "vignette",
)

#: Augmentation ops that rewrite the bounding boxes.
GEOMETRIC_OPS: Tuple[str, ...] = ("flip_180", "horizontal_flip")

_KNOWN_OPS: Tuple[str, ...] = PHOTOMETRIC_OPS + GEOMETRIC_OPS


class DatasetConfigError(ValueError):
    """Raised when ``dataset.yaml`` is missing something or breaks a rule."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise DatasetConfigError(f"{where}: missing required key '{key}'")
    return section[key]


def _as_float(value: Any, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetConfigError(f"{where}: expected a number, got {value!r}") from exc


def _as_int(value: Any, where: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise DatasetConfigError(f"{where}: expected an integer, got {value!r}") from exc


def _as_vec(value: Any, where: str, length: int) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise DatasetConfigError(f"{where}: expected {length} numbers, got {value!r}")
    return tuple(_as_float(v, where) for v in value)


def _as_str_list(value: Any, where: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, (list, tuple)):
        raise DatasetConfigError(f"{where}: expected a list of strings, got {value!r}")
    return [str(v) for v in value]


# --------------------------------------------------------------------------- #
# value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Range:
    """An inclusive uniform sampling range."""

    lo: float
    hi: float

    def sample(self, rng: Any) -> float:
        return float(rng.uniform(self.lo, self.hi))

    def as_list(self) -> List[float]:
        return [self.lo, self.hi]

    def contains(self, value: float) -> bool:
        return self.lo <= value <= self.hi


@dataclass(frozen=True)
class RandomOp:
    """One augmentation with a probability and named parameter ranges."""

    name: str
    prob: float
    params: Dict[str, Range]

    @property
    def geometric(self) -> bool:
        return self.name in GEOMETRIC_OPS

    def draws(self, rng: Any) -> Dict[str, float]:
        return {key: value.sample(rng) for key, value in self.params.items()}


@dataclass(frozen=True)
class ClassDef:
    index: int
    name: str
    description: str
    color: Tuple[float, float, float]


@dataclass(frozen=True)
class Region:
    name: str
    rect: Tuple[float, float, float, float]
    weight: float

    @property
    def x0(self) -> float:
        return self.rect[0]

    @property
    def y0(self) -> float:
        return self.rect[1]

    @property
    def x1(self) -> float:
        return self.rect[2]

    @property
    def y1(self) -> float:
        return self.rect[3]


@dataclass(frozen=True)
class PosePlan:
    margin: float
    regions: List[Region]
    heading: str
    heading_jitter_deg: float
    min_obstacle_clearance: float
    min_crater_clearance: float
    max_tries: int
    seed_namespace: str


@dataclass(frozen=True)
class CaptureCfg:
    poses_per_scene: int
    frames_per_pose: int
    cameras: List[str]
    settle_seconds: float
    pose_gz_attempts: int
    world_name: str
    model_name: str
    teleport_z: float
    pose_plan: PosePlan


@dataclass(frozen=True)
class LabelingCfg:
    ground_z: float
    boulder_silhouette_samples: int
    crater_rim_samples: int
    min_box_width_px: float
    min_box_height_px: float
    min_visible_fraction: float
    keep_truncated: bool
    truncation_margin_px: float
    occluded_policy: str
    occlusion_fraction: float
    min_box_area_px: float


@dataclass(frozen=True)
class AugmentCfg:
    enabled: bool
    seed_namespace: str
    ops: Dict[str, RandomOp]

    def op(self, name: str) -> RandomOp:
        return self.ops[name]

    @property
    def geometric_ops(self) -> List[RandomOp]:
        return [self.ops[name] for name in GEOMETRIC_OPS if name in self.ops]


@dataclass(frozen=True)
class SceneRandomization:
    count: int
    start_seed: int
    seed_namespace: str
    sand_hue_shift: Range
    sand_value_scale: Range
    terrain_noise_amplitude: Range
    sun_elevation_deg: Range
    sun_azimuth_deg: Range
    sun_warm_cool: Range
    sun_intensity_scale: Range
    sun_cast_shadows_prob: float
    ambient: Range
    background: Range
    mount_translation_m: Range
    mount_rotation_deg: Range


@dataclass(frozen=True)
class SplitCfg:
    val_fraction: float
    seed: int
    seed_namespace: str
    stratify_by_source: bool


@dataclass(frozen=True)
class SamplingCfg:
    target_real_fraction: float
    max_repeat: int
    sim_repeat: int


@dataclass(frozen=True)
class PathsCfg:
    arena_config: str
    rover_config: str
    dataset_root: str
    scenes_dir: str
    image_format: str
    image_quality: int
    classes_file: str
    data_yaml: str
    write_sidecar: bool


@dataclass(frozen=True)
class FinetuneCfg:
    base_model: str
    imgsz: int
    epochs: int
    batch: int
    patience: int
    workers: int
    device: str
    project: str
    name: str
    seed: int
    freeze: int
    notes: List[str]


@dataclass(frozen=True)
class DatasetConfig:
    schema_version: int
    path: str
    raw: Dict[str, Any]
    name: str
    rule_note: str
    classes: List[ClassDef]
    paths: PathsCfg
    capture: CaptureCfg
    labeling: LabelingCfg
    augment: AugmentCfg
    scene: SceneRandomization
    split: SplitCfg
    sampling: SamplingCfg
    finetune: FinetuneCfg

    # -- convenience ------------------------------------------------------- #
    @property
    def class_names(self) -> List[str]:
        return [c.name for c in self.classes]

    def class_index(self, name: str) -> int:
        for cls in self.classes:
            if cls.name == name:
                return cls.index
        raise DatasetConfigError(f"unknown class '{name}'")

    def class_by_name(self, name: str) -> ClassDef:
        for cls in self.classes:
            if cls.name == name:
                return cls
        raise DatasetConfigError(f"unknown class '{name}'")

    def config_sha256(self) -> str:
        with open(self.path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()


# --------------------------------------------------------------------------- #
# asset resolution (install space first, source tree fallback) -- the module-2
# pattern, required because ``colcon test`` imports from the install space.
# --------------------------------------------------------------------------- #
def package_share_dir() -> Optional[str]:
    try:
        from ament_index_python.packages import get_package_share_directory

        return get_package_share_directory("rover_dataset")
    except Exception:
        return None


def _asset_path(relative: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    share = package_share_dir()
    if share:
        candidates.append(os.path.join(share, relative))
    candidates.append(os.path.join(here, "..", relative))
    candidates.append(os.path.join(os.path.dirname(here), "share", "rover_dataset", relative))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    raise DatasetConfigError(f"cannot locate {relative}")


def default_config_path() -> str:
    return _asset_path(os.path.join("config", "dataset.yaml"))


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def _parse_classes(raw: Any) -> List[ClassDef]:
    if not isinstance(raw, list) or not raw:
        raise DatasetConfigError("classes: expected a non-empty list")
    out: List[ClassDef] = []
    seen = set()
    for i, entry in enumerate(raw):
        where = f"classes[{i}]"
        if not isinstance(entry, dict):
            raise DatasetConfigError(f"{where}: expected a mapping")
        name = str(_require(entry, "name", where))
        if name in seen:
            raise DatasetConfigError(f"{where}: duplicate class name '{name}'")
        seen.add(name)
        color = _as_vec(entry.get("color", [1.0, 1.0, 1.0]), f"{where}.color", 3)
        for c in color:
            if not 0.0 <= c <= 1.0:
                raise DatasetConfigError(f"{where}.color: components must be in [0, 1]")
        out.append(ClassDef(i, name, str(entry.get("description", "")), color))
    return out


def _parse_range(value: Any, where: str, *, minimum: Optional[float] = None) -> Range:
    if isinstance(value, dict):
        if "min" not in value or "max" not in value:
            raise DatasetConfigError(f"{where}: expected {{min, max}}")
        lo = _as_float(value["min"], f"{where}.min")
        hi = _as_float(value["max"], f"{where}.max")
    elif isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise DatasetConfigError(f"{where}: expected [min, max]")
        lo = _as_float(value[0], where)
        hi = _as_float(value[1], where)
    else:
        lo = hi = _as_float(value, where)
    if lo > hi:
        raise DatasetConfigError(f"{where}: min ({lo}) must be <= max ({hi})")
    if minimum is not None and lo < minimum:
        raise DatasetConfigError(f"{where}: min ({lo}) must be >= {minimum}")
    return Range(lo, hi)


def _parse_paths(raw: Dict[str, Any]) -> PathsCfg:
    fmt = str(raw.get("image_format", "jpg")).lower()
    if fmt not in ("jpg", "jpeg", "png"):
        raise DatasetConfigError("paths.image_format: must be jpg or png")
    quality = _as_int(raw.get("image_quality", 95), "paths.image_quality")
    if not 1 <= quality <= 100:
        raise DatasetConfigError("paths.image_quality: must be in [1, 100]")
    return PathsCfg(
        arena_config=str(raw.get("arena_config", "")),
        rover_config=str(raw.get("rover_config", "")),
        dataset_root=str(raw.get("dataset_root", "dataset")),
        scenes_dir=str(raw.get("scenes_dir", "_scenes")),
        image_format="jpg" if fmt == "jpeg" else fmt,
        image_quality=quality,
        classes_file=str(raw.get("classes_file", "classes.txt")),
        data_yaml=str(raw.get("data_yaml", "data.yaml")),
        write_sidecar=bool(raw.get("write_sidecar", True)),
    )


def _parse_pose_plan(raw: Dict[str, Any]) -> PosePlan:
    regions_raw = raw.get("regions", [])
    if not isinstance(regions_raw, list) or not regions_raw:
        raise DatasetConfigError("capture.pose_plan.regions: expected a non-empty list")
    regions: List[Region] = []
    for i, entry in enumerate(regions_raw):
        where = f"capture.pose_plan.regions[{i}]"
        rect = _as_vec(_require(entry, "rect", where), f"{where}.rect", 4)
        if rect[2] - rect[0] <= 0 or rect[3] - rect[1] <= 0:
            raise DatasetConfigError(f"{where}.rect: must have positive width and height")
        weight = _as_float(entry.get("weight", 1.0), f"{where}.weight")
        if weight <= 0:
            raise DatasetConfigError(f"{where}.weight: must be > 0")
        regions.append(Region(str(_require(entry, "name", where)), rect, weight))
    names = [r.name for r in regions]
    if len(set(names)) != len(names):
        raise DatasetConfigError("capture.pose_plan.regions: duplicate region names")

    heading = str(raw.get("heading", "toward_obstacles"))
    if heading not in ("toward_obstacles", "toward_region_centre", "random"):
        raise DatasetConfigError(
            "capture.pose_plan.heading: must be toward_obstacles, toward_region_centre or random"
        )
    jitter = _as_float(raw.get("heading_jitter_deg", 25.0), "capture.pose_plan.heading_jitter_deg")
    if not 0.0 <= jitter <= 180.0:
        raise DatasetConfigError("capture.pose_plan.heading_jitter_deg: must be in [0, 180]")
    return PosePlan(
        margin=_as_float(raw.get("margin", 0.45), "capture.pose_plan.margin"),
        regions=regions,
        heading=heading,
        heading_jitter_deg=jitter,
        min_obstacle_clearance=_as_float(
            raw.get("min_obstacle_clearance", 0.75), "capture.pose_plan.min_obstacle_clearance"
        ),
        min_crater_clearance=_as_float(
            raw.get("min_crater_clearance", 0.55), "capture.pose_plan.min_crater_clearance"
        ),
        max_tries=_as_int(raw.get("max_tries", 400), "capture.pose_plan.max_tries"),
        seed_namespace=str(raw.get("seed_namespace", "pose_plan")),
    )


def _parse_capture(raw: Dict[str, Any], class_names: Sequence[str]) -> CaptureCfg:
    poses = _as_int(raw.get("poses_per_scene", 60), "capture.poses_per_scene")
    if poses <= 0:
        raise DatasetConfigError("capture.poses_per_scene: must be > 0")
    frames = _as_int(raw.get("frames_per_pose", 1), "capture.frames_per_pose")
    if frames <= 0:
        raise DatasetConfigError("capture.frames_per_pose: must be > 0")
    cameras = _as_str_list(raw.get("cameras", ["cam_nav"]), "capture.cameras")
    if not cameras:
        raise DatasetConfigError("capture.cameras: must not be empty")
    settle = _as_float(raw.get("settle_seconds", 0.7), "capture.settle_seconds")
    if settle < 0:
        raise DatasetConfigError("capture.settle_seconds: must be >= 0")
    teleport_z = _as_float(raw.get("teleport_z", 0.12), "capture.teleport_z")
    if teleport_z < 0:
        raise DatasetConfigError("capture.teleport_z: must be >= 0")
    attempts = _as_int(raw.get("pose_gz_attempts", 3), "capture.pose_gz_attempts")
    if attempts <= 0:
        raise DatasetConfigError("capture.pose_gz_attempts: must be > 0")
    return CaptureCfg(
        poses_per_scene=poses,
        frames_per_pose=frames,
        cameras=cameras,
        settle_seconds=settle,
        pose_gz_attempts=attempts,
        world_name=str(raw.get("world_name", "cat_arena")),
        model_name=str(raw.get("model_name", "rover")),
        teleport_z=teleport_z,
        pose_plan=_parse_pose_plan(dict(raw.get("pose_plan", {}))),
    )


def _parse_labeling(raw: Dict[str, Any]) -> LabelingCfg:
    policy = str(raw.get("occluded_policy", "drop"))
    if policy not in ("drop", "keep"):
        raise DatasetConfigError("labeling.occluded_policy: must be drop or keep")
    min_visible = _as_float(raw.get("min_visible_fraction", 0.30), "labeling.min_visible_fraction")
    if not 0.0 < min_visible <= 1.0:
        raise DatasetConfigError("labeling.min_visible_fraction: must be in (0, 1]")
    occlusion = _as_float(raw.get("occlusion_fraction", 0.98), "labeling.occlusion_fraction")
    if not 0.0 < occlusion <= 1.0:
        raise DatasetConfigError("labeling.occlusion_fraction: must be in (0, 1]")
    samples_b = _as_int(raw.get("boulder_silhouette_samples", 512), "labeling.boulder_silhouette_samples")
    samples_c = _as_int(raw.get("crater_rim_samples", 256), "labeling.crater_rim_samples")
    if samples_b < 16 or samples_c < 16:
        raise DatasetConfigError("labeling: silhouette sample counts must be >= 16")
    return LabelingCfg(
        ground_z=_as_float(raw.get("ground_z", 0.0), "labeling.ground_z"),
        boulder_silhouette_samples=samples_b,
        crater_rim_samples=samples_c,
        min_box_width_px=_as_float(raw.get("min_box_width_px", 6.0), "labeling.min_box_width_px"),
        min_box_height_px=_as_float(raw.get("min_box_height_px", 6.0), "labeling.min_box_height_px"),
        min_visible_fraction=min_visible,
        keep_truncated=bool(raw.get("keep_truncated", True)),
        truncation_margin_px=_as_float(raw.get("truncation_margin_px", 1.0), "labeling.truncation_margin_px"),
        occluded_policy=policy,
        occlusion_fraction=occlusion,
        min_box_area_px=_as_float(raw.get("min_box_area_px", 40.0), "labeling.min_box_area_px"),
    )


def _parse_augment(raw: Dict[str, Any]) -> AugmentCfg:
    ops: Dict[str, RandomOp] = {}
    for name, value in raw.items():
        if name in ("enabled", "seed_namespace"):
            continue
        if name not in _KNOWN_OPS:
            raise DatasetConfigError(
                f"augment.{name}: unknown op (known: {', '.join(_KNOWN_OPS)})"
            )
        if not isinstance(value, dict):
            raise DatasetConfigError(f"augment.{name}: expected a mapping with 'prob'")
        prob = _as_float(value.get("prob", 0.0), f"augment.{name}.prob")
        if not 0.0 <= prob <= 1.0:
            raise DatasetConfigError(f"augment.{name}.prob: must be in [0, 1]")
        params: Dict[str, Range] = {}
        for key, param in value.items():
            if key == "prob":
                continue
            params[key] = _parse_range(param, f"augment.{name}.{key}")
        ops[name] = RandomOp(name, prob, params)
    return AugmentCfg(
        enabled=bool(raw.get("enabled", True)),
        seed_namespace=str(raw.get("seed_namespace", "augment")),
        ops=ops,
    )


def _parse_scene(raw: Dict[str, Any]) -> SceneRandomization:
    sun = dict(raw.get("sun", {}))
    scene = dict(raw.get("scene", {}))
    sand = dict(raw.get("sand_color", {}))
    terrain = dict(raw.get("terrain", {}))
    jitter = dict(raw.get("camera_mount_jitter", {}))
    shadows = _as_float(sun.get("cast_shadows_prob", 0.85), "scene.sun.cast_shadows_prob")
    if not 0.0 <= shadows <= 1.0:
        raise DatasetConfigError("scene.sun.cast_shadows_prob: must be in [0, 1]")
    count = _as_int(raw.get("count", 8), "scene.count")
    if count <= 0:
        raise DatasetConfigError("scene.count: must be > 0")
    return SceneRandomization(
        count=count,
        start_seed=_as_int(raw.get("start_seed", 1000), "scene.start_seed"),
        seed_namespace=str(raw.get("seed_namespace", "scene")),
        sand_hue_shift=_parse_range(sand.get("hue_shift", {"min": 0.0, "max": 0.0}), "scene.sand_color.hue_shift"),
        sand_value_scale=_parse_range(sand.get("value_scale", {"min": 1.0, "max": 1.0}), "scene.sand_color.value_scale", minimum=0.0),
        terrain_noise_amplitude=_parse_range(terrain.get("noise_amplitude", {"min": 0.0, "max": 0.0}), "scene.terrain.noise_amplitude", minimum=0.0),
        sun_elevation_deg=_parse_range(sun.get("elevation_deg", {"min": 45.0, "max": 45.0}), "scene.sun.elevation_deg"),
        sun_azimuth_deg=_parse_range(sun.get("azimuth_deg", {"min": -45.0, "max": -45.0}), "scene.sun.azimuth_deg"),
        sun_warm_cool=_parse_range(sun.get("warm_cool", {"min": 0.0, "max": 0.0}), "scene.sun.warm_cool"),
        sun_intensity_scale=_parse_range(sun.get("intensity_scale", {"min": 1.0, "max": 1.0}), "scene.sun.intensity_scale", minimum=0.0),
        sun_cast_shadows_prob=shadows,
        ambient=_parse_range(scene.get("ambient", {"min": 0.55, "max": 0.55}), "scene.scene.ambient", minimum=0.0),
        background=_parse_range(scene.get("background", {"min": 0.8, "max": 0.8}), "scene.scene.background", minimum=0.0),
        mount_translation_m=_parse_range(jitter.get("translation_m", {"min": 0.0, "max": 0.0}), "scene.camera_mount_jitter.translation_m", minimum=0.0),
        mount_rotation_deg=_parse_range(jitter.get("rotation_deg", {"min": 0.0, "max": 0.0}), "scene.camera_mount_jitter.rotation_deg", minimum=0.0),
    )


def _parse_split(raw: Dict[str, Any]) -> SplitCfg:
    val = _as_float(raw.get("val_fraction", 0.25), "split.val_fraction")
    if not 0.0 <= val < 1.0:
        raise DatasetConfigError("split.val_fraction: must be in [0, 1)")
    return SplitCfg(
        val_fraction=val,
        seed=_as_int(raw.get("seed", 0), "split.seed"),
        seed_namespace=str(raw.get("seed_namespace", "split")),
        stratify_by_source=bool(raw.get("stratify_by_source", True)),
    )


def _parse_sampling(raw: Dict[str, Any]) -> SamplingCfg:
    target = _as_float(raw.get("target_real_fraction", 0.6), "sampling.target_real_fraction")
    if not 0.0 <= target < 1.0:
        raise DatasetConfigError("sampling.target_real_fraction: must be in [0, 1)")
    max_repeat = _as_int(raw.get("max_repeat", 8), "sampling.max_repeat")
    if max_repeat < 1:
        raise DatasetConfigError("sampling.max_repeat: must be >= 1")
    sim_repeat = _as_int(raw.get("sim_repeat", 1), "sampling.sim_repeat")
    if sim_repeat < 1:
        raise DatasetConfigError("sampling.sim_repeat: must be >= 1")
    return SamplingCfg(target, max_repeat, sim_repeat)


def _parse_finetune(raw: Dict[str, Any]) -> FinetuneCfg:
    return FinetuneCfg(
        base_model=str(raw.get("base_model", "yolov8n.pt")),
        imgsz=_as_int(raw.get("imgsz", 640), "finetune.imgsz"),
        epochs=_as_int(raw.get("epochs", 100), "finetune.epochs"),
        batch=_as_int(raw.get("batch", 16), "finetune.batch"),
        patience=_as_int(raw.get("patience", 30), "finetune.patience"),
        workers=_as_int(raw.get("workers", 4), "finetune.workers"),
        device=str(raw.get("device", "")),
        project=str(raw.get("project", "runs")),
        name=str(raw.get("name", "boulder_crater")),
        seed=_as_int(raw.get("seed", 0), "finetune.seed"),
        freeze=_as_int(raw.get("freeze", 0), "finetune.freeze"),
        notes=[str(n) for n in (raw.get("notes") or [])],
    )


def load_config(path: Optional[str] = None) -> DatasetConfig:
    """Load and fully validate ``dataset.yaml``."""
    import yaml  # local import keeps --help paths light

    resolved = os.path.abspath(os.path.expanduser(path)) if path else default_config_path()
    if not os.path.isfile(resolved):
        raise DatasetConfigError(f"dataset config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise DatasetConfigError("dataset.yaml must contain a mapping at the top level")

    schema_version = _as_int(raw.get("schema_version", 1), "schema_version")
    if schema_version != 1:
        raise DatasetConfigError(f"unsupported schema_version {schema_version} (expected 1)")

    meta = dict(raw.get("meta", {}))
    classes = _parse_classes(raw.get("classes"))
    capture = _parse_capture(dict(_require(raw, "capture", "dataset.yaml")), [c.name for c in classes])
    labeling = _parse_labeling(dict(raw.get("labeling", {})))
    augment = _parse_augment(dict(raw.get("augment", {})))
    scene = _parse_scene(dict(_require(raw, "scene", "dataset.yaml")))
    split = _parse_split(dict(raw.get("split", {})))
    sampling = _parse_sampling(dict(raw.get("sampling", {})))
    paths = _parse_paths(dict(raw.get("paths", {})))
    finetune = _parse_finetune(dict(raw.get("finetune", {})))

    cfg = DatasetConfig(
        schema_version=schema_version,
        path=resolved,
        raw=raw,
        name=str(meta.get("name", "rover_dataset")),
        rule_note=str(meta.get("rule_note", "")),
        classes=classes,
        paths=paths,
        capture=capture,
        labeling=labeling,
        augment=augment,
        scene=scene,
        split=split,
        sampling=sampling,
        finetune=finetune,
    )

    # The camera names in the capture plan must be known to module 2; resolve
    # them now so a typo fails before a simulation is launched.
    known = _known_camera_names()
    if known:
        for cam in capture.cameras:
            if cam not in known:
                raise DatasetConfigError(
                    f"capture.cameras: '{cam}' is not a rover_description camera ({sorted(known)})"
                )
    return cfg


def _known_camera_names() -> List[str]:
    """Camera names from the shipped ``rover.yaml`` (or [] if unavailable).

    Read straight from module 2's *config* rather than its generated
    calibration, because ``rover.yaml`` always ships while ``calibration.yaml``
    is a build artefact.  This is what turns a typo in ``capture.cameras`` into a
    config error instead of a launch failure.
    """
    try:
        import yaml

        from .scene import resolve_rover_config_path

        with open(resolve_rover_config_path(), "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        cameras = (raw.get("sensors") or {}).get("cameras") or []
        return sorted(str(entry.get("name")) for entry in cameras if entry.get("name"))
    except Exception:
        return []
