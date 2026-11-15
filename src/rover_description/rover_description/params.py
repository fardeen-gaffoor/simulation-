"""Typed, validated Stage 1 configuration loader.

Loads the four PROMPT_ROVER_SIM.md Stage 1 YAML files
(``robot_params.yaml``, ``motors.yaml``, ``battery.yaml``, ``cameras.yaml``)
and exposes immutable dataclasses.  Every numeric value carries a ``tag``
(``"given"`` / ``"assumed"`` / ``"derived"``) so the requirements report can
list every number that was not stated in the prompt.

This module is importable from plain Python and from pytest.  No ROS imports.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


class RoverParamsError(ValueError):
    """Raised when a Stage 1 config file is missing or inconsistent."""


def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise RoverParamsError(f"{where}: missing required key '{key}'")
    return section[key]


def _val(entry: Any, where: str) -> Any:
    """Extract the ``value`` from a {value, tag, comment} dict, or return raw."""
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def _tag(entry: Any, where: str) -> str:
    """Return the tag from a {value, tag, comment} dict, or "assumed" if absent."""
    if isinstance(entry, dict) and "tag" in entry:
        return str(entry["tag"])
    return "assumed"


def _as_float(entry: Any, where: str) -> float:
    raw = _val(entry, where)
    if raw is None:
        return 0.0  # computed fields (e.g. frame_mass_kg) have value: null in YAML
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise RoverParamsError(f"{where}: expected a number, got {raw!r}") from exc


def _as_int(entry: Any, where: str) -> int:
    try:
        return int(_val(entry, where))
    except (TypeError, ValueError) as exc:
        raise RoverParamsError(f"{where}: expected an integer, got {entry!r}") from exc


def _as_str(entry: Any, where: str) -> str:
    return str(_val(entry, where))


def _as_vec(entry: Any, where: str, length: int) -> Tuple[float, ...]:
    raw = _val(entry, where)
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise RoverParamsError(f"{where}: expected {length} numbers, got {raw!r}")
    return tuple(float(v) for v in raw)


# --------------------------------------------------------------------------- #
# robot_params.yaml dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LimitEntry:
    value: float
    tag: str
    comment: str


@dataclass(frozen=True)
class Limits:
    envelope_length_x: LimitEntry
    envelope_width_y: LimitEntry
    envelope_height_z: LimitEntry
    mass_ceiling_kg: LimitEntry


@dataclass(frozen=True)
class MassEntry:
    value: Optional[float]
    tag: str
    comment: str


@dataclass(frozen=True)
class MassBudget:
    total_max_kg: MassEntry
    wheel_mass_kg: MassEntry
    motor_mass_kg: MassEntry
    num_motors: MassEntry
    battery_mass_kg: MassEntry
    ladder_mass_kg: MassEntry
    tub_funnel_mass_kg: MassEntry
    electronics_wiring_mass_kg: MassEntry
    frame_mass_kg: MassEntry

    def frame_mass_computed(self) -> float:
        """50 kg minus the sum of all other masses."""
        total = self.total_max_kg.value or 0.0
        others = (
            4.0 * (self.wheel_mass_kg.value or 0.0)
            + 4.0 * (self.motor_mass_kg.value or 0.0)
            + (self.battery_mass_kg.value or 0.0)
            + (self.ladder_mass_kg.value or 0.0)
            + (self.tub_funnel_mass_kg.value or 0.0)
            + (self.electronics_wiring_mass_kg.value or 0.0)
        )
        return total - others


@dataclass(frozen=True)
class GeometryEntry:
    value: float
    tag: str
    comment: str


@dataclass(frozen=True)
class Geometry:
    overall_length_m: GeometryEntry
    body_width_m: GeometryEntry
    overall_width_over_wheels_m: GeometryEntry
    track_m: GeometryEntry
    wheelbase_m: GeometryEntry
    ground_clearance_m: GeometryEntry
    gap_frame_to_wheel_m: GeometryEntry


@dataclass(frozen=True)
class ExcavationParams:
    ladder_height_m: GeometryEntry
    belt_width_m: GeometryEntry
    ladder_pitch_min_deg: GeometryEntry
    ladder_pitch_max_deg: GeometryEntry
    belt_speed_mps: GeometryEntry
    flight_spacing_m: GeometryEntry


@dataclass(frozen=True)
class DriveParams:
    num_wheels: MassEntry
    drive_type: str
    target_speed_mps: GeometryEntry


@dataclass(frozen=True)
class RobotParams:
    schema_version: int
    path: str
    raw: Dict[str, Any]
    name: str
    declared_forward: str
    limits: Limits
    mass: MassBudget
    geometry: Geometry
    excavation: ExcavationParams
    drive: DriveParams
    assumptions: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# motors.yaml dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MotorEntry:
    value: Any
    tag: str
    comment: str


@dataclass(frozen=True)
class MotorElectrical:
    voltage_nominal_v: MotorEntry
    no_load_current_a: MotorEntry
    load_current_advised_a: MotorEntry
    stall_current_a: MotorEntry


@dataclass(frozen=True)
class MotorMechanical:
    no_load_output_rpm: MotorEntry
    base_motor_rpm: MotorEntry
    gearbox_stages: MotorEntry
    shaft_diameter_mm: MotorEntry
    gearbox_diameter_mm: MotorEntry
    motor_weight_kg: MotorEntry


@dataclass(frozen=True)
class TorqueSpec:
    rated_torque_kgcm: MotorEntry
    advised_max_torque_kgcm: MotorEntry
    gearbox_breaking_torque_kgcm: MotorEntry
    stall_torque_max_kgcm: MotorEntry
    stall_torque_requirement_kgcm: MotorEntry
    steady_torque_requirement_kgcm: MotorEntry
    speed_requirement_rpm: MotorEntry
    speed_requirement_mps: MotorEntry


@dataclass(frozen=True)
class Conversion:
    kgcm_to_nm: MotorEntry


@dataclass(frozen=True)
class Drivetrain:
    chain_ratio: MotorEntry
    chain_efficiency: MotorEntry
    encoder_cpr: MotorEntry


@dataclass(frozen=True)
class MotorsConfig:
    schema_version: int
    path: str
    raw: Dict[str, Any]
    model: str
    electrical: MotorElectrical
    mechanical: MotorMechanical
    torque: TorqueSpec
    conversion: Conversion
    drivetrain: Drivetrain
    assumptions: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# battery.yaml dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BatteryEntry:
    value: Any
    tag: str
    comment: str


@dataclass(frozen=True)
class BatteryElectrical:
    voltage_nominal_v: BatteryEntry
    capacity_mah: BatteryEntry
    capacity_wh: BatteryEntry
    voltage_cutoff_v: BatteryEntry
    internal_resistance_ohm: BatteryEntry
    chemistry: BatteryEntry


@dataclass(frozen=True)
class PwmSpec:
    duty_cap: BatteryEntry
    duty_cap_14_6v: BatteryEntry


@dataclass(frozen=True)
class BatteryConfig:
    schema_version: int
    path: str
    raw: Dict[str, Any]
    electrical: BatteryElectrical
    pwm: PwmSpec
    assumptions: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# cameras.yaml dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraSpec:
    mount: str
    tilt_down_deg: Optional[float]
    hfov_deg: Optional[float]
    placeholder: bool
    role: str
    tag: str


@dataclass(frozen=True)
class CameraPreset:
    default: str
    image_format: str
    resolution: Tuple[int, int]
    update_rate_hz: float
    low_res_option: bool


@dataclass(frozen=True)
class PresetB:
    name: str
    purpose: str
    enabled: bool


@dataclass(frozen=True)
class ToFSensors:
    count: int
    enabled: bool


@dataclass(frozen=True)
class CamerasConfig:
    schema_version: int
    path: str
    raw: Dict[str, Any]
    preset: CameraPreset
    cameras: Dict[str, CameraSpec]
    preset_b: PresetB
    tof_sensors: ToFSensors
    assumptions: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RoverParamsError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    if not isinstance(doc, dict):
        raise RoverParamsError(f"{path}: top level must be a mapping")
    return doc


def _section(doc: Dict[str, Any], name: str, where: str) -> Dict[str, Any]:
    sec = doc.get(name)
    if not isinstance(sec, dict):
        raise RoverParamsError(f"{where}: missing section '{name}'")
    return sec


def load_robot_params(path: Optional[str] = None) -> RobotParams:
    resolved = os.path.abspath(os.path.expanduser(path)) if path else _default("robot_params.yaml")
    doc = _load_yaml(resolved)
    meta = _section(doc, "meta", resolved)
    limits_raw = _section(doc, "limits", resolved)
    mass_raw = _section(doc, "mass", resolved)
    geom_raw = _section(doc, "geometry", resolved)
    exc_raw = _section(doc, "excavation", resolved)
    drive_raw = _section(doc, "drive", resolved)

    limits = Limits(
        envelope_length_x=_entry(limits_raw, "envelope.length_x", resolved),
        envelope_width_y=_entry(limits_raw, "envelope.width_y", resolved),
        envelope_height_z=_entry(limits_raw, "envelope.height_z", resolved),
        mass_ceiling_kg=_entry(limits_raw, "mass_ceiling_kg", resolved),
    )
    mass = MassBudget(
        total_max_kg=_entry(mass_raw, "total_max_kg", resolved),
        wheel_mass_kg=_entry(mass_raw, "wheel_mass_kg", resolved),
        motor_mass_kg=_entry(mass_raw, "motor_mass_kg", resolved),
        num_motors=_entry(mass_raw, "num_motors", resolved),
        battery_mass_kg=_entry(mass_raw, "battery_mass_kg", resolved),
        ladder_mass_kg=_entry(mass_raw, "ladder_mass_kg", resolved),
        tub_funnel_mass_kg=_entry(mass_raw, "tub_funnel_mass_kg", resolved),
        electronics_wiring_mass_kg=_entry(mass_raw, "electronics_wiring_mass_kg", resolved),
        frame_mass_kg=_entry(mass_raw, "frame_mass_kg", resolved),
    )
    geometry = Geometry(
        overall_length_m=_g_entry(geom_raw, "overall_length_m", resolved),
        body_width_m=_g_entry(geom_raw, "body_width_m", resolved),
        overall_width_over_wheels_m=_g_entry(geom_raw, "overall_width_over_wheels_m", resolved),
        track_m=_g_entry(geom_raw, "track_m", resolved),
        wheelbase_m=_g_entry(geom_raw, "wheelbase_m", resolved),
        ground_clearance_m=_g_entry(geom_raw, "ground_clearance_m", resolved),
        gap_frame_to_wheel_m=_g_entry(geom_raw, "gap_frame_to_wheel_m", resolved),
    )
    excavation = ExcavationParams(
        ladder_height_m=_g_entry(exc_raw, "ladder_height_m", resolved),
        belt_width_m=_g_entry(exc_raw, "belt_width_m", resolved),
        ladder_pitch_min_deg=_g_entry(exc_raw, "ladder_pitch_min_deg", resolved),
        ladder_pitch_max_deg=_g_entry(exc_raw, "ladder_pitch_max_deg", resolved),
        belt_speed_mps=_g_entry(exc_raw, "belt_speed_mps", resolved),
        flight_spacing_m=_g_entry(exc_raw, "flight_spacing_m", resolved),
    )
    drive = DriveParams(
        num_wheels=_entry(drive_raw, "num_wheels", resolved),
        drive_type=_as_str(drive_raw.get("drive_type", {}), "drive.drive_type"),
        target_speed_mps=_g_entry(drive_raw, "target_speed_mps", resolved),
    )

    assumptions = [
        "P1: wheelbase 0.65 m assumed - CONFLICTS with 0.81 m overall length + 0.20 m wheel dia (needs 0.85 m)",
        "P2: frame mass computed as remainder of the 50 kg budget",
        "P3: track 0.54 m derived from overall width over wheels minus wheel width",
    ]

    return RobotParams(
        schema_version=int(doc.get("schema_version", 1)),
        path=resolved,
        raw=doc,
        name=str(meta.get("name", "rover")),
        declared_forward=str(meta.get("declared_forward", "+x")),
        limits=limits,
        mass=mass,
        geometry=geometry,
        excavation=excavation,
        drive=drive,
        assumptions=assumptions,
    )


def load_motors(path: Optional[str] = None) -> MotorsConfig:
    resolved = os.path.abspath(os.path.expanduser(path)) if path else _default("motors.yaml")
    doc = _load_yaml(resolved)
    meta = _section(doc, "meta", resolved)
    elec_raw = _section(doc, "electrical", resolved)
    mech_raw = _section(doc, "mechanical", resolved)
    trq_raw = _section(doc, "torque_kgcm", resolved)
    conv_raw = _section(doc, "conversion", resolved)
    dt_raw = _section(doc, "drivetrain", resolved)

    electrical = MotorElectrical(
        voltage_nominal_v=_m_entry(elec_raw, "voltage_nominal_v", resolved),
        no_load_current_a=_m_entry(elec_raw, "no_load_current_a", resolved),
        load_current_advised_a=_m_entry(elec_raw, "load_current_advised_a", resolved),
        stall_current_a=_m_entry(elec_raw, "stall_current_a", resolved),
    )
    mechanical = MotorMechanical(
        no_load_output_rpm=_m_entry(mech_raw, "no_load_output_rpm", resolved),
        base_motor_rpm=_m_entry(mech_raw, "base_motor_rpm", resolved),
        gearbox_stages=_m_entry(mech_raw, "gearbox_stages", resolved),
        shaft_diameter_mm=_m_entry(mech_raw, "shaft_diameter_mm", resolved),
        gearbox_diameter_mm=_m_entry(mech_raw, "gearbox_diameter_mm", resolved),
        motor_weight_kg=_m_entry(mech_raw, "motor_weight_kg", resolved),
    )
    torque = TorqueSpec(
        rated_torque_kgcm=_m_entry(trq_raw, "rated_torque_kgcm", resolved),
        advised_max_torque_kgcm=_m_entry(trq_raw, "advised_max_torque_kgcm", resolved),
        gearbox_breaking_torque_kgcm=_m_entry(trq_raw, "gearbox_breaking_torque_kgcm", resolved),
        stall_torque_max_kgcm=_m_entry(trq_raw, "stall_torque_max_kgcm", resolved),
        stall_torque_requirement_kgcm=_m_entry(trq_raw, "stall_torque_requirement_kgcm", resolved),
        steady_torque_requirement_kgcm=_m_entry(trq_raw, "steady_torque_requirement_kgcm", resolved),
        speed_requirement_rpm=_m_entry(trq_raw, "speed_requirement_rpm", resolved),
        speed_requirement_mps=_m_entry(trq_raw, "speed_requirement_mps", resolved),
    )
    conversion = Conversion(kgcm_to_nm=_m_entry(conv_raw, "kgcm_to_nm", resolved))
    drivetrain = Drivetrain(
        chain_ratio=_m_entry(dt_raw, "chain_ratio", resolved),
        chain_efficiency=_m_entry(dt_raw, "chain_efficiency", resolved),
        encoder_cpr=_m_entry(dt_raw, "encoder_cpr", resolved),
    )

    assumptions = [
        "M1: stall current 20 A assumed (not given in listing)",
        "M2: chain_ratio 1.0 default, sweep 1.0 to 2.0",
        "M3: chain efficiency 0.95 assumed",
    ]

    return MotorsConfig(
        schema_version=int(doc.get("schema_version", 1)),
        path=resolved,
        raw=doc,
        model=str(meta.get("motor_model", "Robokites Rhino")),
        electrical=electrical,
        mechanical=mechanical,
        torque=torque,
        conversion=conversion,
        drivetrain=drivetrain,
        assumptions=assumptions,
    )


def load_battery(path: Optional[str] = None) -> BatteryConfig:
    resolved = os.path.abspath(os.path.expanduser(path)) if path else _default("battery.yaml")
    doc = _load_yaml(resolved)
    elec_raw = _section(doc, "electrical", resolved)
    pwm_raw = _section(doc, "pwm", resolved)

    electrical = BatteryElectrical(
        voltage_nominal_v=_b_entry(elec_raw, "voltage_nominal_v", resolved),
        capacity_mah=_b_entry(elec_raw, "capacity_mah", resolved),
        capacity_wh=_b_entry(elec_raw, "capacity_wh", resolved),
        voltage_cutoff_v=_b_entry(elec_raw, "voltage_cutoff_v", resolved),
        internal_resistance_ohm=_b_entry(elec_raw, "internal_resistance_ohm", resolved),
        chemistry=_b_entry(elec_raw, "chemistry", resolved),
    )
    pwm = PwmSpec(
        duty_cap=_b_entry(pwm_raw, "duty_cap", resolved),
        duty_cap_14_6v=_b_entry(pwm_raw, "duty_cap_14_6v", resolved),
    )

    assumptions = [
        "B1: voltage cutoff 10.0 V assumed",
        "B2: internal resistance 0.05 ohm assumed",
        "B3: PWM duty cap 1.0 default",
    ]

    return BatteryConfig(
        schema_version=int(doc.get("schema_version", 1)),
        path=resolved,
        raw=doc,
        electrical=electrical,
        pwm=pwm,
        assumptions=assumptions,
    )


def load_cameras(path: Optional[str] = None) -> CamerasConfig:
    resolved = os.path.abspath(os.path.expanduser(path)) if path else _default("cameras.yaml")
    doc = _load_yaml(resolved)
    preset_raw = _section(doc, "preset", resolved)
    cams_raw = _section(doc, "cameras", resolved)
    pb_raw = _section(doc, "preset_b", resolved)
    tof_raw = _section(doc, "tof_sensors", resolved)

    preset = CameraPreset(
        default=_as_str(preset_raw.get("default", {}), "preset.default"),
        image_format=_as_str(preset_raw.get("image_format", {}), "preset.image_format"),
        resolution=_as_vec(preset_raw.get("resolution", {}), "preset.resolution", 2),
        update_rate_hz=_as_float(preset_raw.get("update_rate_hz", {}), "preset.update_rate_hz"),
        low_res_option=bool(preset_raw.get("low_res_option", False)),
    )

    cameras: Dict[str, CameraSpec] = {}
    for name, entry in cams_raw.items():
        cameras[name] = CameraSpec(
            mount=_as_str(entry.get("mount", {}), f"cameras.{name}.mount"),
            tilt_down_deg=_opt_float(entry.get("tilt_down_deg"), f"cameras.{name}.tilt_down_deg"),
            hfov_deg=_opt_float(entry.get("lens_deg") or entry.get("hfov_deg"), f"cameras.{name}.lens_deg"),
            placeholder=bool(entry.get("placeholder", False)),
            role=_as_str(entry.get("role", {}), f"cameras.{name}.role"),
            tag=_tag(entry, f"cameras.{name}"),
        )

    preset_b = PresetB(
        name=_as_str(pb_raw.get("name", {}), "preset_b.name"),
        purpose=_as_str(pb_raw.get("purpose", {}), "preset_b.purpose"),
        enabled=bool(pb_raw.get("enabled", False)),
    )
    tof = ToFSensors(
        count=_as_int(tof_raw.get("count", {}), "tof_sensors.count"),
        enabled=bool(tof_raw.get("enabled", False)),
    )

    assumptions = [
        "C1: all camera poses are placeholders (layout not decided)",
        "C2: preset B (no cameras) for fast physics tests",
        "C3: ToF drop sensors disabled by default",
    ]

    return CamerasConfig(
        schema_version=int(doc.get("schema_version", 1)),
        path=resolved,
        raw=doc,
        preset=preset,
        cameras=cameras,
        preset_b=preset_b,
        tof_sensors=tof,
        assumptions=assumptions,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _default(name: str) -> str:
    """Resolve a config file name in install share or source tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    # Installed share directory (colcon)
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("rover_description")
        if share:
            candidates.append(os.path.join(share, "config", name))
    except Exception:
        pass
    # Source tree fallback (pytest / direct run)
    candidates.append(os.path.join(here, "..", "config", name))
    candidates.append(os.path.join(os.path.dirname(here), "share", "rover_description", name))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    # Return the most likely path even if missing (error will be raised by caller)
    return os.path.join(here, "..", "config", name)


def _entry(doc: Dict[str, Any], dotted: str, where: str) -> MassEntry:
    cur: Any = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise RoverParamsError(f"{where}: missing '{dotted}'")
        cur = cur[part]
    if not isinstance(cur, dict) or "value" not in cur:
        raise RoverParamsError(f"{where}: '{dotted}' is not a {{value, tag, comment}} entry")
    return MassEntry(
        value=_as_float(cur, where),
        tag=_as_str(cur.get("tag", {}), where),
        comment=_as_str(cur.get("comment", ""), where),
    )


def _g_entry(doc: Dict[str, Any], key: str, where: str) -> GeometryEntry:
    cur = _require(doc, key, where)
    if not isinstance(cur, dict) or "value" not in cur:
        raise RoverParamsError(f"{where}: '{key}' is not a {{value, tag, comment}} entry")
    return GeometryEntry(
        value=_as_float(cur, where),
        tag=_as_str(cur.get("tag", {}), where),
        comment=_as_str(cur.get("comment", ""), where),
    )


def _m_entry(doc: Dict[str, Any], key: str, where: str) -> MotorEntry:
    cur = _require(doc, key, where)
    if not isinstance(cur, dict) or "value" not in cur:
        raise RoverParamsError(f"{where}: '{key}' is not a {{value, tag, comment}} entry")
    return MotorEntry(
        value=_val(cur, where),
        tag=_as_str(cur.get("tag", {}), where),
        comment=_as_str(cur.get("comment", ""), where),
    )


def _b_entry(doc: Dict[str, Any], key: str, where: str) -> BatteryEntry:
    cur = _require(doc, key, where)
    if not isinstance(cur, dict) or "value" not in cur:
        raise RoverParamsError(f"{where}: '{key}' is not a {{value, tag, comment}} entry")
    return BatteryEntry(
        value=_val(cur, where),
        tag=_as_str(cur.get("tag", {}), where),
        comment=_as_str(cur.get("comment", ""), where),
    )


def _opt_float(entry: Any, where: str) -> Optional[float]:
    if entry is None:
        return None
    if isinstance(entry, dict) and "value" in entry:
        return _as_float(entry, where)
    return _as_float(entry, where)


__all__ = [
    "RoverParamsError",
    "load_robot_params",
    "load_motors",
    "load_battery",
    "load_cameras",
    "RobotParams",
    "MotorsConfig",
    "BatteryConfig",
    "CamerasConfig",
    "MassBudget",
    "Geometry",
    "TorqueSpec",
    "Drivetrain",
    "CameraSpec",
]
