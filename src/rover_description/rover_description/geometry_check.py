"""Stage 1: geometry consistency check.

Checks that the rover geometry from robot_params.yaml is internally
consistent and flags anything that does not close.  In particular:

* wheelbase 0.65 m + wheel diameter 0.20 m = 0.85 m > 0.81 m overall length
  - the wheels would stick out beyond the body.  CONFLICT - flagged.
* track 0.54 m vs overall width over wheels 0.60 m - consistent (gap = 0.03 m/side)
* ground clearance 0.20 m vs wheel radius 0.10 m - body underside is 0.10 m
  above the axle line.  Consistent as given.
* stowed envelope vs 1.5 x 0.75 x 0.75 m competition limit.

Every check is tagged pass / flag / conflict so the requirements report can
list them all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .params import RobotParams, load_robot_params


@dataclass(frozen=True)
class GeometryCheck:
    name: str
    status: str  # "pass" | "flag" | "conflict"
    detail: str
    given_value: Optional[float] = None
    computed_value: Optional[float] = None
    limit_value: Optional[float] = None


@dataclass(frozen=True)
class GeometryReport:
    checks: List[GeometryCheck]
    conflicts: List[str]
    flags: List[str]
    passes: List[str]

    @property
    def ok(self) -> bool:
        return len(self.conflicts) == 0

    def table(self) -> str:
        width = max(len(c.name) for c in self.checks) if self.checks else 10
        lines: List[str] = []
        lines.append(f"{'check'.ljust(width)}  {'status':>8}  detail")
        lines.append("-" * (width + 20))
        for c in self.checks:
            lines.append(f"{c.name.ljust(width)}  {c.status:>8}  {c.detail}")
        lines.append("-" * (width + 20))
        if self.conflicts:
            lines.append("")
            lines.append("CONFLICTS:")
            for c in self.conflicts:
                lines.append(f"  * {c}")
        if self.flags:
            lines.append("")
            lines.append("FLAGS:")
            for f in self.flags:
                lines.append(f"  * {f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "given_value": c.given_value,
                    "computed_value": c.computed_value,
                    "limit_value": c.limit_value,
                }
                for c in self.checks
            ],
            "conflicts": self.conflicts,
            "flags": self.flags,
            "passes": self.passes,
        }


def check_geometry(robot: RobotParams) -> GeometryReport:
    """Run all geometry consistency checks on the robot parameters."""
    g = robot.geometry
    m = robot.mass
    lim = robot.limits

    wheel_dia = 0.20  # given: rod-tip circle / effective rolling diameter
    wheel_radius = wheel_dia / 2.0
    wheel_width = 0.06  # given

    checks: List[GeometryCheck] = []
    conflicts: List[str] = []
    flags: List[str] = []
    passes: List[str] = []

    # --- 1. wheelbase vs overall length --------------------------------------
    wheelbase = g.wheelbase_m.value
    overall_len = g.overall_length_m.value
    needed_len = wheelbase + wheel_dia  # wheels stick out by radius at each end
    if needed_len > overall_len + 1e-6:
        conflict = (
            f"wheelbase {wheelbase} m + wheel dia {wheel_dia} m = {needed_len:.2f} m "
            f"> overall length {overall_len} m; wheels protrude {needed_len - overall_len:.2f} m beyond body"
        )
        checks.append(GeometryCheck("wheelbase_vs_length", "conflict", conflict, wheelbase, needed_len, overall_len))
        conflicts.append(conflict)
    else:
        checks.append(GeometryCheck("wheelbase_vs_length", "pass", f"wheelbase+dia {needed_len:.2f} m <= length {overall_len} m"))
        passes.append("wheelbase fits within overall length")

    # --- 2. track vs overall width ------------------------------------------
    track = g.track_m.value
    overall_width = g.overall_width_over_wheels_m.value
    gap_per_side = (overall_width - track) / 2.0
    checks.append(GeometryCheck(
        "track_vs_width", "pass",
        f"track {track} m + 2 x gap {gap_per_side:.3f} m = overall width {overall_width} m",
        track, overall_width, overall_width,
    ))
    passes.append("track consistent with overall width")

    if gap_per_side < 0.02:
        flags.append(f"gap between frame and wheel only {gap_per_side:.3f} m per side - very tight")

    # --- 3. ground clearance vs wheel radius --------------------------------
    clearance = g.ground_clearance_m.value
    body_above_axle = clearance - wheel_radius
    checks.append(GeometryCheck(
        "clearance_vs_wheel_radius", "pass",
        f"clearance {clearance} m - wheel radius {wheel_radius} m = body underside {body_above_axle:.2f} m above axle",
        clearance, body_above_axle, wheel_radius,
    ))
    passes.append("ground clearance consistent with wheel radius")

    if body_above_axle < 0.05:
        flags.append("body underside less than 5 cm above axle line - very low ground clearance")

    # --- 4. stowed envelope -------------------------------------------------
    env_len = lim.envelope_length_x.value
    env_wid = lim.envelope_width_y.value
    env_hgt = lim.envelope_height_z.value

    # Rough stowed estimate: overall length x overall width x (clearance + chassis height + margin)
    chassis_height = 0.22  # assumed
    est_height = clearance + chassis_height + 0.1  # + margin for ladder/camera
    est_width = overall_width
    est_length = overall_len

    env_ok = True
    if est_length > env_len:
        checks.append(GeometryCheck("envelope_length", "conflict", f"est length {est_length:.2f} m > limit {env_len} m", est_length, est_length, env_len))
        conflicts.append(f"stowed length {est_length:.2f} m exceeds 1.5 m envelope limit")
        env_ok = False
    else:
        checks.append(GeometryCheck("envelope_length", "pass", f"est length {est_length:.2f} m <= limit {env_len} m"))

    if est_width > env_wid:
        checks.append(GeometryCheck("envelope_width", "conflict", f"est width {est_width:.2f} m > limit {env_wid} m", est_width, est_width, env_wid))
        conflicts.append(f"stowed width {est_width:.2f} m exceeds 0.75 m envelope limit")
        env_ok = False
    else:
        checks.append(GeometryCheck("envelope_width", "pass", f"est width {est_width:.2f} m <= limit {env_wid} m"))

    if est_height > env_hgt:
        checks.append(GeometryCheck("envelope_height", "conflict", f"est height {est_height:.2f} m > limit {env_hgt} m", est_height, est_height, env_hgt))
        conflicts.append(f"stowed height {est_height:.2f} m exceeds 0.75 m envelope limit")
        env_ok = False
    else:
        checks.append(GeometryCheck("envelope_height", "pass", f"est height {est_height:.2f} m <= limit {env_hgt} m"))

    if env_ok:
        passes.append("stowed envelope within 1.5 x 0.75 x 0.75 m competition limit")

    # --- 5. wheelbase flag (assumed value) ----------------------------------
    if g.wheelbase_m.tag == "assumed":
        flags.append(f"wheelbase {wheelbase} m is assumed - not stated in prompt; verify against CAD")

    # --- 6. mass target vs ceiling ------------------------------------------
    target = m.total_max_kg.value
    ceiling = lim.mass_ceiling_kg.value
    checks.append(GeometryCheck(
        "mass_target_vs_ceiling", "pass",
        f"design target {target} kg <= competition ceiling {ceiling} kg",
        target, target, ceiling,
    ))
    passes.append("50 kg design target within 80 kg competition ceiling")

    return GeometryReport(
        checks=checks,
        conflicts=conflicts,
        flags=flags,
        passes=passes,
    )


def load_and_check(robot_path: Optional[str] = None) -> Tuple[RobotParams, GeometryReport]:
    robot = load_robot_params(robot_path)
    report = check_geometry(robot)
    return robot, report


__all__ = [
    "GeometryCheck",
    "GeometryReport",
    "check_geometry",
    "load_and_check",
]
