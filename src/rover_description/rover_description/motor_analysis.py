"""Stage 1: motor and torque analysis.

Uses the 50 kg design weight, the rod-tip wheel radius (0.10 m), the chain
ratio, a rolling-resistance sweep (0.15 to 0.4) and slopes of 0, 5, 10 degrees.
For each case prints:

* torque per motor (N.m and kg.cm) against the rated (40 kg.cm), advised
  maximum (75 kg.cm) and gearbox breaking limit (100 kg.cm)
* speed achievable at that torque (m/s)
* motor current (A)

Honesty rule: the listing's gearbox breaks at 100 kg.cm, so the 125 kg.cm
stall-torque requirement is NOT met - reported plainly.  Do not tune parameters
to force a pass.

Skid-steer model (rigid ground approximation - Gazebo ground is rigid, it
cannot show real sand sinkage or slip):
  F_needed = m * g * (sin(slope) + rr * cos(slope))     # total tractive force
  F_per_wheel = F_needed / 4                             # 4WD, equal load share
  torque_wheel = F_per_wheel * wheel_radius              # N.m at the wheel
  torque_motor = torque_wheel * chain_ratio / eff        # referred to motor shaft
  speed_mps = (motor_rpm / 60) * 2*pi * wheel_radius * eff / chain_ratio
  current_A = no_load_I + (torque_kgcm / 75.0) * (load_I - no_load_I)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .params import MotorsConfig, RobotParams, load_motors, load_robot_params

G = 9.81  # m/s^2
KG_CM_TO_NM = 0.0980665  # given conversion


@dataclass(frozen=True)
class MotorCase:
    slope_deg: float
    rr: float                 # rolling resistance coefficient
    total_force_n: float      # N, total tractive force
    force_per_wheel_n: float  # N
    torque_wheel_nm: float    # N.m at wheel
    torque_motor_nm: float    # N.m referred to motor shaft
    torque_motor_kgcm: float  # kg.cm
    speed_mps: float          # achievable speed at no-load (gear-limited)
    current_a: float          # estimated motor current
    vs_rated: str             # "OK" / "EXCEEDS"
    vs_advised: str           # "OK" / "EXCEEDS"
    vs_gearbox: str           # "OK" / "EXCEEDS"


@dataclass(frozen=True)
class MotorAnalysis:
    mass_kg: float
    wheel_radius_m: float
    chain_ratio: float
    chain_efficiency: float
    motor_no_load_rpm: float
    motor_rated_torque_kgcm: float
    motor_advised_max_kgcm: float
    motor_gearbox_break_kgcm: float
    motor_stall_req_kgcm: float
    no_load_current_a: float
    load_current_advised_a: float
    cases: List[MotorCase]

    @property
    def stall_requirement_met(self) -> bool:
        return self.motor_stall_req_kgcm <= self.motor_gearbox_break_kgcm

    def table(self) -> str:
        width = 10
        lines: List[str] = []
        lines.append("Motor & torque analysis (50 kg design mass, rod-tip wheel radius 0.10 m)")
        lines.append("=" * 78)
        lines.append(f"mass              : {self.mass_kg} kg")
        lines.append(f"wheel radius      : {self.wheel_radius_m} m")
        lines.append(f"chain ratio       : {self.chain_ratio}")
        lines.append(f"chain efficiency  : {self.chain_efficiency}")
        lines.append(f"motor no-load rpm : {self.motor_no_load_rpm}")
        lines.append("")
        lines.append(f"{'slope'.ljust(width)}  {'rr'.ljust(width)}  {'F_total'.ljust(width)}  {'F/wheel'.ljust(width)}  {'T_motor'.ljust(width)}  {'T_motor'.ljust(width)}  {'speed'.ljust(width)}  {'current'.ljust(width)}  {'rated'.ljust(width)}  {'advised'.ljust(width)}  {'gearbox'.ljust(width)}")
        lines.append(f"{'deg'.ljust(width)}  {'coeff'.ljust(width)}  {'(N)'.ljust(width)}  {'(N)'.ljust(width)}  {'(N.m)'.ljust(width)}  {'(kg.cm)'.ljust(width)}  {'(m/s)'.ljust(width)}  {'(A)'.ljust(width)}  {'check'.ljust(width)}  {'check'.ljust(width)}  {'check'.ljust(width)}")
        lines.append("-" * 78)
        for c in self.cases:
            lines.append(
                f"{c.slope_deg:<{width}.1f}  {c.rr:<{width}.2f}  {c.total_force_n:<{width}.1f}  {c.force_per_wheel_n:<{width}.1f}  "
                f"{c.torque_motor_nm:<{width}.3f}  {c.torque_motor_kgcm:<{width}.2f}  {c.speed_mps:<{width}.3f}  {c.current_a:<{width}.2f}  "
                f"{c.vs_rated:<{width}}  {c.vs_advised:<{width}}  {c.vs_gearbox:<{width}}"
            )
        lines.append("-" * 78)
        lines.append("")
        lines.append(f"Rated torque     : {self.motor_rated_torque_kgcm} kg.cm ({self.motor_rated_torque_kgcm * KG_CM_TO_NM:.3f} N.m)")
        lines.append(f"Advised max      : {self.motor_advised_max_kgcm} kg.cm ({self.motor_advised_max_kgcm * KG_CM_TO_NM:.3f} N.m)")
        lines.append(f"Gearbox break    : {self.motor_gearbox_break_kgcm} kg.cm ({self.motor_gearbox_break_kgcm * KG_CM_TO_NM:.3f} N.m)")
        lines.append(f"Stall requirement: {self.motor_stall_req_kgcm} kg.cm - {'MET' if self.stall_requirement_met else 'NOT MET - motor gearbox breaks at 100 kg.cm'}")

        # Find worst case
        worst = max(self.cases, key=lambda c: c.torque_motor_kgcm)
        lines.append("")
        lines.append(f"Worst case: slope={worst.slope_deg} deg, rr={worst.rr} -> {worst.torque_motor_kgcm:.2f} kg.cm per motor")
        if worst.vs_gearbox == "EXCEEDS":
            lines.append("  => EXCEEDS gearbox breaking torque - this operating point is not reachable without damage.")

        return "\n".join(lines)


def _compare(value_kgcm: float, limit_kgcm: float) -> str:
    return "EXCEEDS" if value_kgcm > limit_kgcm + 1e-9 else "OK"


def analyze_motor_torque(
    robot: RobotParams,
    motors: MotorsConfig,
    mass_kg: Optional[float] = None,
    wheel_radius_m: Optional[float] = None,
    chain_ratio: Optional[float] = None,
    chain_efficiency: Optional[float] = None,
    rr_sweep: Optional[Tuple[float, ...]] = None,
    slopes_deg: Optional[Tuple[float, ...]] = None,
) -> MotorAnalysis:
    """Run the motor/torque analysis over a grid of (slope, rolling resistance)."""
    mass = mass_kg if mass_kg is not None else 50.0  # given design mass
    wr = wheel_radius_m if wheel_radius_m is not None else 0.10  # given rod-tip radius
    cr = chain_ratio if chain_ratio is not None else float(motors.drivetrain.chain_ratio.value or 1.0)
    ce = chain_efficiency if chain_efficiency is not None else float(motors.drivetrain.chain_efficiency.value or 0.95)
    no_load_rpm = float(motors.mechanical.no_load_output_rpm.value or 60.0)
    rated_tq = float(motors.torque.rated_torque_kgcm.value or 40.0)
    advised_tq = float(motors.torque.advised_max_torque_kgcm.value or 75.0)
    break_tq = float(motors.torque.gearbox_breaking_torque_kgcm.value or 100.0)
    stall_req = float(motors.torque.stall_torque_requirement_kgcm.value or 125.0)
    no_load_i = float(motors.electrical.no_load_current_a.value or 0.8)
    load_i = float(motors.electrical.load_current_advised_a.value or 7.5)

    rr_values = rr_sweep or (0.15, 0.25, 0.40)
    slope_values = slopes_deg or (0.0, 5.0, 10.0)

    cases: List[MotorCase] = []
    for slope in slope_values:
        slope_rad = math.radians(slope)
        for rr in rr_values:
            # Total tractive force needed on the slope
            f_needed = mass * G * (math.sin(slope_rad) + rr * math.cos(slope_rad))
            f_per_wheel = f_needed / 4.0
            t_wheel = f_per_wheel * wr  # N.m at wheel
            t_motor = t_wheel * cr / ce  # N.m referred to motor shaft
            t_motor_kgcm = t_motor / KG_CM_TO_NM

            # Achievable speed (no-load, gear-reduced)
            speed_mps = (no_load_rpm / 60.0) * 2.0 * math.pi * wr * ce / cr

            # Current: linear from no-load (0.8 A at 0 kg.cm) to load (7.5 A at 75 kg.cm)
            if t_motor_kgcm <= 0:
                current_a = no_load_i
            else:
                current_a = no_load_i + (t_motor_kgcm / 75.0) * (load_i - no_load_i)

            cases.append(MotorCase(
                slope_deg=slope,
                rr=rr,
                total_force_n=f_needed,
                force_per_wheel_n=f_per_wheel,
                torque_wheel_nm=t_wheel,
                torque_motor_nm=t_motor,
                torque_motor_kgcm=t_motor_kgcm,
                speed_mps=speed_mps,
                current_a=current_a,
                vs_rated=_compare(t_motor_kgcm, rated_tq),
                vs_advised=_compare(t_motor_kgcm, advised_tq),
                vs_gearbox=_compare(t_motor_kgcm, break_tq),
            ))

    return MotorAnalysis(
        mass_kg=mass,
        wheel_radius_m=wr,
        chain_ratio=cr,
        chain_efficiency=ce,
        motor_no_load_rpm=no_load_rpm,
        motor_rated_torque_kgcm=rated_tq,
        motor_advised_max_kgcm=advised_tq,
        motor_gearbox_break_kgcm=break_tq,
        motor_stall_req_kgcm=stall_req,
        no_load_current_a=no_load_i,
        load_current_advised_a=load_i,
        cases=cases,
    )


def load_and_analyze(
    robot_path: Optional[str] = None,
    motors_path: Optional[str] = None,
    mass_kg: Optional[float] = None,
) -> Tuple[RobotParams, MotorsConfig, MotorAnalysis]:
    robot = load_robot_params(robot_path)
    motors = load_motors(motors_path)
    analysis = analyze_motor_torque(robot, motors, mass_kg=mass_kg)
    return robot, motors, analysis


__all__ = [
    "G",
    "KG_CM_TO_NM",
    "MotorCase",
    "MotorAnalysis",
    "analyze_motor_torque",
    "load_and_analyze",
]
