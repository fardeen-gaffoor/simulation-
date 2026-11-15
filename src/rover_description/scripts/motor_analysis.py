#!/usr/bin/env python3
"""``ros2 run rover_description motor_analysis``

Stage 1 of PROMPT_ROVER_SIM.md: motor and torque analysis over a grid of
(slope, rolling resistance).  Uses the 50 kg design weight, the rod-tip wheel
radius (0.10 m), the chain ratio, and slopes of 0, 5, 10 degrees.

    ros2 run rover_description motor_analysis
    ros2 run rover_description motor_analysis -- --json
    ros2 run rover_description motor_analysis --config robot_params.yaml --motors motors.yaml --mass 50.0
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from rover_description.motor_analysis import analyze_motor_torque, load_and_analyze


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="motor_analysis",
        description="Stage 1: motor and torque analysis for the rover drive.",
    )
    parser.add_argument("--config", default=None, help="path to robot_params.yaml")
    parser.add_argument("--motors", default=None, help="path to motors.yaml")
    parser.add_argument("--mass", type=float, default=None, help="rover mass in kg (default 50.0 from config)")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    args = parser.parse_args(argv)

    robot, motors, analysis = load_and_analyze(
        robot_path=args.config,
        motors_path=args.motors,
        mass_kg=args.mass,
    )

    if args.json:
        out = {
            "mass_kg": analysis.mass_kg,
            "wheel_radius_m": analysis.wheel_radius_m,
            "chain_ratio": analysis.chain_ratio,
            "chain_efficiency": analysis.chain_efficiency,
            "motor_no_load_rpm": analysis.motor_no_load_rpm,
            "rated_torque_kgcm": analysis.motor_rated_torque_kgcm,
            "advised_max_torque_kgcm": analysis.motor_advised_max_kgcm,
            "gearbox_break_torque_kgcm": analysis.motor_gearbox_break_kgcm,
            "stall_torque_requirement_kgcm": analysis.motor_stall_req_kgcm,
            "stall_requirement_met": analysis.stall_requirement_met,
            "no_load_current_a": analysis.no_load_current_a,
            "load_current_advised_a": analysis.load_current_advised_a,
            "cases": [
                {
                    "slope_deg": c.slope_deg,
                    "rr": c.rr,
                    "total_force_n": round(c.total_force_n, 3),
                    "force_per_wheel_n": round(c.force_per_wheel_n, 3),
                    "torque_wheel_nm": round(c.torque_wheel_nm, 4),
                    "torque_motor_nm": round(c.torque_motor_nm, 4),
                    "torque_motor_kgcm": round(c.torque_motor_kgcm, 3),
                    "speed_mps": round(c.speed_mps, 4),
                    "current_a": round(c.current_a, 3),
                    "vs_rated": c.vs_rated,
                    "vs_advised": c.vs_advised,
                    "vs_gearbox": c.vs_gearbox,
                }
                for c in analysis.cases
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(analysis.table())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
