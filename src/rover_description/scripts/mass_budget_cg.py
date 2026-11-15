#!/usr/bin/env python3
"""``ros2 run rover_description mass_budget_cg``

Stage 1 of PROMPT_ROVER_SIM.md: print the rover mass budget with centre of
gravity and inertia tensor, plus front-to-rear weight split sweeps.

    ros2 run rover_description mass_budget_cg
    ros2 run rover_description mass_budget_cg -- --json
    ros2 run rover_description mass_budget_cg --config robot_params.yaml --motors motors.yaml --battery battery.yaml --cameras cameras.yaml
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from rover_description.mass_cg import (
    compute_mass_budget,
    load_all_configs,
)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mass_budget_cg",
        description="Stage 1: rover mass budget with centre of gravity and inertia tensor.",
    )
    parser.add_argument("--config", default=None, help="path to robot_params.yaml")
    parser.add_argument("--motors", default=None, help="path to motors.yaml")
    parser.add_argument("--battery", default=None, help="path to battery.yaml")
    parser.add_argument("--cameras", default=None, help="path to cameras.yaml")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    args = parser.parse_args(argv)

    robot, motors, battery, cameras = load_all_configs(
        robot_path=args.config,
        motors_path=args.motors,
        battery_path=args.battery,
        cameras_path=args.cameras,
    )

    budget = compute_mass_budget(robot, motors, battery, cameras)

    if args.json:
        out = {
            "mass_budget": {
                "rows": [
                    {"name": r.name, "mass_kg": round(r.mass_kg, 4), "category": r.category, "tag": r.tag, "comment": r.comment}
                    for r in budget.rows
                ],
                "structural_mass_kg": round(budget.structural_mass_kg, 4),
                "payload_mass_kg": round(budget.payload_mass_kg, 4),
                "total_mass_kg": round(budget.total_mass_kg, 4),
                "design_target_kg": budget.design_target_kg,
                "competition_ceiling_kg": budget.competition_ceiling_kg,
                "margin_to_target": round(budget.margin_to_target, 4),
                "margin_to_ceiling": round(budget.margin_to_ceiling, 4),
                "ok_target": budget.ok_target,
                "ok_ceiling": budget.ok_ceiling,
            },
            "centre_of_gravity": {
                "x": round(budget.cg.x, 6),
                "y": round(budget.cg.y, 6),
                "z": round(budget.cg.z, 6),
                "total_mass_kg": round(budget.cg.total_mass_kg, 4),
            },
            "inertia_about_cg": {
                "ixx": round(budget.inertia_about_cg.ixx, 6),
                "ixy": round(budget.inertia_about_cg.ixy, 6),
                "ixz": round(budget.inertia_about_cg.ixz, 6),
                "iyy": round(budget.inertia_about_cg.iyy, 6),
                "iyz": round(budget.inertia_about_cg.iyz, 6),
                "izz": round(budget.inertia_about_cg.izz, 6),
                "principal_moments": [round(v, 6) for v in budget.inertia_about_cg.principal_moments()],
            },
            "weight_splits": budget.weight_splits,
        }
        print(json.dumps(out, indent=2))
    else:
        print("=== Rover Mass Budget (Stage 1, PROMPT_ROVER_SIM.md) ===")
        print()
        print(budget.table())
        print()
        print(budget.cg_table())
        print()
        print(budget.inertia_table())
        print()
        print("=== Weight split sweeps ===")
        for s in budget.weight_splits:
            print(f"  {s['split']}: front {s['front_mass_kg']:.2f} kg, rear {s['rear_mass_kg']:.2f} kg")

    return 0 if (budget.ok_target and budget.ok_ceiling) else 1


if __name__ == "__main__":
    raise SystemExit(main())
