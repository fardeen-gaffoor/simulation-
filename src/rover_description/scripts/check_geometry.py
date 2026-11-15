#!/usr/bin/env python3
"""``ros2 run rover_description check_geometry``

Stage 1 of PROMPT_ROVER_SIM.md: check rover geometry consistency.  Flags the
wheelbase conflict (0.65 m + 0.20 m dia = 0.85 m > 0.81 m overall length) and
anything else that does not close.

    ros2 run rover_description check_geometry
    ros2 run rover_description check_geometry -- --json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from rover_description.geometry_check import check_geometry, load_and_check


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_geometry",
        description="Stage 1: rover geometry consistency check.",
    )
    parser.add_argument("--config", default=None, help="path to robot_params.yaml")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    args = parser.parse_args(argv)

    try:
        robot, report = load_and_check(robot_path=args.config)
    except Exception as exc:  # noqa: BLE001 - report config errors cleanly
        print(f"check_geometry: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("=== Rover Geometry Consistency Check (Stage 1, PROMPT_ROVER_SIM.md) ===")
        print()
        print(report.table())
        print()
        if report.ok:
            print("RESULT: All geometry checks passed (no conflicts).")
        else:
            print("RESULT: Geometry conflicts found - see CONFLICTS above.")

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
