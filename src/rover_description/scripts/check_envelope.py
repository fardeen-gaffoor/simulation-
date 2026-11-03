#!/usr/bin/env python3
"""Assert the rover fits the competition stowed envelope and the 80 kg ceiling.

    ros2 run rover_description check_envelope
    ros2 run rover_description check_envelope -- --json

Exit status is 0 only when both the envelope and the mass budget pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from rover_description.config import RoverConfigError, load_config
from rover_description.envelope import check_envelope
from rover_description.mass import mass_budget


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_envelope",
        description="Check the rover's stowed envelope and mass budget against the limits.",
    )
    parser.add_argument("--config", default=None, help="path to rover.yaml")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except RoverConfigError as exc:
        print(f"check_envelope: {exc}", file=sys.stderr)
        return 2

    envelope = check_envelope(cfg)
    mass = mass_budget(cfg)

    if args.json:
        print(json.dumps({"envelope": envelope.to_dict(), "mass": mass.to_dict()}, indent=2))
    else:
        print("Stowed envelope (competition limit %.2f x %.2f x %.2f m):"
              % (envelope.limit.length_x, envelope.limit.width_y, envelope.limit.height_z))
        print(envelope.table())
        print()
        print("Mass budget (competition limit %.1f kg):" % mass.mass_limit)
        print(mass.table())

    return 0 if (envelope.ok and mass.ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
