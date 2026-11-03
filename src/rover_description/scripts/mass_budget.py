#!/usr/bin/env python3
"""Print the rover mass budget table (module 2 deliverable).

    ros2 run rover_description mass_budget
    ros2 run rover_description mass_budget -- --json
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from rover_description.config import load_config
from rover_description.mass import mass_budget


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mass_budget")
    parser.add_argument("--config", default=None, help="path to rover.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    budget = mass_budget(cfg)
    if args.json:
        print(json.dumps(budget.to_dict(), indent=2))
    else:
        print(budget.table())
    return 0 if budget.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
