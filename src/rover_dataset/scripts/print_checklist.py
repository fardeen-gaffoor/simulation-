#!/usr/bin/env python3
"""Print the labelling checklist (installed alongside the package).

    ros2 run rover_dataset print_checklist.py

Read it before labelling real photographs; the machine-checkable items are
enforced by ``import_real_photos.py`` and ``verify_dataset.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence


def checklist_path() -> str:
    relative = os.path.join("docs", "labeling_checklist.md")
    try:
        from ament_index_python.packages import get_package_share_directory

        share = os.path.join(get_package_share_directory("rover_dataset"), relative)
        if os.path.isfile(share):
            return share
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(here, "..", relative),
        os.path.join(here, "..", "share", "rover_dataset", relative),
    ):
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    raise FileNotFoundError("labeling_checklist.md not found")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="print_checklist", description="print the labelling checklist")
    parser.parse_args(argv)
    try:
        path = checklist_path()
    except FileNotFoundError as exc:
        print(f"[checklist] {exc}", file=sys.stderr)
        return 1
    with open(path, "r", encoding="utf-8") as handle:
        sys.stdout.write(handle.read())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
