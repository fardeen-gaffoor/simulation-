#!/usr/bin/env python3
"""Draw a contact sheet of labelled frames so a human can eyeball the dataset.

    ros2 run rover_dataset preview_labels.py --out-dir dataset
    ros2 run rover_dataset preview_labels.py --out-dir dataset --count 12 --columns 4
    ros2 run rover_dataset preview_labels.py --out-dir dataset --split val

Open the written PNG (default ``<root>/preview/grid.png``).  Green boxes are
class 0 (boulder), magenta class 1 (crater).  If a box is not sitting on its
object, the projection is wrong and no amount of training will fix it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from rover_dataset.config import load_config
from rover_dataset.preview import build_contact_sheet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preview_labels",
        description="contact sheet of labelled dataset frames",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="dataset.yaml (empty = installed default)")
    parser.add_argument("--out-dir", default="", help="dataset root (overrides paths.dataset_root)")
    parser.add_argument("--out", default="", help="output PNG (default <root>/preview/grid.png)")
    parser.add_argument("--count", type=int, default=9, help="how many frames to draw")
    parser.add_argument("--columns", type=int, default=3, help="grid columns")
    parser.add_argument("--cell", type=int, default=320, help="cell size in pixels")
    parser.add_argument("--split", default="", help="train, val or empty for every split present")
    parser.add_argument("--json", action="store_true", help="also print the summary as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config or None)
    root = os.path.abspath(os.path.expanduser(args.out_dir or cfg.paths.dataset_root))
    splits = [args.split] if args.split else None

    try:
        result = build_contact_sheet(
            root,
            count=args.count,
            columns=args.columns,
            cell_px=args.cell,
            output=args.out or None,
            classes=cfg.class_names,
            splits=splits,
        )
    except FileNotFoundError as exc:
        print(f"[preview] {exc}", file=sys.stderr)
        return 1

    print(f"[preview] {result.cells} frame(s) from {root}")
    print(
        f"[preview] {result.boxes} box(es) {result.per_class}, "
        f"{result.empty_images} background-only, per split {result.per_split}"
    )
    if result.missing_labels:
        print(
            f"[preview] WARNING: {len(result.missing_labels)} image(s) have no label file, "
            f"e.g. {result.missing_labels[0]}",
            file=sys.stderr,
        )
    print(f"[preview] wrote {result.output}")
    if args.json:
        print(json.dumps(result.as_dict(), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
