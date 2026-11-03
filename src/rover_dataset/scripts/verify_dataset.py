#!/usr/bin/env python3
"""Verify a built dataset and print a report.

    ros2 run rover_dataset verify_dataset.py
    ros2 run rover_dataset verify_dataset.py --duplicates --json

Checks: every image has a label, every label is a legal YOLO box, classes match
``classes.txt``, no scene leaks across train/val, and (optionally) no identical
image appears in both splits.  Exits non-zero if anything is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from rover_dataset.config import load_config
from rover_dataset.dataset import DatasetLayout, verify_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_dataset",
        description="check dataset integrity and report statistics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="dataset.yaml (empty = installed default)")
    parser.add_argument("--out-dir", default="", help="dataset root (overrides paths.dataset_root)")
    parser.add_argument("--duplicates", action="store_true", help="hash every image (slower)")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config or None)
    root = os.path.abspath(os.path.expanduser(args.out_dir or cfg.paths.dataset_root))
    layout = DatasetLayout(
        root=root,
        image_format=cfg.paths.image_format,
        classes_file=cfg.paths.classes_file,
        data_yaml=cfg.paths.data_yaml,
    )
    report = verify_dataset(layout, cfg.class_names, check_duplicates=args.duplicates)
    if args.json:
        print(json.dumps(report.as_dict(), indent=1, sort_keys=True))
    else:
        stats = report.stats.get("splits", {})
        for split in ("train", "val"):
            entry = stats.get(split, {})
            print(
                f"{split:5s}: {entry.get('images', 0):5d} images  "
                f"{entry.get('boxes', 0):6d} boxes  "
                f"{entry.get('empty_images', 0):5d} background-only  "
                f"{entry.get('per_class', {})}"
            )
        print(f"scenes: {len(report.scene_split)}")
        print(f"image sizes: {', '.join(report.stats.get('image_sizes', []))}")
        for problem in report.problems:
            print(f"PROBLEM {problem}")
        print("OK" if report.ok else f"FAILED ({len(report.problems)} problem(s))")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
