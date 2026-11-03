#!/usr/bin/env python3
"""Build the train/val split from the staged dataset.

    ros2 run rover_dataset split_dataset.py
    ros2 run rover_dataset split_dataset.py --copy     # self-contained copy, no symlinks

The split is BY SCENE (never by frame) and deterministic from ``split.seed``.
Real-photo batches are oversampled here so that they carry at least
``sampling.target_real_fraction`` of the effective training set.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

from rover_dataset.config import load_config
from rover_dataset.dataset import DatasetLayout, verify_dataset
from rover_dataset.registry import all_scenes
from rover_dataset.split import build_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="split_dataset",
        description="materialise images/{train,val} from the raw staging area",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="dataset.yaml (empty = installed default)")
    parser.add_argument("--out-dir", default="", help="dataset root (overrides paths.dataset_root)")
    parser.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    parser.add_argument("--no-verify", action="store_true", help="skip the integrity check")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config or None)
    root = os.path.abspath(os.path.expanduser(args.out_dir or cfg.paths.dataset_root))

    scenes = all_scenes(root)
    if not scenes:
        print("[split] no scenes found; run collect_dataset.py or import_real_photos.py first")
        return 1

    result = build_split(cfg, root, scenes, use_links=not args.copy)
    sim = sum(1 for s in scenes if s.get("source") == "sim")
    real = sum(1 for s in scenes if s.get("source") == "real")
    print(f"[split] {len(scenes)} scene(s): {sim} sim, {real} real")
    print(f"[split] train={result.written.get('train', 0)}  val={result.written.get('val', 0)}")
    print(
        f"[split] real-photo fraction: before {result.sampling.get('real_fraction_before', 0.0):.2f} "
        f"-> after {result.sampling.get('real_fraction_after', 0.0):.2f} "
        f"(target {result.sampling.get('target_real_fraction', 0.0):.2f}, "
        f"real_repeat={result.sampling.get('real_repeat', 1)})"
    )
    if not result.sampling.get("reachable", True):
        print(
            "[split] WARNING: the real-photo target cannot be reached with the current photos "
            "(see sampling.max_repeat); import more real photographs",
        )

    if args.no_verify:
        return 0
    layout = DatasetLayout(
        root=root,
        image_format=cfg.paths.image_format,
        classes_file=cfg.paths.classes_file,
        data_yaml=cfg.paths.data_yaml,
    )
    report = verify_dataset(layout, cfg.class_names, expected_scene_split=result.scene_split)
    stats = report.stats["splits"]
    for split in ("train", "val"):
        entry = stats[split]
        print(
            f"[split] {split}: {entry['images']} images, {entry['boxes']} boxes, "
            f"{entry['empty_images']} background-only, {entry['per_class']}"
        )
    if not report.ok:
        for problem in report.problems[:20]:
            print(f"[split] PROBLEM {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
