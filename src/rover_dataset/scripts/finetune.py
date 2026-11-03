#!/usr/bin/env python3
"""Fine-tune the boulder/crater detector (YOLOv8n) on the built dataset.

    ros2 run rover_dataset finetune.py                 # print the plan
    ros2 run rover_dataset finetune.py --run           # actually train (needs ultralytics)
    ros2 run rover_dataset finetune.py --json

The plan is derived from ``finetune`` in ``dataset.yaml`` and audited against the
split: it warns when there are no real photos, when the real-photo fraction
target was not reached, or when the validation split is empty.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from rover_dataset.config import load_config
from rover_dataset.finetune import build_training_plan, run_training, ultralytics_available


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finetune",
        description="build (and optionally run) the YOLOv8n fine-tuning recipe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="dataset.yaml (empty = installed default)")
    parser.add_argument("--out-dir", default="", help="dataset root (overrides paths.dataset_root)")
    parser.add_argument("--dataset-yaml", default="", help="override data.yaml path")
    parser.add_argument("--run", action="store_true", help="run the training (requires ultralytics)")
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config or None)
    root = os.path.abspath(os.path.expanduser(args.out_dir or cfg.paths.dataset_root))
    dataset_yaml = args.dataset_yaml or os.path.join(root, cfg.paths.data_yaml)

    try:
        plan = build_training_plan(cfg, dataset_yaml)
    except FileNotFoundError as exc:
        print(f"[finetune] {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(plan.as_dict(), indent=1, sort_keys=True))
    else:
        available, detail = ultralytics_available()
        print(f"[finetune] dataset     : {plan.dataset_yaml}")
        print(f"[finetune] ultralytics : {'yes' if available else 'NO'} ({detail})")
        for warning in plan.warnings:
            print(f"[finetune] WARNING: {warning}")
        print("[finetune] command:")
        print("  " + " ".join(plan.command))
        print("[finetune] python:")
        for line in plan.snippet.rstrip("\n").splitlines():
            print("  " + line)

    if not args.run:
        return 0
    return run_training(cfg, dataset_yaml)


if __name__ == "__main__":
    raise SystemExit(main())
