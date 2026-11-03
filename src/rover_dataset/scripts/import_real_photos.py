#!/usr/bin/env python3
"""Import a batch of labelled real photographs into the dataset.

    ros2 run rover_dataset import_real_photos.py \
        --photos ~/field_photos/batch1 --labels ~/field_photos/batch1/labels \
        --batch batch1 --camera cam_nav --mount-height 0.58

Each photo must have a matching YOLO ``.txt`` next to it (or in ``--labels``).
Class indices must match ``classes`` in ``dataset.yaml`` (0 = boulder,
1 = crater).  Read ``docs/labeling_checklist.md`` before labelling.

The batch becomes a scene (``real_<batch>``) and joins the scene-level split; it
is oversampled at split time so real photos carry most of the training weight.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from rover_dataset.config import load_config
from rover_dataset.realphotos import import_real_photos
from rover_dataset.registry import IMPORTS_FILE, load_imports, upsert_scene


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="import_real_photos",
        description="import labelled real rover photographs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="dataset.yaml (empty = installed default)")
    parser.add_argument("--out-dir", default="", help="dataset root (overrides paths.dataset_root)")
    parser.add_argument("--photos", required=True, help="directory of photographs")
    parser.add_argument("--labels", default="", help="directory of .txt labels (default: alongside)")
    parser.add_argument("--batch", required=True, help="batch name -> scene id 'real_<batch>'")
    parser.add_argument("--camera", required=True, help="camera the photos were taken with")
    parser.add_argument("--mount-height", type=float, default=None, help="measured camera height (m)")
    parser.add_argument("--calibration", default="", help="calibration.yaml override")
    parser.add_argument(
        "--unlabeled-as-negative",
        action="store_true",
        help="photos without a label file become background negatives instead of being skipped",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config or None)
    root = os.path.abspath(os.path.expanduser(args.out_dir or cfg.paths.dataset_root))

    result = import_real_photos(
        cfg,
        root,
        args.photos,
        args.labels or None,
        args.batch,
        args.camera,
        mount_height_m=args.mount_height,
        calibration_path=args.calibration or None,
        unlabeled_as_negative=args.unlabeled_as_negative,
    )

    for skipped in result.skipped:
        print(f"[import] skipped {skipped}")
    for problem in result.problems:
        print(f"[import] PROBLEM {problem}", file=sys.stderr)
    print(
        f"[import] scene {result.scene_id}: imported {result.imported} photo(s) "
        f"with camera {result.camera}"
    )
    if result.calibration_height_m is not None:
        print(
            f"[import] calibration camera height {result.calibration_height_m:.3f} m"
            + (f", measured {result.mount_height_m:.3f} m" if result.mount_height_m else "")
        )

    if result.imported == 0:
        print("[import] nothing was imported", file=sys.stderr)
        return 1
    if not result.ok:
        # A batch with any problem is not safe to train on: keep the staged
        # files for inspection, but keep it out of the split until it is fixed.
        print(
            f"[import] NOT registering {result.scene_id}: fix the {len(result.problems)} "
            "problem(s) above and re-run.\n"
            "[import] The photos are staged under raw/ but will not join the split.",
            file=sys.stderr,
        )
        return 1

    record = result.as_scene_record()
    upsert_scene(root, record, IMPORTS_FILE)
    imports = load_imports(root)
    print(
        f"[import] registry {IMPORTS_FILE}: {len(imports.get('scenes') or [])} real scene(s). "
        "Next: ros2 run rover_dataset split_dataset.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
