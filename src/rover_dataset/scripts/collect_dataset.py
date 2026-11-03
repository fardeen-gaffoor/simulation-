#!/usr/bin/env python3
"""Collect the whole simulated dataset, one randomized scene at a time.

    ros2 run rover_dataset collect_dataset.py --scenes 2 --poses 12 --headless

For each scene this script:

1. writes a randomized ``arena.yaml`` + ``rover.yaml`` (sand colour, lighting,
   shadows, sand ripples, camera-mount jitter) into ``<dataset>/_scenes/<id>``,
2. launches ``collect.launch.py``, which brings up one gz world plus the rover
   and the collector node,
3. waits for the collector to finish and reads its per-scene report.

It finishes by writing ``<dataset>/collection.json`` and, unless disabled,
building the scene-level train/val split.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

from rover_dataset.config import DatasetConfig, load_config
from rover_dataset.scene import prepare_scenes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_dataset",
        description="collect the simulated boulder/crater dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="dataset.yaml (empty = installed default)")
    parser.add_argument("--out-dir", default="", help="dataset root (overrides paths.dataset_root)")
    parser.add_argument("--scenes-dir", default="", help="scene staging dir (default <root>/_scenes)")
    parser.add_argument("--scenes", type=int, default=0, help="number of scenes (0 = dataset.yaml)")
    parser.add_argument("--seed", type=int, default=-1, help="master seed (-1 = scene.start_seed)")
    parser.add_argument("--poses", type=int, default=0, help="poses per scene (0 = dataset.yaml)")
    parser.add_argument("--frames", type=int, default=0, help="frames per pose (0 = dataset.yaml)")
    parser.add_argument("--cameras", default="", help="comma-separated cameras (empty = dataset.yaml)")
    parser.add_argument(
        "--scene-timeout", type=float, default=0.0, help="seconds per scene (0 = estimate)"
    )
    parser.add_argument("--headless", default="true", choices=["true", "false"])
    parser.add_argument("--headless-rendering", default="true", choices=["true", "false"])
    parser.add_argument("--spawn-delay", type=float, default=10.0)
    parser.add_argument("--launch-extra", default="", help="extra 'arg:=value' pairs, comma separated")
    parser.add_argument("--no-build-split", action="store_true", help="skip the train/val split")
    parser.add_argument("--dry-run", action="store_true", help="write the scene configs and stop")
    return parser


def resolve_config(args: argparse.Namespace) -> DatasetConfig:
    cfg = load_config(args.config or None)
    root = os.path.abspath(os.path.expanduser(args.out_dir or cfg.paths.dataset_root))
    scenes_dir = os.path.abspath(
        os.path.expanduser(args.scenes_dir or os.path.join(root, cfg.paths.scenes_dir))
    )
    return replace(cfg, paths=replace(cfg.paths, dataset_root=root, scenes_dir=scenes_dir))


def estimate_timeout(cfg: DatasetConfig, poses: int, frames: int) -> float:
    per_frame = 0.9
    per_pose = float(cfg.capture.settle_seconds) + 0.6
    body = poses * (per_pose + frames * per_frame)
    return max(300.0, 120.0 + body)


def launch_command(
    cfg: DatasetConfig,
    scene_files,
    spec,
    args: argparse.Namespace,
    timeout_note: bool = False,
) -> List[str]:
    command = [
        "ros2", "launch", "rover_dataset", "collect.launch.py",
        f"scene_dir:={scene_files.directory}",
        f"scene_id:={spec.scene_id}",
        f"scene_index:={spec.index}",
        f"seed:={spec.seed}",
        f"master_seed:={cfg.scene.start_seed if args.seed < 0 else args.seed}",
        f"dataset_root:={cfg.paths.dataset_root}",
        f"arena_config:={scene_files.arena_config}",
        f"rover_config:={scene_files.rover_config}",
        f"headless:={args.headless}",
        f"headless_rendering:={args.headless_rendering}",
        f"spawn_delay:={args.spawn_delay}",
        f"poses:={args.poses}",
        f"frames:={args.frames}",
    ]
    if args.cameras:
        command.append(f"cameras:={args.cameras}")
    if args.launch_extra:
        for pair in args.launch_extra.split(","):
            pair = pair.strip()
            if pair:
                command.append(pair)
    return command


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = resolve_config(args)

    count = args.scenes or cfg.scene.count
    master_seed = cfg.scene.start_seed if args.seed < 0 else args.seed
    os.makedirs(cfg.paths.dataset_root, exist_ok=True)
    os.makedirs(cfg.paths.scenes_dir, exist_ok=True)

    print(f"[collect] dataset root : {cfg.paths.dataset_root}")
    print(f"[collect] scenes dir   : {cfg.paths.scenes_dir}")
    print(f"[collect] scenes       : {count}  (master seed {master_seed})")
    print(f"[collect] cameras      : {args.cameras or ','.join(cfg.capture.cameras)}")

    specs, files = prepare_scenes(cfg, master_seed=master_seed, count=count)
    print(f"[collect] wrote {len(specs)} scene config(s)")

    if args.dry_run:
        print("[collect] dry run: stopping before collection")
        return 0

    poses = args.poses or cfg.capture.poses_per_scene
    frames = args.frames or cfg.capture.frames_per_pose
    timeout = args.scene_timeout or estimate_timeout(cfg, poses, frames)

    records: List[Dict[str, Any]] = []
    failures = 0
    for spec, scene_files in zip(specs, files):
        report_path = os.path.join(scene_files.directory, "collect_report.json")
        command = launch_command(cfg, scene_files, spec, args)
        print(f"[collect] scene {spec.scene_id}: launching (timeout {timeout:.0f} s)")
        started = time.time()
        try:
            completed = subprocess.run(command, timeout=timeout)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = -1
            print(f"[collect] scene {spec.scene_id}: TIMED OUT")
        except KeyboardInterrupt:
            print("[collect] interrupted")
            return 130
        elapsed = time.time() - started

        record: Dict[str, Any] = {
            "scene_id": spec.scene_id,
            "index": spec.index,
            "source": spec.source,
            "seed": spec.seed,
            "ok": False,
            "launch_returncode": code,
            "seconds": round(elapsed, 1),
        }
        if os.path.isfile(report_path):
            with open(report_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
            record.update(report)
            record["ok"] = bool(report.get("ok", False)) and code == 0
        else:
            record["errors"] = ["no collect_report.json (launch failed?)"]
        if not record.get("ok"):
            failures += 1
        records.append(record)
        print(
            f"[collect] scene {spec.scene_id}: ok={record.get('ok')} "
            f"images={record.get('images_written', 0)} boxes={record.get('boxes_written', 0)} "
            f"in {elapsed:.0f} s"
        )

    collection = {
        "schema_version": 1,
        "dataset_config_sha256": cfg.config_sha256(),
        "master_seed": master_seed,
        "poses_per_scene": poses,
        "frames_per_pose": frames,
        "cameras": (args.cameras.split(",") if args.cameras else cfg.capture.cameras),
        "scenes": [
            {
                "scene_id": record["scene_id"],
                "source": record.get("source", "sim"),
                "index": record.get("index"),
                "seed": record.get("seed"),
                "ok": record.get("ok", False),
                "images": record.get("images_written", 0),
                "boxes": record.get("boxes_written", 0),
                "per_class": record.get("per_class", {}),
                "seconds": record.get("seconds", 0.0),
                "report": record.get("report", ""),
            }
            for record in records
        ],
        "failures": failures,
    }
    collection_path = os.path.join(cfg.paths.dataset_root, "collection.json")
    with open(collection_path, "w", encoding="utf-8") as handle:
        json.dump(collection, handle, indent=1, sort_keys=True)
        handle.write("\n")
    print(f"[collect] wrote {collection_path}")

    if failures:
        print(f"[collect] {failures} scene(s) failed; not building the split")
        return 1
    if args.no_build_split:
        return 0

    from rover_dataset.split import build_split

    scenes = [dict(record, images=record.get("images_written", 0)) for record in records]
    result = build_split(cfg, cfg.paths.dataset_root, scenes)
    print(
        f"[collect] split: train={result.written.get('train', 0)} "
        f"val={result.written.get('val', 0)} images"
    )
    print(
        f"[collect] real-photo fraction now {result.sampling.get('real_fraction_after', 0.0):.2f} "
        f"(target {result.sampling.get('target_real_fraction', 0.0):.2f}); "
        "import real photos to raise it"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
