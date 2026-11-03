#!/usr/bin/env python3
"""Write the runtime tag catalogue, the tag bitmaps and the spawnable board SDFs.

    ros2 run rover_perception generate_tags.py
    ros2 run rover_perception generate_tags.py --markers ~/cat/worlds/seed_1/markers.yaml
    ros2 run rover_perception generate_tags.py --out-dir /tmp/opencode/tags --json

Inputs (both are generated artefacts, never Python imports):

* ``cat_arena``'s ``markers.yaml`` - where the three boards physically are,
* this package's ``perception.yaml`` - the tag family, board standoff and
  bitmap resolution.

Outputs, all in ``--out-dir`` (default ``~/.config``-free
``$CAT_PERCEPTION_ARTIFACTS`` or ``~/cat/perception``):

* ``tags.yaml`` - id -> tag pose in the map frame + size: what ``tag_localizer``
  reads at runtime,
* ``tag_<id>_tag36h11.png`` - the real 36h11 bitmap for each id,
* ``tag_board_<id>.sdf`` - a thin textured-box static model ``ros_gz_sim
  create`` can spawn at the pose in ``tags.yaml`` (a ``<plane>`` does not
  texture in gz Harmonic, so a box is used),
* ``tag_boards.yaml`` - the spawn plan (poses + ready-made command lines).

Run it before ``perception.launch.py``; the launch file does it automatically
when ``generate_tags:=true`` (the default).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from rover_perception import tagboard, tags
from rover_perception.config import PerceptionConfigError, load_config
from rover_perception.markers_ref import MarkersError, load_markers
from rover_perception.nodeutil import default_artifacts_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_tags",
        description="generate tags.yaml, tag bitmaps and tag board SDFs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="perception.yaml (empty = installed default)")
    parser.add_argument("--markers", default="", help="markers.yaml (empty = newest generated world)")
    parser.add_argument("--out-dir", default="", help="output directory (empty = the default artefacts dir)")
    parser.add_argument("--tags-file", default="", help="catalogue file name (empty = perception.yaml)")
    parser.add_argument("--world", default="cat_arena", help="gz world name for the spawn commands")
    parser.add_argument("--no-textures", action="store_true", help="write tags.yaml only")
    parser.add_argument("--no-tags-file", action="store_true", help="write bitmaps/SDFs but not tags.yaml")
    parser.add_argument("--quiet", action="store_true", help="only errors")
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument("--print-paths", action="store_true", help="print just the written paths")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config or None)
    markers = load_markers(args.markers or config.artifacts.markers or None)
    out_dir = os.path.abspath(
        os.path.expanduser(args.out_dir or config.artifacts.output_dir or default_artifacts_dir())
    )
    tags_name = args.tags_file or config.artifacts.tags_file or "tags.yaml"
    os.makedirs(out_dir, exist_ok=True)

    document = tags.build_document(markers, config.tags.board, frame_id=config.arena.frame_id)
    tags_path = os.path.join(out_dir, tags_name)
    if not args.no_tags_file:
        tags.write_catalogue(tags_path, document)
    catalogue = tags.load_tags(tags_path) if not args.no_tags_file else None

    boards: List[tagboard.TagBoard] = []
    plan_path = ""
    if not args.no_textures:
        source = catalogue
        if source is None:  # --no-tags-file: build a catalogue in memory for the plan
            source = tags.load_tags(tags_path) if os.path.isfile(tags_path) else None
            if source is None:
                raise PerceptionConfigError(
                    "--no-tags-file needs an existing catalogue; run once without it first"
                )
        boards = tagboard.build_plan(source, config.tags.board, out_dir, write=True)
        plan = tagboard.plan_document(boards, out_dir, world=args.world)
        plan_path = tagboard.write_plan(os.path.join(out_dir, "tag_boards.yaml"), plan)

    report = {
        "schema_version": 1,
        "config": config.path,
        "output_dir": out_dir,
        "markers": markers.path,
        "markers_seed": markers.seed,
        "family": document["family"],
        "tags_file": "" if args.no_tags_file else tags_path,
        "plan_file": plan_path,
        "count": document["count"],
        "standoff_m": document["source"]["standoff_m"],
        "tags": document["tags"],
        "boards": [board.to_dict() for board in boards],
        "paths": [path for path in (tags_path if not args.no_tags_file else "", plan_path) if path],
    }
    report["paths"] += [board.texture_path for board in boards]
    report["paths"] += [board.sdf_path for board in boards]
    return report


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (PerceptionConfigError, MarkersError, tags.TagsError, tagboard.TagBoardError) as exc:
        print(f"generate_tags: {exc}", file=sys.stderr)
        return 1
    if args.print_paths:
        for path in report["paths"]:
            print(path)
        return 0
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=False))
        return 0
    if not args.quiet:
        print(f"config   {report['config']}")
        print(f"markers  {report['markers']} (seed {report['markers_seed']})")
        print(f"output   {report['output_dir']}")
        print(f"tags     {report['count']} x {report['family']} (standoff {report['standoff_m']:.4f} m)")
        for tag in report["tags"]:
            x, y, z = tag["position"]
            print(
                f"  id {tag['id']:>2}  {tag['mount']:<10} ({x:.3f}, {y:.3f}, {z:.3f})  "
                f"board {tag['size_m']:.3f} m  bitmap {tag['drawing_size_m']:.3f} m"
            )
        if report["tags_file"]:
            print(f"wrote    {report['tags_file']}")
        for board in report["boards"]:
            print(f"wrote    {board['texture']}")
            print(f"wrote    {board['sdf']}")
        if report["plan_file"]:
            print(f"wrote    {report['plan_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
