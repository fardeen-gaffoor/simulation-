#!/usr/bin/env python3
"""Render the tag bitmaps module 4 puts on the arena's marker boards.

    ros2 run rover_perception preview_tags.py                    # the arena's ids
    ros2 run rover_perception preview_tags.py --ids 0,1,2,3,4,5
    ros2 run rover_perception preview_tags.py --out /tmp/opencode/tags.png --json

This is the "look at it before you trust it" tool: it draws exactly the bitmaps
``generate_tags.py`` writes, side by side, so a wrong table, a wrong quiet zone
or a mirrored layout is visible immediately.  ``test/test_tagimages.py`` goes
further and makes the real detector read the rendered ids back.

With no ``--ids`` the ids come from the newest generated ``markers.yaml``, so
the preview shows the tags this arena will actually use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from rover_perception import tagimages
from rover_perception.config import PerceptionConfigError, load_config
from rover_perception.markers_ref import MarkersError, load_markers
from rover_perception.nodeutil import default_artifacts_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preview_tags",
        description="draw a contact sheet of AprilTag bitmaps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="perception.yaml (empty = installed default)")
    parser.add_argument("--markers", default="", help="markers.yaml (empty = newest generated world)")
    parser.add_argument("--ids", default="", help="comma separated tag ids (empty = the arena's ids)")
    parser.add_argument("--count", type=int, default=6, help="how many ids when none are given and no markers exist")
    parser.add_argument("--family", default="", help="override the tag family")
    parser.add_argument("--columns", type=int, default=3, help="contact sheet columns")
    parser.add_argument("--texture-size-px", type=int, default=0, help="bitmap size (0 = perception.yaml)")
    parser.add_argument("--out", default="", help="output PNG (empty = <artefacts>/tags_preview.png)")
    parser.add_argument("--quiet", action="store_true", help="only errors")
    parser.add_argument("--json", action="store_true", help="print a JSON report")
    parser.add_argument("--print-paths", action="store_true", help="print just the written paths")
    return parser


def _ids_from(value: str) -> List[int]:
    ids: List[int] = []
    for chunk in str(value).replace(" ", "").split(","):
        if not chunk:
            continue
        ids.append(int(chunk))
    return ids


def resolve_ids(args: argparse.Namespace) -> List[int]:
    ids = _ids_from(args.ids) if args.ids else []
    if ids:
        return ids
    try:
        markers = load_markers(args.markers or None)
    except MarkersError:
        return list(range(max(1, int(args.count))))
    return markers.ids


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = load_config(args.config or None)
    family = args.family or config.tags.family
    size_px = int(args.texture_size_px) if args.texture_size_px > 0 else config.tags.board.texture_size_px
    ids = resolve_ids(args)
    if not ids:
        raise PerceptionConfigError("no tag ids to draw (pass --ids)")
    tag_size = None
    try:
        markers = load_markers(args.markers or config.artifacts.markers or None)
        tag_size = markers.tag_size
    except MarkersError:
        tag_size = 0.18  # markers.yaml absent: draw at the competition board size
    image, rendered = tagimages.sheet_from_ids(
        ids,
        family_name=family,
        tag_size_m=float(tag_size),
        texture_size_px=size_px,
        quiet_zone_modules=config.tags.board.quiet_zone_modules,
        columns=max(1, int(args.columns)),
    )
    out_dir = os.path.abspath(
        os.path.expanduser(config.artifacts.output_dir or default_artifacts_dir())
    )
    out_path = args.out or os.path.join(out_dir, "tags_preview.png")
    out_path = os.path.abspath(os.path.expanduser(out_path))
    tagimages.write_png(out_path, image)
    return {
        "config": config.path,
        "family": family,
        "tag_size_m": float(tag_size),
        "ids": ids,
        "image": out_path,
        "pixels": list(image.shape[::-1]),
        "tags": [tag.summary() for tag in rendered],
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run(args)
    except (PerceptionConfigError, MarkersError, tagimages.TagImageError) as exc:
        print(f"preview_tags: {exc}", file=sys.stderr)
        return 1
    if args.print_paths:
        print(report["image"])
        return 0
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=False))
        return 0
    if not args.quiet:
        print(f"family  {report['family']}   ids {report['ids']}")
        print(f"image   {report['image']}  {report['pixels'][0]}x{report['pixels'][1]}")
        for tag in report["tags"]:
            print(
                f"  id {tag['id']:>3}  {tag['pixels'][0]}x{tag['pixels'][1]} px  "
                f"{tag['modules']} modules  {tag['drawing_size_m']:.4f} m across"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
