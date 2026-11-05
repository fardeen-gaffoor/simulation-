"""Ground-truth export for one seeded arena.

The ground-truth file is the *oracle*: exact obstacle positions, crater
geometry, zone rectangles, start pose and marker poses.  It is used by

* the module-1 acceptance tests,
* the module-3 dataset auto-labeller,
* the module-5 batch evaluation (obstacle contacts / crater crossings).

It is **never** read by the autonomy stack.  The header of the emitted file says
so, and :func:`assert_autonomy_clean` lets tests prove that no perception or
planning package in the workspace opens it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import yaml

from .config import ArenaConfig, Rect
from .markers import marker_poses
from .obstacles import ObstacleLayout
from .start_pose import StartPose
from .terrain import Crater

__all__ = ["build_ground_truth", "write_ground_truth", "GROUND_TRUTH_HEADER"]

GROUND_TRUTH_HEADER = """\
# =============================================================================
# TEST / LABEL USE ONLY -- GROUND TRUTH
#
# This file describes hidden arena state (obstacle positions, crater geometry).
# Giving this information to the autonomy stack is FORBIDDEN by the challenge
# rules.  It exists exclusively for:
#   * unit / acceptance tests in module 1,
#   * automatic labelling and scene splits in module 3,
#   * batch scoring in module 5 (obstacle contacts, crater crossings).
# Nothing under rover_perception / rover_navigation / rover_excavation may open
# this file; a test in module 1 enforces that.
# =============================================================================
"""


def _zone_dict(cfg: ArenaConfig) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for zone in cfg.zones.values():
        out[zone.name] = {
            "rect": [round(v, 6) for v in zone.rect.as_list()],
            "center": [round(v, 6) for v in zone.rect.center],
            "size": [round(zone.rect.width, 6), round(zone.rect.height, 6)],
            "tiling": zone.tiling,
            "fill": zone.fill,
            "obstacles_allowed": zone.obstacles_allowed,
            "color": list(zone.color),
            "description": zone.description,
        }
    return out


def build_ground_truth(
    cfg: ArenaConfig,
    seed: int,
    layout: ObstacleLayout,
    mesh,  # Heightmap mock object
    terrain_stats: Dict[str, Any],
    start_pose: StartPose,
    files: Dict[str, Any],
    versions: Dict[str, str],
) -> Dict[str, Any]:
    """Assemble the full ground-truth document for one run.

    ``terrain_stats`` is the dictionary returned by
    :func:`cat_arena.terrain.write_obj_mesh`, optionally extended with the
    ``albedo`` and ``heightmap`` statistics dictionaries.
    """
    boulders = layout.boulder_dicts()
    craters = layout.crater_dicts()
    start_zone = cfg.zone("start").rect

    def _footprint(center: Sequence[float], radius: float) -> Rect:
        return Rect(center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)

    open_crossings = [
        b["id"] for b in boulders if start_zone.intersects(_footprint(b["center"], b["radius"]))
    ]
    crater_crossings = [
        c["id"] for c in craters if start_zone.intersects(_footprint(c["center"], c["radius"]))
    ]
    albedo = terrain_stats.get("diffuse") if isinstance(terrain_stats.get("diffuse"), dict) else None
    normal_map = terrain_stats.get("normal") if isinstance(terrain_stats.get("normal"), dict) else None
    heightmap = terrain_stats.get("heightmap") if isinstance(terrain_stats.get("heightmap"), dict) else None

    rules = {
        "boulder_count_ok": len(boulders) >= 3,
        "crater_count_ok": len(craters) >= 3,
        "no_boulders_in_start_zone": not open_crossings,
        "no_craters_in_start_zone": not crater_crossings,
        "boulders_in_allowed_zones": all(
            cfg.zone(b["zone"]).obstacles_allowed for b in boulders
        ),
        "craters_in_allowed_zones": all(
            cfg.zone(c["zone"]).obstacles_allowed for c in craters
        ),
        "crater_depths_negative": all(c["depth"] > 0.0 for c in craters),
        "berm_target_inside_construction": cfg.zone("construction").rect.contains_rect(
            cfg.zone("berm_target").rect
        ),
        "start_pose_inside_start_zone": all(
            start_zone.contains(cx, cy, margin=cfg.start_pose.margin - 1e-9)
            for cx, cy in start_pose.corners
        ),
        "marker_count_within_limit": len(cfg.markers.placements) <= cfg.markers.max_count,
        "marker_board_size_within_limit": cfg.markers.board_size <= 0.25 + 1e-9,
        "marker_top_within_limit": (
            cfg.markers.center_height + 0.5 * cfg.markers.board_size <= 0.60 + 1e-9
        ),
    }
    rules["all"] = all(bool(v) for v in rules.values())

    return {
        "schema_version": 1,
        "seed": seed,
        "world_name": cfg.world_name,
        "config": {"path": cfg.path, "sha256": cfg.config_sha256()},
        "frame": {
            "description": "origin = west-south arena corner at mean sand level",
            "x": "east",
            "y": "north",
            "z": "up",
            "units": "metres, radians",
        },
        "arena": {
            "length_x": cfg.length_x,
            "width_y": cfg.width_y,
            "sand_depth": cfg.sand_depth,
            "friction": {"mu": cfg.friction.mu, "mu2": cfg.friction.mu2},
            "walls": {
                "present": cfg.walls.present,
                "height": cfg.walls.height,
                "thickness": cfg.walls.thickness,
                "assumption": "wall height is not stated by the rules; flagged assumption",
            },
            "spawn_z_offset": cfg.spawn_z,
        },
        "zones": _zone_dict(cfg),
        "terrain": {
            "use_heightmap": cfg.terrain.use_heightmap,
            "heightmap_resolution": cfg.terrain.heightmap_resolution,
            "max_relief": cfg.terrain.max_relief,
            "heightmap_size": [terrain_stats.get("rows"), terrain_stats.get("cols")],
            "z_min": terrain_stats.get("z_min"),
            "z_max": terrain_stats.get("z_max"),
            "skirt_depth": round(mesh.skirt_depth, 6),
            "files": {
                "heightmap": heightmap.get("file") if heightmap else None,
                "albedo": albedo.get("file") if albedo else None,
                "normal": normal_map.get("file") if normal_map else None,
            },
            "heightmap_encoding": heightmap,
        },
        "craters": craters,
        "boulders": boulders,
        "start_pose": start_pose.to_dict(),
        "markers": {
            "family": cfg.markers.family,
            "board_size": cfg.markers.board_size,
            "tag_size": cfg.markers.tag_size,
            "center_height": cfg.markers.center_height,
            "notes": "visual-only boards; module 4 draws the AprilTag bitmaps",
            "placements": [p.to_dict() for p in marker_poses(cfg)],
        },
        "rule_compliance": rules,
        "placement_attempts": layout.attempts,
        "assumptions": cfg.assumptions_report(),
        "files": files,
        "versions": versions,
    }


def write_ground_truth(path: str, ground_truth: Dict[str, Any]) -> str:
    """Write the ground-truth document as YAML with an explicit usage warning."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(GROUND_TRUTH_HEADER)
        yaml.safe_dump(ground_truth, handle, sort_keys=False, default_flow_style=False, width=100)
    return path


def assert_autonomy_clean(package_dirs: Sequence[str], ground_truth_names: Sequence[str] = ("ground_truth.yaml", "ground_truth.json")) -> List[str]:
    """Return violations where an autonomy package references ground truth.

    Used both as a unit test (module 1) and as a pre-flight check before
    competition runs: if this ever returns anything, the run would be cheating.
    """
    import os

    violations: List[str] = []
    for root in package_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith((".py", ".cpp", ".hpp", ".yaml", ".xml", ".launch")):
                    continue
                full = os.path.join(dirpath, filename)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as handle:
                        text = handle.read()
                except OSError:
                    continue
                for token in ground_truth_names:
                    if token in text:
                        violations.append(f"{full}: references '{token}'")
    return violations
