"""``cat_arena`` world generator.

One call produces a complete, self-describing arena for a given seed:

    arena.sdf            Gazebo world (terrain mesh, walls, zone markings, markers, boulders)
    terrain.obj / .mtl   triangle terrain with real crater pits (visual + collision)
    sand_albedo.png      procedural dry-sand texture
    terrain_height.png   heightfield artifact for analysis/labels (never loaded by gz)
    ground_truth.yaml    oracle data - TESTS AND LABELS ONLY
    markers.yaml         marker board poses + apriltag_ros handoff block for module 4
    start_pose.yaml      the sampled random spawn pose
    manifest.json        seed, file hashes, versions, assumptions

Determinism: for a fixed seed and configuration every generated file except
``manifest.json``'s timestamp is byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from .config import ArenaConfig, load_config
from .ground_truth import build_ground_truth, write_ground_truth
from .markers import markers_yaml_document
from .obstacles import ObstacleLayout, place_obstacles
from .seeding import resolve_master_seed, rng_numpy
from .sdf import WorldAssets, build_world_sdf
from .start_pose import StartPose, sample_start_pose
from .terrain import (
    Crater,
    TerrainField,
    build_mesh,
    write_heightmap_png,
    write_obj_mesh,
    write_sand_albedo,
)

__all__ = ["GenerationResult", "GenerationError", "generate", "main"]


class GenerationError(RuntimeError):
    """Raised when the arena could not be generated or validated."""


@dataclass
class GenerationResult:
    """Everything a caller (launch file, test, batch runner) needs to know."""

    seed: int
    master_seed: int
    directory: str
    world_path: str
    ground_truth_path: str
    markers_path: str
    start_pose_path: str
    manifest_path: str
    config: ArenaConfig
    layout: ObstacleLayout
    start_pose: StartPose
    terrain_stats: Dict[str, Any] = field(default_factory=dict)
    files: Dict[str, Any] = field(default_factory=dict)
    validation_output: str = ""

    @property
    def terrain_mesh(self) -> str:
        return os.path.join(self.directory, str(self.terrain_stats.get("obj", "terrain.obj")))

    def spawn_arguments(self) -> Tuple[float, float, float, float]:
        """``(x, y, z, yaw)`` for ``ros_gz_sim create``."""
        pose = self.start_pose
        return (pose.x, pose.y, pose.z, pose.yaw)

    def summary(self) -> str:
        layout = self.layout
        return (
            f"seed {self.seed} (master {self.master_seed}) -> {self.directory}\n"
            f"  world      : {self.world_path}\n"
            f"  boulders   : {len(layout.boulders)} "
            f"({', '.join(sorted({b.zone for b in layout.boulders})) or '-'})\n"
            f"  craters    : {len(layout.craters)} "
            f"({', '.join(sorted({c.zone for c in layout.craters})) or '-'})\n"
            f"  start pose : ({self.start_pose.x:.3f}, {self.start_pose.y:.3f}) m, "
            f"yaw {self.start_pose.yaw_deg:.1f} deg\n"
            f"  mesh       : {self.terrain_stats.get('triangles')} triangles, "
            f"{self.terrain_stats.get('cell_size')} m cells\n"
            f"  validate   : {self.validation_output or 'skipped'}"
        )


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gz_sim_version() -> str:
    exe = shutil.which("gz")
    if not exe:
        return "unavailable"
    try:
        proc = subprocess.run(
            [exe, "sim", "--version"], capture_output=True, text=True, timeout=20, check=False
        )
        text = (proc.stdout or proc.stderr).strip().splitlines()
        return text[0].strip() if text else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _validate_sdf(path: str, timeout: float = 60.0) -> str:
    """Run ``gz sdf -k`` on the generated world; raise on failure.

    Two things count as failure, not just a non-zero exit code:

    * ``error`` in the output, and
    * ``not defined in SDF`` - an element Gazebo does not know about.  That is
      how a typo'd element or a version mismatch shows up, and it is silent at
      run time, so it must not slip through the generator's own validation.
    """
    exe = shutil.which("gz")
    if not exe:
        return "skipped: gz CLI not found"
    try:
        proc = subprocess.run(
            [exe, "sdf", "-k", path], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GenerationError(f"failed to run 'gz sdf -k {path}': {exc}") from exc
    combined = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode != 0:
        raise GenerationError(f"'gz sdf -k {path}' failed (rc={proc.returncode}):\n{combined}")
    lowered = combined.lower()
    if "error" in lowered or "not defined in sdf" in lowered:
        raise GenerationError(f"'gz sdf -k {path}' reported problems:\n{combined}")
    return "gz sdf -k: OK"


def generate(
    seed: Optional[int] = None,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    overview_camera: Optional[bool] = None,
    robot_plugins: bool = False,
    validate: bool = True,
    verbose: bool = True,
    include_timestamp: bool = True,
) -> GenerationResult:
    """Generate one complete arena and return the result.

    :param seed: run seed; ``None`` or negative draws a random seed.
    :param config_path: ``arena.yaml`` override.
    :param output_dir: parent directory; output lands in ``<output_dir>/seed_<N>``.
    :param overview_camera: force the verification camera on/off.
    :param robot_plugins: add the gz camera (sensors) and IMU systems to the
        world so a spawned robot's camera/IMU sensors work (module 2 sets this).
    :param validate: run ``gz sdf -k`` on the world before returning.
    :param verbose: print a one-line progress trace to stdout.
    :param include_timestamp: include ``created_utc`` in the manifest.
    """
    cfg = load_config(config_path)
    master = resolve_master_seed(seed)
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "worlds")
    target = os.path.join(os.path.abspath(os.path.expanduser(output_dir)), f"seed_{master}")
    os.makedirs(target, exist_ok=True)

    # -- 1. obstacles ------------------------------------------------------
    # Boulders are seated on the *ripple-only* surface: boulder/crater spacing is
    # enforced, so no boulder can sit inside a crater.
    ripple_field = TerrainField(cfg, [], phases=rng_numpy(master, "terrain_phases").random(3))
    layout = place_obstacles(cfg, master, ripple_field)
    if verbose:
        print(f"[cat_arena] seed {master}: {len(layout.boulders)} boulders, "
              f"{len(layout.craters)} craters")

    # -- 2. terrain (craters carved into the mesh) -------------------------
    field = TerrainField(cfg, layout.craters, phases=rng_numpy(master, "terrain_phases").random(3))
    mesh = build_mesh(cfg, field)
    obj_path = os.path.join(target, "terrain.obj")
    terrain_stats: Dict[str, Any] = write_obj_mesh(obj_path, cfg, mesh)

    albedo_file: Optional[str] = None
    if cfg.terrain.albedo_texture:
        albedo_path = os.path.join(target, "sand_albedo.png")
        terrain_stats["albedo"] = write_sand_albedo(albedo_path, cfg, rng_numpy(master, "albedo"))
        albedo_file = os.path.basename(albedo_path)

    heightmap_file: Optional[str] = None
    if cfg.terrain.heightmap_png:
        z_min = mesh.z_min - 0.01
        z_max = max(mesh.z_max, 0.0) + 0.01
        hm_path = os.path.join(target, "terrain_height.png")
        terrain_stats["heightmap"] = write_heightmap_png(hm_path, mesh, z_min, z_max)
        heightmap_file = os.path.basename(hm_path)

    assets = WorldAssets(
        obj_file=os.path.basename(obj_path),
        mtl_file=str(terrain_stats["mtl"]),
        albedo_file=albedo_file,
        heightmap_file=heightmap_file,
    )

    # -- 3. spawn pose -----------------------------------------------------
    start_pose = sample_start_pose(cfg, master, surface=field.height)

    # -- 4. world ----------------------------------------------------------
    world_text = build_world_sdf(
        cfg=cfg,
        assets=assets,
        output_dir=target,
        boulders=layout.boulders,
        craters=layout.craters,
        seed=master,
        include_overview_camera=overview_camera,
        include_robot_plugins=robot_plugins,
    )
    world_path = os.path.join(target, "arena.sdf")
    with open(world_path, "w", encoding="utf-8") as handle:
        handle.write(world_text)

    # -- 5. sidecar documents ---------------------------------------------
    versions = {
        "cat_arena": _package_version(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": f"{platform.system()} {platform.machine()}",
        "gz_sim": _gz_sim_version(),
    }
    files: Dict[str, Any] = {}
    for name in ("arena.sdf", "terrain.obj", "terrain.mtl", "sand_albedo.png", "terrain_height.png"):
        candidate = os.path.join(target, name)
        if os.path.isfile(candidate):
            files[name] = {"bytes": os.path.getsize(candidate), "sha256": _sha256(candidate)}

    ground_truth = build_ground_truth(
        cfg=cfg,
        seed=master,
        layout=layout,
        mesh=mesh,
        terrain_stats=terrain_stats,
        start_pose=start_pose,
        files=files,
        versions=versions,
    )
    ground_truth_path = os.path.join(target, "ground_truth.yaml")
    write_ground_truth(ground_truth_path, ground_truth)

    markers_path = os.path.join(target, "markers.yaml")
    with open(markers_path, "w", encoding="utf-8") as handle:
        handle.write("# cat_arena marker boards - module 4 consumes the apriltag_ros block\n")
        yaml.safe_dump(markers_yaml_document(cfg, master), handle, sort_keys=False, default_flow_style=False)

    start_pose_path = os.path.join(target, "start_pose.yaml")
    with open(start_pose_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {"schema_version": 1, "seed": master, "frame": "arena/map", **start_pose.to_dict()},
            handle,
            sort_keys=False,
            default_flow_style=False,
        )

    validation = ""
    if validate:
        validation = _validate_sdf(world_path)

    manifest = {
        "schema_version": 1,
        "tool": "cat_arena.generate",
        "seed": master,
        "seed_requested": seed,
        "world_name": cfg.world_name,
        "config": {"path": cfg.path, "sha256": cfg.config_sha256()},
        "versions": versions,
        "assumptions": cfg.assumptions_report(),
        "outputs": files,
        "counts": {
            "boulders": len(layout.boulders),
            "craters": len(layout.craters),
            "markers": len(cfg.markers.placements),
        },
        "rule_compliance": ground_truth["rule_compliance"],
        "validation": validation,
        "reproduce": f"ros2 run cat_arena generate_arena.py --seed {master} --output-dir {os.path.dirname(target)}",
    }
    if include_timestamp:
        manifest["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path = os.path.join(target, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    if verbose:
        print(f"[cat_arena] wrote {world_path}")
        if validation:
            print(f"[cat_arena] {validation}")

    return GenerationResult(
        seed=master,
        master_seed=master,
        directory=target,
        world_path=world_path,
        ground_truth_path=ground_truth_path,
        markers_path=markers_path,
        start_pose_path=start_pose_path,
        manifest_path=manifest_path,
        config=cfg,
        layout=layout,
        start_pose=start_pose,
        terrain_stats=terrain_stats,
        files=files,
        validation_output=validation,
    )


def _package_version() -> str:
    from . import __version__

    return __version__


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_arena",
        description="Generate a seeded Caterpillar Autonomy Challenge arena (world + mesh + labels).",
    )
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="run seed (>=0); omit or -1 for a random seed")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="arena.yaml override (defaults to the installed cat_arena config)")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="parent directory for worlds (default: ./worlds)")
    parser.add_argument("--overview-camera", action="store_true", default=None,
                        help="force the top-down verification camera into the world")
    parser.add_argument("--no-overview-camera", dest="overview_camera", action="store_false",
                        help="force the verification camera off")
    parser.add_argument("--no-validate", dest="validate", action="store_false",
                        help="skip 'gz sdf -k' validation")
    parser.add_argument("--quiet", "-q", action="store_true", help="only print the output directory")
    parser.add_argument("--json", action="store_true", help="print a machine readable summary")
    parser.add_argument("--print-paths", action="store_true",
                        help="print shell assignments for the generated paths")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    try:
        result = generate(
            seed=args.seed,
            config_path=args.config,
            output_dir=args.output_dir,
            overview_camera=args.overview_camera,
            validate=args.validate,
            verbose=not args.quiet and not args.json,
        )
    except GenerationError as exc:
        print(f"[cat_arena] generation failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # configuration / placement problems
        print(f"[cat_arena] error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "seed": result.seed,
            "directory": result.directory,
            "world": result.world_path,
            "ground_truth": result.ground_truth_path,
            "markers": result.markers_path,
            "start_pose": {
                "x": result.start_pose.x,
                "y": result.start_pose.y,
                "z": result.start_pose.z,
                "yaw_rad": result.start_pose.yaw,
                "yaw_deg": result.start_pose.yaw_deg,
            },
            "boulders": len(result.layout.boulders),
            "craters": len(result.layout.craters),
            "validation": result.validation_output,
        }
        print(json.dumps(payload, indent=2))
    elif args.quiet:
        print(result.directory)
    elif args.print_paths:
        print(f"export CAT_ARENA_SEED={result.seed}")
        print(f"export CAT_ARENA_DIR={result.directory}")
        print(f"export CAT_ARENA_WORLD={result.world_path}")
        print(f"export CAT_ARENA_START_X={result.start_pose.x:.6f}")
        print(f"export CAT_ARENA_START_Y={result.start_pose.y:.6f}")
        print(f"export CAT_ARENA_START_Z={result.start_pose.z:.6f}")
        print(f"export CAT_ARENA_START_YAW={result.start_pose.yaw:.6f}")
    else:
        print(result.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
