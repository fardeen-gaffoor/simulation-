#!/usr/bin/env python3
"""``ros2 run rover_excavation excavation_info.py`` - what module 6 will actually do.

Prints the resolved ``excavation.yaml``, the mechanism and sand model derived
from it, and then **dry-runs the work cycles offline**: it builds the real
:class:`~rover_excavation.mechanism.Mechanism`,
:class:`~rover_excavation.sand.Tub` and
:class:`~rover_excavation.sand.BermField` that the node uses, plays
``--cycles`` dig/dump pairs through them with no ROS and no Gazebo, and reports
how long each cycle took and what the berm came out as.

That is the tool to run before a competition run: it answers "if the sequencer
asks for six cycles, what will the berm be" and "does my dump land inside the
target" without starting a simulator.

    ros2 run rover_excavation excavation_info.py
    ros2 run rover_excavation excavation_info.py --cycles 9
    ros2 run rover_excavation excavation_info.py --json
    ros2 run rover_excavation excavation_info.py --deps
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from rover_excavation.config import ExcavationConfig, load_config
from rover_excavation.cycle import WorkCycle
from rover_excavation.mechanism import Mechanism
from rover_excavation.sand import BermField, Tub

__all__ = ["main", "report", "dry_run", "dependency_report"]

#: Integration points module 6 does not own, and what happens without them.
_OPTIONAL = {
    "serial": "the real digger's serial link (backend.kind: hardware needs it)",
    "rover_navigation": "not a dependency: module 6 only speaks rover_navigation_msgs",
    "rclpy": "needed by the node, not by this tool",
}

#: Dig poses are module 5's business, so the dry run only needs *a* legal pose in
#: the excavation zone: a dig cycle deposits nothing, it only fills the tub.
_DIG_SAMPLE_POSE = (8.0, 2.5, 1.570796)


def _dump_poses_x(config: ExcavationConfig) -> List[float]:
    """Module 5's dump x positions along the berm target, from the copy here."""
    target = config.berm_target
    inset = config.nav_dump.inset_m
    lo = target.x0 + inset
    hi = target.x1 - inset
    points = max(1, config.nav_dump.points)
    if points == 1:
        return [0.5 * (lo + hi)]
    return [lo + (hi - lo) * index / (points - 1) for index in range(points)]


def dry_run(config: ExcavationConfig, cycles: int) -> Dict[str, Any]:
    """Play ``cycles`` dig/dump pairs through the real models, offline."""
    mechanism = Mechanism(config.mechanism)
    tub = Tub(config.mechanism.tub)
    berm = BermField(config)
    dump_y = config.berm_dump_pose_y
    yaw = config.nav_dump.heading_rad
    xs = _dump_poses_x(config)

    dt = 0.01
    now = 0.0
    cycle_reports: List[Dict[str, Any]] = []

    def play(phase: str, pose: tuple, index: int) -> WorkCycle:
        nonlocal now
        cycle = WorkCycle(config, phase, pose, mechanism, tub, berm, cycle_id=index, now=now)
        started = now
        while not cycle.done and now - started < 600.0:
            mechanism.step(dt)
            cycle.step(now)
            now += dt
        return cycle

    for index in range(max(0, cycles)):
        dig = play("excavate", _DIG_SAMPLE_POSE, index)
        tub_after_dig = tub.fraction
        dump_x = xs[index % len(xs)]
        dump = play("dump", (dump_x, dump_y, yaw), index)
        cycle_reports.append(
            {
                "cycle": index,
                "dig": {
                    "success": dig.success,
                    "elapsed_s": round(dig.elapsed_s, 2),
                    "tub_fraction": round(tub_after_dig, 4),
                    "amount_m3": round(dig.amount_moved, 5),
                    "message": dig.message,
                },
                "dump": {
                    "success": dump.success,
                    "elapsed_s": round(dump.elapsed_s, 2),
                    "x": round(dump_x, 4),
                    "amount_m3": round(dump.amount_moved, 5),
                    "message": dump.message,
                },
                "berm_completion": round(berm.completion, 4),
            }
        )

    # The last dump leaves the ladder on its way up; the node's very next idle
    # tick would report STOWED, so the dry run has to as well or it would claim
    # the rover is not ready to drive when it is.
    mechanism.settle_idle()

    return {
        "cycles": cycle_reports,
        "total_s": round(now, 1),
        "berm": {
            "target_volume_m3": round(berm.target_volume_m3, 5),
            "placed_volume_m3": round(berm.placed_volume_m3, 5),
            "spilled_volume_m3": round(berm.spilled_volume_m3, 5),
            "spill_fraction": round(
                berm.spilled_volume_m3 / max(1e-9, berm.placed_volume_m3 + berm.spilled_volume_m3), 4
            ),
            "completion": round(berm.completion, 4),
            "coverage": round(berm.coverage, 4),
            "mean_height_m": round(berm.mean_height_m, 4),
            "peak_height_m": round(berm.peak_height_m, 4),
            "dumps": berm.dumps,
            "cells_total": berm.cells_total,
            "cells_filled": berm.cells_filled,
        },
        "tub_fraction_end": round(tub.fraction, 4),
        "mechanism": mechanism.snapshot().__dict__,
    }


def _deps() -> Dict[str, Any]:
    import importlib

    required = {}
    for name in ("numpy", "yaml", "rover_excavation_msgs", "rover_navigation_msgs"):
        try:
            importlib.import_module(name)
            required[name] = True
        except Exception:
            required[name] = False
    optional = {}
    for name in _OPTIONAL:
        try:
            importlib.import_module(name)
            optional[name] = True
        except Exception:
            optional[name] = False
    return {"required": required, "optional": optional}


def dependency_report() -> List[str]:
    deps = _deps()
    lines = ["dependencies:"]
    for name, present in deps["required"].items():
        lines.append(f"  {'ok      ' if present else 'MISSING '} {name}")
    lines.append("optional:")
    for name, present in deps["optional"].items():
        lines.append(f"  {'present ' if present else 'absent  '} {name}  - {_OPTIONAL[name]}")
    return lines


def report(config: ExcavationConfig, cycles: int, with_dry_run: bool = True) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "config": {
            "path": config.path,
            "sha256": config.sha256,
            "schema_version": config.schema_version,
        },
        "frames": {"map": config.frames.map, "base": config.frames.base},
        "backend": {
            "kind": config.backend.kind,
            "ladder_cmd_topic": config.backend.ladder_cmd_topic,
            "belt_cmd_topic": config.backend.belt_cmd_topic,
            "joint_states_topic": config.backend.joint_states_topic,
        },
        "mechanism": {
            "rate_hz": config.mechanism.rate_hz,
            "ladder": {
                "joint": config.mechanism.ladder.joint,
                "dig_pitch_rad": config.mechanism.ladder.dig_pitch_rad,
                "stowed_pitch_rad": config.mechanism.ladder.stowed_pitch_rad,
                "min_dig_pitch_rad": round(config.mechanism.ladder.min_dig_pitch_rad, 4),
                "dig_depth_m": round(config.mechanism.ladder.tip_depth_m(config.mechanism.ladder.dig_pitch_rad), 4),
                "engaged_width_m": round(config.mechanism.ladder.engaged_width_m, 4),
                "max_rate_radps": config.mechanism.ladder.max_rate_radps,
            },
            "belt": {
                "joint": config.mechanism.belt.joint,
                "dig_velocity_radps": config.mechanism.belt.dig_velocity_radps,
                "dump_velocity_radps": config.mechanism.belt.dump_velocity_radps,
                "dig_surface_mps": round(
                    config.mechanism.belt.surface_speed_mps(config.mechanism.belt.dig_velocity_radps), 4
                ),
            },
            "tub": {
                "box_m": config.mechanism.tub.box_m,
                "capacity_m3": config.mechanism.tub.capacity_m3,
                "capacity_l": round(config.mechanism.tub.capacity_m3 * 1000.0, 2),
            },
            "interlocks": {
                "travel_pitch_max_rad": config.mechanism.interlocks.travel_pitch_max_rad,
                "max_depth_m": config.mechanism.interlocks.max_depth_m,
            },
        },
        "sand": {
            "dig_rate_m3_s": round(
                _dig_rate(config), 5
            ),
            "dig_seconds_to_full_tub": round(
                config.mechanism.tub.capacity_m3
                * config.cycle.dig.target_fill
                / max(1e-9, _dig_rate(config)),
                2,
            ),
            "dump_rate_m3_s": config.sand.dump.rate_m3_s,
            "repose_deg": config.sand.dump.repose_deg,
            "pile_basal_radius_m": round(config.pile_basal_radius_m, 4),
            "landing_aft_m": round(config.landing_aft_m, 4),
            "spread_m": config.sand.dump.spread_m,
        },
        "berm": {
            "target_rect": config.berm_target.as_tuple(),
            "target_area_m2": round(config.berm_target.area, 4),
            "target_height_m": config.sand.berm.target_height_m,
            "target_volume_m3": round(config.berm_target_volume_m3, 5),
            "cell_m": config.sand.berm.cell_m,
            "dump_pose_y": round(config.berm_dump_pose_y, 4),
            "dump_poses_x": [round(x, 4) for x in _dump_poses_x(config)],
            "dump_lands_inside": dump_lands_inside(config),
        },
        "cycles_target_hint": cycles,
    }
    if with_dry_run:
        data["dry_run"] = dry_run(config, cycles)
    return data


def dump_lands_inside(config: ExcavationConfig) -> Dict[str, Any]:
    """Does a tub emptied from module 5's dump poses land inside the target?"""
    yaw = config.nav_dump.heading_rad
    y = config.berm_dump_pose_y
    results = []
    for x in _dump_poses_x(config):
        lx, ly = config.landing_point(x, y, yaw)
        inside = config.berm_target.contains(lx, ly)
        results.append({"x": round(x, 4), "landing_x": round(lx, 4), "landing_y": round(ly, 4), "inside": inside})
    return {"all_inside": all(record["inside"] for record in results), "poses": results}


def _dig_rate(config: ExcavationConfig) -> float:
    from rover_excavation.config import dig_rate_m3_s

    return dig_rate_m3_s(
        config.sand.dig,
        config.mechanism,
        config.mechanism.ladder.dig_pitch_rad,
        config.mechanism.belt.dig_velocity_radps,
    )


def _print_text(data: Dict[str, Any], deps: bool, quiet: bool) -> None:
    if not quiet:
        config = data["config"]
        print(f"excavation.yaml: {config['path']}")
        print(f"  sha256 {config['sha256'][:16]}  schema {config['schema_version']}")
        frames = data["frames"]
        print(f"frames: map='{frames['map']}' base='{frames['base']}'")
        backend = data["backend"]
        print(
            f"backend: {backend['kind']}  ladder <- {backend['ladder_cmd_topic']}  "
            f"belt <- {backend['belt_cmd_topic']}  joints <- {backend['joint_states_topic']}"
        )
        mechanism = data["mechanism"]
        ladder = mechanism["ladder"]
        belt = mechanism["belt"]
        tub = mechanism["tub"]
        print(
            f"ladder: {ladder['joint']}, dig {ladder['dig_pitch_rad']:.3f} rad -> "
            f"{ladder['dig_depth_m']:g} m deep, stowed {ladder['stowed_pitch_rad']:.3f} rad, "
            f"cut width {ladder['engaged_width_m']:g} m, slew {ladder['max_rate_radps']:g} rad/s"
        )
        print(
            f"belt: {belt['joint']}, dig {belt['dig_velocity_radps']:g} rad/s "
            f"({belt['dig_surface_mps']:g} m/s), dump {belt['dump_velocity_radps']:g} rad/s"
        )
        print(f"tub: {tub['box_m']} m -> {tub['capacity_l']:g} L usable")
        print(f"mechanism: control loop {mechanism['rate_hz']:g} Hz")
        sand = data["sand"]
        print(
            f"sand: dig {sand['dig_rate_m3_s'] * 1000.0:g} L/s (tub in "
            f"{sand['dig_seconds_to_full_tub']:g} s), dump {sand['dump_rate_m3_s'] * 1000.0:g} L/s, "
            f"repose {sand['repose_deg']:g} deg"
        )
        berm = data["berm"]
        print(
            f"berm: target {berm['target_rect']} ({berm['target_area_m2']:g} m^2 x "
            f"{berm['target_height_m']:g} m = {berm['target_volume_m3'] * 1000.0:g} L), "
            f"cell {berm['cell_m']:g} m"
        )
        print(
            f"dump: poses y={berm['dump_pose_y']:g} x={berm['dump_poses_x']}, pile lands "
            f"{sand['landing_aft_m']:g} m behind base_link, all inside={berm['dump_lands_inside']['all_inside']}"
        )
    dry = data.get("dry_run")
    if dry:
        print(f"dry run: {len(dry['cycles'])} cycles in {dry['total_s']:g} s")
        for cycle in dry["cycles"]:
            dig = cycle["dig"]
            dump = cycle["dump"]
            print(
                f"  cycle {cycle['cycle']}: dig {dig['elapsed_s']:5.1f}s "
                f"({dig['amount_m3'] * 1000:5.2f} L) -> dump at x={dump['x']:.3f} "
                f"{dump['elapsed_s']:5.1f}s ({dump['amount_m3'] * 1000:5.2f} L) "
                f"completion {100 * cycle['berm_completion']:5.1f}%"
            )
        berm = dry["berm"]
        print(
            f"  berm: {berm['placed_volume_m3'] * 1000:.2f}/{berm['target_volume_m3'] * 1000:.2f} L "
            f"({100 * berm['completion']:.0f}%), coverage {100 * berm['coverage']:.0f}%, "
            f"mean height {berm['mean_height_m']:.3f} m, peak {berm['peak_height_m']:.3f} m, "
            f"spilled {berm['spilled_volume_m3'] * 1000:.2f} L ({100 * berm['spill_fraction']:.0f}%)"
        )
    if deps:
        for line in dependency_report():
            print(line)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog="excavation_info.py",
        description="report what module 6 is configured to do, and dry-run the work cycles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="excavation.yaml (empty = installed default)")
    parser.add_argument("--cycles", type=int, default=6, help="work cycles to dry-run")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--deps", action="store_true", help="also print the dependency report")
    parser.add_argument("--no-dry-run", action="store_true", help="skip the offline cycle simulation")
    parser.add_argument("--quiet", action="store_true", help="only print the dry run (and deps)")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        config = load_config(args.config or None)
    except Exception as exc:
        print(f"excavation_info.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    data = report(config, max(0, args.cycles), with_dry_run=not args.no_dry_run)
    if args.json:
        payload: Dict[str, Any] = dict(data)
        if args.deps:
            payload["dependencies"] = _deps()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    _print_text(data, deps=args.deps, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
