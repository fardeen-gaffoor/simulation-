#!/usr/bin/env python3
"""``ros2 run rover_navigation navigation_info.py`` - what module 5 will actually do.

Prints the resolved ``navigation.yaml``, the geometry derived from it and the
mission poses it generates, then checks each generated pose against the same
footprint test the navigator applies to a goal.  This is the tool to run before
a competition run: it answers "can the rover legally be everywhere it is about to
be told to go" without starting Gazebo.

    ros2 run rover_navigation navigation_info.py
    ros2 run rover_navigation navigation_info.py --json
    ros2 run rover_navigation navigation_info.py --deps
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from rover_navigation import footprint
from rover_navigation.config import NavigationConfig, load_config
from rover_navigation.geometry import Pose3
from rover_navigation.mission import dump_poses, dig_poses

__all__ = ["main", "report", "dependency_report"]

_OPTIONAL = {
    "robot_localization": "the wheel+IMU EKF that publishes /odometry/filtered "
    "(module 5 falls back to /odom without it)",
    "nav2_bringup": "not used: module 5 plans and controls itself",
    "apriltag_ros": "not used: module 4 uses dt-apriltags",
}


def _pose_reports(config: NavigationConfig, poses: List[Any], label: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for waypoint in poses:
        pose = Pose3.from_xy_yaw(waypoint.x, waypoint.y, waypoint.yaw)
        blocked = footprint.footprint_block(config, pose, [], [], clearance=0.0)
        out.append(
            {
                "label": f"{label}:{waypoint.label}",
                "x": round(waypoint.x, 4),
                "y": round(waypoint.y, 4),
                "yaw": round(waypoint.yaw, 4),
                "legal": blocked is None,
                "blocked_by": blocked or "",
            }
        )
    return out


def _deps() -> Dict[str, Any]:
    import importlib

    available: Dict[str, bool] = {}
    for name in ("rclpy", "nav_msgs", "geometry_msgs", "std_msgs", "rover_navigation_msgs", "rover_perception_msgs"):
        try:
            importlib.import_module(name)
            available[name] = True
        except Exception:
            available[name] = False
    optional: Dict[str, bool] = {}
    for name in _OPTIONAL:
        try:
            importlib.import_module(name)
            optional[name] = True
        except Exception:
            optional[name] = False
    return {"required": available, "optional": optional}


def dependency_report() -> List[str]:
    deps = _deps()
    lines = ["dependencies:"]
    for name, present in deps["required"].items():
        lines.append(f"  {'ok      ' if present else 'MISSING '} {name}")
    lines.append("optional:")
    for name, present in deps["optional"].items():
        lines.append(f"  {'present ' if present else 'absent  '} {name}  - {_OPTIONAL[name]}")
    return lines


def report(config: NavigationConfig, with_poses: bool = True) -> Dict[str, Any]:
    poses = []
    if with_poses:
        poses.extend(_pose_reports(config, dig_poses(config), "dig"))
        poses.extend(_pose_reports(config, dump_poses(config), "dump"))
        park = config.mission.park
        poses.extend(
            _pose_reports(
                config,
                [type("W", (), {"x": park.x, "y": park.y, "yaw": park.yaw, "label": "park"})()],
                "park",
            )
        )
    return {
        "config": {
            "path": config.path,
            "sha256": config.sha256,
            "schema_version": config.schema_version,
        },
        "frames": {
            "map": config.frames.map,
            "odom": config.frames.odom,
            "robot": config.frames.robot,
        },
        "arena": {
            "length_x": config.arena.length_x,
            "width_y": config.arena.width_y,
            "wall_margin_m": config.arena.wall_margin_m,
            "zones": {name: rect.as_tuple() for name, rect in config.arena.zones.items()},
            "free_rect": config.free_rect.as_tuple(),
            "free_area_m2": round(config.free_rect.area, 3),
        },
        "robot": {
            "operational_footprint_m": (config.robot.length_m, config.robot.width_m),
            "competition_envelope_m": (config.robot.build_length_m, config.robot.build_width_m),
            "inscribed_radius_m": round(config.robot.inscribed_radius_m, 4),
            "circumscribed_radius_m": round(config.robot.circumscribed_radius_m, 4),
            "planner_radius_m": round(config.robot.radius_m, 4),
            "obstacle_inflation_m": round(config.robot.obstacle_inflation_m, 4),
            "max_speed_mps": config.robot.max_speed_mps,
            "max_yaw_rate_radps": config.robot.max_yaw_rate_radps,
        },
        "costmap": {
            "resolution_m": config.costmap.resolution_m,
            "inflation_m": config.costmap.inflation_m,
            "soft_cost": config.costmap.soft_cost,
            "memory_s": config.costmap.obstacle_memory_s,
        },
        "planner": {
            "allow_diagonal": config.planner.allow_diagonal,
            "heuristic_weight": config.planner.heuristic_weight,
            "smoothing": config.planner.smoothing,
        },
        "controller": {
            "lookahead_m": config.controller.lookahead_m,
            "in_place_turn_rad": config.controller.in_place_turn_rad,
            "goal_timeout_s": config.controller.goal_timeout_s,
        },
        "safety": {
            "require_localized": config.safety.require_localized,
            "sigma_xy_slow_m": config.safety.sigma_xy_slow_m,
            "sigma_xy_stop_m": config.safety.sigma_xy_stop_m,
            "max_time_since_fix_s": config.safety.max_time_since_fix_s,
            "emergency_stop_m": config.safety.emergency_stop_m,
            "gate_rate_hz": config.safety.gate_rate_hz,
        },
        "mission": {
            "enabled": config.mission.enabled,
            "work_mode": config.mission.work_mode,
            "cycles_target": config.mission.cycles_target,
            "excavation_action": config.mission.excavation_action,
        },
        "topics": {
            "obstacles": config.topics.obstacles,
            "localization_state": config.topics.localization_state,
            "odometry_topics": config.topics.odometry_topics,
            "navigate_action": config.topics.navigate_action,
            "nav_cmd_vel": config.topics.nav_cmd_vel,
            "cmd_vel": config.topics.cmd_vel,
            "nav_status": config.topics.nav_status,
            "gate_status": config.topics.gate_status,
            "costmap": config.topics.costmap,
            "mission_status": config.topics.mission_status,
            "perception_mode": config.topics.perception_mode,
            "estop": config.topics.estop,
        },
        "poses": poses,
    }


def _print_text(data: Dict[str, Any], deps: bool, quiet: bool) -> None:
    if not quiet:
        config = data["config"]
        print(f"navigation.yaml: {config['path']}")
        print(f"  sha256 {config['sha256'][:16]}  schema {config['schema_version']}")
        frames = data["frames"]
        print(f"frames: map='{frames['map']}' odom='{frames['odom']}' robot='{frames['robot']}'")
        arena = data["arena"]
        print(
            f"arena: {arena['length_x']:g} x {arena['width_y']:g} m, wall margin "
            f"{arena['wall_margin_m']:g} m, free rect {arena['free_rect']} "
            f"({arena['free_area_m2']:g} m^2)"
        )
        robot = data["robot"]
        print(
            f"robot: operational {robot['operational_footprint_m']} m "
            f"(envelope limit {robot['competition_envelope_m']} m), "
            f"planner radius {robot['planner_radius_m']} m, "
            f"obstacle inflation {robot['obstacle_inflation_m']} m"
        )
        print(
            f"limits: {robot['max_speed_mps']:g} m/s, {robot['max_yaw_rate_radps']:g} rad/s"
        )
        costmap = data["costmap"]
        print(
            f"costmap: {costmap['resolution_m']:g} m cells, inflation {costmap['inflation_m']:g} m, "
            f"soft cost {costmap['soft_cost']:g}, memory {costmap['memory_s']:g} s"
        )
        safety = data["safety"]
        print(
            f"safety: localized-required={safety['require_localized']}, "
            f"sigma stop {safety['sigma_xy_stop_m']:g} m, brake {safety['emergency_stop_m']:g} m, "
            f"gate {safety['gate_rate_hz']:g} Hz"
        )
        mission = data["mission"]
        print(
            f"mission: work mode '{mission['work_mode']}', {mission['cycles_target']} cycles, "
            f"action {mission['excavation_action']}, enabled={mission['enabled']}"
        )
        topics = data["topics"]
        print(f"topics: {topics['navigate_action']} -> {topics['nav_cmd_vel']} -> "
              f"{topics['cmd_vel']} (gated)")
    poses = data.get("poses") or []
    if poses:
        illegal = [pose for pose in poses if not pose["legal"]]
        print(f"generated poses: {len(poses)} ({len(illegal)} illegal on an empty map)")
        for pose in poses:
            marker = "ok " if pose["legal"] else "BAD"
            detail = "" if pose["legal"] else f"  <- {pose['blocked_by']}"
            print(
                f"  {marker} {pose['label']:16s} x={pose['x']:6.3f} y={pose['y']:6.3f} "
                f"yaw={pose['yaw']:6.3f}{detail}"
            )
    if deps:
        for line in dependency_report():
            print(line)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog="navigation_info.py",
        description="report what module 5 is configured to do",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="navigation.yaml (empty = installed default)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--deps", action="store_true", help="also print the dependency report")
    parser.add_argument("--no-poses", action="store_true", help="skip the generated mission poses")
    parser.add_argument("--quiet", action="store_true", help="only print the poses (and deps)")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        config = load_config(args.config or None)
    except Exception as exc:
        print(f"navigation_info.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    data = report(config, with_poses=not args.no_poses)
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
