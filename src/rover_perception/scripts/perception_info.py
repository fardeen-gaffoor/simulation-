#!/usr/bin/env python3
"""Print the effective module-4 configuration and the state of this machine.

    ros2 run rover_perception perception_info.py
    ros2 run rover_perception perception_info.py --json
    ros2 run rover_perception perception_info.py --deps

This is the "what will actually run?" tool.  It resolves the same paths and
loads the same artefacts the nodes do - so if it prints a missing
``calibration.yaml`` or an empty tag catalogue, the nodes would fail the same
way, and you find out before launching anything.  It also reports, per camera,
the rate the phase manager would command right now, which is the fastest way to
sanity-check a ``perception.yaml`` edit.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from rover_perception import deps
from rover_perception.calibration_ref import CalibrationError, load_calibration, resolve_calibration_path
from rover_perception.camera_manager import PhaseController
from rover_perception.config import PerceptionConfigError, load_config
from rover_perception.markers_ref import MarkersError, load_markers
from rover_perception.nodeutil import default_artifacts_dir, resolve_tags_path
from rover_perception.tags import TagsError, load_tags
from rover_perception.thermal import ThermalMonitor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="perception_info",
        description="print the effective rover_perception configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="", help="perception.yaml (empty = installed default)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    parser.add_argument("--deps", action="store_true", help="also print the dependency report")
    parser.add_argument("--no-artefacts", action="store_true", help="do not try to load generated artefacts")
    parser.add_argument("--quiet", action="store_true", help="only print problems")
    parser.add_argument("--json-deps", action="store_true", help=argparse.SUPPRESS)
    return parser


def _artifact_report(cfg: Any, load_artefacts: bool) -> Dict[str, Any]:
    """Resolve (and optionally load) calibration, markers and the tag catalogue."""
    report: Dict[str, Any] = {}

    calibration_path = ""
    try:
        calibration_path = resolve_calibration_path(cfg.artifacts.calibration)
        report["calibration"] = {"path": calibration_path, "state": "present"}
        if load_artefacts:
            calibration = load_calibration(calibration_path)
            report["calibration"]["cameras"] = {
                name: {
                    "fx": round(camera.fx, 3),
                    "fy": round(camera.fy, 3),
                    "cx": round(camera.cx, 3),
                    "cy": round(camera.cy, 3),
                    "width": camera.width,
                    "height": camera.height,
                    "hfov_deg": round(camera.hfov_rad * 180.0 / 3.141592653589793, 2),
                    "mount_z_m": round(float(camera.t_base_cam[2]), 4),
                }
                for name, camera in calibration.cameras.items()
            }
    except CalibrationError as exc:
        report["calibration"] = {"path": calibration_path, "state": "MISSING", "detail": str(exc)}

    try:
        markers = load_markers(cfg.artifacts.markers or None)
        report["markers"] = {
            "path": markers.path,
            "state": "present",
            "seed": markers.seed,
            "family": markers.family,
            "ids": markers.ids,
            "board_size": markers.board_size,
            "tag_size": markers.tag_size,
        }
    except MarkersError as exc:
        report["markers"] = {"path": cfg.artifacts.markers, "state": "MISSING", "detail": str(exc)}

    tags_path = resolve_tags_path(cfg)
    report["tags"] = {"path": tags_path, "state": "present"}
    try:
        catalogue = load_tags(tags_path)
        report["tags"].update(
            {
                "ids": list(catalogue.ids),
                "count": catalogue.count,
                "family": catalogue.family,
                "source_markers": catalogue.source_markers,
            }
        )
    except TagsError as exc:
        report["tags"].update({"state": "MISSING", "detail": str(exc)})
    report["artifacts_dir"] = cfg.artifacts.output_dir or default_artifacts_dir()
    return report


def _phase_report(cfg: Any) -> Dict[str, Any]:
    """What the phase manager would command for the configured initial mode."""
    controller = PhaseController(
        cfg.camera_manager,
        cfg.camera_names,
        cfg.active_camera_for_mode,
    )
    command = controller.update(0.0)
    return {
        "mode": command.mode,
        "active_camera": command.active_camera,
        "active_rate_hz": command.active_rate_hz,
        "idle_rate_hz": command.idle_rate_hz,
        "derate_index": command.derate_index,
        "derate_factor": command.derate_factor,
        "rates": {item.name: item.rate_hz for item in command.commands},
        "bandwidth_limit_kbps": command.limit_kbps,
        "thermal_thresholds_c": {
            "warn": cfg.camera_manager.thermal.warn_c,
            "throttle": cfg.camera_manager.thermal.throttle_c,
            "critical": cfg.camera_manager.thermal.critical_c,
        },
    }


def build_report(cfg: Any, load_artefacts: bool = True) -> Dict[str, Any]:
    thermal = ThermalMonitor(cfg.camera_manager.thermal)
    reading = thermal.read()
    return {
        "config": {"path": cfg.path, "sha256": cfg.sha256, "schema_version": cfg.schema_version},
        "arena": {
            "frame_id": cfg.arena.frame_id,
            "size": [cfg.arena.length_x, cfg.arena.width_y],
            "ground_z": cfg.arena.ground_z,
            "wall_margin_m": cfg.arena.wall_margin_m,
            "zones": {name: rect.as_tuple() for name, rect in cfg.arena.zones.items()},
        },
        "cameras": [
            {
                "name": camera.name,
                "role": camera.role,
                "topic": camera.topic,
                "camera_info_topic": camera.camera_info_topic,
                "size": [camera.width, camera.height],
                "encoding": camera.encoding,
                "payload_bytes": camera.bytes_per_pixel * camera.width * camera.height,
            }
            for camera in cfg.cameras
        ],
        "camera_manager": {
            "enabled": cfg.camera_manager.enabled,
            "mode_topic": cfg.camera_manager.mode_topic,
            "phase_topic": cfg.camera_manager.phase_topic,
            "status_topic": cfg.camera_manager.status_topic,
            "active_rate_hz": cfg.camera_manager.active_rate_hz,
            "idle_rate_hz": cfg.camera_manager.idle_rate_hz,
            "min_rate_hz": cfg.camera_manager.min_rate_hz,
            "derate_factors": list(cfg.camera_manager.bandwidth.derate_factors),
            "window_s": cfg.camera_manager.bandwidth.window_s,
            "thermal": {
                "enabled": cfg.camera_manager.thermal.enabled,
                "state": reading.state_name,
                "temperature_c": reading.temperature_c,
                "hottest_zone": reading.zone,
            },
            "initial": _phase_report(cfg),
        },
        "detector": {
            "enabled": cfg.detector.enabled,
            "camera": cfg.detector.camera,
            "backend": cfg.detector.backend,
            "publish_hz": cfg.detector.publish_hz,
            "max_frame_age_s": cfg.detector.max_frame_age_s,
            "obstacles_topic": cfg.detector.obstacles_topic,
            "stats_topic": cfg.detector.stats_topic,
            "cloud": {"enabled": cfg.detector.cloud_enabled, "topic": cfg.detector.cloud_topic},
            "ground": {
                "min_range_m": cfg.detector.ground.min_range_m,
                "max_range_m": cfg.detector.ground.max_range_m,
                "contact_point": cfg.detector.ground.contact_point,
                "sigma_xy_m": cfg.detector.ground.default_sigma_xy_m,
            },
            "classical": {
                "min_area_px": cfg.detector.classical.min_area_px,
                "min_fill": cfg.detector.classical.min_fill,
                "ambiguous_kind": cfg.detector.classical.ambiguous_kind,
            },
            "onnx": {
                "model": cfg.detector.onnx.model,
                "input_size": cfg.detector.onnx.input_size,
                "classes": list(cfg.detector.onnx.classes),
            },
        },
        "tracking": {
            "min_hits": cfg.tracking.min_hits,
            "max_misses": cfg.tracking.max_misses,
            "min_confidence": cfg.tracking.min_confidence,
            "associate_radius_m": cfg.tracking.associate_radius_m,
            "crater_bias": cfg.tracking.crater_bias,
        },
        "tags": {
            "enabled": cfg.tags.enabled,
            "camera": cfg.tags.camera,
            "family": cfg.tags.family,
            "topic": cfg.tags.topic,
            "decision_margin_min": cfg.tags.detection.decision_margin_min,
            "hamming_max": cfg.tags.detection.hamming_max,
            "board_enabled": cfg.tags.board.enabled,
            "standoff_m": cfg.tags.board.standoff_m,
        },
        "localization": {
            "enabled": cfg.localization.enabled,
            "odometry_topics": list(cfg.localization.odometry_topics),
            "state_topic": cfg.localization.state_topic,
            "fix_topic": cfg.localization.fix_topic,
            "publish_hz": cfg.localization.publish_hz,
            "fix_sigma_xy_m": cfg.localization.fix_sigma_xy_m,
            "growth_m_per_m": cfg.localization.growth_m_per_m,
            "max_correction_m": cfg.localization.max_correction_m,
            "max_correction_yaw_rad": cfg.localization.max_correction_yaw_rad,
        },
        "health": {
            "enabled": cfg.health.enabled,
            "topic": cfg.health.topic,
            "diagnostics_topic": cfg.health.diagnostics_topic,
            "publish_hz": cfg.health.publish_hz,
        },
        "artefacts": _artifact_report(cfg, load_artefacts),
    }


def _print_text(report: Dict[str, Any], show_deps: bool) -> None:
    config = report["config"]
    print(f"config      {config['path']}  (sha256 {config['sha256'][:12]}, schema {config['schema_version']})")
    arena = report["arena"]
    print(
        f"arena       {arena['size'][0]} x {arena['size'][1]} m, frame '{arena['frame_id']}', "
        f"ground z {arena['ground_z']}, wall margin {arena['wall_margin_m']} m"
    )
    for name, rect in arena["zones"].items():
        print(f"              zone {name:<12} x {rect[0]:.2f}..{rect[2]:.2f}  y {rect[1]:.2f}..{rect[3]:.2f}")
    print("cameras")
    for camera in report["cameras"]:
        width, height = camera["size"]
        print(
            f"  {camera['name']:<16} role {camera['role']:<11} {width}x{height} "
            f"{camera['encoding']:<5} {camera['topic']}"
        )
    manager = report["camera_manager"]
    initial = manager["initial"]
    print(
        f"camera mgr  enabled {manager['enabled']}  mode '{initial['mode']}' active '{initial['active_camera']}' "
        f"({initial['active_rate_hz']:.1f} Hz) idle {initial['idle_rate_hz']:.1f} Hz  "
        f"budget {initial['bandwidth_limit_kbps']:.0f} kbps"
    )
    print(f"              derate ladder {manager['derate_factors']}, window {manager['window_s']:.0f} s")
    thermal = manager["thermal"]
    print(
        f"              thermal enabled {thermal['enabled']} state {thermal['state']} "
        f"({thermal['temperature_c']} C on {thermal['hottest_zone']})"
    )
    detector = report["detector"]
    print(
        f"detector    backend '{detector['backend']}' on {detector['camera']} at {detector['publish_hz']:.0f} Hz "
        f"-> {detector['obstacles_topic']}"
    )
    ground = detector["ground"]
    print(
        f"              ground {ground['min_range_m']}..{ground['max_range_m']} m via {ground['contact_point']}, "
        f"sigma {ground['sigma_xy_m']} m"
    )
    tags_report = report["tags"]
    print(
        f"tags        family {tags_report['family']} on {tags_report['camera']} -> {tags_report['topic']} "
        f"(board standoff {tags_report['standoff_m']:.4f} m)"
    )
    localization = report["localization"]
    print(
        f"localization odom {localization['odometry_topics']} -> {localization['state_topic']} "
        f"at {localization['publish_hz']:.0f} Hz"
    )
    print(
        f"              fix sigma {localization['fix_sigma_xy_m']} m, growth "
        f"{localization['growth_m_per_m']} m/m, max correction {localization['max_correction_m']} m"
    )
    print("artefacts")
    for key in ("calibration", "markers", "tags"):
        entry = report["artefacts"][key]
        state = entry.get("state", "?")
        detail = f"  <- {entry['detail']}" if state == "MISSING" and entry.get("detail") else ""
        print(f"  {key:<12} {state:<8} {entry.get('path', '')}{detail}")
        if key == "calibration" and entry.get("cameras"):
            for name, camera in entry["cameras"].items():
                print(
                    f"    {name:<16} fx {camera['fx']:.1f} fy {camera['fy']:.1f} "
                    f"cx {camera['cx']:.1f} cy {camera['cy']:.1f} mount {camera['mount_z_m']:.3f} m"
                )
        if key == "markers" and entry.get("ids"):
            print(f"    seed {entry.get('seed')} ids {entry.get('ids')} tag_size {entry.get('tag_size')}")
        if key == "tags" and entry.get("ids"):
            print(f"    ids {entry.get('ids')} from {entry.get('source_markers')}")
    if show_deps:
        print("dependencies")
        for row in deps.report().splitlines():
            print(f"  {row}")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config or None)
    except PerceptionConfigError as exc:
        print(f"perception_info: {exc}", file=sys.stderr)
        return 1
    report = build_report(config, load_artefacts=not args.no_artefacts)
    if args.json or args.json_deps:
        payload = dict(report)
        if args.json_deps or args.deps:
            payload["dependencies"] = deps.summary_dict()
        print(json.dumps(payload, indent=2, sort_keys=False))
        return 0
    problems: List[str] = []
    for key in ("calibration", "markers", "tags"):
        if report["artefacts"][key].get("state") == "MISSING":
            problems.append(key)
    missing_required = deps.missing_required()
    if args.quiet:
        for key in problems:
            entry = report["artefacts"][key]
            print(f"perception_info: {key} unavailable: {entry.get('detail', entry.get('path', ''))}", file=sys.stderr)
        for row in missing_required:
            print(f"perception_info: required dependency missing: {row.name} ({row.install})", file=sys.stderr)
        return 0
    _print_text(report, args.deps)
    if problems:
        print("")
        print("not ready: " + ", ".join(f"{key} unavailable" for key in problems))
    if missing_required:
        print("not ready: " + ", ".join(f"{row.name} missing" for row in missing_required))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
