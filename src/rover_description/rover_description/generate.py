"""Generate every derived ``rover_description`` artifact from ``rover.yaml``.

One call produces a complete, self-describing robot for a given config:

    inertias.yaml     link mass + inertia tensor, consumed by the xacro model
    rover.urdf        the expanded URDF (robot_state_publisher + gz spawn)
    rover.sdf         gz SDF converted from the URDF (best effort, dev hosts)
    calibration.yaml  the ONE intrinsics/extrinsics file (module 4 projection)
    mass_budget.yaml  the auditable mass table
    envelope.yaml     measured stowed envelope vs the competition limit
    manifest.json     hashes, versions, assumptions

``rover.urdf`` is deterministic: for a fixed ``rover.yaml`` every generated file
except ``manifest.json``'s timestamp is byte-identical.
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
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from .calibration import calibration_document
from .config import (
    CAMERA_TOPICS,
    RoverConfig,
    RoverConfigError,
    default_config_path,
    load_config,
    package_share_dir,
)
from .envelope import EnvelopeReport, check_envelope
from .geometry import link_inertias, wheel_centers
from .mass import MassBudget, mass_budget

__all__ = [
    "RoverResult",
    "generate",
    "main",
    "default_urdf_path",
    "default_output_dir",
    "inertias_document",
    "validate_urdf",
    "expected_links",
    "expected_joints",
]

ROBOT_NAME = "rover"


def default_urdf_path() -> str:
    """Located ``urdf/rover.urdf.xacro`` (installed or source tree)."""
    here = os.path.dirname(os.path.abspath(__file__))
    relative = os.path.join("urdf", "rover.urdf.xacro")
    candidates = []
    share = package_share_dir()
    if share:
        candidates.append(os.path.join(share, relative))
    candidates.append(os.path.join(here, "..", relative))
    candidates.append(os.path.join(os.path.dirname(here), "share", "rover_description", relative))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    raise RoverConfigError("cannot locate urdf/rover.urdf.xacro")


def default_output_dir() -> str:
    base = "/tmp/opencode" if os.path.isdir("/tmp/opencode") else tempfile.gettempdir()
    return os.path.join(base, "rover_description")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _write_yaml(path: str, data: Any) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
def inertias_document(cfg: RoverConfig) -> Dict[str, Any]:
    """``{link_name: {mass, ixx, ...}}`` with link names at the top level so the
    xacro model can read ``iner[<link>]`` directly."""
    doc: Dict[str, Any] = {
        "_meta": {
            "source": "rover_description.generate (derived from config/rover.yaml)",
            "config_sha256": cfg.config_sha256(),
            "note": "analytic inertias about each link's centre of mass",
        }
    }
    for name, iner in link_inertias(cfg).items():
        doc[name] = {
            "mass": round(iner.mass, 6),
            "ixx": round(iner.ixx, 9),
            "ixy": round(iner.ixy, 9),
            "ixz": round(iner.ixz, 9),
            "iyy": round(iner.iyy, 9),
            "iyz": round(iner.iyz, 9),
            "izz": round(iner.izz, 9),
        }
    return doc


def expected_links(cfg: RoverConfig) -> List[str]:
    names = ["base_link", "ladder_link", "belt_link", "tub_link", "funnel_link"]
    names += list(wheel_centers(cfg).keys())
    for cam in cfg.sensors.cameras:
        names += [cam.frame, cam.optical_frame]
    if cfg.sensors.imu.enabled:
        names.append(cfg.sensors.imu.link)
    return names


def expected_joints(cfg: RoverConfig) -> List[str]:
    names = ["ladder_pitch_joint", "belt_joint", "tub_joint", "funnel_joint"]
    for wheel in wheel_centers(cfg):
        names.append(f"{wheel}_joint")
    for cam in cfg.sensors.cameras:
        names += [f"{cam.frame}_joint", f"{cam.name}_optical_joint"]
    if cfg.sensors.imu.enabled:
        names.append(f"{cfg.sensors.imu.link}_joint")
    return names


def validate_urdf(urdf_text: str, cfg: RoverConfig) -> List[str]:
    """Structural + competition-rule validation of the generated URDF.

    Returns a list of problems (empty means valid).  The same checks run in the
    test suite, so a broken model fails the build, not the robot.
    """
    problems: List[str] = []
    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError as exc:  # pragma: no cover - exercised by a test
        return [f"URDF is not well-formed XML: {exc}"]

    if root.tag != "robot":
        problems.append(f"root element is <{root.tag}>, expected <robot>")
    if root.get("name") != ROBOT_NAME:
        problems.append(f"robot name is {root.get('name')!r}, expected {ROBOT_NAME!r}")

    links = {e.get("name") for e in root.findall("link")}
    joints = {e.get("name") for e in root.findall("joint")}
    for name in expected_links(cfg):
        if name not in links:
            problems.append(f"missing link {name}")
    for name in expected_joints(cfg):
        if name not in joints:
            problems.append(f"missing joint {name}")

    # sensor ban: only camera + imu are allowed anywhere in the model
    allowed = {"camera", "imu"}
    camera_topics: List[str] = []
    for sensor in root.iter("sensor"):
        stype = (sensor.get("type") or "").lower()
        if stype not in allowed:
            problems.append(
                f"forbidden sensor type {stype!r} (name={sensor.get('name')!r}); "
                f"allowed: {sorted(allowed)}"
            )
        if stype == "camera":
            topic_el = sensor.find("topic")
            if topic_el is not None and topic_el.text:
                camera_topics.append(topic_el.text.strip())
    if sorted(camera_topics) != sorted(CAMERA_TOPICS):
        problems.append(f"camera topics {camera_topics} != {list(CAMERA_TOPICS)}")

    # forbidden hardware strings (belt-and-braces on top of the sensor scan)
    lowered = urdf_text.lower()
    for banned in ("magnetometer", "gps", "navsat", "sonar", "ultrasonic"):
        if banned in lowered:
            problems.append(f"forbidden token {banned!r} found in the model")

    plugins = [f"{e.get('filename') or ''} {e.get('name') or ''}" for e in root.iter("plugin")]
    if not any("diff-drive" in p for p in plugins):
        problems.append("gz diff-drive plugin not found")
    if not any("joint-state-publisher" in p for p in plugins):
        problems.append("gz joint-state-publisher plugin not found")

    return problems


# --------------------------------------------------------------------------- #
# xacro / sdf
# --------------------------------------------------------------------------- #
def _process_xacro(urdf_xacro: str, config_path: str, inertias_path: str) -> str:
    import xacro  # ROS package; only needed when generating

    document = xacro.process_file(
        urdf_xacro, mappings={"config": config_path, "inertias": inertias_path}
    )
    text = document.toprettyxml(indent="  ")
    if not text.endswith("\n"):
        text += "\n"
    return text


def _urdf_to_sdf(urdf_path: str) -> Optional[str]:
    if shutil.which("gz") is None:
        return None
    try:
        proc = subprocess.run(
            ["gz", "sdf", "-p", urdf_path],
            capture_output=True, text=True, timeout=90,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    text = proc.stdout
    if not text.endswith("\n"):
        text += "\n"
    return text


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
@dataclass
class RoverResult:
    output_dir: str
    config: RoverConfig
    urdf_path: str
    sdf_path: Optional[str]
    inertias_path: str
    calibration_path: str
    mass_path: str
    envelope_path: str
    manifest_path: str
    envelope: EnvelopeReport
    mass: MassBudget
    warnings: List[str] = field(default_factory=list)
    file_hashes: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.envelope.ok and self.mass.ok and not self.warnings

    def summary(self) -> str:
        lines = [
            f"rover_description -> {self.output_dir}",
            f"  envelope : {'PASS' if self.envelope.ok else 'FAIL'}  {self.envelope.measured.as_tuple()}",
            f"  mass     : {'PASS' if self.mass.ok else 'FAIL'}  {self.mass.total_mass:.2f} kg "
            f"(limit {self.mass.mass_limit:.1f} kg)",
            f"  urdf     : {self.urdf_path}",
        ]
        if self.warnings:
            lines.append("  warnings :")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


def generate(
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    urdf: Optional[str] = None,
    write_sdf: bool = True,
    validate: bool = True,
) -> RoverResult:
    cfg = load_config(config_path)
    out = os.path.abspath(os.path.expanduser(output_dir)) if output_dir else default_output_dir()
    os.makedirs(out, exist_ok=True)
    xacro_path = urdf or default_urdf_path()
    warnings: List[str] = []

    # 1. inertias (must exist before xacro runs)
    inertias_path = os.path.join(out, "inertias.yaml")
    _write_yaml(inertias_path, inertias_document(cfg))

    # 2. xacro -> urdf
    urdf_text = _process_xacro(xacro_path, cfg.path, inertias_path)
    urdf_path = os.path.join(out, "rover.urdf")
    _write_text(urdf_path, urdf_text)

    if validate:
        warnings.extend(validate_urdf(urdf_text, cfg))

    # 3. gz sdf (best effort)
    sdf_path: Optional[str] = None
    if write_sdf:
        sdf_text = _urdf_to_sdf(urdf_path)
        if sdf_text:
            sdf_path = os.path.join(out, "rover.sdf")
            _write_text(sdf_path, sdf_text)
        else:
            warnings.append("gz unavailable: rover.sdf not generated (fine on the robot)")

    # 4. calibration
    calibration_path = os.path.join(out, "calibration.yaml")
    _write_yaml(calibration_path, calibration_document(cfg))

    # 5. mass + envelope reports
    mass = mass_budget(cfg)
    envelope = check_envelope(cfg)
    mass_path = os.path.join(out, "mass_budget.yaml")
    _write_yaml(mass_path, mass.to_dict())
    envelope_path = os.path.join(out, "envelope.yaml")
    _write_yaml(envelope_path, envelope.to_dict())

    if not envelope.ok:
        warnings.append("measured envelope exceeds the competition limit")
    if not mass.ok:
        warnings.append("mass budget exceeds the competition limit")

    # 6. manifest
    tracked = {
        "inertias.yaml": inertias_path,
        "rover.urdf": urdf_path,
        "calibration.yaml": calibration_path,
        "mass_budget.yaml": mass_path,
        "envelope.yaml": envelope_path,
    }
    if sdf_path:
        tracked["rover.sdf"] = sdf_path
    file_hashes = {name: _sha256_file(path) for name, path in tracked.items()}

    manifest = {
        "module": "rover_description",
        "package_version": _package_version(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_path": cfg.path,
        "config_sha256": cfg.config_sha256(),
        "urdf_source": xacro_path,
        "robot_name": ROBOT_NAME,
        "declared_forward": cfg.declared_forward,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "files": file_hashes,
        "envelope": envelope.to_dict(),
        "mass": mass.to_dict(),
        "assumptions": cfg.assumptions_report(),
        "warnings": warnings,
    }
    manifest_path = os.path.join(out, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
        handle.write("\n")

    return RoverResult(
        output_dir=out,
        config=cfg,
        urdf_path=urdf_path,
        sdf_path=sdf_path,
        inertias_path=inertias_path,
        calibration_path=calibration_path,
        mass_path=mass_path,
        envelope_path=envelope_path,
        manifest_path=manifest_path,
        envelope=envelope,
        mass=mass,
        warnings=warnings,
        file_hashes=file_hashes,
    )


def _package_version() -> str:
    from . import __version__

    return __version__


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="generate_robot",
        description="Generate the rover URDF/SDF, calibration, mass and envelope artifacts.",
    )
    parser.add_argument("--config", default=None, help="path to rover.yaml")
    parser.add_argument("--output-dir", default=None, help="where to write the artifacts")
    parser.add_argument("--urdf", default=None, help="override the urdf/xacro path")
    parser.add_argument("--no-sdf", action="store_true", help="skip gz sdf conversion")
    parser.add_argument("--no-validate", action="store_true", help="skip URDF validation")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = generate(
            config_path=args.config,
            output_dir=args.output_dir,
            urdf=args.urdf,
            write_sdf=not args.no_sdf,
            validate=not args.no_validate,
        )
    except RoverConfigError as exc:
        print(f"rover_description: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        print(result.summary())
    return 0 if result.envelope.ok and result.mass.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
