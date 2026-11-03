"""What module 4 needs on this machine, what is missing, and how to install it.

Written because module 4 is deliberately boring about dependencies: it runs the
bootstrap detector with numpy alone, and the *only* non-ROS package it truly
needs is ``dt-apriltags`` (a real AprilTag 3 detector).  Everything else -
``onnxruntime`` for the trained model, ``robot_localization`` for the fused
odometry, the CSI hub's driver stack on the Pi - is optional and this module
says so out loud instead of failing at an import.

``scripts/perception_info.py`` prints this; ``scripts/*.py`` nodes use it to warn
once at startup rather than crash on a missing optional piece.

``cv2`` and ``cv_bridge`` are listed as *refused*, not missing: this machine's
numpy 2.2 shadows the apt OpenCV build and ``cv2`` cannot even import, so module
4 is written to never need them (see docs/detector.md, docs/tags.md).
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

__all__ = [
    "Dependency",
    "check_dependencies",
    "report",
    "missing_required",
    "summary_dict",
]

#: Dependency kinds, used for grouping in the report.
PYTHON = "python"
ROS = "ros"
SYSTEM = "system"


@dataclass(frozen=True)
class Dependency:
    """One thing module 4 may need, and whether it is here."""

    name: str
    kind: str
    #: required = the classical sim/real stack cannot run without it
    required: bool
    available: bool
    detail: str
    install: str = ""

    @property
    def state(self) -> str:
        if self.available:
            return "ok"
        return "MISSING" if self.required else "optional"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
            "available": self.available,
            "state": self.state,
            "detail": self.detail,
            "install": self.install,
        }


def _present(module: str) -> Tuple[bool, str]:
    """``(importable, detail)`` for a Python module, without importing it."""
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return (False, "not found")
    if spec is None:
        return (False, "not found")
    version = ""
    try:  # cheap version lookup, only for modules that import cleanly
        loaded = importlib.import_module(module)
        version = str(getattr(loaded, "__version__", "") or "")
    except Exception:
        return (True, "present")
    return (True, version or "present")


def check_dependencies() -> Tuple[Dependency, ...]:
    """Probe every dependency module 4 knows about (never raises)."""
    rows: List[Dependency] = []

    def add(
        name: str,
        module: str,
        kind: str,
        required: bool,
        install: str = "",
        note: str = "",
    ) -> None:
        available, detail = _present(module)
        if note and available:
            detail = f"{detail} ({note})"
        rows.append(
            Dependency(
                name=name,
                kind=kind,
                required=required,
                available=available,
                detail=detail,
                install=install if not available else "",
            )
        )

    # -- the four soft dependencies of the algorithms ----------------------- #
    add(
        "numpy",
        "numpy",
        PYTHON,
        True,
        "sudo apt install python3-numpy   (or pip install --user numpy)",
        "every algorithm",
    )
    add(
        "yaml",
        "yaml",
        PYTHON,
        True,
        "sudo apt install python3-yaml",
        "config and artefacts",
    )
    add(
        "Pillow",
        "PIL",
        PYTHON,
        True,
        "sudo apt install python3-pil",
        "tag bitmaps / debug images",
    )
    add(
        "dt-apriltags",
        "dt_apriltags",
        PYTHON,
        True,
        "pip install --user dt-apriltags",
        "real AprilTag 3 detector (no cv2 needed)",
    )

    # -- optional: the trained detector ------------------------------------- #
    add(
        "onnxruntime",
        "onnxruntime",
        PYTHON,
        False,
        "pip install --user onnxruntime",
        "detector.backend: onnx",
    )

    # -- ROS: nodes --------------------------------------------------------- #
    for name in (
        "rclpy",
        "std_msgs",
        "sensor_msgs",
        "geometry_msgs",
        "nav_msgs",
        "diagnostic_msgs",
        "tf2_ros",
    ):
        add(name, name, ROS, True, "sudo apt install ros-humble-<name>")
    add("rover_perception_msgs", "rover_perception_msgs", ROS, True, "colcon build --packages-select rover_perception_msgs")
    add("rover_perception", "rover_perception", ROS, True, "colcon build --packages-select rover_perception")

    # -- optional: robot_localization, sim plumbing ------------------------- #
    add(
        "robot_localization",
        "robot_localization",
        ROS,
        False,
        "sudo apt install ros-humble-robot-localization",
        "the EKF that fuses wheel odom + IMU",
    )
    add("ros_gz_sim", "ros_gz_sim", ROS, False, "sudo apt install ros-humble-ros-gz-sim", "spawning tag boards in sim")
    add("launch_ros", "launch_ros", ROS, False, "sudo apt install ros-humble-launch-ros", "the launch files")

    # -- refused on purpose ------------------------------------------------- #
    rows.append(
        Dependency(
            name="cv2",
            kind=PYTHON,
            required=False,
            available=False,
            detail="refused by design: this machine's numpy shadows apt OpenCV",
            install="",
        )
    )
    rows.append(
        Dependency(
            name="cv_bridge",
            kind=ROS,
            required=False,
            available=False,
            detail="refused by design: it needs cv2 (rover_perception.imagecodec replaces it)",
            install="",
        )
    )
    return tuple(rows)


def missing_required(dependencies: Sequence[Dependency] = ()) -> Tuple[Dependency, ...]:
    """Only the missing *required* rows (what actually blocks a run)."""
    rows = dependencies or check_dependencies()
    return tuple(row for row in rows if row.required and not row.available)


def summary_dict(dependencies: Sequence[Dependency] = ()) -> Dict[str, Any]:
    rows = dependencies or check_dependencies()
    missing = [row for row in rows if row.required and not row.available]
    optional_missing = [row for row in rows if not row.required and not row.available and row.install]
    return {
        "checked": len(rows),
        "missing_required": [row.name for row in missing],
        "missing_optional": [row.name for row in optional_missing],
        "ok": not missing,
        "dependencies": [row.to_dict() for row in rows],
    }


def report(dependencies: Sequence[Dependency] = ()) -> str:
    """A printable table (``perception_info.py --deps``)."""
    rows = dependencies or check_dependencies()
    width = max(len(row.name) for row in rows)
    lines = []
    for row in rows:
        marker = row.state
        lines.append(f"{row.name.ljust(width)}  {row.kind:<7} {marker:<9} {row.detail}")
        if row.install:
            lines.append(f"{' ' * width}  {'':<7} {'':<9} install: {row.install}")
    missing = missing_required(rows)
    if missing:
        lines.append("")
        lines.append(f"{len(missing)} REQUIRED dependency/ies missing: {', '.join(row.name for row in missing)}")
    return "\n".join(lines)
