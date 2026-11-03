#!/usr/bin/env python3
"""Environment check for the Caterpillar Autonomy Challenge stack.

Verifies, in order of how much pain each failure would cause:

1. platform, Python, ROS 2 distribution and middleware
2. Gazebo Harmonic (``gz sim`` 8.x) and the plugins the arena needs
3. **ros_gz bridge compatibility**: the installed ``ros_gz`` binaries are linked
   against exactly the Gazebo libraries that are on the machine (this is the
   check that catches "Harmonic binaries against a Garden bridge")
4. ROS packages required by module 1 and by the upcoming modules
5. a live smoke test: start a headless ``gz sim``, bridge ``/clock`` into ROS 2
   and prove that messages actually arrive
6. optionally (``--render``) prove that a camera sensor renders headless, and
   (``--arena``) generate a full arena and validate it with ``gz sdf -k``

Exit code is 0 when nothing FAILed; warnings never fail the check.

Usage::

    ros2 run cat_arena env_check.py
    ros2 run cat_arena env_check.py --no-smoke --json
    ros2 run cat_arena env_check.py --render --arena --install-hints
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

#: ROS packages per module.  ``required`` ones must exist right now.
_PACKAGES: Sequence[Tuple[str, str, str]] = (
    # (ros package, module, why)
    ("rclpy", "1", "start-pose publisher / all rclpy nodes"),
    ("ros_gz_sim", "1", "spawning the rover and starting gz sim from a launch file"),
    ("ros_gz_bridge", "1", "Gazebo <-> ROS 2 message bridge"),
    ("geometry_msgs", "1", "PoseWithCovarianceStamped start pose"),
    ("launch", "1", "launch files"),
    ("launch_ros", "1", "ros2 launch integration"),
    ("xacro", "2", "rover_description macro expansion"),
    ("robot_state_publisher", "2", "URDF -> TF"),
    ("joint_state_publisher", "2", "wheel/joint states in RViz"),
    ("urdf", "2", "URDF parsing helpers"),
    ("tf2_ros", "2", "TF2 plumbing"),
    ("cv_bridge", "3", "dataset capture / image conversion"),
    ("image_transport", "3", "camera topics"),
    ("vision_msgs", "4", "detection messages"),
    ("diagnostic_msgs", "4", "inference FPS / CPU / temperature reporting"),
    ("apriltag_ros", "4", "cam_tag localization (NOT installed by default)"),
    ("robot_localization", "4", "EKF (wheel odom + IMU, no magnetometer)"),
    ("nav2_bringup", "5", "navigation bringup"),
    ("nav2_map_server", "5", "arena bounds / zone map"),
    ("nav2_costmap_2d", "5", "global + rolling local costmaps"),
    ("nav2_smac_planner", "5", "Theta* / Smac 2D planner"),
    ("nav2_dwb_controller", "5", "DWB controller"),
    ("nav2_bt_navigator", "5", "behaviour trees"),
    ("nav2_lifecycle_manager", "5", "lifecycle management"),
)

#: Gazebo system plugins the generated world depends on.
_GZ_PLUGINS: Sequence[Tuple[str, str]] = (
    ("gz-sim-physics-system", "physics"),
    ("gz-sim-user-commands-system", "spawning entities (/world/<name>/create)"),
    ("gz-sim-scene-broadcaster-system", "scene/pose publication, gz model queries"),
    ("gz-sim-contact-system", "contact reporting"),
    ("gz-sim-sensors-system", "camera sensors (modules 3 and 4)"),
)

#: Gazebo components whose exact installed version is worth reporting.
_GZ_COMPONENTS: Sequence[Tuple[str, str]] = (
    ("gz-sim8", "the simulator"),
    ("gz-transport13", "transport (bridge)"),
    ("gz-msgs10", "messages (bridge)"),
    ("gz-rendering8", "rendering (cameras)"),
    ("gz-sensors8", "sensors (cameras)"),
    ("gz-physics7", "physics"),
    ("gz-common5", "common"),
    ("gz-math7", "math"),
)

#: Sibling Gazebo libraries that must be present for the bridge to work.
_GZ_LIBS: Sequence[str] = (
    "libgz-sim8.so",
    "libgz-transport13.so",
    "libgz-msgs10.so",
    "libgz-rendering8.so",
    "libgz-sensors8.so",
    "libgz-physics7.so",
    "libgz-common5.so",
    "libgz-math7.so",
)

#: ros_gz shared objects whose dynamic links are verified against the system.
_ROS_GZ_LIBS: Sequence[str] = (
    "libros_gz_bridge_lib.so",
    "libros_gz_sim.so",
)

_APT_HINTS: Dict[str, str] = {
    "joint_state_publisher": "ros-humble-joint-state-publisher ros-humble-joint-state-publisher-gui",
    "apriltag_ros": "ros-humble-apriltag-ros",
    "robot_localization": "ros-humble-robot-localization",
    "nav2_bringup": "ros-humble-nav2-bringup",
    "nav2_map_server": "ros-humble-nav2-map-server",
    "nav2_costmap_2d": "ros-humble-nav2-costmap-2d",
    "nav2_smac_planner": "ros-humble-nav2-smac-planner",
    "nav2_dwb_controller": "ros-humble-nav2-dwb-controller",
    "nav2_bt_navigator": "ros-humble-nav2-bt-navigator",
    "nav2_lifecycle_manager": "ros-humble-nav2-lifecycle-manager",
    "vision_msgs": "ros-humble-vision-msgs",
    "diagnostic_msgs": "ros-humble-diagnostic-msgs",
}


@dataclass
class Result:
    section: str
    name: str
    status: str
    detail: str = ""

    def as_dict(self) -> Dict[str, str]:
        return {"section": self.section, "name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class Report:
    results: List[Result] = field(default_factory=list)

    def add(self, section: str, name: str, status: str, detail: str = "") -> None:
        self.results.append(Result(section, name, status, detail))

    def counts(self) -> Dict[str, int]:
        counts = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return counts

    def ok(self) -> bool:
        return self.counts()[FAIL] == 0

    def render(self, use_color: bool = True) -> str:
        colors = {
            PASS: "\033[32m",
            WARN: "\033[33m",
            FAIL: "\033[31m",
            SKIP: "\033[90m",
        }
        reset = "\033[0m" if use_color else ""
        lines: List[str] = []
        section = None
        for result in self.results:
            if result.section != section:
                section = result.section
                lines.append("")
                lines.append(f"== {section} ==")
            color = colors.get(result.status, "") if use_color else ""
            detail = f"  {result.detail}" if result.detail else ""
            lines.append(f"  [{color}{result.status:4s}{reset}] {result.name}{detail}")
        counts = self.counts()
        lines.append("")
        lines.append(
            f"summary: {counts[PASS]} pass, {counts[WARN]} warn, "
            f"{counts[FAIL]} fail, {counts[SKIP]} skip"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _run(cmd: Sequence[str], timeout: float = 30.0) -> Tuple[int, str]:
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return 127, f"{cmd[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(cmd)}: timed out after {timeout:.0f}s"
    except OSError as exc:  # pragma: no cover
        return 126, str(exc)
    return proc.returncode, ((proc.stdout or "") + (proc.stderr or "")).strip()


def _dpkg_version(package: str) -> Optional[str]:
    rc, out = _run(["dpkg-query", "-W", "-f=${Version}", package], timeout=15)
    return out if rc == 0 and out else None


_IMPORT_CACHE: Dict[str, Tuple[bool, str]] = {}


def _importable(module: str) -> Tuple[bool, str]:
    """Import ``module`` once, caching the outcome.

    stderr is captured (and dropped): a module compiled against NumPy 1.x dumps
    a multi-line ABI traceback there when imported under NumPy 2.x, which is
    noise - the returned error string already carries the useful part.
    """
    cached = _IMPORT_CACHE.get(module)
    if cached is not None:
        return cached
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            mod = importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - environment dependent
            result = (False, str(exc))
        else:
            version = getattr(mod, "__version__", "")
            result = (True, version or "ok")
    _IMPORT_CACHE[module] = result
    return result


def _opencv_numpy_conflict_hint() -> Optional[str]:
    """Detect the classic "cv2 built against NumPy 1.x, NumPy 2.x installed" clash.

    Ubuntu 22.04 ships ``python3-opencv`` built against NumPy 1.21; if a newer
    NumPy 2.x shadow it (e.g. a pip install in ``~/.local``), ``import cv2``
    fails with ``_ARRAY_API not found`` / ``numpy.core.multiarray failed to
    import``.  ROS 2 Humble's ``cv_bridge`` needs the apt cv2, so the fix is to
    keep NumPy 1.x rather than to replace OpenCV.
    """
    if _importable("cv2")[0]:
        return None
    ok, numpy_detail = _importable("numpy")
    if not ok:
        return None
    try:
        import numpy

        major = int(str(numpy.__version__).split(".")[0])
    except Exception:  # pragma: no cover - environment dependent
        return None
    if major < 2:
        return None
    return (
        f"NumPy {numpy.__version__} shadows the apt NumPy 1.x that python3-opencv "
        "was built against; keep NumPy 1.x for ROS 2 Humble "
        "(e.g. 'pip uninstall numpy' or run with PYTHONNOUSERSITE=1)"
    )


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


# --------------------------------------------------------------------------- #
# 1. platform
# --------------------------------------------------------------------------- #
def check_platform(report: Report) -> None:
    section = "platform"
    report.add(
        section,
        "os",
        PASS,
        f"{platform.system()} {platform.release()} ({platform.machine()})",
    )
    try:
        os_release: Dict[str, str] = {}
        with open("/etc/os-release", "r", encoding="utf-8") as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.rstrip().split("=", 1)
                    os_release[key] = value.strip().strip('"')
        pretty = os_release.get("PRETTY_NAME", "unknown")
        version_id = os_release.get("VERSION_ID", "")
        status = PASS if version_id == "22.04" else WARN
        report.add(section, "ubuntu 22.04", status, pretty)
    except OSError as exc:
        report.add(section, "ubuntu 22.04", WARN, f"cannot read /etc/os-release: {exc}")

    report.add(section, "python", PASS, sys.version.split()[0])
    report.add(section, "python executable", PASS, sys.executable)

    ros_distro = os.environ.get("ROS_DISTRO", "")
    report.add(
        section,
        "ROS_DISTRO",
        PASS if ros_distro == "humble" else FAIL,
        ros_distro or "not set - source /opt/ros/humble/setup.bash",
    )
    report.add(
        section,
        "ROS_VERSION",
        PASS if os.environ.get("ROS_VERSION") == "2" else WARN,
        os.environ.get("ROS_VERSION", "not set"),
    )
    report.add(
        section,
        "RMW_IMPLEMENTATION",
        PASS,
        os.environ.get("RMW_IMPLEMENTATION", "default (rmw_fastrtps_cpp)"),
    )
    report.add(section, "ROS_DOMAIN_ID", PASS, os.environ.get("ROS_DOMAIN_ID", "0 (default)"))
    display = os.environ.get("DISPLAY", "")
    report.add(
        section,
        "DISPLAY",
        PASS if display else WARN,
        display or "unset - GUI worlds need one; headless server mode still works",
    )

    for module in ("numpy", "yaml"):
        ok, detail = _importable(module)
        report.add(section, f"python module {module}", PASS if ok else FAIL, detail)
    for module in ("cv2", "PIL"):
        ok, detail = _importable(module)
        if not ok and module == "cv2":
            detail = _opencv_numpy_conflict_hint() or detail
        report.add(
            section,
            f"python module {module}",
            PASS if ok else WARN,
            detail if ok else f"{detail} (needed by module 3 dataset tooling)",
        )


# --------------------------------------------------------------------------- #
# 2. gazebo
# --------------------------------------------------------------------------- #
def check_gazebo(report: Report) -> None:
    section = "gazebo (Harmonic)"
    gz = shutil.which("gz")
    if not gz:
        report.add(section, "gz CLI", FAIL, "'gz' not on PATH")
        return
    report.add(section, "gz CLI", PASS, gz)

    rc, out = _run([gz, "sim", "--version"], timeout=30)
    version = _first_line(out)
    if rc != 0:
        report.add(section, "gz sim version", FAIL, version or f"rc={rc}")
    else:
        major = ""
        for token in version.replace(",", " ").split():
            if token.replace(".", "").isdigit():
                major = token.split(".")[0]
                break
        report.add(
            section,
            "gz sim version",
            PASS if major == "8" else WARN,
            f"{version} (Harmonic == gz-sim 8.x)",
        )

    for component, why in _GZ_COMPONENTS:
        version_text = _gz_component_version(component)
        report.add(
            section,
            component,
            PASS if version_text else WARN,
            f"{version_text} ({why})" if version_text else f"version unknown ({why})",
        )

    # system plugins: the short name from the SDF, plus what breaks without it
    plugin_dirs = _gz_plugin_dirs()
    if not plugin_dirs:
        report.add(section, "gz plugin directory", FAIL, "no gz-sim-8/plugins directory found")
    for plugin, why in _GZ_PLUGINS:
        library = _find_gz_plugin(plugin)
        if not library:
            report.add(
                section,
                f"plugin {plugin}",
                FAIL,
                f"library not found in {', '.join(plugin_dirs) or '(no plugin directory)'}",
            )
            continue
        # dlopen it: this also catches an ABI mismatch between gz-sim and the plugin
        rc, out = _run([gz, "plugin", "--info", "-p", library], timeout=30)
        loaded = rc == 0 and "Found 0 plugins" not in out and "gz::sim::v8::System" in out
        report.add(
            section,
            f"plugin {plugin}",
            PASS if loaded else FAIL,
            why if loaded else f"{library} does not load: {_first_line(out)}",
        )

    engine = _find_render_engine("ogre2")
    report.add(
        section,
        "render engine ogre2",
        PASS if engine else WARN,
        engine or "not found - camera sensors (modules 3/4) need it; verify with --render",
    )

    for lib in _GZ_LIBS:
        found = _find_library(lib)
        report.add(section, lib, PASS if found else FAIL, found or "not found in the dynamic linker path")

    # DART bullet collision detector: required for triangle-mesh terrain
    bullet = _find_library("libdart-collision-bullet")
    report.add(
        section,
        "DART bullet collision",
        PASS if bullet else FAIL,
        bullet or "required for mesh terrain with dartsim (libdart-collision-bullet.so)",
    )
    physics_bullet = _find_library("libgz-physics7-bullet")
    report.add(
        section,
        "gz-physics bullet engine",
        PASS if physics_bullet else WARN,
        physics_bullet or "optional alternative physics engine",
    )


def _gz_component_version(component: str) -> Optional[str]:
    """Exact version of a Gazebo component, from its CMake package config.

    The ``gz`` CLI in this build only implements ``sim``/``model``/``topic``/...,
    so ``gz transport --version`` just prints the top-level help.  The CMake
    config files carry the authoritative version string.
    """
    triplet = _lib_triplet()
    for directory in (
        f"/usr/lib/{triplet}/cmake/{component}",
        "/usr/lib/cmake/" + component,
        "/usr/local/lib/cmake/" + component,
        f"/usr/share/cmake/{component}",
    ):
        path = os.path.join(directory, f"{component}-config-version.cmake")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if line.startswith("set(PACKAGE_VERSION") and '"' in line:
                        value = line.split('"')[1].strip()
                        if value and value.upper() != "UNKNOWN":
                            return value
        except OSError:
            continue
    return None


def _lib_triplet() -> str:
    machine = platform.machine()
    return {
        "x86_64": "x86_64-linux-gnu",
        "amd64": "x86_64-linux-gnu",
        "aarch64": "aarch64-linux-gnu",
        "arm64": "aarch64-linux-gnu",
    }.get(machine, f"{machine}-linux-gnu")


def _gz_plugin_dirs() -> List[str]:
    """Directories Gazebo searches for system plugins, best-guess order."""
    triplet = _lib_triplet()
    dirs: List[str] = []
    for entry in os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "").split(":"):
        if entry:
            dirs.append(entry)
    dirs.extend(
        [
            f"/usr/lib/{triplet}/gz-sim-8/plugins",
            "/usr/lib/gz-sim-8/plugins",
            "/usr/local/lib/gz-sim-8/plugins",
        ]
    )
    return [d for d in dirs if os.path.isdir(d)]


def _find_gz_plugin(short_name: str) -> Optional[str]:
    """``gz-sim-physics-system`` -> ``.../libgz-sim8-physics-system.so``."""
    suffix = short_name[len("gz-sim-"):] if short_name.startswith("gz-sim-") else short_name
    for stem in (f"libgz-sim8-{suffix}", f"libgz-sim-{suffix}"):
        for directory in _gz_plugin_dirs():
            candidate = os.path.join(directory, stem + ".so")
            if os.path.isfile(candidate):
                return candidate
    return None


def _find_render_engine(name: str) -> Optional[str]:
    triplet = _lib_triplet()
    for directory in (
        f"/usr/lib/{triplet}/gz-rendering-8/engine-plugins",
        "/usr/lib/gz-rendering-8/engine-plugins",
        "/usr/local/lib/gz-rendering-8/engine-plugins",
    ):
        for stem in (f"libgz-rendering8-{name}", f"libgz-rendering-{name}"):
            candidate = os.path.join(directory, stem + ".so")
            if os.path.isfile(candidate):
                return candidate
    return None


def _find_library(stem: str) -> Optional[str]:
    rc, out = _run(["ldconfig", "-p"], timeout=30)
    if rc == 0:
        for line in out.splitlines():
            if stem in line and "=>" in line:
                path = line.split("=>", 1)[1].strip()
                if os.path.exists(path):
                    return path
    for directory in ("/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu", "/usr/local/lib"):
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if name.startswith(stem):
                candidate = os.path.join(directory, name)
                if os.path.isfile(candidate) or os.path.islink(candidate):
                    return candidate
    return None


# --------------------------------------------------------------------------- #
# 3. ros_gz bridge compatibility
# --------------------------------------------------------------------------- #
def check_bridge_compatibility(report: Report) -> None:
    section = "ros_gz bridge compatibility"
    version = _dpkg_version("ros-humble-ros-gzharmonic")
    report.add(
        section,
        "ros-humble-ros-gzharmonic",
        PASS if version else FAIL,
        version or "not installed - this is the Harmonic-matched bridge package",
    )
    if version:
        report.add(
            section,
            "bridge release train",
            PASS if version.startswith("0.244.") else WARN,
            f"{version} (Harmonic pairing is the 0.244.x series)",
        )

    prefix = "/opt/ros/humble/lib"
    if not os.path.isdir(prefix):
        report.add(section, "ros install prefix", FAIL, prefix)
        return

    for lib in _ROS_GZ_LIBS:
        path = os.path.join(prefix, lib)
        if not os.path.isfile(path):
            report.add(section, lib, FAIL, f"{path} missing")
            continue
        rc, out = _run(["ldd", path], timeout=30)
        if rc != 0:
            report.add(section, lib, WARN, out or f"ldd rc={rc}")
            continue
        missing = [line.split()[0] for line in out.splitlines() if "not found" in line]
        gz_links = [
            line.split("=>")[0].strip()
            for line in out.splitlines()
            if "libgz-" in line and "=>" in line
        ]
        detail = ", ".join(gz_links) if gz_links else "no libgz-* link detected"
        report.add(
            section,
            f"{lib} dynamic links",
            FAIL if missing else PASS,
            f"missing: {', '.join(missing)}" if missing else detail,
        )

    # Cross-check the SONAMEs the bridge resolved against the installed gz libs
    rc, out = _run(["ldd", os.path.join(prefix, "libros_gz_bridge_lib.so")], timeout=30)
    if rc == 0:
        resolved = {}
        for line in out.splitlines():
            if "libgz-" in line and "=>" in line:
                left, right = line.split("=>", 1)
                resolved[left.strip()] = right.strip().split("(")[0].strip()
        mismatched: List[str] = []
        for name, target in resolved.items():
            if not os.path.exists(target):
                mismatched.append(f"{name} -> {target}")
        report.add(
            section,
            "bridge <-> gazebo SONAME resolution",
            FAIL if mismatched else PASS,
            "; ".join(mismatched) if mismatched else "every bridge dependency resolves on this machine",
        )


# --------------------------------------------------------------------------- #
# 4. ROS packages
# --------------------------------------------------------------------------- #
def check_packages(report: Report, install_hints: bool) -> None:
    section = "ros packages"
    try:
        from ament_index_python.packages import get_package_share_directory
    except Exception as exc:  # pragma: no cover
        report.add(section, "ament_index_python", FAIL, str(exc))
        return

    missing: List[str] = []
    for package, module, why in _PACKAGES:
        try:
            get_package_share_directory(package)
            present = True
        except Exception:
            present = False
        if present:
            report.add(section, package, PASS, f"module {module}")
        else:
            missing.append(package)
            report.add(section, package, WARN, f"module {module}: {why}")

    if missing and install_hints:
        apt_names: List[str] = []
        for package in missing:
            apt_names.extend(_APT_HINTS.get(package, f"ros-humble-{package.replace('_', '-')}").split())
        report.add(
            section,
            "install hint",
            WARN,
            "sudo apt install -y " + " ".join(sorted(set(apt_names))),
        )


# --------------------------------------------------------------------------- #
# 5. smoke tests
# --------------------------------------------------------------------------- #
def _smoke_world(path: str, world_name: str, with_camera: bool = False, camera_topic: str = "/env_check/image") -> str:
    camera_block = ""
    if with_camera:
        camera_block = """
    <model name="env_check_camera">
      <static>true</static>
      <pose>0 0 1.5 0 0.5 0</pose>
      <link name="link">
        <sensor name="rgb" type="camera">
          <always_on>1</always_on>
          <update_rate>10</update_rate>
          <visualize>false</visualize>
          <topic>env_check/image</topic>
          <camera>
            <horizontal_fov>1.2</horizontal_fov>
            <image><width>320</width><height>240</height><format>R8G8B8</format></image>
            <clip><near>0.1</near><far>100</far></clip>
          </camera>
        </sensor>
      </link>
    </model>"""
    text = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{world_name}">
    <physics name="1ms" type="dartsim">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <dart><collision_detector>bullet</collision_detector></dart>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="check_ground">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
          <material><ambient>0.8 0.75 0.55 1</ambient><diffuse>0.8 0.75 0.55 1</diffuse></material>
        </visual>
      </link>
    </model>
    <model name="check_box">
      <pose>1.5 0 0.25 0 0 0</pose>
      <link name="link">
        <inertial><mass>1.0</mass><inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.02</izz></inertia></inertial>
        <collision name="collision"><geometry><box><size>0.5 0.5 0.5</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>0.5 0.5 0.5</size></box></geometry>
          <material><ambient>1 0.2 0.2 1</ambient><diffuse>1 0.2 0.2 1</diffuse></material>
        </visual>
      </link>
    </model>{camera_block}
  </world>
</sdf>
"""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _start_process(cmd: Sequence[str], log_path: str) -> subprocess.Popen:
    handle = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        list(cmd),
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=dict(os.environ),
    )


def _stop_process(process: Optional[subprocess.Popen]) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        process.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass


def _wait_for_gz_world(gz: str, world: str, log_path: str, timeout: float = 60.0) -> Tuple[bool, str]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, out = _run([gz, "model", "--list"], timeout=15)
        if rc == 0 and world in out:
            return True, f"world '{world}' reachable"
        time.sleep(1.0)
    tail = ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
            tail = "".join(handle.readlines()[-3:]).strip()
    except OSError:
        pass
    return False, f"world '{world}' not reachable within {timeout:.0f}s. log tail: {tail}"


def _smoke_physics(report: Report, workdir: str) -> None:
    section = "smoke: gazebo physics"
    gz = shutil.which("gz")
    if not gz:
        report.add(section, "gz sim boot", SKIP, "gz CLI missing")
        return
    world_path = _smoke_world(os.path.join(workdir, "smoke.sdf"), "cat_env_check")
    log = os.path.join(workdir, "smoke.log")
    process = _start_process([gz, "sim", "-s", "-r", "-v", "1", world_path], log)
    try:
        ready, detail = _wait_for_gz_world(gz, "cat_env_check", log, timeout=60.0)
        report.add(section, "gz sim boot + world load", PASS if ready else FAIL, detail)
        if ready:
            rc, out = _run([gz, "model", "-m", "check_box", "-p"], timeout=20)
            ok = rc == 0 and "Pose" in out
            report.add(section, "entity query (gz model)", PASS if ok else FAIL, _first_line(out) if ok else out[-200:])
    finally:
        _stop_process(process)


def _smoke_bridge(report: Report, workdir: str) -> None:
    section = "smoke: ros_gz bridge"
    gz = shutil.which("gz")
    ros2 = shutil.which("ros2")
    if not gz or not ros2:
        report.add(section, "/clock bridge", SKIP, "gz or ros2 CLI missing")
        return
    world_name = "cat_env_check"
    world_path = os.path.join(workdir, "smoke.sdf")
    if not os.path.isfile(world_path):
        _smoke_world(world_path, world_name)
    log = os.path.join(workdir, "smoke_bridge.log")
    bridge_log = os.path.join(workdir, "bridge.log")
    sim = _start_process([gz, "sim", "-s", "-r", "-v", "1", world_path], log)
    bridge: Optional[subprocess.Popen] = None
    try:
        ready, detail = _wait_for_gz_world(gz, world_name, log, timeout=60.0)
        if not ready:
            report.add(section, "/clock bridge", SKIP, detail)
            return
        bridge_arg = "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
        bridge = _start_process(
            [ros2, "run", "ros_gz_bridge", "parameter_bridge", bridge_arg], bridge_log
        )
        received, latency = _await_ros_clock(timeout=25.0)
        if received:
            report.add(
                section,
                "/clock bridge",
                PASS,
                f"gz /clock (world '{world_name}') -> ROS /clock within {latency:.2f}s",
            )
        else:
            tail = ""
            try:
                with open(bridge_log, "r", encoding="utf-8", errors="ignore") as handle:
                    tail = "".join(handle.readlines()[-3:]).strip()
            except OSError:
                pass
            report.add(section, "/clock bridge", FAIL, f"no /clock message within 25s. bridge log: {tail}")
    finally:
        _stop_process(bridge)
        _stop_process(sim)


def _await_ros_clock(timeout: float) -> Tuple[bool, float]:
    """Subscribe to ``/clock`` with rclpy; return (received, seconds waited)."""
    try:
        import rclpy
        from rclpy.node import Node
        from rosgraph_msgs.msg import Clock
    except Exception as exc:  # pragma: no cover
        return False, 0.0
    rclpy.init(args=[])
    node = Node("cat_arena_env_check")
    state = {"seen": False, "t": time.time()}

    def _callback(_msg) -> None:
        state["seen"] = True
        state["t"] = time.time()

    node.create_subscription(Clock, "/clock", _callback, 10)
    deadline = time.time() + timeout
    start = time.time()
    try:
        while time.time() < deadline and not state["seen"]:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return bool(state["seen"]), state["t"] - start


def _smoke_render(report: Report, workdir: str) -> None:
    section = "smoke: headless rendering"
    gz = shutil.which("gz")
    if not gz:
        report.add(section, "camera sensor render", SKIP, "gz CLI missing")
        return
    world_name = "cat_env_check_render"
    world_path = _smoke_world(
        os.path.join(workdir, "smoke_render.sdf"), world_name, with_camera=True
    )
    log = os.path.join(workdir, "smoke_render.log")
    sim = _start_process([gz, "sim", "-s", "-r", "-v", "1", world_path], log)
    try:
        ready, detail = _wait_for_gz_world(gz, world_name, log, timeout=60.0)
        if not ready:
            report.add(section, "camera sensor render", FAIL, detail)
            return
        deadline = time.time() + 40.0
        published = False
        while time.time() < deadline:
            rc, out = _run([gz, "topic", "-l"], timeout=20)
            if rc == 0 and "image" in out:
                published = True
                break
            time.sleep(1.5)
        report.add(
            section,
            "camera sensor render",
            PASS if published else FAIL,
            "an image topic is being published (ogre2 renders headless)"
            if published
            else "no image topic appeared - camera sensors will not work; modules 3/4 need this",
        )
    finally:
        _stop_process(sim)


def _arena_check(report: Report, workdir: str) -> None:
    section = "smoke: arena generation"
    try:
        from cat_arena.generate import generate
        from cat_arena.config import load_config
    except Exception as exc:
        report.add(section, "cat_arena import", FAIL, str(exc))
        return

    try:
        cfg = load_config()
    except Exception as exc:
        report.add(section, "arena.yaml", FAIL, str(exc))
        return
    report.add(section, "arena.yaml", PASS, cfg.path)
    report.add(
        section,
        "zones partition the arena",
        PASS,
        ", ".join(f"{z.name}{z.rect.as_list()}" for z in cfg.tiling_zones()),
    )
    report.add(
        section,
        "assumptions flagged",
        WARN,
        " | ".join(cfg.assumptions_report()),
    )

    try:
        result = generate(
            seed=1, config_path=cfg.path, output_dir=workdir, validate=True, verbose=False
        )
    except Exception as exc:
        report.add(section, "generate seed 1", FAIL, str(exc))
        return
    report.add(section, "generate seed 1", PASS, result.directory)
    report.add(
        section,
        "gz sdf -k",
        PASS if result.validation_output.endswith("OK") else FAIL,
        result.validation_output or "not run",
    )
    report.add(
        section,
        "ground truth rule compliance",
        PASS if all(result.layout.boulders and result.layout.craters) else WARN,
        f"{len(result.layout.boulders)} boulders, {len(result.layout.craters)} craters, "
        f"start ({result.start_pose.x:.2f}, {result.start_pose.y:.2f}) yaw {result.start_pose.yaw_deg:.1f} deg",
    )


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="env_check",
        description="Verify the dev-laptop environment for the Caterpillar Autonomy Challenge stack.",
    )
    parser.add_argument("--no-smoke", action="store_true", help="skip the live gazebo/bridge smoke tests")
    parser.add_argument("--render", action="store_true", help="also verify headless camera rendering")
    parser.add_argument("--arena", action="store_true", help="also generate a full arena and validate it")
    parser.add_argument("--install-hints", action="store_true", help="print the apt command for missing packages")
    parser.add_argument("--json", action="store_true", help="machine readable output")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colours")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = Report()

    check_platform(report)
    check_gazebo(report)
    check_bridge_compatibility(report)
    check_packages(report, install_hints=args.install_hints)

    workdir = os.path.join(tempfile.gettempdir(), "cat_arena_env_check")
    os.makedirs(workdir, exist_ok=True)

    if not args.no_smoke:
        _smoke_physics(report, workdir)
        _smoke_bridge(report, workdir)
        if args.render:
            _smoke_render(report, workdir)
    else:
        report.add("smoke: gazebo physics", "gz sim boot + world load", SKIP, "--no-smoke")
        report.add("smoke: ros_gz bridge", "/clock bridge", SKIP, "--no-smoke")
        report.add("smoke: headless rendering", "camera sensor render", SKIP, "--no-smoke")

    if args.arena:
        _arena_check(report, workdir)
    else:
        report.add("smoke: arena generation", "arena generation", SKIP, "pass --arena to run")

    counts = report.counts()
    if args.json:
        print(json.dumps({"ok": report.ok(), "counts": counts, "results": [r.as_dict() for r in report.results]}, indent=2))
    else:
        print(report.render(use_color=not args.no_color and sys.stdout.isatty()))
    return 0 if report.ok() else 1


if __name__ == "__main__":
    raise SystemExit(main())
