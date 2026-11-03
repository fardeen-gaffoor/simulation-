"""Bring up all of module 4 against a running arena simulation.

    ros2 launch rover_perception perception.launch.py
    ros2 launch rover_perception perception.launch.py seed_hint:=1 spawn_delay:=12
    ros2 launch rover_perception perception.launch.py spawn_tags:=false     # arena without tag boards
    ros2 launch rover_perception perception.launch.py publish_tf:=true      # extra map -> odom TF

Assumes ``cat_arena`` (and, for a moving rover, module 2's ``sim.launch.py``)
already started the world: the arena owns ``gz sim``, the clock bridge and the
rover, and this file adds only what is module 4's.

Order of work:

1. **generate the tag artefacts** (``generate_tags.py``'s job, done in-process
   here): ``tags.yaml``, the 36h11 bitmaps and the board SDFs.  If no
   ``markers.yaml`` exists yet the launch logs exactly how to make one and then
   continues without tag boards - the detector and the obstacle pipeline still
   start, so a broken reveal does not turn into a mystery.
2. **spawn the textured tag boards** after ``spawn_delay`` seconds, one
   ``ros_gz_sim create`` per board, at the poses recorded in ``tags.yaml``.
3. **start the five nodes**: camera manager, tag detector, tag localizer,
   obstacle detector, health monitor.

``publish_tf`` is **false** by default: in simulation the gz diff-drive plugin
already publishes ``odom -> base_link`` on the bridged ``/tf``, so module 4 must
not publish that edge as well.  The ``map -> odom`` edge, which gz knows nothing
about, is the one module 4 owns - and ``localization.launch.py`` (the real-robot
entry point) turns it on.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_NODES = {
    "camera_manager": "camera_manager.py",
    "tag_detector": "tag_detector.py",
    "tag_localizer": "tag_localizer.py",
    "obstacle_detector": "obstacle_detector.py",
    "health_monitor": "health_monitor.py",
}


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _default_tags_dir() -> str:
    override = os.environ.get("CAT_PERCEPTION_ARTIFACTS", "")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    return os.path.join(os.path.expanduser("~"), "cat", "perception")


def _generate_tags(config_path: str, markers: str, out_dir: str, world: str) -> Any:
    """Write tags.yaml + bitmaps + board SDFs; returns ``(boards, tags_path, error)``."""
    from rover_perception import tagboard, tags
    from rover_perception.config import load_config
    from rover_perception.markers_ref import MarkersError, load_markers

    config = load_config(config_path or None)
    try:
        markers_doc = load_markers(markers or config.artifacts.markers or None)
    except MarkersError as exc:
        return None, "", str(exc)
    document = tags.build_document(markers_doc, config.tags.board, frame_id=config.arena.frame_id)
    tags_path = os.path.join(out_dir, config.artifacts.tags_file or "tags.yaml")
    tags.write_catalogue(tags_path, document)
    catalogue = tags.load_tags(tags_path)
    boards = tagboard.build_plan(catalogue, config.tags.board, out_dir, write=True)
    tagboard.write_plan(
        os.path.join(out_dir, "tag_boards.yaml"),
        tagboard.plan_document(boards, out_dir, world=world),
    )
    return boards, tags_path, ""


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Any]:
    config_path = LaunchConfiguration("config").perform(context)
    world = LaunchConfiguration("world").perform(context)
    markers = LaunchConfiguration("markers").perform(context)
    tags_dir = os.path.abspath(
        os.path.expanduser(LaunchConfiguration("tags_dir").perform(context) or _default_tags_dir())
    )
    delay = float(LaunchConfiguration("spawn_delay").perform(context))
    use_sim_time = _as_bool(LaunchConfiguration("use_sim_time").perform(context))
    generate_tags = _as_bool(LaunchConfiguration("generate_tags").perform(context))
    spawn_tags = _as_bool(LaunchConfiguration("spawn_tags").perform(context))
    publish_tf = _as_bool(LaunchConfiguration("publish_tf").perform(context))

    actions: List[Any] = []
    tags_path = ""
    if generate_tags:
        try:
            boards, tags_path, error = _generate_tags(config_path, markers, tags_dir, world)
        except Exception as exc:  # noqa: BLE001 - the launch must explain, not die
            boards, tags_path, error = None, "", f"{type(exc).__name__}: {exc}"
        if error:
            actions.append(
                LogInfo(
                    msg=(
                        f"[rover_perception] tag artefacts not generated ({error}); "
                        "generate an arena first ('ros2 launch cat_arena arena.launch.py') "
                        f"or pass markers:=/path/to/markers.yaml. Continuing without tag boards."
                    )
                )
            )
        else:
            actions.append(
                LogInfo(
                    msg=(
                        f"[rover_perception] wrote {tags_path} and {len(boards or [])} tag boards "
                        f"into {tags_dir}"
                    )
                )
            )
            if spawn_tags and boards:
                commands = []
                for board in boards:
                    command = [
                        "ros2",
                        "run",
                        "ros_gz_sim",
                        "create",
                        "-world",
                        world,
                        "-file",
                        board.sdf_path,
                        "-name",
                        board.name,
                        "-x",
                        f"{board.position[0]:.6f}",
                        "-y",
                        f"{board.position[1]:.6f}",
                        "-z",
                        f"{board.position[2]:.6f}",
                        "-R",
                        f"{board.rpy[0]:.9f}",
                        "-P",
                        f"{board.rpy[1]:.9f}",
                        "-Y",
                        f"{board.rpy[2]:.9f}",
                        "-allow_renaming",
                        "false",
                    ]
                    commands.append(
                        ExecuteProcess(cmd=command, output="screen", name=f"spawn_{board.name}")
                    )
                (x, y, z) = boards[0].position
                actions.append(
                    LogInfo(
                        msg=(
                            f"[rover_perception] spawning {len(commands)} tag boards into "
                            f"'{world}' after {delay:.1f}s (first at "
                            f"{x:.3f}, {y:.3f}, {z:.3f})"
                        )
                    )
                )
                actions.append(TimerAction(period=delay, actions=commands))
            elif spawn_tags:
                actions.append(LogInfo(msg="[rover_perception] no tag boards to spawn"))

    parameters: List[Any] = [{"use_sim_time": use_sim_time}]
    if config_path:
        parameters.append({"config": config_path})
    if tags_path:
        parameters.append({"tags_file": tags_path})

    for name, executable in _NODES.items():
        enabled = _as_bool(LaunchConfiguration(name).perform(context))
        if not enabled:
            actions.append(LogInfo(msg=f"[rover_perception] {name} disabled"))
            continue
        node_parameters: List[Any] = list(parameters)
        if name == "tag_localizer":
            node_parameters.append({"publish_tf": publish_tf})
        actions.append(
            Node(
                package="rover_perception",
                executable=executable,
                name=f"perception_{name}",
                output="screen",
                parameters=node_parameters,
            )
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument(
            "config", default_value="", description="perception.yaml (empty = the installed default)"
        ),
        DeclareLaunchArgument(
            "world", default_value="cat_arena", description="gz world/partition name to spawn tags into"
        ),
        DeclareLaunchArgument(
            "markers",
            default_value="",
            description="markers.yaml (empty = the newest generated arena)",
        ),
        DeclareLaunchArgument(
            "tags_dir",
            default_value=_default_tags_dir(),
            description="where tags.yaml and the board textures are written",
        ),
        DeclareLaunchArgument(
            "generate_tags",
            default_value="true",
            description="write tags.yaml + bitmaps + board SDFs before starting the nodes",
        ),
        DeclareLaunchArgument(
            "spawn_tags",
            default_value="true",
            description="spawn the textured tag boards into the running world",
        ),
        DeclareLaunchArgument(
            "spawn_delay",
            default_value="10.0",
            description="seconds to wait before spawning (gz sim boot time)",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true", description="run the nodes on the gz clock"
        ),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="false",
            description="also broadcast map -> odom (leave false in sim: gz publishes odom -> base_link)",
        ),
        DeclareLaunchArgument("camera_manager", default_value="true"),
        DeclareLaunchArgument("tag_detector", default_value="true"),
        DeclareLaunchArgument("tag_localizer", default_value="true"),
        DeclareLaunchArgument("obstacle_detector", default_value="true"),
        DeclareLaunchArgument("health_monitor", default_value="true"),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])


if __name__ == "__main__":  # pragma: no cover
    print("use: ros2 launch rover_perception perception.launch.py", file=sys.stderr)
    raise SystemExit(1)
