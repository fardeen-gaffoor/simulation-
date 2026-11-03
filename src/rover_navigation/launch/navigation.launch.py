"""Bring up module 5's driving stack: the navigator and the velocity gate.

    ros2 launch rover_navigation navigation.launch.py
    ros2 launch rover_navigation navigation.launch.py publish_costmap:=false
    ros2 launch rover_navigation navigation.launch.py navigation:=false

Two nodes, and the split between them is the point:

* ``navigation.py`` turns a destination into a *raw* command on ``/nav/cmd_vel``
  - obstacle memory, costmap, A*, pure pursuit, and the ``NavigateTo`` action
  that is the only way anything asks the rover to move;
* ``velocity_gate.py`` is the **single writer of ``/cmd_vel``**.  It forwards the
  raw command only when it is not e-stopped, not stale, backed by a tag-anchored
  map, and free of anything within the reactive brake distance.

Assumes the arena is already up (``cat_arena``/``rover_description`` in
simulation, or the real driver on the rover computer) and that module 4 is
publishing ``/perception/obstacles`` and ``/perception/localization_state``.
With ``perception:=true`` this file also starts module 4 first, which is the
quick way to try the whole thing in the simulator.

This file deliberately does **not** start the mission sequencer: that is
``mission.launch.py``, and keeping them apart means the navigator can be driven
by hand (``ros2 action send_goal /nav/navigate_to ...``) without a sequencer
sending it work.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_NODES = {
    "navigation": "navigation.py",
    "velocity_gate": "velocity_gate.py",
}


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Any]:
    config_path = LaunchConfiguration("config").perform(context)
    use_sim_time = _as_bool(LaunchConfiguration("use_sim_time").perform(context))
    publish_costmap = _as_bool(LaunchConfiguration("publish_costmap").perform(context))
    publish_status = _as_bool(LaunchConfiguration("publish_status").perform(context))
    start_perception = _as_bool(LaunchConfiguration("perception").perform(context))
    perception_delay = float(LaunchConfiguration("perception_delay").perform(context))
    spawn_delay = LaunchConfiguration("spawn_delay").perform(context)

    actions: List[Any] = []
    if start_perception:
        # Module 4's launch file already knows how to generate and spawn the tag
        # boards; include it rather than duplicating any of that here.
        try:
            from ament_index_python.packages import get_package_share_directory

            perception_launch = os.path.join(
                get_package_share_directory("rover_perception"), "launch", "perception.launch.py"
            )
        except Exception as exc:  # noqa: BLE001 - the launch must explain, not die
            actions.append(
                LogInfo(
                    msg=(
                        f"[rover_navigation] could not find rover_perception's launch file "
                        f"({type(exc).__name__}: {exc}); starting only module 5"
                    )
                )
            )
        else:
            actions.append(
                LogInfo(msg="[rover_navigation] starting module 4 (perception) as well")
            )
            actions.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(perception_launch),
                    launch_arguments={
                        "use_sim_time": "true" if use_sim_time else "false",
                        "spawn_delay": str(spawn_delay),
                    }.items(),
                )
            )

    parameters: List[Any] = [{"use_sim_time": use_sim_time, "publish_costmap": publish_costmap}]
    if config_path:
        parameters.append({"config": config_path})

    for name, executable in _NODES.items():
        if not _as_bool(LaunchConfiguration(name).perform(context)):
            actions.append(LogInfo(msg=f"[rover_navigation] {name} disabled"))
            continue
        node_parameters: List[Any] = list(parameters)
        if name == "velocity_gate":
            node_parameters.append({"publish_status": publish_status})
        actions.append(
            Node(
                package="rover_navigation",
                executable=executable,
                name=f"rover_navigation_{name}",
                output="screen",
                parameters=node_parameters,
            )
        )
    if start_perception and perception_delay > 0.0:
        actions.append(
            LogInfo(
                msg=(
                    f"[rover_navigation] module 5 subscribes to module 4's topics as soon as it "
                    f"starts; the rover will hold (no map pose) until the camera manager and tag "
                    f"localizer are up and a tag has been seen - roughly {perception_delay:.0f}s "
                    "after Gazebo boots."
                )
            )
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument(
            "config", default_value="", description="navigation.yaml (empty = the installed default)"
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true", description="run the nodes on the gz clock"
        ),
        DeclareLaunchArgument(
            "perception", default_value="false", description="also start module 4 (rover_perception)"
        ),
        DeclareLaunchArgument(
            "perception_delay",
            default_value="0.0",
            description="(informational) seconds module 4 needs before it publishes a map pose",
        ),
        DeclareLaunchArgument(
            "spawn_delay",
            default_value="10.0",
            description="passed to module 4: seconds before it spawns the tag boards",
        ),
        DeclareLaunchArgument(
            "publish_costmap", default_value="true", description="publish /nav/costmap (OccupancyGrid)"
        ),
        DeclareLaunchArgument(
            "publish_status", default_value="true", description="publish /nav/gate_status"
        ),
        DeclareLaunchArgument("navigation", default_value="true"),
        DeclareLaunchArgument("velocity_gate", default_value="true"),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])


if __name__ == "__main__":  # pragma: no cover
    print("use: ros2 launch rover_navigation navigation.launch.py", file=sys.stderr)
    raise SystemExit(1)
