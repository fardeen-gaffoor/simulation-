"""Bring up module 6: the excavation mechanism and the berm.

    ros2 launch rover_excavation excavation.launch.py
    ros2 launch rover_excavation excavation.launch.py backend:=null
    ros2 launch rover_excavation excavation.launch.py gz_berm:=false rate_hz:=20.0

One node, and what it owns is the whole point of the module boundary:

* ``excavation.py`` serves the ``WorkCycle`` action that module 5's mission
  sequencer calls, runs the ladder/belt/tub mechanism, models the sand, and
  publishes ``MechanismStatus`` and ``BermStatus``;
* it also draws the berm - an rviz marker array always, and a box in the Gazebo
  world (``gz_berm:=true`` by default) so the build is visible in a screenshot.

It does **not** drive the rover and does **not** start module 5.  A work cycle
only ever happens because module 5's sequencer asked for one, at a pose module 5
put the rover in; starting this file on its own is the right way to check that
the mechanism is wired up, and the node will simply sit stowed.

``excavation.yaml``'s ``backend.kind`` decides where the actuator commands go
(``sim`` = the bridged Gazebo joint controllers, ``hardware`` = the digger's
serial link, ``null`` = nothing).  ``backend:=`` overrides it for one launch.
"""

from __future__ import annotations

import sys
from typing import Any, List

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Any]:
    config_path = LaunchConfiguration("config").perform(context)
    use_sim_time = _as_bool(LaunchConfiguration("use_sim_time").perform(context))
    backend = LaunchConfiguration("backend").perform(context).strip()
    rate_hz = float(LaunchConfiguration("rate_hz").perform(context))
    gz_berm = LaunchConfiguration("gz_berm").perform(context)
    publish_marker = _as_bool(LaunchConfiguration("publish_marker").perform(context))

    parameters: List[Any] = [
        {
            "use_sim_time": use_sim_time,
            "publish_marker": publish_marker,
        }
    ]
    if rate_hz > 0.0:
        # Only when asked.  A declared 0.0 would *override* the node's default
        # with something it then has to clamp to its 1 Hz floor, so "0 = the
        # yaml decides" has to mean "do not declare it at all".
        parameters.append({"rate_hz": rate_hz})
    if config_path:
        parameters.append({"config": config_path})
    if backend:
        parameters.append({"backend": backend})
    if str(gz_berm).strip():
        parameters.append({"gz_berm": _as_bool(gz_berm)})

    actions: List[Any] = [
        LogInfo(
            msg=(
                "[rover_excavation] serving /excavation/work_cycle; module 6 only works when "
                "module 5's mission sequencer sends it a work cycle"
            )
        )
    ]
    if _as_bool(LaunchConfiguration("excavation").perform(context)):
        actions.append(
            Node(
                package="rover_excavation",
                executable="excavation.py",
                name="rover_excavation_mechanism",
                output="screen",
                parameters=parameters,
            )
        )
    else:
        actions.append(LogInfo(msg="[rover_excavation] excavation disabled"))
    return actions


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument(
            "config", default_value="", description="excavation.yaml (empty = the installed default)"
        ),
        DeclareLaunchArgument("use_sim_time", default_value="true", description="run on the gz clock"),
        DeclareLaunchArgument(
            "backend",
            default_value="",
            description="override backend.kind: sim | hardware | null (empty = excavation.yaml)",
        ),
        DeclareLaunchArgument(
            "rate_hz", default_value="0.0", description="control loop rate (0 = excavation.yaml)"
        ),
        DeclareLaunchArgument(
            "gz_berm",
            default_value="true",
            description="draw the berm as a box in the Gazebo world (empty = excavation.yaml)",
        ),
        DeclareLaunchArgument(
            "publish_marker", default_value="true", description="publish the berm marker array for rviz"
        ),
        DeclareLaunchArgument("excavation", default_value="true"),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])


if __name__ == "__main__":  # pragma: no cover
    print("use: ros2 launch rover_excavation excavation.launch.py", file=sys.stderr)
    raise SystemExit(1)
