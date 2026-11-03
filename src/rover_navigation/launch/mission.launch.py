"""Bring up the run: module 5's behaviour sequencer, plus the driving stack.

    ros2 launch rover_navigation mission.launch.py
    ros2 launch rover_navigation mission.launch.py navigate_only:=true
    ros2 launch rover_navigation mission.launch.py autostart:=false
    ros2 launch rover_navigation mission.launch.py navigation:=false   # sequencer only

With ``navigation:=true`` (the default) this includes ``navigation.launch.py``,
so one command brings up the whole of module 5.

``work_mode`` decides what happens at a dig or dump pose:

* ``delegate`` - the competition setting.  The sequencer sends a ``WorkCycle``
  goal to module 6 on ``/excavation/work_cycle`` and counts a cycle only when
  module 6 says it happened.  If no server is present it says so and holds.
* ``navigate_only`` - drive every leg, never claim a cycle happened.  This is how
  module 5 is exercised in the simulator before module 6 exists.  The status
  message always says which mode is in force, so a ``navigate_only`` run can
  never be mistaken for a real one.

``autostart`` waits ``start_delay`` seconds and then begins the run.  The delay
exists because the rover's map pose only becomes available once module 4 has
seen a wall tag; starting before that simply spends the acquire budget waiting.
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


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Any]:
    config_path = LaunchConfiguration("config").perform(context)
    use_sim_time = _as_bool(LaunchConfiguration("use_sim_time").perform(context))
    with_navigation = _as_bool(LaunchConfiguration("navigation").perform(context))
    perception = _as_bool(LaunchConfiguration("perception").perform(context))
    autostart = _as_bool(LaunchConfiguration("autostart").perform(context))
    start_delay = float(LaunchConfiguration("start_delay").perform(context))
    work_mode = LaunchConfiguration("work_mode").perform(context).strip()

    actions: List[Any] = []
    if with_navigation:
        try:
            from ament_index_python.packages import get_package_share_directory

            navigation_launch = os.path.join(
                get_package_share_directory("rover_navigation"), "launch", "navigation.launch.py"
            )
        except Exception as exc:  # noqa: BLE001
            actions.append(
                LogInfo(
                    msg=(
                        f"[rover_navigation] could not find navigation.launch.py "
                        f"({type(exc).__name__}: {exc}); starting only the sequencer"
                    )
                )
            )
        else:
            actions.append(
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(navigation_launch),
                    launch_arguments={
                        "config": config_path,
                        "use_sim_time": "true" if use_sim_time else "false",
                        "perception": "true" if perception else "false",
                    }.items(),
                )
            )

    if work_mode:
        actions.append(
            LogInfo(
                msg=(
                    f"[rover_navigation] mission work_mode '{work_mode}': "
                    + (
                        "work cycles are delegated to module 6 on /excavation/work_cycle"
                        if work_mode == "delegate"
                        else "every leg will be driven but no cycle will be counted or claimed"
                    )
                )
            )
        )

    parameters: List[Any] = [{"use_sim_time": use_sim_time}]
    if config_path:
        parameters.append({"config": config_path})
    if work_mode:
        # A node parameter would need the config loader to accept overrides; the
        # loader is the single source of truth, so the mode is overridden here by
        # rewriting the value the node starts with instead of by a second path.
        parameters.append({"work_mode_override": work_mode})
    parameters.append({"autostart": autostart})
    parameters.append({"start_delay_s": start_delay})

    actions.append(
        Node(
            package="rover_navigation",
            executable="mission.py",
            name="rover_navigation_mission",
            output="screen",
            parameters=parameters,
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
            "navigation",
            default_value="true",
            description="also start the navigator and the velocity gate",
        ),
        DeclareLaunchArgument(
            "perception",
            default_value="false",
            description="also start module 4 (only meaningful with navigation:=true)",
        ),
        DeclareLaunchArgument(
            "autostart", default_value="true", description="begin the run after start_delay_s"
        ),
        DeclareLaunchArgument(
            "start_delay",
            default_value="8.0",
            description="seconds to wait before autostart (module 4 needs a tag fix first)",
        ),
        DeclareLaunchArgument(
            "work_mode",
            default_value="",
            description="override mission.work_mode: 'delegate' or 'navigate_only' (empty = config)",
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])


if __name__ == "__main__":  # pragma: no cover
    print("use: ros2 launch rover_navigation mission.launch.py", file=sys.stderr)
    raise SystemExit(1)
