"""Localisation only: fused odometry + AprilTag fixes -> ``map -> odom``.

    ros2 launch rover_perception localization.launch.py                    # the real robot
    ros2 launch rover_perception localization.launch.py ekf:=false          # gz odom, no EKF
    ros2 launch rover_perception localization.launch.py publish_tf:=false   # topic-only correction

This is the entry point for the rover computer (no Gazebo there): it starts

* ``robot_localization``'s ``ekf_filter_node`` with ``config/ekf.yaml`` - wheel
  odometry + IMU, no magnetometer, ``two_d_mode``, output on
  ``/odometry/filtered``, and
* the two nodes that anchor the arena frame: ``tag_detector`` (AprilTags on
  ``cam_tag``) and ``tag_localizer`` (tag sighting -> ``map -> odom``).

Two defaults differ from ``perception.launch.py`` and both are deliberate:

* ``publish_tf:=true``.  On the robot nothing else publishes ``map -> odom`` or
  ``odom -> base_link``, so the EKF and the localizer each publish their own
  edge.  In simulation gz owns ``odom -> base_link``, which is why the sim entry
  point leaves both off.
* ``use_sim_time:=false``.

``robot_localization`` is an optional dependency (it is not installed on the
development laptop, which runs the simulation on gz odometry instead), so when
it is missing the launch says exactly what to install and carries on with the
``/odom`` fallback that ``localization.odometry_topics`` already lists - the tag
fix chain keeps working, only the filtered odometry is absent.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_DEFAULT_SHARE = None
try:  # ament index is optional so the file can be linted outside a ROS env
    from ament_index_python.packages import get_package_share_directory

    _DEFAULT_SHARE = get_package_share_directory("rover_perception")
except Exception:  # pragma: no cover
    _DEFAULT_SHARE = None


def _share_dir() -> str:
    if _DEFAULT_SHARE:
        return _DEFAULT_SHARE
    return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _robot_localization_available() -> bool:
    try:
        from ament_index_python.packages import get_package_share_directory

        get_package_share_directory("robot_localization")
        return True
    except Exception:
        return False


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> List[Any]:
    config_path = LaunchConfiguration("config").perform(context)
    ekf_config = LaunchConfiguration("ekf_config").perform(context)
    use_ekf = _as_bool(LaunchConfiguration("ekf").perform(context))
    publish_tf = _as_bool(LaunchConfiguration("publish_tf").perform(context))
    use_sim_time = _as_bool(LaunchConfiguration("use_sim_time").perform(context))
    tags_file = LaunchConfiguration("tags_file").perform(context)
    detector = _as_bool(LaunchConfiguration("tag_detector").perform(context))

    actions: List[Any] = []
    parameters: List[Any] = [{"use_sim_time": use_sim_time}]
    if config_path:
        parameters.append({"config": config_path})
    if tags_file:
        parameters.append({"tags_file": tags_file})

    if use_ekf:
        if _robot_localization_available():
            actions.append(
                Node(
                    package="robot_localization",
                    executable="ekf_node",
                    name="ekf_filter_node",
                    output="screen",
                    parameters=[
                        ekf_config,
                        {"publish_tf": publish_tf, "use_sim_time": use_sim_time},
                    ],
                )
            )
            actions.append(
                LogInfo(
                    msg=(
                        f"[rover_perception] ekf_filter_node started with {ekf_config} "
                        f"(publish_tf={publish_tf})"
                    )
                )
            )
        else:
            actions.append(
                LogInfo(
                    msg=(
                        "[rover_perception] robot_localization is not installed: "
                        "'sudo apt install ros-humble-robot-localization' to fuse /odom + /imu. "
                        "Continuing with /odom directly (localization.odometry_topics lists it)."
                    )
                )
            )
    else:
        actions.append(LogInfo(msg="[rover_perception] EKF disabled (ekf:=false)"))

    if detector:
        actions.append(
            Node(
                package="rover_perception",
                executable="tag_detector.py",
                name="perception_tag_detector",
                output="screen",
                parameters=parameters,
            )
        )
    actions.append(
        Node(
            package="rover_perception",
            executable="tag_localizer.py",
            name="perception_tag_localizer",
            output="screen",
            parameters=list(parameters) + [{"publish_tf": publish_tf}],
        )
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    default_config = os.path.join(_share_dir(), "config", "perception.yaml")
    default_ekf = os.path.join(_share_dir(), "config", "ekf.yaml")
    arguments = [
        DeclareLaunchArgument(
            "config", default_value=default_config, description="perception.yaml"
        ),
        DeclareLaunchArgument(
            "ekf_config", default_value=default_ekf, description="robot_localization ekf.yaml"
        ),
        DeclareLaunchArgument("ekf", default_value="true", description="start the robot_localization EKF"),
        DeclareLaunchArgument(
            "publish_tf",
            default_value="true",
            description="EKF publishes odom -> base_link and the localizer map -> odom",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false", description="run on the /clock topic (simulation only)"
        ),
        DeclareLaunchArgument(
            "tags_file",
            default_value="",
            description="tags.yaml (empty = artifacts.output_dir + artifacts.tags_file)",
        ),
        DeclareLaunchArgument("tag_detector", default_value="true", description="start the tag detector"),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])


if __name__ == "__main__":  # pragma: no cover
    print("use: ros2 launch rover_perception localization.launch.py", file=sys.stderr)
    raise SystemExit(1)
