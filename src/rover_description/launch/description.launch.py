"""Inspect the rover model: robot_state_publisher + RViz.  No simulation.

    ros2 launch rover_description description.launch.py
    ros2 launch rover_description description.launch.py use_joint_state_publisher:=true

Useful for eyeballing the URDF, the TF tree and the four camera frames.  With
``use_joint_state_publisher:=true`` the wheel/ladder/belt joints get fake states
so they can be swept in RViz (requires the ``joint_state_publisher`` package,
which may be absent - the launch degrades gracefully).
"""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context, *args, **kwargs):
    from rover_description.generate import default_output_dir, generate

    rover_config = LaunchConfiguration("rover_config").perform(context) or None
    robot_output_dir = LaunchConfiguration("robot_output_dir").perform(context) or default_output_dir()
    use_jsp = _as_bool(LaunchConfiguration("use_joint_state_publisher").perform(context))

    result = generate(config_path=rover_config, output_dir=robot_output_dir, write_sdf=False)
    with open(result.urdf_path, "r", encoding="utf-8") as handle:
        robot_description = handle.read()

    actions = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
    ]

    if use_jsp:
        try:
            from ament_index_python.packages import get_package_share_directory

            get_package_share_directory("joint_state_publisher")
            actions.append(
                Node(
                    package="joint_state_publisher",
                    executable="joint_state_publisher",
                    name="joint_state_publisher",
                    output="screen",
                )
            )
        except Exception:
            actions.append(
                LogInfo(msg="[rover] joint_state_publisher not installed; skipping (RViz shows static pose)")
            )

    rviz_config = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "rviz", "rover.rviz"
    )
    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", os.path.normpath(rviz_config)],
        )
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    import rover_description.generate as _gen

    arguments = [
        DeclareLaunchArgument("rover_config", default_value="", description="rover.yaml override"),
        DeclareLaunchArgument(
            "robot_output_dir",
            default_value=_gen.default_output_dir(),
            description="directory for the generated rover URDF",
        ),
        DeclareLaunchArgument(
            "use_joint_state_publisher",
            default_value="false",
            description="start joint_state_publisher to move the joints in RViz",
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])
