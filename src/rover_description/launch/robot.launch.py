"""Real rover bring-up: model + TF + interface stubs.  NO Gazebo.

    ros2 launch rover_description robot.launch.py

Runs on the Raspberry Pi 5 (ROS 2 Humble in Docker).  It publishes the robot
model and its TF tree and starts the ``/estop`` and ``/battery/energy_wh``
interface stubs.  Camera images on ``/cam_nav``, ``/cam_tag``,
``/cam_excavation``, ``/cam_dump`` are produced by the separate hardware package
``rover_camera_driver`` (the same ROS topics the simulator bridges), so no
consumer of the topic interface changes between sim and robot.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    from rover_description.generate import default_output_dir, generate

    rover_config = LaunchConfiguration("rover_config").perform(context) or None
    robot_output_dir = LaunchConfiguration("robot_output_dir").perform(context) or default_output_dir()

    # On the robot we still expand the model (for robot_state_publisher) but we
    # never need a gz SDF.
    result = generate(config_path=rover_config, output_dir=robot_output_dir, write_sdf=False)
    with open(result.urdf_path, "r", encoding="utf-8") as handle:
        robot_description = handle.read()

    return [
        LogInfo(msg=f"[rover] robot bring-up from {result.urdf_path} (no Gazebo)"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description, "use_sim_time": False}],
        ),
        Node(
            package="rover_description",
            executable="stub_topics.py",
            name="stub_topics",
            output="screen",
            parameters=[
                {
                    "use_sim_time": False,
                    "estop_gate": True,
                    "config": rover_config or "",
                }
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    import rover_description.generate as _gen

    arguments = [
        DeclareLaunchArgument("rover_config", default_value="", description="rover.yaml override"),
        DeclareLaunchArgument(
            "robot_output_dir",
            default_value=_gen.default_output_dir(),
            description="directory for the generated rover URDF",
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])
