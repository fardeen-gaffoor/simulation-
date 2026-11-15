"""Full simulation: a seeded ``cat_arena`` world + the ``rover_description`` rover.

    ros2 launch rover_description sim.launch.py
    ros2 launch rover_description sim.launch.py seed:=7 headless:=true headless_rendering:=true
    ros2 launch rover_description sim.launch.py low_res:=true

What it does, in order:

1. generates the rover (``rover_description.generate``) into
   ``robot_output_dir`` - produces ``rover.urdf`` + ``calibration.yaml``;
2. includes ``cat_arena/arena.launch.py`` with ``robot:=<rover.urdf>`` and
   ``robot_plugins:=true``, so the arena world carries the gz sensor + IMU
   systems the rover needs and the rover spawns at the sampled start pose;
3. starts ``robot_state_publisher`` (URDF -> TF);
4. starts ``ros_gz_bridge`` from ``config/bridges.yaml`` (4 x image +
   camera_info, imu, odom, tf, joint_states, cmd_vel in);
5. starts the ``/estop`` + ``/battery/energy_wh`` interface stub;
6. starts ``skid_steer_controller`` (maps ``/cmd_vel`` to four wheel velocity
   commands).

"""

from __future__ import annotations

import os
import tempfile
from typing import List

import yaml

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import rover_description.config as _cfg_mod
from rover_description.config import default_config_path
from rover_description.generate import default_output_dir, generate


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context, *args, **kwargs):
    from rover_description.bridges import bridge_arguments, load_bridges

    rover_config = LaunchConfiguration("rover_config").perform(context) or None
    robot_output_dir = LaunchConfiguration("robot_output_dir").perform(context) or default_output_dir()
    bridges_file = LaunchConfiguration("bridges").perform(context) or None
    estop_gate = _as_bool(LaunchConfiguration("estop_gate").perform(context))
    low_res = _as_bool(LaunchConfiguration("low_res").perform(context))

    # ---- low-resolution camera option ----------------------------------
    # When low_res:=true, halve every camera width/height in a temp copy of
    # rover.yaml so the generated URDF carries low-res cameras.
    if low_res:
        import copy as _copy
        src = rover_config or default_config_path()
        with open(src, "r", encoding="utf-8") as handle:
            raw = _copy.deepcopy(yaml.safe_load(handle))
        for cam_entry in raw.get("sensors", {}).get("cameras", []):
            cam_entry["width"] = max(160, int(cam_entry.get("width", 640)) // 2)
            cam_entry["height"] = max(120, int(cam_entry.get("height", 480)) // 2)
        tf = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.safe_dump(raw, tf)
        tf.close()
        rover_config = tf.name

    result = generate(config_path=rover_config, output_dir=robot_output_dir, write_sdf=False)
    urdf_path = result.urdf_path
    with open(urdf_path, "r", encoding="utf-8") as handle:
        robot_description = handle.read()

    bridge_args: List[str] = bridge_arguments(load_bridges(bridges_file))

    arena_launch = os.path.join(
        FindPackageShare("cat_arena").perform(context), "launch", "arena.launch.py"
    )

    actions = [
        LogInfo(
            msg=(
                f"[rover] generated {urdf_path}: envelope {result.envelope.measured.as_tuple()} "
                f"(limit {result.envelope.limit.as_tuple()}), mass {result.mass.total_mass:.2f} kg "
                f"(limit {result.mass.mass_limit:.1f} kg)"
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(arena_launch),
            launch_arguments={
                "seed": LaunchConfiguration("seed").perform(context),
                "config": LaunchConfiguration("arena_config").perform(context),
                "output_dir": LaunchConfiguration("arena_output_dir").perform(context),
                "world": LaunchConfiguration("world").perform(context),
                "headless": LaunchConfiguration("headless").perform(context),
                "headless_rendering": LaunchConfiguration("headless_rendering").perform(context),
                "verbose": LaunchConfiguration("verbose").perform(context),
                "validate_world": LaunchConfiguration("validate_world").perform(context),
                "spawn_delay": LaunchConfiguration("spawn_delay").perform(context),
                "use_sim_time": "true",
                "bridge_clock": "true",
                "publish_initial_pose": "true",
                "spawn_robot": "true",
                "robot": urdf_path,
                "robot_name": "rover",
                "robot_plugins": "true",
            }.items(),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description, "use_sim_time": True}],
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="rover_bridge",
            output="screen",
            arguments=bridge_args,
            parameters=[{"use_sim_time": True}],
        ),
        Node(
            package="rover_description",
            executable="stub_topics.py",
            name="stub_topics",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "estop_gate": estop_gate,
                    "config": rover_config or "",
                }
            ],
        ),
        # -- skid-steer controller: /cmd_vel -> four wheel velocities ------
        Node(
            package="rover_description",
            executable="skid_steer_controller.py",
            name="skid_steer_controller",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "rover_config": rover_config or "",
                    "effective_track_correction": LaunchConfiguration(
                        "skid_steer_effective_track_correction"
                    ).perform(context),
                    "max_wheel_velocity": LaunchConfiguration(
                        "max_wheel_velocity"
                    ).perform(context),
                }
            ],
        ),
        # -- motor + battery model: wheel velocities -> current, torque, energy ---
        Node(
            package="rover_sim_tools",
            executable="motor_battery_model.py",
            name="motor_battery_model",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "rover_config": rover_config or "",
                    "motors_config": "",
                    "battery_config": "",
                    "soft_start_tau_s": 0.5,
                    "pwm_duty_cap": 1.0,
                    "rolling_resistance_rr": 0.25,
                    "publish_rate_hz": 50.0,
                }
            ],
        ),
        # -- excavation mechanism: manual dig/dump (V/B in the teleop) --------
        Node(
            package="rover_excavation",
            executable="excavation.py",
            name="rover_excavation_mechanism",
            output="screen",
            parameters=[
                {
                    "use_sim_time": True,
                    "backend": "sim",
                    "publish_marker": True,
                    "manual_topic": "/excavation/manual",
                }
            ],
        ),
        # Keyboard teleop is NOT started here: it needs a real terminal for raw
        # keystrokes.  Run it in a second terminal with:
        #   ros2 run rover_description teleop_keyboard.py
    ]
    return actions


def generate_launch_description() -> LaunchDescription:
    import rover_description.generate as _gen

    arguments = [
        DeclareLaunchArgument("seed", default_value="-1", description="arena run seed (-1 = random)"),
        DeclareLaunchArgument("headless", default_value="false", description="gz sim server-only"),
        DeclareLaunchArgument(
            "headless_rendering",
            default_value="false",
            description="with headless:=true, render camera sensors offscreen",
        ),
        DeclareLaunchArgument("verbose", default_value="2", description="gz sim verbosity"),
        DeclareLaunchArgument("validate_world", default_value="true", description="run 'gz sdf -k'"),
        DeclareLaunchArgument("spawn_delay", default_value="8.0", description="seconds before spawning"),
        DeclareLaunchArgument("arena_config", default_value="", description="arena.yaml override"),
        DeclareLaunchArgument(
            "arena_output_dir", default_value="", description="cat_arena world output directory"
        ),
        DeclareLaunchArgument("world", default_value="", description="reuse an existing arena.sdf"),
        DeclareLaunchArgument("rover_config", default_value="", description="rover.yaml override"),
        DeclareLaunchArgument(
            "robot_output_dir",
            default_value=_gen.default_output_dir(),
            description="directory for the generated rover URDF/calibration",
        ),
        DeclareLaunchArgument("bridges", default_value="", description="bridges.yaml override"),
        DeclareLaunchArgument(
            "estop_gate",
            default_value="true",
            description="stub_topics zeroes /cmd_vel while /estop is active",
        ),
        DeclareLaunchArgument(
            "low_res",
            default_value="false",
            description="halve camera resolutions for faster rendering",
        ),
        DeclareLaunchArgument(
            "skid_steer_effective_track_correction",
            default_value="2.0",
            description="effective track = track * this factor (skid-steer turn gain)",
        ),
        DeclareLaunchArgument(
            "max_wheel_velocity",
            default_value="12.0",
            description="maximum wheel angular velocity (rad/s)",
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])
