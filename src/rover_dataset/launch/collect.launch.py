"""One collection scene: a randomized world + the rover + the collector node.

    ros2 launch rover_dataset collect.launch.py scene_dir:=... scene_id:=... seed:=7

It includes ``rover_description/sim.launch.py`` (which generates the rover and
includes ``cat_arena/arena.launch.py``) with this scene's randomized
``arena.yaml`` and ``rover.yaml``, then adds ``collector_node.py``.  When the
collector exits the whole launch shuts down, which is how ``collect_dataset.py``
knows a scene is finished.
"""

from __future__ import annotations

import os
from typing import List

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    Shutdown,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _setup(context, *args, **kwargs):
    scene_dir = os.path.abspath(os.path.expanduser(LaunchConfiguration("scene_dir").perform(context)))
    if not scene_dir or scene_dir == os.path.abspath("."):
        raise RuntimeError("collect.launch.py requires scene_dir:=<path>")
    seed = LaunchConfiguration("seed").perform(context)
    scene_id = LaunchConfiguration("scene_id").perform(context)
    scene_index = LaunchConfiguration("scene_index").perform(context)
    master_seed = LaunchConfiguration("master_seed").perform(context)
    dataset_root = os.path.abspath(os.path.expanduser(LaunchConfiguration("dataset_root").perform(context)))

    arena_config = LaunchConfiguration("arena_config").perform(context) or os.path.join(scene_dir, "arena.yaml")
    rover_config = LaunchConfiguration("rover_config").perform(context) or os.path.join(scene_dir, "rover.yaml")
    arena_output_dir = LaunchConfiguration("arena_output_dir").perform(context) or os.path.join(scene_dir, "world")
    robot_output_dir = LaunchConfiguration("robot_output_dir").perform(context) or os.path.join(scene_dir, "rover")
    report = LaunchConfiguration("report").perform(context) or os.path.join(scene_dir, "collect_report.json")
    ground_truth = os.path.join(arena_output_dir, f"seed_{seed}", "ground_truth.yaml")
    calibration = os.path.join(robot_output_dir, "calibration.yaml")

    cameras_text = LaunchConfiguration("cameras").perform(context)
    if cameras_text:
        cameras: List[str] = [name.strip() for name in cameras_text.split(",") if name.strip()]
    else:
        from rover_dataset.config import load_config

        config_path = LaunchConfiguration("dataset_config").perform(context) or None
        cameras = list(load_config(config_path).capture.cameras)

    sim_launch = os.path.join(
        FindPackageShare("rover_description").perform(context), "launch", "sim.launch.py"
    )

    collector = Node(
        package="rover_dataset",
        executable="collector_node.py",
        name="rover_dataset_collector",
        output="screen",
        parameters=[
            {
                "scene_id": scene_id,
                "scene_index": int(scene_index),
                "source": "sim",
                "master_seed": int(master_seed),
                "ground_truth": ground_truth,
                "calibration": calibration,
                "dataset_root": dataset_root,
                "report": report,
                "cameras": cameras,
                "poses_per_scene": int(LaunchConfiguration("poses").perform(context)),
                "frames_per_pose": int(LaunchConfiguration("frames").perform(context)),
                "settle_seconds": float(LaunchConfiguration("settle_seconds").perform(context)),
                "frame_timeout": float(LaunchConfiguration("frame_timeout").perform(context)),
                "dry_run": LaunchConfiguration("dry_run").perform(context).lower() == "true",
            }
        ],
        on_exit=[Shutdown(reason=f"scene {scene_id} collected")],
    )

    return [
        LogInfo(
            msg=(
                f"[collect] scene {scene_id} (seed {seed}) -> {dataset_root}\n"
                f"[collect]   arena : {arena_config}\n"
                f"[collect]   rover : {rover_config}\n"
                f"[collect]   cams  : {cameras}"
            )
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_launch),
            launch_arguments={
                "seed": seed,
                "arena_config": arena_config,
                "arena_output_dir": arena_output_dir,
                "rover_config": rover_config,
                "robot_output_dir": robot_output_dir,
                "headless": LaunchConfiguration("headless").perform(context),
                "headless_rendering": LaunchConfiguration("headless_rendering").perform(context),
                "spawn_delay": LaunchConfiguration("spawn_delay").perform(context),
                "verbose": LaunchConfiguration("verbose").perform(context),
                "validate_world": LaunchConfiguration("validate_world").perform(context),
                "estop_gate": "false",
            }.items(),
        ),
        collector,
    ]


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("scene_dir", default_value="", description="scene directory"),
        DeclareLaunchArgument("scene_id", default_value="sim_0000"),
        DeclareLaunchArgument("scene_index", default_value="0"),
        DeclareLaunchArgument("seed", default_value="7", description="arena seed"),
        DeclareLaunchArgument("master_seed", default_value="1000", description="dataset master seed"),
        DeclareLaunchArgument("dataset_root", default_value="dataset"),
        DeclareLaunchArgument("dataset_config", default_value=""),
        DeclareLaunchArgument("arena_config", default_value=""),
        DeclareLaunchArgument("rover_config", default_value=""),
        DeclareLaunchArgument("arena_output_dir", default_value=""),
        DeclareLaunchArgument("robot_output_dir", default_value=""),
        DeclareLaunchArgument("report", default_value=""),
        DeclareLaunchArgument("cameras", default_value=""),
        DeclareLaunchArgument("poses", default_value="0"),
        DeclareLaunchArgument("frames", default_value="0"),
        DeclareLaunchArgument("settle_seconds", default_value="-1.0"),
        DeclareLaunchArgument("frame_timeout", default_value="5.0"),
        DeclareLaunchArgument("dry_run", default_value="false"),
        DeclareLaunchArgument("headless", default_value="true"),
        DeclareLaunchArgument("headless_rendering", default_value="true"),
        DeclareLaunchArgument("spawn_delay", default_value="10.0"),
        DeclareLaunchArgument("verbose", default_value="1"),
        DeclareLaunchArgument("validate_world", default_value="true"),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_setup)])
