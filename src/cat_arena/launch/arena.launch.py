"""Launch a seeded arena and spawn the rover at a random pose in the start zone.

    ros2 launch cat_arena arena.launch.py                     # random seed, GUI
    ros2 launch cat_arena arena.launch.py seed:=7             # reproducible
    ros2 launch cat_arena arena.launch.py seed:=7 headless:=true
    ros2 launch cat_arena arena.launch.py seed:=7 robot:=/path/to/rover.sdf

What it does, in order:

1. generates the world for the run seed (``cat_arena.generate``) - synchronous,
   so ``gz sim`` never starts before ``arena.sdf`` and ``terrain.obj`` exist,
2. starts ``gz sim`` on that world (GUI by default, ``headless:=true`` for the server only),
3. bridges ``/world/<world>/clock`` to ROS ``/clock``,
4. spawns the robot at the sampled start pose via ``ros_gz_sim create`` after
   ``spawn_delay`` seconds, with the exact ``-x -y -z -Y`` the generator sampled,
5. publishes that pose on ``/cat_arena/initial_pose`` (transient-local) for the
   localisation and navigation modules.

``robot`` defaults to the bundled :file:`test_proxy/model.sdf` test fixture so
that module 1 is verifiable on its own.  Module 2 passes the real rover model.
"""

from __future__ import annotations

import os
import sys

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    Shutdown,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_DEFAULT_SHARE = None
try:  # ament index is optional so the file can be linted outside a ROS env
    from ament_index_python.packages import get_package_share_directory

    _DEFAULT_SHARE = get_package_share_directory("cat_arena")
except Exception:  # pragma: no cover
    _DEFAULT_SHARE = None


def _default_output_dir() -> str:
    return os.environ.get(
        "CAT_ARENA_WORLDS", os.path.join(os.path.expanduser("~"), "cat", "worlds")
    )


def _default_robot_model() -> str:
    """Path of the bundled spawn proxy (module 2 passes the real rover SDF).

    ``assets/`` sits next to ``launch/`` both in the source tree and in the
    installed share directory, so the relative fallback works in both.
    """
    candidates = []
    if _DEFAULT_SHARE:
        candidates.append(os.path.join(_DEFAULT_SHARE, "assets", "test_proxy", "model.sdf"))
    candidates.append(
        os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "test_proxy", "model.sdf")
        )
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return ""


def _as_bool(text: str) -> bool:
    return str(text).strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context, *args, **kwargs):
    from cat_arena.generate import generate

    seed = int(LaunchConfiguration("seed").perform(context))
    config = LaunchConfiguration("config").perform(context) or None
    output_dir = LaunchConfiguration("output_dir").perform(context) or _default_output_dir()
    world_override = LaunchConfiguration("world").perform(context)
    headless = _as_bool(LaunchConfiguration("headless").perform(context))
    headless_rendering = _as_bool(LaunchConfiguration("headless_rendering").perform(context))
    verbose = int(LaunchConfiguration("verbose").perform(context))
    spawn_delay = float(LaunchConfiguration("spawn_delay").perform(context))
    spawn_robot = _as_bool(LaunchConfiguration("spawn_robot").perform(context))
    robot = LaunchConfiguration("robot").perform(context) or _default_robot_model()
    robot_name = LaunchConfiguration("robot_name").perform(context)
    use_sim_time = _as_bool(LaunchConfiguration("use_sim_time").perform(context))
    bridge_clock = _as_bool(LaunchConfiguration("bridge_clock").perform(context))
    publish_pose = _as_bool(LaunchConfiguration("publish_initial_pose").perform(context))
    gui_fullscreen = _as_bool(LaunchConfiguration("gui_fullscreen").perform(context))

    robot_plugins = _as_bool(LaunchConfiguration("robot_plugins").perform(context))

    validate = _as_bool(LaunchConfiguration("validate_world").perform(context))

    actions = []
    if world_override:
        world = os.path.abspath(os.path.expanduser(world_override))
        world_name = os.path.splitext(os.path.basename(world))[0]
        resolved_seed = seed
        start_pose = None
        actions.append(LogInfo(msg=f"[cat_arena] using existing world {world}"))
    else:
        result = generate(
            seed=seed,
            config_path=config,
            output_dir=output_dir,
            overview_camera=None,
            robot_plugins=robot_plugins,
            validate=validate,
            verbose=False,
        )
        world = result.world_path
        world_name = result.config.world_name
        resolved_seed = result.seed
        start_pose = result.start_pose
        actions.append(
            LogInfo(
                msg=(
                    f"[cat_arena] seed {result.seed}: world {world} | "
                    f"boulders {len(result.layout.boulders)}, craters {len(result.layout.craters)}"
                )
            )
        )
        actions.append(
            LogInfo(
                msg=(
                    f"[cat_arena] start pose x={start_pose.x:.4f} y={start_pose.y:.4f} "
                    f"z={start_pose.z:.4f} yaw={start_pose.yaw_deg:.2f} deg "
                    f"(zone '{start_pose.zone}', declared forward "
                    f"{start_pose.declared_forward_deg} deg)"
                )
            )
        )

    # -- gazebo ---------------------------------------------------------------
    gz_cmd = ["gz", "sim", "-v", str(verbose)]
    if gui_fullscreen:
        gz_cmd.append("-f")
    if headless:
        gz_cmd.append("-s")
        if headless_rendering:
            gz_cmd.append("--headless-rendering")
    gz_cmd.extend(["-r", world])
    actions.append(
        ExecuteProcess(cmd=gz_cmd, output="screen", on_exit=Shutdown(reason="gz sim exited"))
    )

    # -- clock bridge ---------------------------------------------------------
    # gz publishes the world clock both as /world/<name>/clock and as /clock;
    # bridging /clock is what ROS 2 tooling and use_sim_time expect.
    if bridge_clock:
        # launched as a Node (not `ros2 run ...`) so SIGINT is handled by rclcpp
        # and the bridge shuts down cleanly instead of being SIGTERM-escalated
        actions.append(
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="cat_arena_clock_bridge",
                arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
                output="log",
            )
        )

    # -- spawn the robot ------------------------------------------------------
    if spawn_robot and not world_override:
        if not robot or not os.path.isfile(robot):
            actions.append(
                LogInfo(
                    msg=(
                        "[cat_arena] spawn_robot is true but no robot model was found "
                        f"(robot='{robot}'); nothing will be spawned"
                    )
                )
            )
        else:
            # ``spawn_robot.py`` retries the create service until the world
            # has finished loading; ``ros_gz_sim create`` gives up after 5 s
            # and leaves the arena without a robot on slow machines.
            spawn_cmd = [
                "ros2",
                "run",
                "cat_arena",
                "spawn_robot.py",
                "-world",
                world_name,
                "-file",
                robot,
                "-name",
                robot_name,
                "-x",
                f"{start_pose.x:.6f}",
                "-y",
                f"{start_pose.y:.6f}",
                "-z",
                f"{start_pose.z:.6f}",
                "-R",
                "0.0",
                "-P",
                "0.0",
                "-Y",
                f"{start_pose.yaw:.6f}",
                "-timeout",
                "120.0",
            ]
            actions.append(
                LogInfo(
                    msg=(
                        f"[cat_arena] spawning '{robot_name}' from {robot} at "
                        f"({start_pose.x:.3f}, {start_pose.y:.3f}, {start_pose.z:.3f}) "
                        f"yaw {start_pose.yaw_deg:.1f} deg after {spawn_delay:.1f}s"
                    )
                )
            )
            actions.append(
                TimerAction(
                    period=spawn_delay,
                    actions=[ExecuteProcess(cmd=spawn_cmd, output="screen")],
                )
            )

    # -- start pose on a topic ------------------------------------------------
    if publish_pose:
        parameters = {
            "seed": resolved_seed,
            "use_sim_time": use_sim_time,
            "frame_id": "map",
        }
        if config:
            parameters["config"] = config
        actions.append(
            Node(
                package="cat_arena",
                executable="publish_start_pose.py",
                name="cat_arena_start_pose",
                output="screen",
                parameters=[parameters],
            )
        )
    return actions


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument(
            "seed",
            default_value="-1",
            description="run seed; -1 (default) draws a random seed, which is logged",
        ),
        DeclareLaunchArgument(
            "config", default_value="", description="arena.yaml override (default: installed config)"
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value=_default_output_dir(),
            description="parent directory for generated worlds (<dir>/seed_<N>)",
        ),
        DeclareLaunchArgument(
            "world", default_value="", description="skip generation and use this arena.sdf"
        ),
        DeclareLaunchArgument(
            "headless", default_value="false", description="run gz sim server-only (no GUI)"
        ),
        DeclareLaunchArgument(
            "headless_rendering",
            default_value="false",
            description="with headless:=true, also enable --headless-rendering for camera sensors",
        ),
        DeclareLaunchArgument("verbose", default_value="2", description="gz sim verbosity"),
        DeclareLaunchArgument(
            "validate_world", default_value="true", description="run 'gz sdf -k' on the generated world"
        ),
        DeclareLaunchArgument(
            "spawn_robot", default_value="true", description="spawn a robot model at the sampled pose"
        ),
        DeclareLaunchArgument(
            "robot",
            default_value=_default_robot_model(),
            description="robot SDF (default: the bundled module-1 test proxy)",
        ),
        DeclareLaunchArgument("robot_name", default_value="rover", description="spawned model name"),
        DeclareLaunchArgument(
            "spawn_delay",
            default_value="8.0",
            description="seconds to wait before spawning (gz sim boot time)",
        ),
        DeclareLaunchArgument(
            "publish_initial_pose", default_value="true", description="publish /cat_arena/initial_pose"
        ),
        DeclareLaunchArgument(
            "bridge_clock", default_value="true", description="bridge the gz clock to /clock"
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="false", description="publish the start pose with sim time"
        ),
        DeclareLaunchArgument(
            "gui_fullscreen", default_value="false", description="start the gz GUI fullscreen"
        ),
        DeclareLaunchArgument(
            "robot_plugins",
            default_value="false",
            description=(
                "add the gz camera (sensors) and IMU world systems so a spawned "
                "robot's camera/IMU sensors work (module 2 sets this true)"
            ),
        ),
    ]
    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup)])


if __name__ == "__main__":  # pragma: no cover
    print("use: ros2 launch cat_arena arena.launch.py", file=sys.stderr)
    raise SystemExit(1)
