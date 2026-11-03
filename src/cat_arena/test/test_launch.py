"""Launch file tests.

The launch file is the only supported way to start a run, so what it *does* is
part of the contract: it must generate the world for the run seed before starting
Gazebo, spawn the robot with the exact ``-x -y -z -Y`` the generator sampled, and
publish that same pose on ``/cat_arena/initial_pose``.

These tests build the launch description and call the setup function directly
(no processes are started - ``ExecuteProcess`` entities are only descriptions).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
from typing import List

import pytest

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, TimerAction
from launch.utilities import perform_substitutions
from launch_ros.utilities import evaluate_parameters

from cat_arena.config import load_config
from cat_arena.start_pose import sample_start_pose

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAUNCH = os.path.join(os.path.dirname(_HERE), "launch", "arena.launch.py")

_DEFAULTS = {
    "seed": "17",
    "config": "",
    "output_dir": "",
    "world": "",
    "headless": "true",
    "headless_rendering": "false",
    "verbose": "1",
    "validate_world": "false",
    "spawn_robot": "true",
    "robot": "",
    "robot_name": "rover",
    "spawn_delay": "0.1",
    "publish_initial_pose": "true",
    "bridge_clock": "true",
    "use_sim_time": "false",
    "gui_fullscreen": "false",
    "robot_plugins": "false",
}


@dataclasses.dataclass
class Setup:
    context: LaunchContext
    actions: list

    def flatten(self) -> List[object]:
        return _flatten(self.actions)

    def processes(self) -> list:
        """``ExecuteProcess`` entities that are not ros2 nodes.

        ``launch_ros``'s ``Node`` subclasses ``ExecuteProcess``, so a plain
        ``isinstance`` check would pick the pose node up as well.
        """
        return [
            entity
            for entity in self.flatten()
            if type(entity).__name__ == "ExecuteProcess"
        ]

    def commands(self):
        return [
            [perform_substitutions(self.context, part) for part in entity.cmd]
            for entity in self.processes()
        ]

    def nodes(self):
        return [e for e in self.flatten() if type(e).__name__ == "Node"]

    def nodes_of(self, package: str):
        return [node for node in self.nodes() if node.node_package == package]

    def bridge_nodes(self):
        """The ``ros_gz_bridge`` nodes (currently the clock bridge)."""
        return self.nodes_of("ros_gz_bridge")

    def pose_nodes(self):
        """The ``cat_arena`` start-pose publisher node."""
        return self.nodes_of("cat_arena")

    def logs(self):
        return [e for e in self.flatten() if type(e).__name__ == "LogInfo"]


@pytest.fixture(scope="module")
def launch_module():
    spec = importlib.util.spec_from_file_location("cat_arena_arena_launch", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arguments(description) -> dict:
    context = LaunchContext()
    out = {}
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            out[entity.name] = (
                perform_substitutions(context, entity.default_value)
                if entity.default_value is not None
                else None
            )
    return out


def _setup(launch_module, **overrides) -> Setup:
    config = dict(_DEFAULTS)
    config.update(overrides)
    context = LaunchContext()
    context.launch_configurations.update(config)
    return Setup(context=context, actions=launch_module._launch_setup(context))


def _flatten(entities) -> List[object]:
    out: List[object] = []
    for entity in entities:
        out.append(entity)
        nested = getattr(entity, "_TimerAction__actions", None)
        if nested:
            out.extend(_flatten(nested))
    return out


# --------------------------------------------------------------------------- #
# declared arguments
# --------------------------------------------------------------------------- #
def test_launch_lists_every_documented_argument(launch_module):
    args = _arguments(launch_module.generate_launch_description())
    for name in (
        "seed",
        "config",
        "output_dir",
        "headless",
        "spawn_robot",
        "robot",
        "robot_name",
        "spawn_delay",
        "publish_initial_pose",
        "bridge_clock",
        "use_sim_time",
        "validate_world",
        "robot_plugins",
    ):
        assert name in args, f"launch argument '{name}' is missing"


def test_default_seed_is_random(launch_module):
    args = _arguments(launch_module.generate_launch_description())
    assert int(args["seed"]) < 0, "the default must draw a random seed per run"


def test_default_robot_is_the_bundled_test_proxy(launch_module):
    args = _arguments(launch_module.generate_launch_description())
    assert args["robot"].endswith(os.path.join("assets", "test_proxy", "model.sdf"))
    assert os.path.isfile(args["robot"])


def test_default_output_dir_is_overridable_by_env(launch_module, monkeypatch, tmp_path):
    monkeypatch.setenv("CAT_ARENA_WORLDS", str(tmp_path / "worlds"))
    args = _arguments(launch_module.generate_launch_description())
    assert args["output_dir"] == str(tmp_path / "worlds")


def test_launch_description_contains_a_setup_function(launch_module):
    description = launch_module.generate_launch_description()
    assert any(isinstance(e, OpaqueFunction) for e in description.entities)


# --------------------------------------------------------------------------- #
# what the setup function builds
# --------------------------------------------------------------------------- #
def test_setup_generates_the_world_then_starts_gazebo(launch_module, tmp_path):
    setup = _setup(launch_module, output_dir=str(tmp_path))
    gz = [c for c in setup.commands() if c[0] == "gz"]
    assert len(gz) == 1
    cmd = gz[0]
    assert cmd[:2] == ["gz", "sim"]
    assert "-s" in cmd, "headless:=true must produce a server-only gz sim"
    assert cmd[-1].endswith("arena.sdf")
    assert os.path.isfile(cmd[-1]), "the world must exist before gz sim is started"


def test_setup_spawns_the_robot_at_the_sampled_pose(launch_module, tmp_path, config_path):
    setup = _setup(launch_module, output_dir=str(tmp_path))
    create = [c for c in setup.commands() if "ros_gz_sim" in c and "create" in c]
    assert len(create) == 1, "exactly one spawn command is expected"
    cmd = create[0]

    expected = sample_start_pose(load_config(config_path), 17)
    mapping = {"-x": expected.x, "-y": expected.y, "-z": expected.z, "-Y": expected.yaw}
    for flag, value in mapping.items():
        assert flag in cmd, f"spawn command is missing {flag}"
        assert float(cmd[cmd.index(flag) + 1]) == pytest.approx(value, abs=1e-6)
    assert cmd[cmd.index("-world") + 1] == "cat_arena"
    assert cmd[cmd.index("-file") + 1].endswith("model.sdf")
    assert cmd[cmd.index("-name") + 1] == "rover"


def test_setup_bridges_the_clock(launch_module, tmp_path):
    setup = _setup(launch_module, output_dir=str(tmp_path))
    bridges = setup.bridge_nodes()
    assert len(bridges) == 1
    bridge = bridges[0]
    assert bridge.node_executable == "parameter_bridge"
    arguments = [
        part.perform(setup.context) if hasattr(part, "perform") else str(part)
        for part in bridge._Node__arguments
    ]
    assert arguments == ["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]


def test_setup_publishes_the_start_pose_with_the_run_seed(launch_module, tmp_path):
    setup = _setup(launch_module, output_dir=str(tmp_path))
    nodes = setup.pose_nodes()
    assert len(nodes) == 1
    node = nodes[0]
    assert node.node_package == "cat_arena"
    assert node.node_executable == "publish_start_pose.py"
    parameters = evaluate_parameters(setup.context, node._Node__parameters)
    assert parameters[0]["seed"] == pytest.approx(17)
    assert parameters[0]["use_sim_time"] is False


def test_spawn_can_be_disabled(launch_module, tmp_path):
    commands = _setup(launch_module, output_dir=str(tmp_path), spawn_robot="false").commands()
    assert not [c for c in commands if "create" in c]


def test_pose_publication_can_be_disabled(launch_module, tmp_path):
    setup = _setup(launch_module, output_dir=str(tmp_path), publish_initial_pose="false")
    assert setup.pose_nodes() == []


def test_clock_bridge_can_be_disabled(launch_module, tmp_path):
    setup = _setup(launch_module, output_dir=str(tmp_path), bridge_clock="false")
    assert setup.bridge_nodes() == []


def test_spawn_is_deferred_until_gazebo_is_up(launch_module, tmp_path):
    actions = _setup(launch_module, output_dir=str(tmp_path), spawn_delay="3.5").actions
    timers = [a for a in actions if isinstance(a, TimerAction)]
    assert len(timers) == 1
    assert float(timers[0]._TimerAction__period) == pytest.approx(3.5)


def test_headless_false_starts_the_gui(launch_module, tmp_path):
    gz = [c for c in _setup(launch_module, output_dir=str(tmp_path), headless="false").commands()
          if c[0] == "gz"][0]
    assert "-s" not in gz


def test_use_sim_time_is_forwarded_to_the_node(launch_module, tmp_path):
    setup = _setup(launch_module, output_dir=str(tmp_path), use_sim_time="true")
    node = setup.pose_nodes()[0]
    parameters = evaluate_parameters(setup.context, node._Node__parameters)
    assert parameters[0]["use_sim_time"] is True


def test_existing_world_skips_generation(launch_module, tmp_path, generated):
    setup = _setup(launch_module, output_dir=str(tmp_path), world=generated.world_path)
    gz = [c for c in setup.commands() if c[0] == "gz"][0]
    assert gz[-1] == generated.world_path
    assert not [c for c in setup.commands() if "create" in c], "spawning needs the sampled pose"


def test_missing_robot_model_is_reported_not_crashed(launch_module, tmp_path):
    setup = _setup(launch_module, output_dir=str(tmp_path), robot=str(tmp_path / "nope.sdf"))
    messages = [perform_substitutions(setup.context, log.msg) for log in setup.logs()]
    assert any("nothing will be spawned" in m for m in messages)
    assert not [c for c in setup.commands() if "create" in c]


def test_setup_logs_the_seed_and_the_start_pose(launch_module, tmp_path):
    setup = _setup(launch_module, output_dir=str(tmp_path))
    messages = " | ".join(perform_substitutions(setup.context, log.msg) for log in setup.logs())
    assert "seed 17" in messages
    assert "start pose" in messages


def _generated_world_text(output_dir: str) -> str:
    import glob

    worlds = sorted(glob.glob(os.path.join(output_dir, "seed_*", "arena.sdf")))
    assert worlds, "no arena.sdf was generated"
    with open(worlds[0], "r", encoding="utf-8") as handle:
        return handle.read()


def test_robot_plugins_off_keeps_the_world_module1(launch_module, tmp_path):
    # default: no IMU system (module 1 world), whatever the overview camera does
    _setup(launch_module, output_dir=str(tmp_path))
    assert "gz-sim-imu-system" not in _generated_world_text(str(tmp_path))


def test_robot_plugins_on_adds_sensors_and_imu_systems(launch_module, tmp_path):
    # module 2 spawns a rover with cameras + IMU, so the world needs both systems
    _setup(launch_module, output_dir=str(tmp_path), robot_plugins="true")
    world = _generated_world_text(str(tmp_path))
    assert "gz-sim-sensors-system" in world
    assert "gz-sim-imu-system" in world
