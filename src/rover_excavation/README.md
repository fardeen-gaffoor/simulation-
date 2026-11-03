# rover_excavation — module 6

The digging mechanism and the berm: the last module of the Caterpillar Autonomy
Challenge stack, and the one that actually moves sand.

```
rover_dataset (3) ──► rover_perception (4) ──► rover_navigation (5) ──► rover_excavation (6)
   training data          live map + pose          where and when            how, and what came of it
```

## What it owns

| | |
|---|---|
| **The mechanism** | the ladder (chain + buckets) on its pitch joint, the belt roller, the tub gate, and the interlocks that stop them |
| **The sand** | how fast sand enters the tub, where it lands when the tub is emptied, and the **berm height field** over `arena.zones.berm_target` |
| **The work cycle** | `STARTING → MOVING → WORKING → RETRACTING → DONE`, served as the `WorkCycle` action module 5 calls |
| **The travel interlock** | `travel_ready` — module 5 may only drive the rover while this is true, and only module 6 knows it |

## What it does **not** do

* It never sends a wheel command. Module 5 owns `/cmd_vel`, the planner and the
  velocity gate.
* It never decides *where* the rover works. Module 5 puts the rover at the dig or
  dump pose and then sends a `WorkCycle` goal with that pose in it.
* It never reads `ground_truth.yaml`. It knows the berm target because the rules
  say where it is, and nothing else about the arena.
* It does not model soil physics. Gazebo has no sand (module 1's terrain is a
  static triangle mesh), so module 6 models the sand itself — see
  [docs/sand.md](docs/sand.md) for exactly what that means and where the model
  can be wrong.

## Packages

* **`rover_excavation`** (this one, Python + ament_cmake): the node and the models.
* **`rover_excavation_msgs`** (C++/ament_cmake): `MechanismStatus` and
  `BermStatus`.
* The *call* into module 6 — `rover_navigation_msgs/action/WorkCycle` — lives in
  module 5's message package, because module 5 is the caller and had to exist
  first. Module 6 is its server.

## Run it

In simulation, with the arena, the rover, module 4 and module 5 already up:

```bash
ros2 launch rover_excavation excavation.launch.py
```

That starts one node, `rover_excavation_mechanism`, which:

* serves `/excavation/work_cycle`,
* publishes `/excavation/status`, `/excavation/berm` and `/excavation/berm_marker`,
* publishes the actuator commands on `/excavation/ladder_cmd` and
  `/excavation/belt_cmd` (bridged into the Gazebo joint controllers), and
* raises a box out of the ground in Gazebo as the berm grows.

On its own it does nothing but sit stowed; a work cycle only happens because
module 5's mission sequencer asks for one:

```bash
ros2 launch cat_arena arena.launch.py            # 1: the world
ros2 launch rover_description sim.launch.py       # 2: the rover (bridges + gz plugins)
ros2 run rover_perception perception               # 4: live map and pose
ros2 launch rover_navigation navigation.launch.py  # 5: navigator + velocity gate
ros2 launch rover_navigation mission.launch.py     # 5: the behaviour sequencer
ros2 launch rover_excavation excavation.launch.py  # 6: the digger
```

Session options:

```bash
ros2 launch rover_excavation excavation.launch.py backend:=null       # dry run: drive nothing
ros2 launch rover_excavation excavation.launch.py gz_berm:=false      # rviz marker only
ros2 launch rover_excavation excavation.launch.py rate_hz:=20.0
ros2 launch rover_excavation excavation.launch.py config:=/path/to/excavation.yaml
```

On the rover computer (Raspberry Pi 5, no Gazebo):

```bash
ros2 run rover_excavation excavation.py --ros-args -p backend:=hardware
```

with `python3-serial` installed; `backend.kind: hardware` opens
`backend.serial.port` and speaks the digger's protocol
(see [docs/mechanism.md](docs/mechanism.md#the-serial-protocol)).

## Check it before a run

```bash
ros2 run rover_excavation excavation_info.py
```

which prints the resolved config **and dry-runs the work cycles offline**: it
plays N dig/dump pairs through the real mechanism, tub and berm models and
reports how long each cycle takes, whether the pile lands inside the target, and
what the berm comes out as.

```bash
ros2 run rover_excavation excavation_info.py --cycles 6
ros2 run rover_excavation excavation_info.py --json
ros2 run rover_excavation excavation_info.py --deps
```

## Interfaces

| topic | type | direction |
|---|---|---|
| `/excavation/work_cycle` | `rover_navigation_msgs/action/WorkCycle` | module 5 → module 6 |
| `/excavation/status` | `rover_excavation_msgs/MechanismStatus` | module 6 → everyone |
| `/excavation/berm` | `rover_excavation_msgs/BermStatus` | module 6 → everyone |
| `/excavation/berm_marker` | `visualization_msgs/MarkerArray` | module 6 → rviz |
| `/excavation/ladder_cmd` | `std_msgs/Float64` | module 6 → gz joint controller |
| `/excavation/belt_cmd` | `std_msgs/Float64` | module 6 → gz joint controller |
| `/perception/mode` | `std_msgs/String` | module 5 → everyone (module 6 only reads it for logging) |
| `/estop` | `std_msgs/Bool` (latched) | module 2 → everyone |
| `/joint_states` | `sensor_msgs/JointState` | gz → module 6 (tracking check only) |

The same topics in simulation and on the rover. Nothing above the backend layer
knows whether Gazebo or a serial link is on the other end.

## Documentation

* [docs/excavation.md](docs/excavation.md) — the node: action semantics, topic
  contract, e-stop and cancel behaviour, the integration checklist.
* [docs/mechanism.md](docs/mechanism.md) — the mechanism model, the interlocks,
  and the serial protocol.
* [docs/sand.md](docs/sand.md) — the sand and berm model, the dump-landing
  coupling to module 5, and how to re-tune it.
* [docs/cycle.md](docs/cycle.md) — the work-cycle state machine.

## Tests

```bash
colcon test --packages-select rover_excavation rover_excavation_msgs
colcon test-result --all --verbose
```

No test in this package reads `ground_truth.yaml`; one of them checks that by
scanning the source. The cross-module tests compare the copies of the arena,
robot and dump geometry against modules 1, 2 and 5 directly, and skip (rather
than pass) when those source trees are not reachable.
