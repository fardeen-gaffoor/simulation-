# rover_navigation — module 5, deciding where the rover goes and how fast

Module 5 turns a destination into wheel speeds, and turns the competition task
into an order of operations. It is the only part of the stack that may write
`/cmd_vel`, and the only part that may tell module 4 what the rover is doing.

Three nodes, and the split is the design:

| node | name | job |
|---|---|---|
| `navigation.py` | `rover_navigation_navigator` | obstacle memory → costmap → A\* → pure pursuit → **`/nav/cmd_vel`** (raw, ungated) + the `NavigateTo` action |
| `velocity_gate.py` | `rover_navigation_velocity_gate` | the **single writer of `/cmd_vel`**: e-stop, staleness, localisation gating, reactive brake, clamping, rate limiting |
| `mission.py` | `rover_navigation_mission` | the behaviour sequencer: acquire → travel → dig → haul → dump → repeat → park, and the only writer of `/perception/mode` |

Everything a node does is decided by a ROS-free module (`costmap`, `planner`,
`controller`, `footprint`, `safety`, `mission`) that is unit-tested without a
simulator. The nodes are wiring. `nav2` is deliberately **not** used: it is not
installed on the rover computer, and module 5's arena is 9 × 5 m with a handful
of static obstacles, where a custom planner is both smaller and easier to reason
about.

There is **no reverse** anywhere in module 5. `cam_nav` is forward-facing with a
100° horizontal field of view mounted 18° down; reversing would drive blind, and
no scoring objective needs it. Everything that looks like "back up" is a turn.

---

## The pipeline

```
/perception/obstacles ──► ObstacleMemory ──► Costmap ──► Planner (A* + string pull)
                              ▲                             │
                              │                             ▼
                     /perception/localization_state   PurePursuitController
                              │                             │
/odom, /odometry/filtered ────┴──► PoseTracker              ▼
                                       (map_T_base)   /nav/cmd_vel ──► velocity_gate ──► /cmd_vel
                                                                          ▲
                                  /estop, /perception/*, odometry ────────┘

/mission/status ◄── mission ──► /perception/mode
                     │  └─ NavigateTo/goal ──► navigator
                     └──── WorkCycle/goal ──► module 6 (rover_excavation)
```

Module 4 owns `map -> odom`. Gazebo (or the real drive) owns
`odom -> base_link`. Module 5 publishes **no TF at all**: it composes the two
into the rover's map pose, which is the same composition module 4's obstacle
projection uses, so the obstacles and the rover's own position agree.

---

## Layout

```
rover_navigation/
  rover_navigation/            the algorithms (importable, ROS-free)
    geometry.py                wrap_angle, Pose3, rpy/quaternion maths
    config.py                  navigation.yaml loader, range-checked, derived geometry
    costmap.py                 ObstacleMemory + the inflated lethal/cost grid
    planner.py                 A* (8-connected, no corner cutting) + string pulling
    controller.py              pure pursuit with an in-place turn phase
    footprint.py               the exact rotated-rectangle tests (goal pose, in-place turn)
    safety.py                  the velocity gate's decision function
    mission.py                 the behaviour sequencer and the generated poses
    nodeutil.py                all the rclpy glue (params, clock, QoS, PoseTracker, main)
    nodes/                     one class per node, the rclpy layer
  rover_navigation_msgs/       (separate package) the wire contract
    msg/{NavigationStatus,GateStatus,MissionStatus}.msg
    action/{NavigateTo,WorkCycle}.action
  config/navigation.yaml       MODULE 5 SINGLE SOURCE OF TRUTH
  launch/navigation.launch.py  navigator + gate
  launch/mission.launch.py     the sequencer, including navigation.launch.py by default
  scripts/*.py                 the three nodes + navigation_info.py
  docs/{planner,controller,safety,mission}.md
  test/                        458 tests
```

---

## Topics and actions

| direction | name | type |
|---|---|---|
| in | `/perception/obstacles` | `rover_perception_msgs/ObstacleArray` |
| in | `/perception/localization_state` | `rover_perception_msgs/LocalizationState` |
| in | `/odom`, `/odometry/filtered` | `nav_msgs/Odometry` |
| in | `/estop` | `std_msgs/Bool` (latched) |
| out | `/nav/cmd_vel` | `geometry_msgs/Twist` — raw, ungated |
| out | `/cmd_vel` | `geometry_msgs/Twist` — the **only** writer is the gate |
| out | `/nav/path`, `/nav/costmap`, `/nav/status` | `Path`, `OccupancyGrid`, `NavigationStatus` |
| out | `/nav/gate_status` | `GateStatus` |
| out | `/mission/status` | `MissionStatus` |
| out | `/perception/mode` | `std_msgs/String` |
| out | `/mission/berm_started` | `std_msgs/Bool` (latched) — the berm is now a planner keep-out |
| server | `/nav/navigate_to` | `NavigateTo` action |
| client | `/excavation/work_cycle` | `WorkCycle` action — **module 6 implements this** |

Subscriptions are best-effort (they match both best-effort and reliable
publishers, which is what `ros_gz_bridge` produces). `/estop` and
`/mission/berm_started` are reliable + transient-local, because a latched value
is exactly the one a late subscriber must not miss.

---

## Running it

```bash
# the whole of module 5 against a running simulation
ros2 launch rover_navigation mission.launch.py

# just the driving stack, to drive it by hand
ros2 launch rover_navigation navigation.launch.py perception:=true

# drive every leg but never claim sand moved (module 6 does not exist yet)
ros2 launch rover_navigation mission.launch.py work_mode:=navigate_only

# what will actually happen: geometry, poses, and whether each pose is legal
ros2 run rover_navigation navigation_info.py
```

`navigation_info.py` is the pre-run check. It prints the resolved config, the
derived geometry and every generated dig/dump/park pose, and runs each pose
through the same exact footprint test the navigator applies to a goal — so "can
the rover legally be everywhere it is about to be told to go" is answered
without starting Gazebo.

---

## Configuration

`config/navigation.yaml` is the module's single source of truth, and it
deliberately **duplicates the arena geometry** from `cat_arena` and the measured
envelope from `rover_description`, because neither package is installed on the
Raspberry Pi. `test/test_cross_checks.py` reads the other two packages' configs
and fails if the copy drifts.

Two numbers carry most of the design:

* `robot.length_m × width_m` (1.30 × 0.70 m) is the **operational** footprint —
  module 2's measured body (1.2396 × 0.655 m) padded by `footprint_clearance_m`
  (0.06) on every side.
* `robot.build_length_m × build_width_m` (1.50 × 0.75 m) is the competition's
  hard envelope. It is kept as the ceiling and cross-checked, but *planning* for
  it makes the seeded arena impassable: a 0.84 m circumscribed circle around each
  35 cm boulder seals the 5.5 m obstacle zone. `footprint.py` covers the
  difference exactly instead of by inflation.

See `docs/planner.md`, `docs/controller.md`, `docs/safety.md` and
`docs/mission.md` for the reasoning behind each piece.

---

## Tests

```bash
colcon test --packages-select rover_navigation   # 458 tests
cd src/rover_navigation && python3 -m pytest test/ -q
```

No test reads `ground_truth.yaml`. Module 5's premise is that obstacles arrive
from live perception and nothing else, so every obstacle scenario in the tests is
synthetic and stated in the test that needs it; the one place the real world
appears is the live simulator acceptance run.
