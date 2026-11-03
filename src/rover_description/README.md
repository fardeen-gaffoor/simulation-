# rover_description (Module 2)

The rover model for the Caterpillar Autonomy Challenge berm-construction rover:
one Xacro description that produces the URDF, the Gazebo SDF, the single camera
calibration file, the mass budget and the stowed-envelope report, plus the
interface stubs (`/estop`, `/battery/energy_wh`) the rest of the stack depends on.

Everything is derived from one file: [`config/rover.yaml`](config/rover.yaml).
There are no hand-maintained numbers anywhere else.

```text
config/rover.yaml ──► rover_description.generate ──► inertias.yaml  (xacro input)
                                                  ├► rover.urdf     (RSP + gz spawn)
                                                  ├► rover.sdf      (inspection/dev)
                                                  ├► calibration.yaml  (module 4)
                                                  ├► mass_budget.yaml
                                                  ├► envelope.yaml
                                                  └► manifest.json
```

## The rover

| property            | value                                                        |
| ------------------- | ------------------------------------------------------------ |
| drivetrain          | 4-wheel skid steer, `gz-sim-diff-drive-system`               |
| chassis             | 0.70 x 0.60 x 0.28 m box, centre 0.34 m above the wheel plane |
| wheels              | radius 0.10 m, width 0.05 m, track 0.60 m, wheelbase 0.60 m  |
| excavation          | ladder (pitch joint) → belt (roller joint) → tub → funnel    |
| mass (structural + payload) | 59.75 kg (competition limit 80 kg)                   |
| stowed envelope     | 1.240 x 0.655 x 0.607 m (limit 1.5 x 0.75 x 0.75 m)          |
| cameras             | 4 x plain RGB (no depth), see below                          |
| IMU                 | 3-axis gyro + accelerometer, **no magnetometer**             |
| odometry            | wheel odometry from the diff-drive plugin                    |

### Cameras

| name             | where                    | resolution | HFOV | mount                    |
| ---------------- | ------------------------ | ---------- | ---- | ------------------------ |
| `cam_nav`        | front, high, tilted      | 640x480    | 100° | 18° down                 |
| `cam_tag`        | front, level, high-res   | 1280x720   | 65°  | 0°, at most 60 cm high   |
| `cam_excavation` | over the ladder          | 640x480    | 90°  | 50° down                 |
| `cam_dump`       | rear, over the tub       | 640x480    | 90°  | 180° yaw, 40° down       |

Each camera publishes `sensor_msgs/Image` + `sensor_msgs/CameraInfo` on
`/cam_<name>` and `/cam_<name>/camera_info`. The image `frame_id` is the camera's
optical frame (`<name>_optical_frame`), a child of the camera link with
`rpy = (-π/2, 0, -π/2)`, so the two ROS conventions (URDF link axes vs. optical
axes) are handled in the model and nothing downstream has to think about it.

## Frame conventions

* The robot's **declared forward is `+x`** (competition requires the team to
  declare it; assumption A2). `base_link` is at `+x` forward, `+y` left, `+z` up.
* `base_link`'s origin is the projection of the chassis centre onto the ground
  plane, so the wheel contact plane is `z = 0`.
* Rotations use ZYX rpy; **positive pitch tilts local `+x` down** (used for both
  the ladder and the camera mounts).

## Build

```bash
cd ~/cat
source /opt/ros/humble/setup.bash
colcon build --packages-select rover_description --symlink-install
source install/setup.bash
```

## Generate the artifacts

```bash
ros2 run rover_description generate_robot.py --output-dir /tmp/rover
```

An explicit path is passed to xacro for both the config and the inertias, so the
generated URDF never depends on the source tree.

## Check the competition limits

```bash
ros2 run rover_description check_envelope.py       # envelope + mass, exit 0/1
ros2 run rover_description mass_budget.py          # the mass table
```

## Run in the simulator

```bash
# GUI, random seed
ros2 launch rover_description sim.launch.py

# reproducible + headless (renders camera sensors offscreen)
ros2 launch rover_description sim.launch.py seed:=7 headless:=true headless_rendering:=true
```

`sim.launch.py` generates the rover, includes `cat_arena/arena.launch.py` with
`robot:=<rover.urdf>` and `robot_plugins:=true` (which adds the gz **sensors** and
**IMU** world systems), then starts `robot_state_publisher`, the `ros_gz_bridge`
(below) and the interface stubs. The rover spawns at the arena's seeded start
pose.

Drive it:

```bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.4}}'
```

## Run on the robot

```bash
ros2 launch rover_description robot.launch.py     # model + TF + stubs, no Gazebo
```

## Inspect the model

```bash
ros2 launch rover_description description.launch.py   # RViz: model, TF, 4 images
```

## Simulator ↔ ROS interface

The complete interface is [`config/bridges.yaml`](config/bridges.yaml) (13 bridges;
`/clock` is owned by `cat_arena`):

| ROS topic                    | type                            | direction |
| ---------------------------- | ------------------------------- | --------- |
| `/cam_{nav,tag,excavation,dump}` | `sensor_msgs/Image`          | gz → ros  |
| `/cam_{...}/camera_info`     | `sensor_msgs/CameraInfo`        | gz → ros  |
| `/imu`                       | `sensor_msgs/Imu`               | gz → ros  |
| `/odom`                      | `nav_msgs/Odometry`             | gz → ros  |
| `/tf`                        | `tf2_msgs/TFMessage`            | gz → ros  |
| `/joint_states`              | `sensor_msgs/JointState`        | gz → ros  |
| `/cmd_vel`                   | `geometry_msgs/Twist`           | ros → gz  |

These are the **same ROS topics on the real rover**; on the robot the images come
from `rover_camera_driver` and `/odom`/`/tf`/`/joint_states` from the drive
firmware. No consumer of the interface changes between sim and robot.

The TF tree is `odom → base_link → {wheels, ladder, belt, tub, funnel, cameras,
imu_link}`: `odom → base_link` comes from the diff-drive plugin, everything else
from `robot_state_publisher` / the URDF.

## Interface stubs

`stub_topics` owns the two topics the rest of the stack needs but that only exist
fully with hardware:

* `/battery/energy_wh` (`std_msgs/Float64`) — remaining pack energy, published at
  1 Hz, drained at `battery.idle_draw_w`.
* `/estop` (`std_msgs/Bool`, **latched**) — the e-stop state. The node publishes
  it (so late subscribers see the current value), mirrors any value published by
  another node, and while it is latched publishes a zero `/cmd_vel` at
  `gate_rate_hz` as a backstop.

The authoritative stop is the autonomy subscribing to `/estop` (module 5); the
zero-command gate is a simulator/hardware-independent backstop, not a substitute
for a hardware e-stop chain.

## Layout

```text
rover_description/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   ├── rover.yaml            # single source of truth
│   └── bridges.yaml          # the sim <-> ROS topic table
├── urdf/
│   ├── macros.xacro          # robot-agnostic macros
│   └── rover.urdf.xacro      # the model
├── rviz/rover.rviz
├── launch/
│   ├── sim.launch.py
│   ├── robot.launch.py
│   └── description.launch.py
├── scripts/
│   ├── generate_robot.py
│   ├── check_envelope.py
│   ├── mass_budget.py
│   └── stub_topics.py
├── rover_description/
│   ├── config.py             # typed + validated config
│   ├── geometry.py           # rpy, inertias, envelope
│   ├── calibration.py        # intrinsics + extrinsics
│   ├── mass.py               # mass budget
│   ├── bridges.py            # bridges.yaml loader
│   ├── envelope.py           # envelope check
│   ├── generate.py           # the orchestrator
│   └── stub_topics.py        # the interface stub node
└── test/                     # pytest suite
```

## Tests

```bash
colcon test --packages-select rover_description && colcon test-result --verbose
```

Covers the config validation, the analytic inertias, the envelope, the URDF
structure and the competition bans (only `camera`/`imu` sensors; no `gps`,
`magnetometer`, …), byte-for-byte generation determinism, the bridge table, the
launch wiring and the stub node.

## Module 1 integration

`cat_arena` gained one backwards-compatible option, `robot_plugins` (default
`false`), which adds `gz-sim-sensors-system` **and** `gz-sim-imu-system` to the
world. `sim.launch.py` sets it to `true`; module 1 behaviour is otherwise
unchanged.
