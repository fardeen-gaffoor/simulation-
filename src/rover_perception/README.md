# rover_perception — module 4, camera-only perception

Module 4 turns the rover's four cameras into the three things modules 5 and
beyond are allowed to consume:

1. a **camera manager** that keeps the rover inside the 4000 kbps average comms
   rule and reacts to thermal throttling, by commanding a per-phase frame-rate
   policy,
2. an **obstacle map** — boulders and craters, in arena (map) coordinates,
   confidence-weighted, tracked over time, with a conservative bias for craters,
3. a **map ⇄ odom anchor** — AprilTags on the walls turn into a `map -> odom`
   correction whose covariance grows while the rover dead-reckons and collapses
   on every accepted fix. There is **no AMCL** in this stack.

There is **no depth sensor anywhere**: the only senses are the four cameras and
wheel odometry + IMU. Both the classical and the learned detector are camera-only,
and every obstacle position is measured live. The competition forbids giving the
autonomy a precomputed obstacle list, so this package never reads
`ground_truth.yaml`, `labels/`, `data.yaml` or `scenes.json`.

---

## The pipeline

```
/cam_nav ──► obstacle_detector ──► tracker ──► wall mask ──► /perception/obstacles
   │              │ classical (numpy)                                                  │
   │              └ onnx (module 3's YOLOv8n)                                           ▼
   │                                        flat-ground projection  ◄── calibration.yaml
   │                                                                    (module 2)
   ├──► camera_manager  ──► /perception/camera_phase  ──► rover_camera_driver (real HW)
   │         │ status_topic, thermal, 4000 kbps meter
   │         └──► /perception/health ──► /diagnostics
/cam_tag ──► tag_detector ──► /perception/tag_detections ─┐
                                                          ├──► tag_localizer
/odom, /odometry/filtered (EKF) ──────────────────────────┘         │
                                                                    ▼
                                       /perception/localization_state + map->odom TF
```

Module 4 owns the *map frame*. It never owns the *rover*: it consumes odometry
and produces a correction, so a wrong perception can only slow the rover down,
not drive it into anything.

---

## Layout

```
rover_perception/
  rover_perception/               the algorithms (importable, ROS-free)
    config.py                     perception.yaml loader, range-checked
    geometry.py                   Pose3, rpy/quaternion maths
    calibration_ref.py            module 2's calibration.yaml reader
    markers_ref.py                module 1's markers.yaml reader
    imagecodec.py                 Image <-> numpy without cv_bridge/cv2
    camera_manager.py             phase + bandwidth + thermal state machine
    bandwidth.py                  the sliding-window 4000 kbps meter
    thermal.py                    /sys thermal-zone monitor
    cpu.py                        fps, latency, /proc CPU meters
    detector.py                   Detection, Backend, create_backend, annotate
    backends/classical.py         numpy-only segmentation backend
    backends/onnx_yolo.py         YOLOv8 ONNX backend (letterbox/decode/NMS)
    groundproj.py                 pixel -> arena ground projection
    tracking.py                   multi-frame Tracker, confidence decay
    obstacles.py                  Obstacle, arena-wall mask, PointCloud2, covariance
    tagcodes.py                   the tag36h11 code table + payload -> modules
    tagimages.py                  bitmap renderer (Pillow + numpy)
    tags.py                       markers.yaml -> tags.yaml catalogue
    tagboard.py                   thin textured box per board + spawn plan
    tagdetect.py                  dt-apriltags wrapper (camera-frame poses)
    localizer.py                  tag fix -> map->odom, OdomBuffer
    deps.py                       dependency probe for perception_info.py
    nodeutil.py                   all the rclpy glue (params, time, QoS, main)
    nodes/                        one class per node (+ main), the rclpy layer
  config/
    perception.yaml               MODULE 4 SINGLE SOURCE OF TRUTH
    ekf.yaml                      robot_localization (wheel odom + IMU, 2-D)
  launch/
    perception.launch.py          the whole module against a running sim
    localization.launch.py        the rover computer (EKF + tags, no Gazebo)
  scripts/                        the ros2-run entry points (5 node shims + 3 tools)
  docs/                           detector.md, localization.md, camera_manager.md, tags.md
  test/                           504 tests, no Gazebo and no network
```

`rover_perception_msgs` (a sibling C++/rosidl package) is the wire contract.
Nothing may read module 4's internals; module 5 talks to it only through these
messages.

---

## Topics

| Topic | Type | Dir | Owner | What it means |
|---|---|---|---|---|
| `/cam_nav` `/cam_tag` `/cam_excavation` `/cam_dump` | `sensor_msgs/Image` | in | camera driver | the four feeds |
| `/cam_*/camera_info` | `sensor_msgs/CameraInfo` | in | camera driver | intrinsics (scales the calibration) |
| `/odom` | `nav_msgs/Odometry` | in | gz / drive firmware | wheel dead reckoning |
| `/odometry/filtered` | `nav_msgs/Odometry` | in | EKF | fused odom (preferred when present) |
| `/imu` | `sensor_msgs/Imu` | in | gz / IMU | gyro + accel (no magnetometer) |
| `/perception/mode` | `std_msgs/String` | in | module 5 | phase request |
| `/perception/camera_phase` | `CameraPhase` | out | camera manager | the rate command for the driver |
| `/perception/camera_status` | `CameraStatusArray` | out | camera manager | measured per-camera cost |
| `/perception/tag_detections` | `TagDetectionArray` | out | tag detector | raw sightings (camera frame) |
| `/perception/tag_debug` | `sensor_msgs/Image` | out | tag detector | overlay, off by default |
| `/perception/obstacles` | `ObstacleArray` | out | obstacle detector | **the obstacle map** |
| `/perception/obstacles_cloud` | `sensor_msgs/PointCloud2` | out | obstacle detector | for a nav2 costmap layer |
| `/perception/detector_stats` | `std_msgs/Float32MultiArray` | out | obstacle detector | fps/latency/cpu, read by health |
| `/perception/localization_state` | `LocalizationState` | out | tag localizer | how well the map is anchored |
| `/localization/tag_fix` | `geometry_msgs/PoseWithCovarianceStamped` | out | tag localizer | the correction itself |
| `/perception/health` | `PerceptionHealth` | out | health monitor | the per-second report |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | out | health monitor | the same, for `rqt_robot_monitor` |
| TF `map`→`odom` | `tf2` | out | tag localizer | only with `publish_tf:=true` |

Every frame id is `map` (module 4's own frame), except the camera-optical poses in
`TagDetection`, which are in the camera's optical frame as REP-103 defines it.

---

## The three rules this module is built around

* **Never trust the seed.** Obstacle positions come only from the cameras.
  `perception.yaml` deliberately duplicates the arena geometry it is *allowed* to
  know (size, zones, wall margin) instead of reading `ground_truth.yaml`, and a
  test cross-checks those constants against `cat_arena/config/arena.yaml` so the
  two copies cannot drift silently.
* **Craters are lethal, boulders are obstacles.** A monocular camera cannot
  measure a crater's depth, so craters are published with `depth = 0.0` and a
  conservative confidence bias; module 5 must treat a crater as a
  non-traversable disc. Ambiguous blobs in the classical backend are classed as
  craters, because a needless detour is much cheaper than a failed run.
* **The hub is private.** The camera manager publishes a phase (rates per
  camera). Only `rover_camera_driver` may interpret it, so whether the CSI hub
  round-robins or streams simultaneously never leaks into perception, navigation
  or excavation code.

---

## Build and run

```bash
cd ~/cat
colcon build --packages-select rover_perception_msgs rover_perception --symlink-install
source install/setup.bash
```

Simulation (the arena owns `gz sim`; this adds module 4's nodes and the textured
tag boards):

```bash
# terminal 1 - world + rover + clock
ros2 launch cat_arena arena.launch.py robot_plugins:=true seed:=1
ros2 launch rover_description sim.launch.py

# terminal 2 - perception
ros2 launch rover_perception perception.launch.py
```

The real robot (no Gazebo there):

```bash
ros2 launch rover_perception localization.launch.py          # EKF + tag chain
ros2 run rover_perception obstacle_detector.py               # + the obstacle map
```

Tools:

```bash
ros2 run rover_perception perception_info.py                 # effective config + artefact states
ros2 run rover_perception perception_info.py --deps          # what is installed / refused
ros2 run rover_perception generate_tags.py --out-dir /tmp/opencode/tags
ros2 run rover_perception preview_tags.py                    # draw the bitmaps side by side
```

Tests:

```bash
colcon test --packages-select rover_perception && colcon test-result --verbose
# or, straight from the source tree:
python3 -m pytest src/rover_perception/test -q
```

---

## Acceptance tests

| Claim | Where it is pinned |
|---|---|
| The 4000 kbps rule is measured, not assumed, and startup cannot trip it | `test/test_bandwidth.py` |
| Every rung of the derate ladder, hysteresis, and each thermal floor | `test/test_camera_manager.py` |
| Thermal zones parse real sysfs files; no zones is a reported nominal | `test/test_thermal.py` |
| fps/latency/CPU arithmetic and the `/proc` parsers | `test/test_cpu.py` |
| A tag rendered as a real 36h11 bitmap decodes back to the same id | `test/test_tagimages.py`, `test/test_tagdetect.py` |
| `tags.yaml` agrees with `markers.yaml` and with `arena.yaml` | `test/test_tags.py`, `test/test_config.py` |
| The tag frame convention is the viewer's frame | `test/test_tagdetect.py::test_head_on_tag_reports_the_viewer_frame` |
| A fix collapses sigma; sigma grows with dead reckoning and is capped | `test/test_localizer.py` |
| Ground projection round-trips against module 2's calibration | `test/test_groundproj.py` |
| Boulder vs crater classification and conservative ambiguity | `test/test_classical_backend.py` |
| The ONNX decode path (letterbox, per-class NMS, un-letterboxing) | `test/test_onnx_backend.py` |
| Tracking: association, escalation, wall-clock confidence decay | `test/test_tracking.py` |
| The arena-wall mask and covariance floors | `test/test_obstacles.py` |
| Every node imports, parses its flags and builds with `--config` honoured | `test/test_nodes_smoke.py` |
| The launch files compose what they promise | `test/test_launch.py` |
| Every `ros2 run` entry point works | `test/test_scripts_cli.py` |

None of these need Gazebo or the network; the live simulation is a separate
manual check (see `docs/`).

---

## Assumptions and open items

* **Arena geometry is duplicated on purpose** (see the first rule above). If the
  organiser confirms the zone dimensions, update `cat_arena/config/arena.yaml`
  and `perception.yaml` together; the cross-check test will tell you if you miss
  one. The corridor height is the one dimension marked as an assumption.
* **Cameras are 180°-offset and the CSI hub is undecided.** Module 4 consumes
  images exactly as delivered; any de-rotation belongs in `rover_camera_driver`.
  Until the hub is chosen, module 3's dataset keeps `flip_180` augmentation off.
* **`classical` is the shippable backend today**; `onnx` activates the model
  trained in module 3 and is exercised in tests through a stub session, because
  `onnxruntime` and the fine-tuned model are not installed on the dev laptop.
  Its measured live recall in the seed-1 arena — good at close range, degraded
  far away, with the painted-zone and cast-shadow false positives it cannot avoid
  — is written up honestly in `docs/detector.md`; the learned backend is what
  fixes it, and the wire contract is identical either way.
* **`robot_localization` is not installed.** `localization.launch.py` logs the
  install command and carries on with `/odom` when it is missing; the EKF itself
  is a config file, not code, and is tested by a real `ekf_filter_node` only on
  the rover.
* **`cv2` and `cv_bridge` are refused by design** on the dev laptop (its numpy
  shadows apt OpenCV). `rover_perception/imagecodec.py` replaces them; a test
  asserts the refusal is deliberate rather than an accident.
