# `cat_arena` — Module 1 of the Caterpillar Autonomy Challenge stack

The arena. One Python package that turns **one integer seed** into a complete,
self-describing, rule-compliant Gazebo Harmonic world, plus the launch file that
drops the rover into it at a random pose.

Nothing else in the stack is allowed to hard-code arena geometry: every
dimension, colour, friction value and obstacle limit lives in
[`config/arena.yaml`](config/arena.yaml).

```
ros2 run cat_arena generate_arena.py --seed 7      # world + mesh + labels
ros2 launch cat_arena arena.launch.py              # random seed, GUI, rover spawned
ros2 run cat_arena env_check.py                    # is this machine ready?
```

---

## 1. What it produces

For a run seed `N`, `generate` (or the launch file) writes `seed_N/`:

| file | what it is | who may read it |
|---|---|---|
| `arena.sdf` | the Gazebo world | everyone |
| `terrain.obj` + `terrain.mtl` | triangle terrain, **visual and collision** | everyone |
| `sand_albedo.png` | procedural dry-sand texture (visual only) | everyone |
| `terrain_height.png` | heightfield rendering of the mesh | analysis / labels |
| `start_pose.yaml` | the sampled spawn pose | everyone |
| `markers.yaml` | marker board poses + AprilTag handoff block | everyone (module 4) |
| `manifest.json` | seed, config hash, file hashes, versions, assumptions, rule compliance | everyone |
| `ground_truth.yaml` | **oracle**: obstacle positions, crater geometry | tests + labels only |

`ground_truth.yaml` is the only file the autonomy stack may never open. It is
written with a loud warning header, and module 1 ships a test
(`test_ground_truth.py::test_autonomy_packages_are_clean_if_they_exist`) plus the
reusable `cat_arena.ground_truth.assert_autonomy_clean()` helper that greps the
autonomy packages for any reference to it. Run that check before every
competition attempt: if it ever prints anything, the run would be cheating.

### Arena layout

```
            north wall (y = 5.0)
   +----------------------------------------------------------+
   |  START        |                                          |
   |  2.0 x 2.3 m  |                                          |
   |  (no obstacles|         OBSTACLE ZONE                    |
   |   by rule)    |         5.5 m wide                       |  EXCAVATION
   +---------------+                                          |  1.5 x 5 m
   |  strip        |         boulders + craters               |  (boulders ok,
   +---------------+                                          |   no craters)
   |  CONSTRUCTION |                                          |  x = 7.5 .. 9.0
   |  2.0 x 1.5 m  |                                          |
   |  [berm target]|                                          |
   +----------------------------------------------------------+
            south wall (y = 0)
   x = 0 .. 2.0        x = 2.0 .. 7.5
```

Origin = west-south inner corner of the arena floor, at mean sand level.
`+x` = east, `+y` = north, `+z` = up. "Top-left" of the challenge diagram is the
north-west corner.

The five `tiling: true` zones must exactly partition the 9 x 5 m arena, and the
config loader refuses to load a file where they do not (a typo in a scaled-from-
the-diagram rectangle fails at load time, not at run time).

### Obstacles

Per run, seeded from the run seed: **≥3 boulders** (30–40 cm, varying exposed
height, positive features, may be in the excavation zone too) and **≥3 craters**
(25–50 cm wide, 6–18 cm deep, **negative** depressions carved into the terrain
mesh, obstacle zone only). Craters are not separate models — they are real dents
in the collision mesh, so a wheel really does drop into them.

Boulders and craters are kept apart from each other (`spacing`) and from the zone
borders (`margin`), and the start zone contains nothing by construction.

---

## 2. File tree

```
src/cat_arena/
├── CMakeLists.txt              ament_cmake + ament_cmake_python + ament_cmake_pytest
├── package.xml
├── README.md
├── config/
│   └── arena.yaml              SINGLE SOURCE OF TRUTH for arena geometry
├── launch/
│   └── arena.launch.py         generate -> gz sim -> clock bridge -> spawn -> pose
├── scripts/                    installed to lib/cat_arena/ (ros2 run <pkg> <file>.py)
│   ├── env_check.py            environment + bridge compatibility verification
│   ├── generate_arena.py       CLI wrapper around cat_arena.generate.main
│   ├── publish_start_pose.py   the /cat_arena/initial_pose publisher node
│   └── render_overview.py      capture a top-down PNG (visual acceptance test)
├── assets/
│   └── test_proxy/model.sdf    module-1 spawn fixture (NOT the rover)
├── cat_arena/                  the Python package
│   ├── __init__.py             __version__
│   ├── config.py               typed, validated arena.yaml (Rect/Zone/…)
│   ├── seeding.py              SHA-256 namespace seed derivation
│   ├── terrain.py              height field, crater bowls, OBJ/MTL writer, textures
│   ├── obstacles.py            seeded boulder + crater placement
│   ├── start_pose.py           deterministic random spawn pose in the start zone
│   ├── markers.py              marker board poses + apriltag_ros handoff
│   ├── sdf.py                  world SDF builder + rotation helpers
│   ├── ground_truth.py         oracle document + assert_autonomy_clean()
│   ├── png.py                  dependency-free PNG read/write (numpy only)
│   ├── generate.py             the orchestrator + CLI
│   └── start_pose_node.py      rclpy node publishing the sampled pose
└── test/
    ├── conftest.py             fixtures (config, one shared generation)
    ├── scripts_loader.py       import scripts/*.py by path
    ├── test_config.py          dimensions, partition, every rule rejection
    ├── test_terrain.py         crater geometry, OBJ normals, textures, PNG
    ├── test_obstacles.py       rule compliance over 24 seeds, determinism
    ├── test_start_pose.py      footprint fit, randomness, determinism
    ├── test_sdf.py             world structure, physics, materials, helpers
    ├── test_markers.py         board offsets, viewing normals, height limits
    ├── test_ground_truth.py    oracle completeness + the leak check
    ├── test_generate_cli.py    artifacts, reproducibility, CLI exit codes
    ├── test_env_check.py       the checker's own behaviour
    └── test_launch.py          what the launch file builds
```

---

## 3. Build

ROS 2 Humble, Ubuntu 22.04. The arena package itself is pure Python — no
compilation, no extra apt packages beyond what is already needed for Gazebo.

```bash
cd ~/cat
source /opt/ros/humble/setup.bash
colcon build --packages-select cat_arena --symlink-install
source install/setup.bash
```

## 4. Run

```bash
ros2 run cat_arena env_check.py                          # before anything else
ros2 run cat_arena generate_arena.py --seed 7            # world into ./worlds/seed_7
ros2 run cat_arena generate_arena.py --seed 7 --json     # machine-readable summary
ros2 launch cat_arena arena.launch.py                    # random seed, GUI + rover
ros2 launch cat_arena arena.launch.py seed:=7            # reproducible run
ros2 launch cat_arena arena.launch.py seed:=7 headless:=true
ros2 launch cat_arena arena.launch.py seed:=7 robot:=/abs/path/rover.sdf   # module 2
ros2 run cat_arena render_overview.py --seed 7 --out /tmp/arena_7.png
```

Useful `generate_arena.py` flags: `--config`, `--output-dir`, `--overview-camera`,
`--no-validate`, `--quiet`, `--print-paths`.

Useful launch arguments: `seed`, `config`, `output_dir`, `world`, `headless`,
`headless_rendering`, `verbose`, `validate_world`, `spawn_robot`, `robot`,
`robot_name`, `spawn_delay`, `publish_initial_pose`, `bridge_clock`,
`use_sim_time`, `gui_fullscreen`.

### Interfaces other modules use

| topic | type | notes |
|---|---|---|
| `/cat_arena/initial_pose` | `geometry_msgs/PoseWithCovarianceStamped` | transient-local, republished every 5 s; the sampled spawn pose in `map` |
| `/initialpose` | same | mirrored for RViz's "2D Pose Estimate" (disable with `also_publish_initialpose:=false`) |
| `/clock` | `rosgraph_msgs/Clock` | bridged from `gz sim` when `bridge_clock:=true` |
| `/world/cat_arena/create` | Gazebo | spawn service used by `ros_gz_sim create` |

Zone rectangles for module 5's map come from `arena.yaml` (arena *geometry* is
public knowledge — it is painted on the ground). Hidden *obstacle instances* come
only from ground truth, which module 5 may use only for scoring, never for
planning.

## 5. Acceptance tests

```bash
colcon test --packages-select cat_arena --event-handlers console_direct+
colcon test-result --all --verbose
```

The suite is the acceptance test. The parts that matter most:

| test | proves |
|---|---|
| `test_terrain.py::test_obj_has_matching_vertex_and_normal_counts` | the OBJ is loadable by dartsim (this used to segfault) |
| `test_terrain.py::test_crater_is_a_negative_depression` | craters are depressions, not bumps |
| `test_obstacles.py` (24 seeds) | ≥3 boulders, ≥3 craters, correct zones, no start-zone leaks, spacing |
| `test_start_pose.py` (40 seeds) | the whole 1.5 x 0.75 m footprint fits the start zone at any heading; the pose is random but seed-deterministic |
| `test_markers.py` | marker boards sit *inside* the arena, offset off the wall face, facing into it, and within the 0.60 m height limit |
| `test_generate_cli.py::test_same_seed_reproduces_byte_identical_artifacts` | a seed is reproducible byte for byte |
| `test_ground_truth.py::test_rule_compliance_passes_for_all_sub_rules` | the oracle says the generated arena is rule-compliant |
| `test_launch.py` | the launch file spawns the rover at exactly the sampled pose |
| `test_env_check.py::test_static_checks_find_no_failures` | this machine can actually run the stack |

Manual verification beyond the unit tests (all of these were run while building
this module — see §7):

```bash
# 1. the world parses and is loadable
python3 -c "from cat_arena.generate import generate; print(generate(seed=7, verbose=False).validation_output)"
# -> "gz sdf -k: OK"

# 2. everything is in the world and it runs
gz sim -s -r -v 2 worlds/seed_7/arena.sdf      # then: gz model --list

# 3. physics: a 5 cm probe sphere rests exactly on the sand and sinks in a crater
#    (flat sand rest z = +0.050000, crater centre rest z = -0.115 for depth 0.179)

# 4. the rover spawns at the sampled pose
ros2 launch cat_arena arena.launch.py seed:=7
#    -> gz model -m rover -p  ==  start_pose.yaml

# 5. the pose topic
ros2 topic echo /cat_arena/initial_pose --once

# 6. eyes on it
ros2 run cat_arena render_overview.py --seed 7 --out /tmp/arena_7.png
```

## 6. Assumptions and open questions

Flagged in `arena.yaml`, in every generated world header, in the manifest and in
`ground_truth.yaml`:

| # | assumption | where to change it |
|---|---|---|
| A1 | **Zone rectangles are scaled from the challenge diagram** (start 2.0x2.3, construction 2.0x1.5, obstacle 5.5 wide, excavation 1.5x5.0, berm target 1.25x0.9) and **need organizer confirmation**. | `zones:` |
| A2 | **Arena wall height = 0.50 m** (the rules only imply walls exist, via the -50 point wall ram). Thickness 0.06 m. | `arena.walls` |
| A3 | **Sand friction mu = mu2 = 0.6** — an assumed starting value for dry tilled sand (literature 0.5–0.7). Tune in one place; the wheels' behaviour depends on it. | `arena.friction` |
| A4 | The **"left corridor"** strip (2.0 x 1.2 m between construction and start) is not named in the rules; it exists only so that the named zones tile the arena exactly. | `zones.left_corridor` |
| A5 | The robot **stowed envelope is 1.5 x 0.75 m** (given as ground truth) and is used only to guarantee the sampled pose fits the start zone. Module 2 must match it. | `start_pose.robot_footprint` |
| A6 | Body **+x is the declared "forward"**. Change `declared_forward_deg` if the rover's forward ends up elsewhere. | `start_pose.declared_forward_deg` |
| A7 | Judged **berm height 0.15 m** (used by module 6). | `marking.berm_height` |
| A8 | Fiducial markers: 3 boards (2 on the start-zone frame + 1 mid-arena north wall), ≤25x25 cm, ≤0.60 m high, ≤5 total — within the rules. The **CSI camera hub is not chosen yet**, which is why modules 3/4 are not built yet. | `markers` |
| A9 | Terrain ripple `noise_amplitude = 0.0`: the sand is flat except for craters, which keeps navigation tests meaningful. Raise it for realism. | `terrain.noise_amplitude` |
| A10 | Sand depth (0.30 m) is used as the mesh skirt depth, so the terrain is a closed solid, not a floating sheet. | `arena.sand_depth` |
| A11 | `maintainer@example.com`-style placeholder contact in `package.xml` uses `team@cat-arena.local`; the package `license` is Apache-2.0. Adjust for the team's real metadata. | `package.xml` |
| A12 | **NumPy/OpenCV conflict on the dev laptop**: a pip `numpy` 2.x in `~/.local` shadows the apt NumPy 1.x that `python3-opencv` was built against, so `import cv2` fails. ROS 2 Humble's `cv_bridge` needs the apt cv2, so keep NumPy 1.x (`pip uninstall numpy`, or run with `PYTHONNOUSERSITE=1`). `cat_arena` itself passes on both. `env_check.py` detects and explains this. | your Python environment |

Not simulated on purpose: the drag/compaction behaviour of real tilled sand, sand
deformation from digging (module 6), and the CSI camera hub. Wheel-sand
interaction is modelled only through `mu`/`mu2` and the terrain mesh.

## 7. Verified facts (do not re-litigate these)

These were established empirically on the dev laptop (Ubuntu 22.04.5, ROS 2
Humble, gz-sim 8.15.0, `ros-humble-ros-gzharmonic` 0.244.12) while building this
module. They are the reason several otherwise-odd-looking choices exist.

1. **The heightmap is not used for collision.** An `<heightmap><collision>` on
   this gz build produces a *perfectly flat* collidable surface (the shape loads
   but the collision geometry does not follow the heights). The terrain is
   therefore a **triangle mesh for both visual and collision**, and the
   heightfield PNG is only an analysis artifact.
2. **Mesh terrain needs `<collision_detector>bullet</collision_detector>`** in
   the DART physics block. With the default detector the mesh does not collide
   properly. Verified: a probe on flat sand rests at z = 0.050000 (exactly the
   radius), a probe in a crater rests depressed by the bowl depth.
3. **OBJ meshes must carry vertex normals.** Without `vn`, dartsim logs
   `does not have a normal count [0] that matches its vertex count` and then
   **segfaults**. `terrain.py` always writes `v`/`vt`/`vn` in lockstep, and a test
   enforces equal counts.
4. **A world with no `<plugin>` block has no systems at all**: no physics, no
   entity creation, nothing. The generator always emits physics, user-commands,
   scene-broadcaster and contact; the sensors system is added only when the
   verification camera is enabled.
5. **`<gui><camera_pose>` is not valid SDF** (it made `gz sdf -k` complain). The
   overview camera is a normal `<sensor>` model instead.
6. **`ros_gz_sim create` overrides the model pose with its `-x -y -z -R -P -Y`
   flags.** The launch file therefore passes the sampled pose as flags, and the
   spawn proxy's own `<pose>` is irrelevant. Spawn the robot *above* the surface
   (`spawn.z = 0.08 m`), because a body spawned intersecting the mesh starts on
   the wrong side of it.
7. **`gz model -m NAME -p` prints the pose over two lines** (`Pose [XYZ]` header,
   then the XYZ triplet, then RPY). Naive one-line parsing silently returns the
   wrong numbers.
8. **`hash()` is salted per process** and must never be used for seeding; all
   sub-streams come from SHA-256 namespaces (`cat_arena/seeding.py`).
9. The pre-installed `ros-humble-ros-gzharmonic` (0.244.12) is linked against
   `libgz-transport13` / `libgz-msgs10`, i.e. exactly the Harmonic libraries on
   this machine. **Never build `ros_gz` from source.** `env_check.py` verifies
   this link-level compatibility and fails loudly if it ever breaks.
10. **The `gz` CLI only implements a subset of subcommands** (`sim`, `model`,
    `topic`, `service`, `msg`, `sdf`, `plugin`, `param`, `log`, `fuel`,
    `launch`, `gui`). `gz transport --version` is *not* a thing — it prints the
    top-level help. Component versions are therefore read from the authoritative
    CMake package configs (`.../cmake/gz-<component>/gz-<component>-config-version.cmake`).
11. **A marker board must be pushed off the wall face *into* the arena**, i.e.
    `centre = wall_point + normal * (stand_off + thickness/2)` with the normal
    pointing into the play area. The opposite sign buries the board inside the
    wall where it renders invisible; `test_markers.py` pins the correct sign.

## 8. Handover notes for module 2 (`rover_description`)

* Spawn your model with `ros2 launch cat_arena arena.launch.py robot:=<your.sdf>`.
  Pass `robot_name:=rover` (the default) and keep the model's own `<pose>`
  irrelevant — the launch file supplies the pose.
* The stowed envelope must be **1.5 x 0.75 x 0.75 m** and **≤80 kg including
  beacons, markers and battery**; `arena.yaml` assumes the 1.5 x 0.75 m footprint.
* Body **+x is declared forward**.
* The sand surface is `z = 0`; the spawn z offset is `0.08 m` above the local
  sand (raise it to your wheel radius + a small margin).
* Use `use_sim_time:=true` together with the `/clock` bridge.
* Cameras must publish on `/cam_nav`, `/cam_tag`, `/cam_excavation`, `/cam_dump`
  with `sensor_msgs/Image` + `CameraInfo` — identical topic names in sim and on
  the real robot.
