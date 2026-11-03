# Localisation — odometry + AprilTags, no AMCL

Module 4 has **no AMCL, no particle filter, no GPS, no compass, no
magnetometer**. The rover's frame graph is built from two things only:

* **wheel odometry + IMU**, fused by `robot_localization`'s EKF into
  `/odometry/filtered` (config `ekf.yaml`), and
* **AprilTags on the walls**, which are the only absolute reference the rules
  allow. A tag sighting measures *where the rover is*, so one observation solves
  the correction exactly.

```
/odom ─┐
       ├─► EKF (ekf.yaml) ─► /odometry/filtered ─┐
/imu  ─┘                                          ├─► tag_localizer ─► map -> odom
/cam_tag ─► tag_detector ─► /perception/tag_detections ┘            │
                                                                    ▼
                                        LocalizationState + (optional) TF map->odom
```

`tag_localizer` is the authority for the `map → odom` edge. The EKF never
produces `map → odom`; `ekf.yaml` keeps `world_frame: odom` for exactly this
reason.

---

## The solve (`localizer.py`)

```
T_map_tag    known   (tags.yaml: the generated catalogue)
T_cam_tag    measured by the detector, tag pose in the camera optical frame
T_base_cam   known   (module 2's calibration extrinsics)
T_odom_base  known   (the EKF's /odometry/filtered, or raw /odom)

T_base_tag = T_base_cam @ T_cam_tag
T_odom_tag = T_odom_base @ T_base_tag
T_map_odom = T_map_tag  @ inv(T_odom_tag)
```

The odometry is sampled from an `OdomBuffer` (capacity 64) at the timestamp
nearest the tag frame's stamp, so a fix is not applied against a stale pose.

Everything is **planar**: the rover drives on flat sand, so only `x`, `y` and
`yaw` of the correction are ever applied. The z/roll/pitch components are
discarded (`planarize`) rather than allowed to tilt the map frame. When several
tags are seen in one frame, `solve_fixes` combines them (weighted by range) into
one `FixSolution`.

## Rejection, never silent application

A fix that is implausible is rejected **with a reason**, so a bad detection can
only fail to help, never teleport the map:

| reason | guard |
|---|---|
| `tag_not_in_catalogue` | the id is not in `tags.yaml` |
| `too_near` / `too_far` | `min_tag_range_m` (0.25) / `max_tag_range_m` (6.0) |
| `non_finite` | the solve produced NaN/Inf |
| `correction_too_large` | `max_correction_m` (1.5) |
| `yaw_correction_too_large` | `max_correction_yaw_rad` (0.6) |
| `pose_invalid` | the detector flagged a degenerate pose |

`xy_gain` / `yaw_gain` (1.0 by default) let a fix be applied fully or smoothed;
`accept_margin_scale` widens the acceptance gate.

## The first fix *establishes* `map -> odom`, it does not correct it

The rover boots genuinely un-localised — no GPS, no compass, no initial pose is
given to the autonomy (a start pose from `cat_arena` exists in simulation, but it
is ground truth and perception must not use it). Gazebo's diff-drive also starts
`/odom` at `(0, 0, 0)` at the rover's **boot pose**, *not* at the arena origin, so
the very first tag fix always implies a correction about the size of the arena.

If the `max_correction_m` / `max_correction_yaw_rad` guards were applied to that
first fix it would be rejected — forever, because nothing ever changes until a fix
is accepted. So `Localizer.update` passes `initialized=self.localized` to
`solve_fixes`: while the transform is un-set, the first fix is taken as the
**initialisation** (guards skipped, gains ignored, reported `correction_*` are
`0.0`), and only subsequent fixes are guarded as corrections. `localized` then
stays true unless the fix stream is lost long enough for the covariance to hit its
cap, which is the signal module 5 needs to slow down and re-acquire.

## Uncertainty grows with dead reckoning

Between fixes the estimate is pure dead reckoning, so its uncertainty grows with
how far the rover has actually driven and turned:

```
sigma_xy  += growth_m_per_m       (0.05)   * distance travelled since the last fix
sigma_yaw += growth_rad_per_rad   (0.03)   * |yaw| turned since the last fix
```

and collapses to the fix sigma (`fix_xy_m` 0.04, `fix_yaw_rad` 0.06) on every
accepted sighting. Both are capped (`max_xy_m` 3.0, `max_yaw_rad` 1.2), so a
long dead-reckon reports a large but finite uncertainty instead of diverging.

This uncertainty is published on `/perception/localization_state`
(`LocalizationState`) precisely so module 5 can slow down, or stop and look
around, when the map is no longer well anchored. Without GPS, that report is the
only safety net — and it is a *contract*: module 5 may depend on
`sigma_xy_m`/`sigma_yaw_rad` meaning what they say.

`/localization/tag_fix` carries the applied correction itself
(`PoseWithCovarianceStamped`) for debug and for a nav2 consumer that wants the
raw edge.

## The EKF (`ekf.yaml`)

`robot_localization`'s `ekf_filter_node`, `two_d_mode: true`:

* `odom0: /odom` supplies `x`, `y`, `yaw` (+ their rates);
* `imu0: /imu` supplies angular velocity and (differential) acceleration, with
  **gravitational acceleration removed** and absolute orientation **not used** —
  a gyro-integrated orientation drifts, and the tags fix yaw. There is no
  magnetometer channel at all.
* `publish_tf: false` by default: in simulation gz's diff-drive already publishes
  `odom → base_link` on the bridged `/tf`, and two publishers of the same TF edge
  is a `tf2` error, not a merge. The EKF's output is consumed as a **topic**.
  `localization.launch.py` takes `publish_tf:=true` for the real robot, where the
  EKF is the only source of `odom → base_link`.

`robot_localization` is not installed on the dev laptop. `localization.launch.py`
detects this, logs the exact `apt install` command, and carries on with `/odom`
so the tag chain still runs. On the rover it is a required dependency.

## Two entry points

| launch file | where | what it starts |
|---|---|---|
| `perception.launch.py` | dev laptop / sim | the whole module; `publish_tf:=false` (gz owns the edge) |
| `localization.launch.py` | the rover computer | EKF + tag detector + tag localizer; `publish_tf:=true`, `use_sim_time:=false` |

Both accept `config:=`, `publish_tf:=`, `use_sim_time:=` and node on/off switches.

## Live check

```bash
# sim
ros2 launch rover_perception perception.launch.py
ros2 topic echo /perception/localization_state --once      # localized?, sigma
ros2 run tf2_ros tf2_echo map odom                         # only with publish_tf:=true
# on the robot
ros2 launch rover_perception localization.launch.py
ros2 topic hz /odometry/filtered
```

## Tests

```bash
python3 -m pytest src/rover_perception/test/test_localizer.py -q
```

They pin the solve against synthetic tag/pose geometry, the rejection reasons,
the sigma growth and collapse, the cap behaviour (a single `propagate` step can
never reach the yaw cap, so the test loops), and the odometry-nearest-stamp
selection.
