# The controller — pure pursuit with an in-place turn phase

`controller.py` turns "here is where I am, here is the path, here is the heading
I want at the end" into one `ControlCommand` — a linear and an angular velocity,
plus the projected state for the status message. It is **stateless**: the same
pose and the same path always give the same tick, which is what makes it testable
without a clock or a simulator.

```python
command = PurePursuitController(config).compute(
    pose, path, goal_yaw,
    max_speed=0.0,            # 0 = use the robot's own limit
    position_tolerance_m=0.0,  # 0 = use the robot's own tolerance
    yaw_tolerance_rad=0.0,
    force_drive=False,         # skip the in-place turn phase
)
```

---

## Why pure pursuit, and why a turn phase

The rover is a **skid-steer** with no reverse (see the README: `cam_nav` is
forward-only, so backing up would drive blind). Pure pursuit is the right
geometry for a path-follower: pick the point on the path one lookahead ahead,
compute the curvature that reaches it, command that. Two things make it fit this
robot:

* **It can turn on the spot.** A skid-steer does not need a minimum turning
  radius, so the awkward "started the leg pointing 90° off" case is not a
  three-point turn — it is a rotation. The controller detects it
  (`abs(alpha) > in_place_turn_rad`, 0.55 rad ≈ 31°) and commands
  `linear = 0, angular = ±in_place_yaw_rate_radps` until the rover is aimed, then
  switches to pursuit.
* **It never reverses.** `compute` only ever returns a non-negative linear
  velocity. Every lookahead target is ahead on the path, and the path is planned
  from the rover's current cell forward.

---

## The tick, in order

1. **Resolve limits.** `max_speed` and the tolerances fall back to
   `config.robot` when the goal asked for nothing (the `0.0` sentinel).
2. **Arrival check.** Within `position_tolerance_m` (0.12 m) of the path's end:
   * if the final heading is within `yaw_tolerance_rad` (0.20 rad) — or
     `force_drive` is set — return `("arrived", 0, 0)`;
   * otherwise **rotate on the spot** to the final heading, shaped so it does not
     ring (below).
3. **Project onto the path.** `_project_arc` finds the arc length of the point
   on the polyline closest to the rover; `arc_remaining` is `total − arc`.
4. **Lookahead.** `lookahead_m` (0.55) plus `lookahead_speed_gain` (0.80) times
   the current speed limit, clamped to `[min_lookahead_m, max_lookahead_m]`
   (0.30 … 0.90) and never past the end of the path — a pursuit point beyond the
   goal makes the rover sail past it.
5. **Heading error.** `alpha = angle_difference(heading_to_target, pose.yaw)`.
6. **Badly misaligned?** `abs(alpha) > in_place_turn_rad` → turn in place, no
   translation.
7. **Pursuit.** `curvature = 2·sin(alpha) / lookahead`, and the speed is the
   limit tapered by the heading error — `speed = limit · max(0, 1 − 0.6·|alpha|/π)`
   — then further by `arc_remaining / slowdown_distance_m` (0.75) as it
   approaches, floored at `min_speed_mps` (0.08). Near the goal and still
   crooked, it turns rather than arcs.
8. **Clamp** the yaw rate to `max_yaw_rate_radps`.

### The turn taper

`_shape_turn` eases the last few degrees:

```python
if abs(error) >= 0.25:
    return yaw_rate
return yaw_rate * min(1.0, abs(error) / 0.25)
```

The taper is **proportional and reaches zero with the error**, so the rover
converges onto a heading instead of oscillating around it, and a heading that is
already correct produces **no command at all**. (The earlier form used
`max(0.15, |error|/0.25)`, which commanded a non-zero turn for zero error — a
rover that twitches forever at its goal.) The sign follows `yaw_rate`, not the
raw `error`, so an intentionally reversed taper cannot flip the turn direction.

---

## The `force_drive` escape hatch

`footprint.rotation_block` answers "does this rover have room to spin here?" with
an **exact** rotated-rectangle test against the walls, obstacles and keep-outs —
not the circumscribed circle. When it says *no*, a sequencer that insisted on an
in-place turn would hold forever and end the run.

`force_drive=True` skips the turn phase and drives a **tight arc** along the path
instead (the lookahead is capped at `min_lookahead_m` so the arc hugs the
certified corridor). It is the better of two imperfect options: along a path the
planner already certified collision-free, arcing is a real manoeuvre; holding is
a guaranteed failure. `NavigateTo` exposes it through the navigator's own
`footprint` pre-check, never on a whim.

---

## Speeds

The configuration is deliberately slow (`max_speed_mps` 0.45, `max_accel_mps2`
0.50) because the arena is loose sand and the wheels slip: module 2's traction
measurement shows odometry over-reporting travel by roughly 25 % on sand. The
gate's rate limiter (see `docs/safety.md`) is the second half of that — the
controller may ask for a step, but the wheels never see one.
