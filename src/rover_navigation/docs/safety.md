# The safety layer — the single writer of `/cmd_vel`

`velocity_gate.py` (node `rover_navigation_velocity_gate`) is the **only node in
the whole stack that publishes `/cmd_vel`**. The navigator publishes its intent
on `/nav/cmd_vel`; the gate vouches for it, or vetoes it. That split is not
decoration — it keeps "what would be a good thing to do" (the planner) and "what
is safe to do right now" (the gate) in different processes, so the safety rules
are not something the planner can be tempted to optimise around, and so *why is
the rover stopped* has exactly one answer, in `GateDecision.reason`.

Module 2's `stub_topics` already latches `/estop` and zeroes `/cmd_vel` while it
is set. The gate is the authoritative second half of that contract, and it is the
half that knows about the arena.

`safety.py` is the pure decision function; the node owns the clock and the
topics. One `VelocityGate.decide(...)` per tick, and the reasons are checked in
this order.

---

## 1. e-stop

```
if estop: hold("e-stop latched")
```

The hardware chain has the last word. `/estop` is subscribed **reliable +
transient-local** (unlike every other subscription, which is best-effort): a
latched stop is precisely the value a late subscriber must not miss, and a
volatile subscriber would miss it by construction. A latched `/estop` makes the
mission node abort the run too.

## 2. Stale command

```
if no command yet, or now - last_command_time > command_timeout_s (0.50): hold
```

If the navigator goes quiet — crashed, mid-replan, link dropped — the wheels stop.
This is not paranoia about the navigator; it is about **Gazebo**. `gz`'s
diff-drive holds the last velocity command *forever*. Silence must therefore mean
*stop*, never "keep going at the last speed you heard." (The same fact is why the
gate republishes a zero twist every tick rather than only on change.)

## 3. Localisation

Every obstacle is in the **map** frame, and the map frame is only meaningful
while module 4's wall AprilTags anchor it. `sigma_xy` and `sigma_yaw` grow with
dead reckoning and collapse on every accepted fix. The gate:

* **stops** if no `LocalizationState` has arrived and `require_localized` is set
  (`"no localization state received"`),
* **stops** if the state says `localized == False` (`"map not anchored by any tag
  fix yet"`),
* **stops** if `time_since_fix_s > max_time_since_fix_s` (20 s),
* otherwise **scales the command** by a linear ramp:

  | quantity | 1.0 below | 0.0 at/above |
  |---|---|---|
  | `sigma_xy_m` | 0.15 | 0.60 |
  | `sigma_yaw_rad` | 0.10 | 0.50 |

  `scale = min(scale_xy, scale_yaw)`, applied to both the linear and angular
  command, and reported as `speed_scale`. A scale of exactly zero stops with a
  reason that says the uncertainty was too large.

The principle: **a rover that does not know where it is must not drive toward a
hole it also does not know the position of.** Smooth scaling means it slows down
before it stops, rather than slamming to a halt every time a fix is late.

## 4. Reactive brake

Perception can see a boulder closer than the plan assumed — most often just after
the plan was made, before the next replan. A cheap forward-bumper test stops the
rover without waiting for the planner:

```
front   = pose + (length_m / 2) * (cos yaw, sin yaw)
clearance = distance(obstacle, front) - obstacle.radius
if clearance < emergency_stop_m (0.45): brake
```

Two details make it behave:

* The test is at the **front bumper**, using the obstacle's **own radius**, so
  the threshold is a real clearance and not a centre-to-centre distance, and an
  obstacle beside or behind the rover does not stall a legal manoeuvre.
* A **pure-turn command is never braked.** `forward = linear > 0.0`, so a
  zero-linear turn skips the check — a rover may always rotate to look for a way
  out, even nose-to-nose with a boulder.

## Then clamp and rate-limit

Only after all four does the gate clamp to `max_speed_mps` / `max_yaw_rate_radps`
and rate-limit by `max_accel_mps2 · dt` and `max_yaw_accel_radps2 · dt`, so the
wheels never see a step change. A held (zero) command is already zero, so no ramp
is applied while stopping — the stop is immediate.

---

## What it publishes

`/cmd_vel` (the twist, zeros included) and `/nav/gate_status` (`GateStatus`):
`estop`, `allowed`, `stale_command`, `emergency_brake`, `localized`,
`speed_scale`, `sigma_xy_m`, `reason`, the `command` it sent, and the
`max_speed_mps` / `max_yaw_rate_radps` in force *after* localisation scaling.
`reason` is empty when the command was allowed at full scale; when the command
was scaled it says so ("speed scaled to 91%"), and when the rover was held it is
the one sentence that explains the stop.

The gate runs at `gate_rate_hz` (20 Hz) — four times the navigator's 5 Hz control
rate — so a stop lands within 50 ms of the input that caused it, and a stale
command is caught even if the navigator dies between its own ticks.

---

## Coupling to module 4

The gate keeps its **own** obstacle records and its **own** pose, composed
independently of the navigator (`nodeutil.PoseTracker` over the same odometry and
`map -> odom`). It does not read the navigator's state; if the navigator's map
went wrong, the gate's brake is still computed from the raw detections and the
raw odometry. That independence is the reason the two are separate nodes.
