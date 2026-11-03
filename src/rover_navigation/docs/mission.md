# The behaviour sequencer — the shape of the run

`mission.py` (node `rover_navigation_mission`) is the part of module 5 that is
not navigation: it decides *when* to drive, dig and dump, and it is the only
writer of `/perception/mode`. Module 4 reacts to that mode (it picks the active
camera and frame rate from it); module 6 will answer the work cycles the
sequencer sends.

The run is the competition task:

```
acquire → travel to the excavation strip → dig → haul to the berm → dump → repeat → return → park
```

`MissionSequencer` is ROS-free. The node calls `start(now)`, then `tick(now,
pose=…, localized=…, work_available=…)` on every control tick, acts on the
returned `MissionCommand` (its `goal` and `work` are edge-triggered — populated
only on the tick that requests them), and reports outcomes back through
`nav_finished(...)` and `work_finished(...)`. That means "what does the rover do
when the second dump pose is unreachable and the tag fix is 30 s old" is a unit
test, not a simulator session.

---

## Phases

| `PHASE_*` | value | `/perception/mode` | what happens |
|---|---|---|---|
| `IDLE` | 0 | `idle` | not started, or disabled by configuration |
| `ACQUIRE` | 1 | `tag` | find a wall tag so the map frame is anchored |
| `TRAVEL_TO_DIG` | 2 | `nav` | drive to the next dig pose |
| `EXCAVATE` | 3 | `excavate` | work cycle at that pose |
| `TRAVEL_TO_DUMP` | 4 | `nav` | haul to the next dump pose |
| `DUMP` | 5 | `dump` | dump work cycle; a completed dump is a completed cycle |
| `RETURN` | 6 | `nav` | drive back to the park pose |
| `PARK` | 7 | `idle` | settle at the park pose (two ticks) |
| `COMPLETE` | 8 | `idle` | terminal success |
| `ABORTED` | 9 | `idle` | terminal failure, with the reason in `detail` |

The numbers are identical to `rover_navigation_msgs/MissionStatus` and a test
pins them together.

---

## Acquire — you cannot plan in the map frame until you are in it

Every obstacle and every goal is expressed in the **map** frame, and the map
frame is anchored by module 4's AprilTag fixes. At a random start pose the rover
has no fix yet, so the first thing the run does is find one.

ACQUIRE has two steps, and the first one is the non-obvious part:

1. **Drive to the park pose.** A rover spawned in a corner of the 2 × 2.3 m start
   zone may not have room to turn where it stands. The park pose is already known
   legal (`navigation_info.py` checks it), it is inside the start zone, so going
   there costs nothing, and it gives the spin somewhere it can actually happen.
   If the staging goal *fails*, the rover tries turning where it is rather than
   giving up — the tag may well be visible from here.
2. **Spin in place.** A goal at the current position with a new heading makes the
   controller rotate (`acquire_spin_rad` = 1.20 rad per step). It keeps doing
   this until `localized` is true or `acquire_timeout_s` (45 s) expires. A timeout
   fails the run with **"the map is still not anchored: no accepted tag fix in
   time"** — a message that points at the arena, not at the route, because that
   is what the failure means.

Once `localized`, it enters `TRAVEL_TO_DIG`.

---

## The poses

Both pose lists are built at `start()`, from the same config the navigator will
check them against.

### Dig poses — boustrophedon passes over the excavation strip

The strip is worked in parallel passes, alternating direction, so it is worked
evenly instead of from one spot:

```
x: 2 passes at 7.55 and 8.45 m        (the excavation zone, inset cross_inset_m 0.05)
y: 3 points per pass at 1.10, 2.50, 3.90 m   (y_inset_m 0.60)
```

Pass 0 runs north→south, pass 1 south→north, with the yaw flipped by π between
them. The heading is +y (`dig.heading_rad`), so the rover drives *along* the
strip and the ladder on its +x face works the floor as it advances. All six
labels are `dig_p<pass>_s<step>`.

### Dump poses — along the berm, standing off by the swept circle

The berm target is 1.25 × 0.9 m at `[0.375, 0.30, 1.625, 1.20]`. Five dump poses
sit along its long axis (`inset_m` 0.30):

```
x: 0.675, 0.8375, 1.000, 1.1625, 1.325 m
y: target.y1 + standoff = 1.20 + 0.7882 = 1.988 m
```

The stand-off is `circumscribed_radius_m + dump.standoff_m` = **0.7382 + 0.05**,
not the inscribed radius. That is the one place the circumscribed radius is used,
and it has to be: the whole **swept circle** must clear the target the navigator
keeps as a keep-out, *and* the aft face (half the body length back) must sit over
the target's edge so the sand lands **inside** the footprint. The heading is +y
(`dump.heading_rad`), facing away from the berm, because module 2 calibrates
`cam_dump` at yaw = π — the dump camera looks aft. The rover must never drive onto
the berm it is building.

### Park pose

`(1.00, 3.80, π)` — back in the start zone, clear of the berm, facing west
(π = away from the construction zone).

---

## Work cycles, and the two work modes

At `EXCAVATE` and `DUMP` the sequencer does not dig — it asks module 6 to. The
delegation contract is the `WorkCycle` action on `/excavation/work_cycle`, which
module 5 *defines* (in `rover_navigation_msgs`) and *calls*; **module 6
implements the server.** That is why the action exists in module 5 already: the
caller comes first.

`mission.work_mode` decides what happens at a pose:

* **`delegate`** — the competition setting. The sequencer sends a `WorkCycle`
  goal and counts a cycle **only when module 6 confirms it**. If no server is
  present it waits `excavation_wait_s` (5 s) and then **fails the run** with "no
  WorkCycle server on /excavation/work_cycle". Pretending the sand moved would be
  a lie in the status and in the score.
* **`navigate_only`** — drive every leg and never claim a cycle happened, for
  exercising module 5 in the simulator before module 6 exists. Every `detail`
  string is prefixed **`navigate_only: `** so a status can never be mistaken for
  a real run.

A dump that succeeds increments `cycles_completed`; at `cycles_target` (6) the
run returns and parks.

---

## Failure, retry, and giving up

* A failed **navigation** goal in `TRAVEL_TO_DIG` or `TRAVEL_TO_DUMP` **skips
  that pose** and advances the index — another one may be reachable. The run only
  dies when nothing works at all.
* A failed **work cycle** is counted too.
* After `MAX_CONSECUTIVE_FAILURES` (4), the run aborts. The message says which
  kind of failure it was: in ACQUIRE it is *"could not turn far enough to find a
  wall tag (N consecutive failures)"* — an arena problem; elsewhere it is the
  generic *"giving up after N consecutive failures"*.
* A **latched `/estop`** makes the mission node abort the run; the gate has
  already stopped the wheels.
* `progress` is reported on **every** tick, including the terminal one, so a
  finished run reports `1.0` rather than a stale mid-run value.

---

## The berm is a keep-out the camera cannot see

A finished berm is **sand** — `cam_nav` will not detect it as an obstacle, and the
navigator would happily plan straight through the thing it was just told to
build. So the mission node latches **`/mission/berm_started`**
(`std_msgs/Bool`, transient-local) after the first dump, and the navigator adds
`berm_target` as a planner **keep-out** from then on. It is the one piece of
"state that is not perception" module 5 allows itself, and it is derived from an
event module 5 itself caused, not from the arena definition.

---

## What the sequencer never does

* It never sends a velocity. It talks only in poses (`NavigateTo`) and work
  requests (`WorkCycle`); the velocity gate is the only thing that touches
  `/cmd_vel`.
* It never reads the arena's seeded layout. The obstacle route is entirely the
  navigator's problem; the sequencer only knows the zones a competitor is told.
* It never drives a leg itself. Every movement is a `NavigateTo` goal, so the
  planner, the footprint checks and the gate apply to the sequencer's wishes
  exactly as they would to a hand `ros2 action send_goal`.
