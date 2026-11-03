# The excavation node

`rover_excavation_mechanism`, started by `launch/excavation.launch.py` or by
`ros2 run rover_excavation excavation.py`. One process, one node, and the only
rclpy code in module 6.

```
                 /perception/mode ─┐
                                   ├─► ExcavationNode ─┬─► /excavation/status
  /excavation/work_cycle (action) ─┤      timer @      ├─► /excavation/berm
                 /estop ───────────┤   mechanism.rate  ├─► /excavation/berm_marker
             /joint_states ────────┘        Hz         ├─► /excavation/ladder_cmd ─┐
                                                       └─► /excavation/belt_cmd  ───┤
                                                                                   │
                                  backend: sim ──► ros_gz_bridge ──► gz joint ctrl ─┘
                                  backend: hardware ──► serial ──► digger firmware
```

## What one tick does

At `mechanism.rate_hz` (10 Hz by default), in order:

1. If `/estop` is latched, command the mechanism to stow and do **not** send
   actuator commands at all. The model still stows, so `MechanismStatus` stays
   truthful about where the ladder is.
2. Advance the mechanism model by one tick — the ladder slews, the belt spins up
   or down, the gate opens or shuts, and the depth interlock is checked.
3. If a `WorkCycle` is running, advance it. If it just finished, wake the goal's
   coroutine with the outcome.
4. If no cycle is running, decide the reported state: `FAULT`, `RETRACTING` (not
   travel-ready) or `STOWED`.
5. Send the actuator targets through the backend.
6. Check `/joint_states` for the ladder pitch against the model, and warn **once**
   if they disagree by more than 0.2 rad for 3 s — that is the signature of a
   broken bridge or a missing gz plugin, i.e. a rover that is not really digging.
7. Publish `MechanismStatus` and `BermStatus`, refresh the rviz marker (on change,
   and every 2 s for a late subscriber), and raise the Gazebo berm box if its mean
   height moved by more than `visual.gz_height_step_m`.

`dt` is clamped to 0.5 s so a paused simulator cannot let the ladder jump through
half its travel in one step.

## The action

`rover_navigation_msgs/action/WorkCycle` — module 5 is the client.

**Goal**

| field | meaning |
|---|---|
| `phase` | `"excavate"` or `"dump"` — the same word module 5 sets on `/perception/mode` |
| `work_pose` | where the rover was told to work, in the map frame. For a dump this is **where the pile lands relative to**; for a dig it is only recorded |
| `timeout_s` | `0` = use the per-phase timeout from `excavation.yaml` |
| `target_amount` | `0` = run until the tub is full (dig) or empty (dump) |

**Feedback** every tick while active: `stage` (`STAGE_*`), `current_pose`,
`progress` (monotone 0..1), `detail` (a one-line human description).

**Result**: `success`, `message`, `amount_moved` (m³), `elapsed_s`.

### Refusals

The goal is **aborted immediately**, with a result, if:

* `phase` is not one of `excavate`/`dump`,
* `/estop` is latched,
* the mechanism is in `FAULT`.

A refusal never disturbs a cycle that is already running; superseding does, and
that is deliberate:

### One cycle at a time, newest wins

The mechanism is a single physical machine. A second goal accepted while one is
running supersedes it: the old cycle is stopped with
`message = "superseded by a newer work cycle"`, the ladder is commanded to stow,
and the old goal is returned as a failure. Its `_execute` notices it is no longer
the newest goal by comparing its own sequence number and returns without
touching anything the new goal owns. That is the same pattern module 5's
navigator uses, for the same reason: two goals can be accepted in the same
control tick and their coroutines interleave.

### Cancellation

Module 5's mission node cancels in-flight goals when the run ends or aborts, so
the server honours it properly: `cancel_callback` accepts, stops the cycle with
`message = "canceled by the caller"`, stows, and the goal is transitioned to
`canceled` with `success = False`.

### E-stop

`/estop` is latched and checked both in the tick and when a goal arrives. On a
latched stop the running cycle is aborted, the mechanism is commanded to stow,
and no actuator commands are published while the stop is held. Module 5 aborts
the whole run on a latched e-stop, so this is the second line of defence, not the
first.

## `travel_ready` — the one fact module 5 needs from module 6

`MechanismStatus.travel_ready` is true only when all three of these hold:

* the ladder pitch is at or above `mechanism.interlocks.travel_pitch_max_rad`,
* the belt is stopped (within `BELT_TOL_RADPS`), and
* the tub gate is shut (within `GATE_TOL`).

Module 5 cannot derive any of that: it does not model the digger. A rover driving
across the arena with the ladder down would catch it on the floor, so
`travel_ready` exists to make that impossible to forget. It is also what ends
every work cycle — the cycle is only `DONE` once the mechanism has put itself
back in the travel attitude.

## The berm in the simulator

Gazebo cannot dig: module 1's terrain is a static triangle mesh with the craters
already baked in. So the berm is module 6's height field, and Gazebo gets a
*picture* of it:

* a box of the berm target's footprint and the target height is created once,
  fully buried, through `gz service /world/cat_arena/create`, and
* its `z` is then moved through `set_pose` so its top sits at the berm's current
  **mean height**.

One create and a handful of cheap `set_pose` calls per run; a screenshot at any
moment shows the berm's real height. It is created **visual-only** by default
(`visual.gz_visual_only`): the authoritative obstacle is module 5's planner
keep-out over `berm_target`, and giving modelled sand a collision shape would
make the simulator assert something it does not simulate. If `gz` is missing or
the service fails, the visual disables itself after one warning and the run
continues.

Disable it with `gz_berm:=false` or `visual.gz_berm: false`.

## Integration checklist

Before a competition run, in order:

1. `ros2 run rover_excavation excavation_info.py` — the cycles dry-run and the
   pile lands inside the target.
2. Bring up everything (see the README) and confirm
   `ros2 topic echo /excavation/status --once` shows `travel_ready: true` and
   `state_name: "STOWED"`.
3. Send one work cycle by hand and watch it:
   ```bash
   ros2 action send_goal /excavation/work_cycle rover_navigation_msgs/action/WorkCycle \
     "{phase: excavate, work_pose: {header: {frame_id: map}, pose: {position: {x: 8.0, y: 2.5}, orientation: {w: 1.0}}}, timeout_s: 0.0, target_amount: 0.0}" --feedback
   ```
4. Confirm the Gazebo box rises and `/excavation/berm` reports `dumps: 1` after a
   dump. If `/excavation/status` never leaves `STOWED` and the ladder does not
   move, the gz joint controllers or their bridges are missing — the node warns
   about exactly that once the model and `/joint_states` disagree.

## Parameters

| parameter | default | meaning |
|---|---|---|
| `config` | `""` | `excavation.yaml`; empty = the installed default |
| `backend` | `""` | overrides `backend.kind`: `sim` \| `hardware` \| `null` |
| `rate_hz` | `0.0` | control loop rate; `0` = `mechanism.rate_hz` |
| `gz_berm` | *(unset)* | overrides `visual.gz_berm` |
| `publish_marker` | *(config)* | publish the rviz marker array |
| `use_sim_time` | `false` | set by the launch file |
