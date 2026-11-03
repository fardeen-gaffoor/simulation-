# The work cycle

`rover_excavation/cycle.py`. One `WorkCycle` object per `WorkCycle` action goal,
ROS-free, driven by the node's control loop.

```
STARTING ──► MOVING ──► WORKING ──► RETRACTING ──► DONE
   │            │           │            │
   └────────────┴───────────┴────────────┴──► DONE (failed)
```

The stage numbers are the action's `STAGE_*` constants exactly, because they are
what the caller watches on feedback.

## STARTING

Latch the clock, clear any previous fault, and aim the mechanism:

* **dig** — command the ladder to `dig_pitch_rad`, belt stopped, gate shut. The
  ladder slews down at `max_rate_radps`.
* **dump** — command the ladder to the **stowed** attitude, belt stopped, and the
  **gate fully open**. A dump discharges from the travel attitude; the ladder does
  not need to be down to empty the tub.

Immediately advance to `MOVING`. `_move_total_s` is recorded so `progress` can be
computed against how long the motion *should* take.

## MOVING

Wait for the mechanism to get where it is going. This is real time: 1.9 s for the
ladder to deploy, 0.8 s for the gate to open.

* **dig** → `WORKING` when the pitch has reached `dig_pitch_rad`, and only then is
  the belt started. Running the belt before the ladder is in the sand would just
  spin it in the air.
* **dump** → `WORKING` when the pitch is stowed **and** the gate is open, and only
  then is the belt reversed.

## WORKING

The belt runs; sand moves.

**Dig.** Each tick:

```
rate = coefficient * engaged_width * engaged_depth * belt_surface_speed
accepted = tub.add(rate * dt)          # the overflow spills at the dig face
```

and the cycle ends when the tub reaches `cycle.dig.target_fill` (0.95) *and* it
has run at least `cycle.dig.min_s` (2 s) — or when `target_amount` m³ has been
accepted, if the goal asked for a specific amount.

**Dump.** Each tick, once the gate is open and the belt is reversing:

```
removed = tub.remove(sand.dump.rate_m3_s * dt)
berm.begin_dump()                      # once, on the first sand that leaves
berm.deposit(landing_x, landing_y, removed)
```

where `landing_point` is the map pose the goal carried, displaced
`landing_aft_m` behind the rover. The cycle ends when the tub reaches
`cycle.dump.target_empty` (0.02) and `cycle.dump.min_s` (1.5 s) have passed.

A dump of an **empty tub is a success that moved nothing**, with
`message = "tub was already empty; nothing to place"`. It is not a failure: module
5 counts failures and retries, and retrying an empty tub achieves nothing. The
berm status shows the truth either way (`dumps` is not incremented, because no
pile was started).

## RETRACTING

Stop the belt and shut the gate **immediately**, but hold the ladder where it is
for `cycle.settle_s` (0.4 s) so the last sand can fall before the ladder moves.
After the settle, command the ladder to the travel attitude.

The stage ends when `travel_ready` is true — pitch at or above the travel
threshold, belt stopped, gate shut. That is the whole point of the stage: module 5
must not drive a rover with the ladder down.

## DONE

`success`, `message`, `amount_moved` (m³) and `elapsed_s` are reported back
through the action result. `progress` is 1.0.

## Progress

Monotone, 0..1, and partitioned by stage so the caller sees something sensible
throughout rather than jumping from 0 to 1:

| stage | budget |
|---|---|
| STARTING | 0 |
| MOVING | 0 → 0.15, by remaining ladder travel |
| WORKING | 0.15 → 0.90, by tub fill (dig) or emptying (dump) |
| RETRACTING | 0.90 → 1.00, by remaining ladder travel |

`progress = max(progress, estimate)` every tick, so it never goes backwards even
when an estimate is noisy.

## Failure

`fail(reason)` sets `done`, `success = False`, `progress = 1.0`, and commands the
mechanism to stow (unless a fault already did). Two ways in:

* **Timeout.** `now - start > timeout_s`, where `timeout_s` is the goal's (if
  non-zero) or `cycle.dig.timeout_s` / `cycle.dump.timeout_s`. The message names
  the stage it was in, e.g.
  `timed out after 60.0s (limit 60.0s) during moving`.
* **Mechanism fault.** Checked at the top of every step, and again inside the
  mechanism's own integration. The cycle fails with
  `mechanism fault: <cause>`.

An e-stop or a cancel is handled by the node, which stops the cycle directly
rather than waiting for the next step.

## Timing, as configured

For the shipped configuration, one full dig + dump pair takes ≈ **36 s**:

| | dig | dump |
|---|---|---|
| deploy / open | 1.9 s | 0.8 s |
| work | 27.7 s | 4.3 s |
| settle + stow | 2.3 s | 1.2 s |

so module 5's six cycles are ≈ 200 s of mechanism time, plus the ~6 s of driving
between the excavation zone and the construction zone.

`ros2 run rover_excavation excavation_info.py` prints these numbers from the real
model, so they can be checked without starting anything.
