# The sand and the berm

`rover_excavation/sand.py`, plus the dig/dump rates in `config.py`.

## Why module 6 models the sand at all

**Gazebo cannot dig.** Module 1's arena terrain is a static triangle mesh with a
0.30 m sand layer and the seeded craters baked into it; there is no dynamic
granular material anywhere in the simulation. So module 6 *models* the sand:

* the **tub** is a volume with a capacity,
* the **berm** is a height field over `arena.zones.berm_target`, and
* `rover_excavation_msgs/BermStatus` *is* the berm as far as the software is
  concerned.

The one thing that is not modelled is the ground itself: digging does not lower
the terrain, and the terrain does not push back. The rover's traction is whatever
module 1's friction says, and the wheel odometry error over sand is a known
simulation artefact, not something module 6 can fix.

The planner in module 5 treats `berm_target` as a keep-out from the moment the
first dump lands (it latches `/mission/berm_started` itself). So the navigator and
this model agree about where the berm is **by construction** rather than by
measurement — which is the only way it can work, since a camera cannot see a
0.15 m mound of the same sand it is standing on.

## The tub

```
capacity_m3 = tub.fill_fraction * product(tub.box_m)
            = 0.85 * (0.42 * 0.44 * 0.22) = 0.0346 m3  (34.6 L)
```

The loader refuses a `capacity_m3` that is not `fill_fraction × box volume`, so
the tub cannot silently disagree with the box module 2 builds. `add` accepts only
what fits (the rest spills at the dig face); `remove` returns only what left. The
tub's fill fraction is what ends a dig cycle (at `cycle.dig.target_fill`) and a
dump cycle (at `cycle.dump.target_empty`).

## The dig rate

```
rate = coefficient * engaged_width * engaged_depth * belt_surface_speed
```

clamped into `[sand.dig.min_rate_m3_s, sand.dig.max_rate_m3_s]`, and **zero**
whenever the ladder is not pointed into the sand or the belt is not running
forward.

| term | value | derivation |
|---|---|---|
| `engaged_width` | 0.30 m | ladder width 0.40 − 2 × frame 0.05, derived not configured |
| `engaged_depth` | 0.037 m | `tip_depth_m(dig_pitch)`, capped by the depth interlock |
| `belt_surface_speed` | 0.25 m/s | `dig_velocity_radps × roller_radius` |
| `coefficient` | 0.5 | the one empirical number |

so ≈ **1.4 L/s**, and a tub's worth (at `target_fill` 0.95) takes **24 s**. The
loader refuses a configuration whose dig rate is zero at the dig attitude,
because that tub would never fill and the failure would only show up 60 s into a
run as a timeout.

## Where a dump lands

This is the part of the model that couples to module 5, so it is derived rather
than tuned.

Sand cannot land under the rover. It falls just behind the aft face and forms a
pile whose base spreads by the angle of repose. So the centre of the pile is:

```
landing_aft_m = robot.half_length_m + basal_radius(tub capacity, repose) + landing_margin_m
```

with the cone radius from `V = (π/3) r² h`, `h = r tan θ`:

```
basal_radius = (3V / (π tan θ))^(1/3)
             = (3 × 0.0346 / (π × tan 34°))^(1/3) = 0.366 m
```

so `landing_aft_m = 0.65 + 0.366 = 1.016 m` — the pile's near edge just reaches
the rover's aft face, which is exactly what a real dump does.

Around that centre the material is modelled as a 2-D Gaussian of width
`sand.dump.spread_m` (0.20 m, of the same order as the pile radius; the loader
refuses a spread that is not comparable with the basal radius). The fraction that
falls inside the target rectangle is computed **analytically** — the product of
two normal CDFs — so the spill number is exact rather than a pixel count, and the
placed volume is distributed over the height-field cells in proportion to the
Gaussian (truncated at 3σ).

### The coupling, and how it is tested

Module 5 puts the rover at `y = berm_target.y1 + dump.standoff_m +
robot.circumscribed_radius_m` (1.9882 m for the shipped configuration) and drives
to five poses along the target's long axis, heading `+π/2`, so the discharge side
faces the target. Module 6 does **not** read module 5 at runtime; instead:

* the dump geometry is **copied** into `excavation.yaml` under `nav_dump`, and
* `test/test_cross_checks.py` compares every copied number against
  `rover_navigation/config/navigation.yaml`, recomputes the dump `y` from module
  5's own numbers, and asserts that the pile centre from **every** module 5 dump
  pose lands inside `berm_target`.

If module 5's stand-off or footprint ever changes, that test fails and says to
re-tune `sand.dump` — it does not quietly start spilling sand outside the berm.

### What the model does *not* say

All five dump poses are at the same `y`, so the model builds a **ridge**, not a
flat prism: the pile lands in a band across the target's near half. The model
reports this honestly rather than hiding it:

* `completion` = placed volume / target volume, capped at 1 — the judged yardstick
  is the berm's size, and six cycles (6 × 32.9 L ≈ 0.197 m³ placed against a
  0.169 m³ target) are designed to just about fill it;
* `coverage` = the fraction of target cells at or above the target height, which
  saturates well below 1 for a ridge.

A *uniform* prism at 0.15 m over the whole footprint needs about 1.4 × the target
volume, because a real pile is not flat. Getting closer to that needs a **dump
pattern with more than one row** across the target's 0.9 m depth, which is a
module 5 mission decision — see the findings in `~/cat/REPORT.md`.

## The berm height field

* `cell_m: 0.05` → 25 × 18 = 450 cells over 1.25 × 0.9 m.
* `deposit(x, y, volume)` adds `volume / (dx·dy)` × the normalised kernel to the
  cells within 3σ of the landing point.
* `begin_dump()` latches one dump per **pile**, not per control tick. The node
  deposits a little sand every tick while the tub empties (so the berm grows
  smoothly and a screenshot is never stale), but `dumps` counts work cycles.

Reported: `placed_volume_m3` (exactly `sum(height) × cell_area`),
`spilled_volume_m3`, `completion`, `coverage`, `mean_height_m`, `peak_height_m`,
`cells_filled`, `dumps`.

## Re-tuning

Everything is in `excavation.yaml` under `sand:`. In order of likelihood:

| symptom | change |
|---|---|
| the pile lands outside the target | re-check module 5's `dump.standoff_m` first; then `sand.dump.landing_margin_m` (negative is not allowed — the margin only pushes further away) |
| too much spill | the pile is on the target's edge: a wider `spread_m` lowers the peak but widens the pile; a *pattern* change in module 5 is the real fix |
| the berm never completes | `cycle.dig.target_fill` (how full the tub is when a dig stops) or module 5's `cycles_target` |
| cycles take too long | `sand.dig.coefficient`, or `cycle.dig.timeout_s` if it is timing out |

`ros2 run rover_excavation excavation_info.py --cycles 6` shows the effect of any
of these in about a second, with no simulator.
