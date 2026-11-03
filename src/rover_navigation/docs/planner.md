# The obstacle map and the planner — what the rover plans on, and how

Two files: `costmap.py` turns a stream of partial, forgetful perception frames
into a map that is *sticky where it should be* and *healing where it must be*;
`planner.py` searches that map for a route. Neither talks to ROS.

---

## Why module 5 keeps its own memory

Module 4's detector sees only what is in front of `cam_nav`, and it decays a
track the moment it leaves the field of view. That is correct for a perception
node — a detector that remembers things it cannot see is a detector that lies —
but it is the wrong input for a planner. If the map forgets a boulder when the
rover turns away from it, the next plan drives straight through the hole that
used to be occupied by rock.

So `ObstacleMemory` accumulates module 4's confirmed observations, and the
competition helps: **the obstacles are fixed in place for the whole run.** A
boulder seen once is at that position for the next ten minutes. Remembering it
is not a hack; it is the only honest use of the fact that `boulders[]` is
seeded once per run and never moves.

The tension is the other direction: module 4's classical bootstrap detector also
produces **false positives** — painted zone lines and cast shadows both show up
as obstacles at long range. A map that remembered those forever would close the
arena on itself. So memory is *finite*.

### Gate, associate, merge

An observation becomes a memory only if it passes two gates:

| gate | default | why |
|---|---|---|
| `min_observations` | 3 | a shadow flickers once; a boulder is seen on every frame it is in view |
| `min_confidence` | 0.30 | module 4's tracker confidence; below this the detection is a guess |

Association is **by distance, not by id.** Module 4's track ids are scoped to a
single detector run and are not stable across a rover that turns away and back;
two observations that are 0.20 m apart are the same boulder, and
`memory_associate_m` (0.35) is the radius within which they are merged. The merge
is deliberately **conservative**:

* position blends with gain 0.5 (a running average, so a single bad frame cannot
  drag a remembered boulder),
* `kind` escalates to **crater** (the more dangerous of the two — a boulder can be
  hit, a crater can be driven into),
* radius, confidence and observation count take the **maximum**.

`obstacle_memory_s` (60 s) is the lifetime. The competition obstacles never move
and the seeding is fixed, so sixty seconds outlives any plausible occlusion; but
it does not outlive a *false* positive, which is the point. It is a config knob:
set it to `0` and the memory keeps a confirmed obstacle for the whole run, which
is the right setting once the learned detector replaces the bootstrap one.

---

## The cost grid

`Costmap` rasterises the arena at `resolution_m` (0.05 m) into a
180 × 100 grid. Each memorised obstacle is drawn as a **lethal disc** whose
radius is the obstacle's own radius plus the rover's *inscribed* radius plus
clearance:

```
lethal_r = obstacle_r * kind_scale + robot.inscribed_radius_m + footprint_clearance_m
```

with `kind_scale` = `boulder_radius_scale` (1.00) for boulders and
`crater_radius_scale` (1.20) for craters — a crater is inflated past its rim
because `module 4` marks it lethal and a wheel near the rim can collapse the
edge. Around the lethal disc sits a **soft-cost band** `inflation_m` (0.10) wide
worth `soft_cost` (6.0) per cell, which the A\* search pays but does not forbid:
it makes the planner prefer the middle of a gap without making a tight gap
illegal.

The robot's *inscribed* radius — half the width, 0.35 m — is what a body
translating along a path actually sweeps. The **circumscribed** radius (0.7382 m)
is only ever used for the dump stand-off and the swept-circle test in
`docs/mission.md`; inflating every obstacle by it seals the arena (a 0.84 m
circle around each 35 cm boulder closes the 5.5 m obstacle zone).

Two details of the grid matter at the edges:

* **`unknown_is_free`.** There is no unknown layer. The arena is bounded and
  small and the camera sees forward only, so treating unseen cells as free is the
  only way to commit to a plan at all. It is safe because the map is rebuilt
  continuously and the rover re-plans every `replan_period_s` (1.0 s).
* **`nearest_free(x, y, max_radius)` is Euclidean-bounded.** A start or goal that
  lands in an obstacle is nudged to the closest free cell *within* a true radius;
  the bound used to be a Chebyshev ring and could return a cell up to
  √2 · max_radius away. The consequence of the fix is honest: a goal inside a
  0.15 m boulder (lethal disc 0.56 m) is now **refused** at the default
  `snap_radius_m` of 0.50, rather than silently snapped to a cell a metre away.

---

## The search

`Planner.plan(costmap, start, goal)`:

1. **Straight shot.** If `segment_clear(start, goal)` — the whole segment stays
   out of every lethal disc — return `[start, goal]` densified, with
   `PlanResult.straight = True`. The common case is "drive down the corridor" and
   it does not deserve a search.
2. **Resolve endpoints.** Snap a blocked start or goal to a free cell within
   `snap_radius_m`; refuse if there is none, with a message that says which
   endpoint failed.
3. **A\*** — 8-connected, octile heuristic, `heuristic_weight` 1.0 (so the search
   is optimal; a weight above 1 makes it greedy and slightly longer, which is why
   it is not the default). Two rules keep it honest for a big rectangular rover:
   * **no corner cutting**: a diagonal step is refused if either orthogonal
     neighbour is lethal. An inflated obstacle's corner is still an obstacle, and
     a corridor that is only diagonal-clear is not a corridor a 0.70 m wide rover
     fits through.
   * diagonal steps cost √2, straight steps cost 1 — both scaled by
     `cost_at`, so routing through the soft band is discouraged but not banned.
   `max_expansions` (200000) bounds the worst case and reports *why* the search
   gave up rather than hanging.
4. **String-pull** (greedy shortcutting): from the current anchor, keep the
   farthest point still `segment_clear` from it, then repeat. This removes the
   grid's staircase from a straight run.
5. **Thin and densify.** Thin drops waypoints closer than `waypoint_spacing_m`
   (0.10); densify resamples so no two waypoints are more than two cells apart,
   which is what the controller's arc-length lookahead wants.

`PlanResult` carries the *why* alongside the path: `expanded` (search effort),
`straight`, `start_snapped`, `goal_snapped`, and a `message` on failure that a
human can act on ("goal is inside an obstacle and no free cell is close enough").
`PlanResult.legs` is `waypoints − 1` — a count of polyline *segments*, not of the
planner's straight legs, because densify has already resampled them.

---

## What this deliberately does not do

* **No nav2.** Not installed on the rover computer, and a 9 × 5 m arena with a
  handful of static obstacles does not need a behaviour tree or a lattice
  planner.
* **No dynamic obstacles.** The seeded boulders and craters do not move. The map
  is rebuilt from `ObstacleMemory` every replan; there is no notion of an
  obstacle with velocity.
* **No precomputed arena.** Module 5 never reads `ground_truth.yaml`. Every
  lethal cell in the grid came from a live detection that passed the gates above.
