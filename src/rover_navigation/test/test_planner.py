"""A* on the costmap, then string-pulling.

The planner is the piece that turns "the rover may not be in these cells" into
"drive here, then here".  The properties worth pinning are:

* **a clear shot is a clear shot** - no search, one leg, exactly the straight
  line, because that is the common case and grid artefacts must not appear in it;
* **every leg of the answer is drivable** - re-checked against the same map, so a
  smoothed or corner-cut path cannot sneak through a lethal cell;
* **the map is respected in both directions** - a wall of obstacles makes the
  goal unreachable and the planner says so instead of inventing a route, and a
  real gap stays usable;
* **it always terminates** - the expansion budget turns a pathological map into a
  refusal, not a hang.
"""

from __future__ import annotations

import math

import pytest

from rover_navigation.config import Rect
from rover_navigation.costmap import Costmap
from rover_navigation.geometry import wrap_angle
from rover_navigation.planner import PlanResult, Planner

from support import boulder, config_with, obstacle_line

def _planner(config) -> Planner:
    return Planner(config)


def _sealed_wall(config, x: float = 4.5, count: int = 40, radius: float = 0.18):
    """A line of boulders tall enough to seal the whole free band."""
    return obstacle_line((x, 0.6), (x, 4.4), count=count, radius=radius)


def _partial_wall(config, x: float = 4.5):
    """A wall that seals the middle of the band but leaves the south gap open.

    With a lethal radius of ~0.59 m each boulder seals y = 1.0 .. 5.0 around
    x = 4.5, so a route along y ~ 0.7 exists and a route along y = 2.5 does not.
    """
    return obstacle_line((x, 1.6), (x, 4.4), count=22, radius=0.18)


def _heading_changes(waypoints, threshold_rad: float) -> int:
    """How many times the densified path changes direction by more than the threshold.

    A staircase of 5 cm cells has a change at nearly every waypoint; a
    string-pulled path has one per corner it rounds.
    """
    changes = 0
    previous = None
    for index in range(1, len(waypoints)):
        ax, ay = waypoints[index - 1]
        bx, by = waypoints[index]
        heading = math.atan2(by - ay, bx - ax)
        if previous is not None and abs(wrap_angle(heading - previous)) > threshold_rad:
            changes += 1
        previous = heading
    return changes


def _assert_path_is_drivable(costmap: Costmap, waypoints) -> None:
    assert len(waypoints) >= 2
    for x, y in waypoints:
        assert costmap.is_free(x, y), f"waypoint ({x}, {y}) is lethal"
    for index in range(1, len(waypoints)):
        ax, ay = waypoints[index - 1]
        bx, by = waypoints[index]
        assert costmap.segment_clear(ax, ay, bx, by), f"leg {index} clips something"


def _max_spacing(waypoints) -> float:
    return max(
        math.hypot(waypoints[i][0] - waypoints[i - 1][0], waypoints[i][1] - waypoints[i - 1][1])
        for i in range(1, len(waypoints))
    )


# =========================================================================== #
# the straight shot
# =========================================================================== #
def test_a_clear_goal_is_a_single_straight_leg(config):
    costmap = Costmap(config, [])
    result = _planner(config).plan(costmap, (1.0, 4.0), (6.0, 4.0))
    assert result.success
    assert result.straight is True
    assert result.expanded == 0
    assert result.length_m == pytest.approx(5.0)
    assert result.waypoints[0] == pytest.approx((1.0, 4.0))
    assert result.waypoints[-1] == pytest.approx((6.0, 4.0))


def test_a_straight_leg_is_densified_for_the_controller(config):
    costmap = Costmap(config, [])
    result = _planner(config).plan(costmap, (1.0, 4.0), (6.0, 4.0))
    assert _max_spacing(result.waypoints) <= 2.0 * config.costmap.resolution_m + 1e-9
    assert len(result.waypoints) > 2


def test_a_straight_leg_is_monotone_along_its_direction(config):
    costmap = Costmap(config, [])
    result = _planner(config).plan(costmap, (1.0, 4.0), (6.0, 4.0))
    xs = [x for x, _ in result.waypoints]
    assert xs == sorted(xs)
    assert all(y == pytest.approx(4.0) for _, y in result.waypoints)


def test_a_zero_length_goal_succeeds_trivially(config):
    costmap = Costmap(config, [])
    result = _planner(config).plan(costmap, (3.0, 3.0), (3.0, 3.0))
    assert result.success
    assert result.length_m == pytest.approx(0.0)


# =========================================================================== #
# around obstacles
# =========================================================================== #
def test_a_wall_of_boulders_is_routed_around(config):
    """A short wall that does not span the arena: there is a way past it."""
    costmap = Costmap(config, _partial_wall(config))
    assert not costmap.segment_clear(1.0, 2.5, 8.0, 2.5)
    result = _planner(config).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert result.success, result.message
    assert result.straight is False
    assert result.expanded > 0
    assert result.length_m > 7.0
    _assert_path_is_drivable(costmap, result.waypoints)


def test_the_path_is_not_a_staircase(config):
    """String-pulling must collapse the cell chain into a handful of legs.

    The waypoints the controller sees are resampled, so counting them says
    nothing about smoothing; counting *direction changes* does.  A raw 8-connected
    cell chain across a 7 m detour changes heading at nearly every one of its
    ~140 cells, a string-pulled path a handful of times.
    """
    costmap = Costmap(config, _partial_wall(config))
    result = _planner(config).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert result.success, result.message
    assert len(result.waypoints) > 60                     # resampled, so many
    assert _heading_changes(result.waypoints, math.radians(15.0)) < 15


def test_a_detour_costs_more_than_the_straight_line(config):
    costmap = Costmap(config, _partial_wall(config))
    planner = _planner(config)
    detour = planner.plan(costmap, (1.0, 2.5), (8.0, 2.5))
    open_map = Costmap(config, [])
    straight = planner.plan(open_map, (1.0, 2.5), (8.0, 2.5))
    assert detour.length_m > straight.length_m


def test_the_path_keeps_its_distance_from_a_lone_boulder(config):
    """The soft cost makes the planner swing wide rather than graze."""
    record = boulder(4.5, 2.5, 0.15)
    costmap = Costmap(config, [record])
    result = _planner(config).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert result.success
    _assert_path_is_drivable(costmap, result.waypoints)
    # The closest approach is outside the lethal disc, not merely outside the
    # boulder, and there is at least one waypoint that is comfortably clear.
    closest = min(math.hypot(x - 4.5, y - 2.5) for x, y in result.waypoints)
    assert closest > costmap.lethal_radius(record.radius, record.kind)


def test_a_corner_is_not_cut(config):
    """Two obstacles forming an L: the path must go round, not through the corner."""
    records = [
        boulder(4.4, 2.0, 0.30),
        boulder(5.0, 1.6, 0.30),
    ]
    costmap = Costmap(config, records)
    result = _planner(config).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert result.success, result.message
    _assert_path_is_drivable(costmap, result.waypoints)


# =========================================================================== #
# refusals
# =========================================================================== #
def test_a_sealed_arena_reports_no_path_rather_than_inventing_one(config):
    costmap = Costmap(config, [], [Rect(4.0, -1.0, 4.3, 6.0)])
    result = _planner(config).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert result.success is False
    assert result.waypoints == []
    assert "no path" in result.message


def test_the_budget_turns_a_big_search_into_a_refusal(config_file, tmp_path):
    cfg = config_with(config_file, tmp_path, {"planner.max_expansions": 1})
    costmap = Costmap(cfg, _sealed_wall(cfg))
    result = Planner(cfg).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert result.success is False
    assert "expansions" in result.message
    assert result.expanded >= 1


def test_a_start_buried_under_a_keep_out_is_refused(config):
    costmap = Costmap(config, [], [Rect(-1.0, -1.0, 10.0, 6.0)])
    result = _planner(config).plan(costmap, (4.0, 2.5), (5.0, 2.5))
    assert result.success is False
    assert "start" in result.message


def test_a_goal_inside_a_boulder_is_rescued_to_the_nearest_free_cell(config_file, tmp_path):
    # A 0.15 m boulder's lethal disc is 0.56 m across, so the rescue needs a
    # radius bigger than the default half metre to find a free cell at all.
    cfg = config_with(config_file, tmp_path, {"planner.snap_radius_m": 1.0})
    record = boulder(6.0, 2.5, 0.15)
    costmap = Costmap(cfg, [record])
    result = Planner(cfg).plan(costmap, (1.0, 2.5), (6.0, 2.5))
    assert result.success
    assert result.goal_snapped is True
    assert costmap.is_free(*result.waypoints[-1])
    # ... and the snapped goal is still within the radius that was asked for.
    distance = math.hypot(result.waypoints[-1][0] - 6.0, result.waypoints[-1][1] - 2.5)
    assert distance <= cfg.planner.snap_radius_m


def test_a_goal_inside_a_boulder_is_refused_when_no_free_cell_is_close_enough(config):
    """The rescue has a radius, and outside it the goal really is unreachable."""
    record = boulder(6.0, 2.5, 0.15)     # lethal disc radius 0.56 m
    costmap = Costmap(config, [record])
    assert config.planner.snap_radius_m < 0.56
    result = _planner(config).plan(costmap, (1.0, 2.5), (6.0, 2.5))
    assert result.success is False
    assert "goal" in result.message


def test_a_start_inside_a_boulder_is_rescued_too(config_file, tmp_path):
    cfg = config_with(config_file, tmp_path, {"planner.snap_radius_m": 1.0})
    record = boulder(1.2, 2.5, 0.15)
    costmap = Costmap(cfg, [record])
    result = Planner(cfg).plan(costmap, (1.2, 2.5), (8.0, 2.5))
    assert result.success
    assert result.start_snapped is True
    assert costmap.is_free(*result.waypoints[0])


def test_a_goal_outside_the_arena_is_refused(config):
    costmap = Costmap(config, [])
    result = _planner(config).plan(costmap, (1.0, 2.5), (20.0, 2.5))
    assert result.success is False
    assert result.message


# =========================================================================== #
# shape of the answer
# =========================================================================== #
def test_waypoints_are_resampled_to_the_configured_spacing(config):
    costmap = Costmap(config, _partial_wall(config))
    result = _planner(config).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert result.success
    assert _max_spacing(result.waypoints) <= 2.0 * config.costmap.resolution_m + 1e-6


def test_legs_is_waypoints_minus_one(config):
    costmap = Costmap(config, [])
    result = _planner(config).plan(costmap, (1.0, 4.0), (6.0, 4.0))
    assert result.legs == len(result.waypoints) - 1


def test_the_reported_length_matches_the_polyline(config):
    costmap = Costmap(config, _partial_wall(config))
    result = _planner(config).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert result.length_m == pytest.approx(
        sum(
            math.hypot(
                result.waypoints[i][0] - result.waypoints[i - 1][0],
                result.waypoints[i][1] - result.waypoints[i - 1][1],
            )
            for i in range(1, len(result.waypoints))
        )
    )


def test_planning_is_deterministic(config):
    costmap = Costmap(config, _partial_wall(config))
    planner = _planner(config)
    first = planner.plan(costmap, (1.0, 2.5), (8.0, 2.5))
    second = planner.plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert first.waypoints == second.waypoints
    assert first.length_m == second.length_m


def test_smoothing_can_be_turned_off_without_breaking_the_path(config_file, tmp_path):
    cfg = config_with(config_file, tmp_path, {"planner.smoothing": False})
    costmap = Costmap(cfg, _partial_wall(cfg))
    result = Planner(cfg).plan(costmap, (1.0, 0.75), (8.0, 0.75))
    assert result.success, result.message
    _assert_path_is_drivable(costmap, result.waypoints)


def test_the_plan_result_defaults_are_empty():
    result = PlanResult(success=False)
    assert result.waypoints == []
    assert result.legs == 0
    assert result.length_m == 0.0
    assert result.message == ""


def test_the_search_expands_cells_but_not_the_whole_grid(config):
    costmap = Costmap(config, _partial_wall(config))
    result = _planner(config).plan(costmap, (1.0, 2.5), (8.0, 2.5))
    assert 0 < result.expanded < costmap.nx * costmap.ny
