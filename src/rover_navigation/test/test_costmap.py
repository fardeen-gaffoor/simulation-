"""Obstacle memory and the costmap it is rasterised into.

Two things have to hold, and they fail in different ways:

* **The memory must not forget an obstacle the rover has driven past.**  Module
  4's tracker decays a track when it leaves the camera's view - correct for
  perception, fatal for planning - so the memory accumulates by proximity and
  only forgets on a *time* budget.
* **The rasterisation must not under-approximate the rover.**  A 1.3 x 0.7 m
  rover that turns in place sweeps a circle, so every obstacle is grown by that
  radius (plus clearance) before a cell is marked; the planner then treats the
  rover as a point.  A test that only checked "the boulder's own cells are
  lethal" would pass while the rover planned through the boulder's corner.

The obstacle scenarios here are synthetic on purpose: module 5's premise is that
obstacles come from live perception, so a test that read ``ground_truth.yaml``
would be testing the wrong pipeline.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rover_navigation.costmap import (
    KIND_BOULDER,
    KIND_CRATER,
    KIND_UNKNOWN,
    Costmap,
    ObstacleMemory,
    ObstacleRecord,
)

from support import boulder, config_with, crater, obstacle_line, point_obstacle

# =========================================================================== #
# ObstacleMemory
# =========================================================================== #
def test_an_empty_memory_is_empty(config):
    memory = ObstacleMemory(config)
    assert len(memory) == 0
    assert memory.records == []
    assert memory.accepted == 0


def test_a_confirmed_observation_is_remembered(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15)], now=1.0)
    assert len(memory) == 1
    record = memory.records[0]
    assert (record.x, record.y) == (4.0, 2.0)
    assert record.first_seen == 1.0
    assert record.last_seen == 1.0
    assert memory.accepted == 1
    assert memory.rejected == 0
    assert memory.merged == 0


def test_a_zero_radius_observation_is_rejected(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.0)], now=0.0)
    assert len(memory) == 0
    assert memory.rejected == 1


def test_a_low_confidence_observation_is_rejected(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, confidence=config.costmap.min_confidence - 0.01)], now=0.0)
    assert len(memory) == 0
    assert memory.rejected == 1


def test_an_under_observed_observation_is_rejected(config):
    """One frame is a candidate, not an obstacle: module 4's tracker decides."""
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, observations=config.costmap.min_observations - 1)], now=0.0)
    assert len(memory) == 0
    assert memory.rejected == 1


def test_an_observation_exactly_on_the_threshold_is_kept(config):
    memory = ObstacleMemory(config)
    memory.update(
        [
            boulder(
                4.0,
                2.0,
                0.15,
                confidence=config.costmap.min_confidence,
                observations=config.costmap.min_observations,
            )
        ],
        now=0.0,
    )
    assert len(memory) == 1


def test_a_weak_observation_does_not_disturb_a_confirmed_one(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="boulder_0")], now=0.0)
    memory.update([boulder(4.01, 2.0, 0.15, identifier="junk", confidence=0.01)], now=1.0)
    assert len(memory) == 1
    assert memory.records[0].identifier == "boulder_0"
    assert memory.records[0].last_seen == 0.0


def test_a_nearby_observation_merges_into_the_same_obstacle(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="first")], now=0.0)
    memory.update([boulder(4.0 + 0.5 * config.costmap.memory_associate_m, 2.0, 0.15, identifier="second")], now=1.0)
    assert len(memory) == 1
    assert memory.merged == 1
    assert memory.records[0].identifier == "first"


def test_an_observation_just_outside_the_association_window_is_a_new_obstacle(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="first")], now=0.0)
    memory.update(
        [boulder(4.0 + 1.5 * config.costmap.memory_associate_m, 2.0, 0.15, identifier="second")], now=0.0
    )
    assert len(memory) == 2


def test_merging_blends_the_position_rather_than_snapping(config):
    """One bad frame must not teleport an obstacle the planner has routed around."""
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="a")], now=0.0)
    memory.update([boulder(4.2, 2.0, 0.15, identifier="b")], now=0.0)
    record = memory.records[0]
    assert record.x == pytest.approx(4.1)   # gain 0.5
    assert record.y == pytest.approx(2.0)


def test_merging_keeps_the_larger_radius(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.10, identifier="a")], now=0.0)
    memory.update([boulder(4.0, 2.0, 0.25, identifier="b")], now=0.0)
    assert memory.records[0].radius == pytest.approx(0.25)
    memory.update([boulder(4.0, 2.0, 0.05, identifier="c")], now=0.0)
    assert memory.records[0].radius == pytest.approx(0.25)


def test_merging_escalates_a_boulder_to_a_crater_and_never_back(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="a")], now=0.0)
    memory.update([crater(4.0, 2.0, 0.15, identifier="b")], now=0.0)
    assert memory.records[0].kind == KIND_CRATER
    # A later boulder sighting must not undo the escalation.
    memory.update([boulder(4.0, 2.0, 0.15, identifier="c")], now=0.0)
    assert memory.records[0].kind == KIND_CRATER


def test_merging_takes_the_best_evidence(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="a", confidence=0.4, observations=3)], now=0.0)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="b", confidence=0.9, observations=12)], now=0.0)
    record = memory.records[0]
    assert record.confidence == pytest.approx(0.9)
    # observations is the tracker's count at this sighting, so the larger wins.
    assert record.observations == 12


def test_merging_keeps_the_conservative_flag(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="a", conservative=True)], now=0.0)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="b", conservative=False)], now=0.0)
    assert memory.records[0].conservative is True


def test_merging_refreshes_last_seen(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="a")], now=0.0)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="b")], now=5.0)
    assert memory.records[0].last_seen == 5.0
    assert memory.records[0].first_seen == 0.0


def test_an_unknown_kind_merging_into_a_boulder_keeps_the_boulder(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="a")], now=0.0)
    unknown = ObstacleRecord("b", KIND_UNKNOWN, 4.0, 2.0, 0.15, 0.9, 5)
    memory.update([unknown], now=0.0)
    assert memory.records[0].kind == KIND_BOULDER


def test_a_boulder_seen_from_the_other_side_is_not_forgotten(config):
    """The whole reason the memory exists: occlusion must not re-open a hole."""
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15)], now=0.0)
    # ... ten seconds of seeing nothing, because the rover drove past it.
    memory.update([], now=10.0)
    assert len(memory) == 1


def test_memory_expires_on_its_time_budget(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15)], now=0.0)
    memory.update([], now=config.costmap.obstacle_memory_s + 0.01)
    assert len(memory) == 0


def test_memory_never_expires_when_the_budget_is_zero(config_file, tmp_path):
    cfg = config_with(config_file, tmp_path, {"costmap.obstacle_memory_s": 0.0})
    memory = ObstacleMemory(cfg)
    memory.update([boulder(4.0, 2.0, 0.15)], now=0.0)
    memory.update([], now=10_000.0)
    assert len(memory) == 1


def test_the_obstacle_count_is_capped(config_file, tmp_path):
    cfg = config_with(config_file, tmp_path, {"costmap.max_obstacles": 3})
    memory = ObstacleMemory(cfg)
    for index in range(10):
        memory.update(
            [boulder(1.0 + index * 1.0, 1.0, 0.1, identifier=f"b{index}", confidence=0.3 + 0.05 * index)],
            now=0.0,
        )
    assert len(memory) == 3
    # The most confident survive.
    assert {record.identifier for record in memory.records} == {"b7", "b8", "b9"}


def test_stale_reports_what_has_not_been_reconfirmed(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15, identifier="old")], now=0.0)
    memory.update([boulder(6.0, 2.0, 0.15, identifier="fresh")], now=10.0)
    stale = memory.stale(now=10.0)
    assert [record.identifier for record in stale] == ["old"]
    assert memory.stale(now=0.0) == []


def test_clear_empties_the_records_but_not_the_counters(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15)], now=0.0)
    memory.clear()
    assert len(memory) == 0
    assert memory.accepted == 1


def test_records_are_returned_as_a_copy(config):
    memory = ObstacleMemory(config)
    memory.update([boulder(4.0, 2.0, 0.15)], now=0.0)
    records = memory.records
    records.clear()
    assert len(memory) == 1


def test_a_record_knows_its_kind_name_and_distance():
    record = ObstacleRecord("b", KIND_CRATER, 3.0, 4.0, 0.2, 0.9, 5)
    assert record.kind_name == "crater"
    assert record.distance_to(0.0, 0.0) == pytest.approx(5.0)
    assert record.age(10.0) == pytest.approx(10.0)  # never re-seen since t=0
    assert record.age(-5.0) == 0.0                   # a clock that went backwards
    assert ObstacleRecord("b", KIND_UNKNOWN, 0.0, 0.0, 0.1).kind_name == "unknown"


def test_a_record_copy_is_independent():
    record = boulder(1.0, 2.0)
    clone = record.copy()
    clone.x = 9.0
    assert record.x == 1.0


# =========================================================================== #
# Costmap: the walls
# =========================================================================== #
def test_the_grid_covers_the_arena_at_the_configured_resolution(config):
    costmap = Costmap(config, [])
    assert costmap.nx == int(round(config.arena.length_x / config.costmap.resolution_m))
    assert costmap.ny == int(round(config.arena.width_y / config.costmap.resolution_m))
    assert costmap.lethal.shape == (costmap.ny, costmap.nx)
    assert costmap.cost.shape == (costmap.ny, costmap.nx)


def test_an_empty_arena_still_has_walls(config):
    costmap = Costmap(config, [])
    # The corners of the arena are outside free_rect and must be lethal.
    assert costmap.is_lethal_cell(0, 0)
    assert costmap.is_lethal_cell(costmap.nx - 1, costmap.ny - 1)
    assert costmap.free_cells() > 0
    assert 0.0 < costmap.lethal_fraction() < 1.0


def test_the_free_region_matches_the_free_rect(config):
    costmap = Costmap(config, [])
    free = config.free_rect
    # A cell just inside the free rect is free; one just outside is not.
    assert costmap.is_free(free.x0 + 0.5 * config.costmap.resolution_m, free.y0 + 0.5 * config.costmap.resolution_m)
    assert not costmap.is_free(free.x0 - config.arena.wall_margin_m, free.y0 + 1.0)
    assert costmap.is_free(0.5 * config.arena.length_x, 0.5 * config.arena.width_y)


def test_out_of_bounds_is_lethal_and_never_free(config):
    costmap = Costmap(config, [])
    assert not costmap.is_free(-1.0, 2.0)
    assert not costmap.is_free(100.0, 2.0)
    assert costmap.is_lethal_cell(-1, 0)
    assert costmap.cost_at(-1, 0) == Costmap.COST_LETHAL


def test_a_segment_through_a_wall_is_not_clear(config):
    costmap = Costmap(config, [])
    assert not costmap.segment_clear(4.0, 2.0, 20.0, 2.0)
    assert not costmap.segment_clear(-0.5, 2.0, 4.0, 2.0)


def test_a_segment_in_the_open_arena_is_clear(config):
    costmap = Costmap(config, [])
    assert costmap.segment_clear(1.0, 4.0, 4.0, 4.0)


# =========================================================================== #
# Costmap: obstacles
# =========================================================================== #
def test_the_lethal_radius_is_the_scaled_radius_plus_the_inflation(config):
    costmap = Costmap(config, [])
    expected = 0.20 * config.costmap.boulder_radius_scale + config.robot.obstacle_inflation_m
    assert costmap.lethal_radius(0.20, KIND_BOULDER) == pytest.approx(expected)


def test_a_crater_is_grown_more_than_a_boulder(config):
    costmap = Costmap(config, [])
    assert costmap.lethal_radius(0.20, KIND_CRATER) > costmap.lethal_radius(0.20, KIND_BOULDER)


def test_a_negative_radius_cannot_produce_a_negative_lethal_radius(config):
    costmap = Costmap(config, [])
    assert costmap.lethal_radius(-1.0, KIND_BOULDER) == pytest.approx(config.robot.obstacle_inflation_m)


def test_an_obstacle_makes_its_own_centre_lethal(config):
    costmap = Costmap(config, [boulder(4.0, 2.0, 0.15)])
    assert not costmap.is_free(4.0, 2.0)
    assert costmap.obstacle_count == 1


def test_the_danger_around_an_obstacle_is_larger_than_the_obstacle(config):
    """The rover is 1.3 x 0.7 m and turns in place, so the map must be bigger."""
    radius = 0.15
    costmap = Costmap(config, [boulder(4.0, 2.0, radius)])
    lethal = costmap.lethal_radius(radius, KIND_BOULDER)
    # Just outside the boulder's own edge but inside the swept circle: lethal.
    probe = 0.5 * (radius + lethal)
    assert probe > radius
    assert not costmap.is_free(4.0 + probe, 2.0)
    # Well outside the swept circle: free.
    assert costmap.is_free(4.0 + lethal + 4 * config.costmap.resolution_m, 2.0)


def test_the_swept_circle_is_the_configured_inflation(config):
    radius = 0.15
    costmap = Costmap(config, [boulder(4.0, 2.0, radius)])
    lethal = costmap.lethal_radius(radius, KIND_BOULDER)
    assert lethal == pytest.approx(radius * config.costmap.boulder_radius_scale + config.robot.obstacle_inflation_m)
    # A point 1 cm inside the boundary is lethal; 20 cm outside is not.
    assert not costmap.is_free(4.0 + lethal - 0.01, 2.0)
    assert costmap.is_free(4.0 + lethal + 0.20, 2.0)


def test_a_wall_of_obstacles_can_seal_a_region(config):
    """The map must be able to *say no* - a false 'there is a way through' is worse."""
    wall = obstacle_line((4.0, 0.6), (4.0, 4.4), count=30, radius=0.15)
    costmap = Costmap(config, wall)
    assert not costmap.segment_clear(2.0, 2.5, 6.0, 2.5)


def test_a_gap_wider_than_the_swept_circle_stays_open(config):
    """And a real gap must not be closed by the inflation."""
    clearance = config.robot.obstacle_inflation_m + 0.15
    gap = 2.0 * clearance + 0.30
    costmap = Costmap(
        config,
        [
            boulder(4.0, 2.5 - gap / 2, 0.15),
            boulder(4.0, 2.5 + gap / 2, 0.15),
        ],
    )
    assert costmap.is_free(4.0, 2.5)
    assert costmap.segment_clear(2.5, 2.5, 5.5, 2.5)


def test_a_crater_is_drawn_bigger_than_a_boulder_of_the_same_radius(config):
    same_radius = 0.18
    boulder_map = Costmap(config, [boulder(4.0, 2.0, same_radius)])
    crater_map = Costmap(config, [crater(4.0, 2.0, same_radius)])
    assert crater_map.lethal.sum() > boulder_map.lethal.sum()


# =========================================================================== #
# Costmap: keep-outs
# =========================================================================== #
def test_a_keep_out_rectangle_is_lethal_inside(config):
    from rover_navigation.config import Rect

    rect = Rect(3.5, 1.5, 4.5, 3.5)
    costmap = Costmap(config, [], [rect])
    assert not costmap.is_free(4.0, 2.5)
    assert costmap.is_free(2.0, 2.5)
    assert costmap.keep_out_count == 1


def test_a_keep_out_is_grown_by_the_swept_circle(config):
    from rover_navigation.config import Rect

    rect = Rect(3.5, 1.5, 4.5, 3.5)
    costmap = Costmap(config, [], [rect])
    inflation = config.robot.obstacle_inflation_m
    res = config.costmap.resolution_m
    # Rasterised: a cell is lethal when its *centre* is inside the grown rect,
    # so the probes stay a cell clear of the boundary on either side.
    assert not costmap.is_free(rect.x1 + inflation - res, 2.5)
    assert costmap.is_free(rect.x1 + inflation + 2 * res, 2.5)


def test_the_source_grid_records_what_made_a_cell_lethal(config):
    from rover_navigation.config import Rect

    costmap = Costmap(config, [boulder(2.5, 2.0, 0.15)], [Rect(6.0, 1.0, 7.0, 4.0)])
    i, j = costmap.world_to_cell(2.5, 2.0)
    assert costmap.source[j, i] == 1                     # obstacle
    i, j = costmap.world_to_cell(6.5, 2.5)
    assert costmap.source[j, i] == 2                     # keep-out
    assert costmap.source[0, 0] == 0                     # wall


# =========================================================================== #
# Costmap: the soft cost band
# =========================================================================== #
def test_free_space_away_from_everything_costs_one(config):
    costmap = Costmap(config, [boulder(4.0, 2.0, 0.15)])
    i, j = costmap.world_to_cell(1.0, 4.0)
    assert costmap.cost_at(i, j) == pytest.approx(1.0)


def test_the_band_next_to_a_lethal_cell_costs_more_but_is_passable(config):
    costmap = Costmap(config, [boulder(4.0, 2.0, 0.15)])
    lethal = costmap.lethal_radius(0.15, KIND_BOULDER)
    # Inside the soft band, outside the lethal disc.
    probe = lethal + 0.5 * config.costmap.inflation_m
    assert costmap.is_free(4.0 + probe, 2.0)
    i, j = costmap.world_to_cell(4.0 + probe, 2.0)
    assert costmap.cost_at(i, j) > 1.0
    assert costmap.cost_at(i, j) < Costmap.COST_LETHAL


def test_a_lethal_cell_costs_infinity(config):
    costmap = Costmap(config, [boulder(4.0, 2.0, 0.15)])
    i, j = costmap.world_to_cell(4.0, 2.0)
    assert costmap.cost_at(i, j) == Costmap.COST_LETHAL
    assert math.isinf(costmap.cost_at(i, j))


def test_turning_the_band_off_leaves_only_lethal_and_one(config_file, tmp_path):
    cfg = config_with(config_file, tmp_path, {"costmap.inflation_m": 0.0})
    costmap = Costmap(cfg, [boulder(4.0, 2.0, 0.15)])
    finite = costmap.cost[np.isfinite(costmap.cost)]
    assert set(np.unique(finite).tolist()) == {1.0}


def test_the_band_is_at_least_one_cell_wide(config):
    """A zero-width band would silently disable the planner's preference."""
    from rover_navigation.config import Rect

    costmap = Costmap(config, [], [Rect(3.0, 1.0, 5.0, 4.0)])
    lethal = Costmap(config, [], [Rect(3.0, 1.0, 5.0, 4.0)]).lethal.copy()
    # There is at least one cell that is free but penalised.
    penalised = (costmap.cost > 1.0) & np.isfinite(costmap.cost)
    assert int(penalised.sum()) > 0
    assert penalised[lethal].sum() == 0


# =========================================================================== #
# Costmap: queries
# =========================================================================== #
def test_world_to_cell_and_back_is_consistent(config):
    costmap = Costmap(config, [])
    i, j = costmap.world_to_cell(3.03, 2.03)
    assert (i, j) == (60, 40)
    x, y = costmap.cell_center(i, j)
    assert (x, y) == pytest.approx((3.025, 2.025))
    assert costmap.world_to_cell(x, y) == (i, j)


def test_in_bounds_rejects_negative_and_large_indices(config):
    costmap = Costmap(config, [])
    assert costmap.in_bounds(0, 0)
    assert costmap.in_bounds(costmap.nx - 1, costmap.ny - 1)
    assert not costmap.in_bounds(-1, 0)
    assert not costmap.in_bounds(costmap.nx, 0)


def test_nearest_free_returns_the_point_itself_when_it_is_free(config):
    costmap = Costmap(config, [boulder(4.0, 2.0, 0.15)])
    assert costmap.nearest_free(1.0, 4.0, 1.0) == (1.0, 4.0)


def test_nearest_free_rescues_a_point_inside_an_obstacle(config):
    """A goal perception has just found to be blocked must still be reachable."""
    costmap = Costmap(config, [boulder(4.0, 2.0, 0.15)])
    rescued = costmap.nearest_free(4.0, 2.0, 1.0)
    assert rescued is not None
    x, y = rescued
    assert math.hypot(x - 4.0, y - 2.0) > 0.0
    assert costmap.is_free(x, y)
    assert math.hypot(x - 4.0, y - 2.0) <= config.robot.obstacle_inflation_m + 0.15 + 0.2


def test_nearest_free_gives_up_when_the_search_radius_is_too_small(config):
    # A keep-out the size of the arena makes everything lethal.
    from rover_navigation.config import Rect

    costmap = Costmap(config, [], [Rect(-1.0, -1.0, 10.0, 6.0)])
    assert costmap.nearest_free(4.0, 2.0, 0.5) is None


def test_the_occupancy_grid_payload_is_row_major_int8(config):
    costmap = Costmap(config, [boulder(4.0, 2.0, 0.15)])
    data, origin = costmap.to_occupancy_grid()
    assert len(data) == costmap.nx * costmap.ny
    assert origin == (0.0, 0.0)
    assert set(np.unique(np.asarray(data)).tolist()) <= {0, 100}
    # Row 0 is the lowest y, so the first cell is the south-west corner: a wall.
    assert data[0] == 100


def test_lethal_fraction_grows_with_obstacles(config):
    empty = Costmap(config, [])
    busy = Costmap(config, obstacle_line((3.0, 0.5), (3.0, 4.5), count=20, radius=0.15))
    assert busy.lethal_fraction() > empty.lethal_fraction()
    assert empty.free_cells() > busy.free_cells()


def test_the_whole_obstacle_zone_can_be_sealed_by_enough_obstacles(config):
    """A sanity check on the rasteriser: 30 boulders across the band shut it."""
    wall = obstacle_line((4.5, 0.3), (4.5, 4.7), count=40, radius=0.18)
    costmap = Costmap(config, wall)
    assert not costmap.segment_clear(1.0, 2.5, 8.0, 2.5)


def test_a_crater_is_as_lethal_as_a_boulder_at_its_centre(config):
    point = crater(4.0, 2.0, 0.2)
    costmap = Costmap(config, [point])
    assert not costmap.is_free(4.0, 2.0)
    assert costmap.lethal_radius(point.radius, KIND_CRATER) > costmap.lethal_radius(point.radius, KIND_BOULDER)


def test_the_costmap_is_rebuilt_from_records_each_time(config):
    first = Costmap(config, [boulder(4.0, 2.0, 0.15)])
    second = Costmap(config, [])
    assert first.free_cells() < second.free_cells()
    assert first.obstacle_count == 1
    assert second.obstacle_count == 0
