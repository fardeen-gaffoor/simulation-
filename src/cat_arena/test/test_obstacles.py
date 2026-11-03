"""Obstacle placement tests.

The competition rules that must hold for *every* seed:

* at least 3 boulders (30-40 cm) and at least 3 craters (positive *depth*, i.e.
  a negative depression in the terrain),
* nothing in the start zone,
* craters only in the obstacle zone, boulders also allowed in the excavation
  zone,
* obstacles never overlap one another (they are solid bodies, and a boulder
  inside a crater would be a different physical situation than either alone).
"""

from __future__ import annotations

import math

import pytest

from cat_arena.config import Rect
from cat_arena.obstacles import obstacles_in_rect, place_obstacles
from cat_arena.terrain import TerrainField

SEEDS = list(range(24))


def _ripple_field(cfg):
    """The surface boulders are seated on (craters are carved afterwards)."""
    return TerrainField(cfg, [])


def _layout(cfg, seed):
    return place_obstacles(cfg, seed, _ripple_field(cfg))


# --------------------------------------------------------------------------- #
# rule compliance over many seeds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_minimum_obstacle_counts(cfg, seed):
    layout = _layout(cfg, seed)
    assert len(layout.boulders) >= 3, "rules: at least 3 boulders"
    assert len(layout.craters) >= 3, "rules: at least 3 craters"
    assert len(layout.boulders) <= cfg.boulders.count_max
    assert len(layout.craters) <= cfg.craters.count_max


@pytest.mark.parametrize("seed", SEEDS)
def test_boulder_geometry_is_within_the_rules(cfg, seed):
    for boulder in _layout(cfg, seed).boulders:
        assert cfg.boulders.diameter_min - 1e-9 <= boulder.diameter <= cfg.boulders.diameter_max + 1e-9
        assert cfg.boulders.exposed_min - 1e-9 <= boulder.exposed_height / boulder.diameter
        assert boulder.exposed_height / boulder.diameter <= cfg.boulders.exposed_max + 1e-9
        assert boulder.buried_depth >= 0.0
        assert boulder.buried_depth <= boulder.diameter + 1e-9
        # a boulder sticks up above the sand, it is never a hole
        assert boulder.top_z > boulder.surface_z - 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_craters_are_negative_depressions(cfg, seed):
    layout = _layout(cfg, seed)
    field = TerrainField(cfg, layout.craters)
    for crater in layout.craters:
        assert 0.25 - 1e-9 <= crater.width <= 0.50 + 1e-9
        assert cfg.craters.depth_min - 1e-9 <= crater.depth <= cfg.craters.depth_max + 1e-9
        assert crater.depth > 0.0, "crater depth must describe a downward depression"
        # and the terrain really is lower there
        assert field.height(crater.x, crater.y) == pytest.approx(-crater.depth, abs=1e-9)
        assert field.height(crater.x, crater.y) < 0.0


@pytest.mark.parametrize("seed", SEEDS)
def test_obstacles_are_in_permitted_zones(cfg, seed):
    layout = _layout(cfg, seed)
    for boulder in layout.boulders:
        assert boulder.zone in cfg.boulders.zones
        assert cfg.zone(boulder.zone).obstacles_allowed
    for crater in layout.craters:
        assert crater.zone == "obstacle", "rules: craters are only in the obstacle zone"
        assert cfg.zone(crater.zone).obstacles_allowed


@pytest.mark.parametrize("seed", SEEDS)
def test_nothing_touches_the_start_zone(cfg, seed):
    layout = _layout(cfg, seed)
    start = cfg.zone("start").rect
    boulders, craters = obstacles_in_rect(layout.boulders, layout.craters, start)
    assert not boulders and not craters
    for boulder in layout.boulders:
        assert not start.intersects(
            Rect(
                boulder.x - boulder.radius,
                boulder.y - boulder.radius,
                boulder.x + boulder.radius,
                boulder.y + boulder.radius,
            )
        )
    for crater in layout.craters:
        assert crater.x - crater.radius > start.x1 or crater.y > start.y1 or crater.y < start.y0 or crater.x + crater.radius < start.x0


@pytest.mark.parametrize("seed", SEEDS)
def test_obstacles_are_separated(cfg, seed):
    layout = _layout(cfg, seed)
    items = [("boulder", b.x, b.y, b.radius) for b in layout.boulders]
    items += [("crater", c.x, c.y, c.radius) for c in layout.craters]
    for i, (kind_a, ax, ay, ar) in enumerate(items):
        for kind_b, bx, by, br in items[i + 1:]:
            separation = math.hypot(ax - bx, ay - by)
            spacing = max(cfg.boulders.spacing, cfg.craters.spacing) if kind_a != kind_b else (
                cfg.boulders.spacing if kind_a == "boulder" else cfg.craters.spacing
            )
            assert separation >= max(spacing, ar + br + 0.05) - 1e-9, (
                f"{kind_a} at ({ax:.2f},{ay:.2f}) overlaps {kind_b} at ({bx:.2f},{by:.2f})"
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_obstacles_respect_the_zone_margin(cfg, seed):
    layout = _layout(cfg, seed)
    for boulder in layout.boulders:
        rect = cfg.zone(boulder.zone).rect
        margin = cfg.boulders.margin
        assert rect.x0 + margin - 1e-9 <= boulder.x <= rect.x1 - margin + 1e-9
        assert rect.y0 + margin - 1e-9 <= boulder.y <= rect.y1 - margin + 1e-9
    for crater in layout.craters:
        rect = cfg.zone(crater.zone).rect
        margin = cfg.craters.margin
        assert rect.x0 + margin - 1e-9 <= crater.x <= rect.x1 - margin + 1e-9
        assert rect.y0 + margin - 1e-9 <= crater.y <= rect.y1 - margin + 1e-9


@pytest.mark.parametrize("seed", SEEDS)
def test_boulders_are_seated_on_the_rendered_terrain(cfg, seed):
    layout = _layout(cfg, seed)
    field = TerrainField(cfg, layout.craters)
    for boulder in layout.boulders:
        # the stored surface height matches the actual sand surface (within one
        # mesh cell, because the mesh is a discretisation of the field)
        assert abs(field.height(boulder.x, boulder.y) - boulder.surface_z) < cfg.terrain.cell_size


# --------------------------------------------------------------------------- #
# determinism / spread
# --------------------------------------------------------------------------- #
def test_same_seed_gives_identical_layouts(cfg):
    a = _layout(cfg, 12345).as_dict()
    b = _layout(cfg, 12345).as_dict()
    assert a == b


def test_different_seeds_give_different_layouts(cfg):
    signatures = {repr(_layout(cfg, seed).as_dict()) for seed in range(12)}
    assert len(signatures) >= 10, "placement is not actually seed-dependent"


def test_seeded_layout_is_master_seed_independent_of_history(cfg):
    """Drawing seed 5 after seed 3 must equal drawing seed 5 alone."""
    first = _layout(cfg, 3).as_dict()
    middle = _layout(cfg, 5).as_dict()
    again = _layout(cfg, 5).as_dict()
    assert middle == again
    assert first != middle


def test_boulders_are_reachable_in_both_zones(cfg):
    zones = set()
    for seed in range(40):
        zones |= {b.zone for b in _layout(cfg, seed).boulders}
    assert "obstacle" in zones
    assert zones <= {"obstacle", "excavation"}
