"""The tub and the berm height field: the sand model, and its exact spill."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rover_excavation.sand import KERNEL_SIGMAS, BermField, Tub, _normal_cdf

from support import dump_pose, rig


# --------------------------------------------------------------------------- #
# the tub
# --------------------------------------------------------------------------- #
def test_tub_starts_empty(config):
    tub = Tub(config.mechanism.tub)
    assert tub.volume_m3 == 0.0
    assert tub.fraction == 0.0
    assert tub.is_empty
    assert not tub.is_full


def test_tub_fills_to_capacity_and_no_further(config):
    tub = Tub(config.mechanism.tub)
    accepted = tub.add(10.0 * config.mechanism.tub.capacity_m3)
    assert accepted == pytest.approx(config.mechanism.tub.capacity_m3)
    assert tub.is_full
    assert tub.fraction == pytest.approx(1.0)
    # A second pour fits nothing.
    assert tub.add(0.01) == 0.0
    assert tub.volume_m3 == pytest.approx(config.mechanism.tub.capacity_m3)


def test_tub_remove_and_empty(config):
    tub = Tub(config.mechanism.tub)
    tub.fill()
    removed = tub.remove(0.01)
    assert removed == pytest.approx(0.01)
    tub.empty()
    assert tub.is_empty
    assert tub.remove(0.01) == 0.0


def test_tub_ignores_nonsense_volumes(config):
    tub = Tub(config.mechanism.tub)
    assert tub.add(0.0) == 0.0
    assert tub.add(-1.0) == 0.0
    assert tub.remove(-1.0) == 0.0


def test_cli_style_construction_can_start_part_full(config):
    tub = Tub(config.mechanism.tub, volume_m3=1e9)
    assert tub.volume_m3 == pytest.approx(config.mechanism.tub.capacity_m3)


# --------------------------------------------------------------------------- #
# the height field
# --------------------------------------------------------------------------- #
def test_field_geometry_matches_the_target(config):
    berm = BermField(config)
    assert berm.rect.as_tuple() == config.berm_target.as_tuple()
    assert berm.dx == pytest.approx(0.05)
    assert berm.dy == pytest.approx(0.05)
    assert berm.cells_total == 25 * 18
    assert berm.target_height_m == pytest.approx(0.15)
    assert berm.target_volume_m3 == pytest.approx(config.berm_target_volume_m3)


def test_empty_field_reports_zero(config):
    berm = BermField(config)
    assert berm.placed_volume_m3 == 0.0
    assert berm.spilled_volume_m3 == 0.0
    assert berm.mean_height_m == 0.0
    assert berm.peak_height_m == 0.0
    assert berm.coverage == 0.0
    assert berm.completion == 0.0
    assert berm.dumps == 0


def test_deposit_centred_on_the_target_keeps_most_of_its_volume(config):
    berm = BermField(config)
    berm.begin_dump()
    result = berm.deposit(*config.berm_target.center, 0.05)
    # Even dead centre, a pile has tails that reach past the footprint; the
    # model says exactly how much, and that is what is placed.
    assert result.inside_fraction > 0.95
    assert result.spilled_m3 == pytest.approx(0.05 * (1.0 - result.inside_fraction))
    assert berm.placed_volume_m3 == pytest.approx(0.05 * result.inside_fraction, rel=1e-9)
    assert berm.dumps == 1


def test_a_dump_is_counted_per_pile_not_per_tick(config):
    # The node deposits a little sand every control tick while the tub empties;
    # that must still be one dump.
    berm = BermField(config)
    berm.begin_dump()
    spilled = 0.0
    for _ in range(20):
        spilled += berm.deposit(*config.berm_target.center, 0.001).spilled_m3
    assert berm.dumps == 1
    assert berm.placed_volume_m3 + spilled == pytest.approx(0.02, rel=1e-9)
    assert berm.spilled_volume_m3 == pytest.approx(spilled, rel=1e-9)


def test_deposit_volume_is_conserved(config):
    berm = BermField(config)
    total = 0.0
    for index in range(6):
        x, y, yaw = dump_pose(config, index)
        lx, ly = config.landing_point(x, y, yaw)
        berm.begin_dump()
        result = berm.deposit(lx, ly, 0.0329)
        total += result.placed_m3 + result.spilled_m3
    assert berm.dumps == 6
    assert total == pytest.approx(6 * 0.0329, rel=1e-9)
    assert berm.placed_volume_m3 == pytest.approx(total - berm.spilled_volume_m3, rel=1e-9)


def test_deposit_far_outside_spills_everything(config):
    berm = BermField(config)
    # Somewhere in the excavation zone: nothing of it reaches the berm.
    result = berm.deposit(8.0, 4.0, 0.02)
    assert result.placed_m3 == 0.0
    assert result.spilled_m3 == pytest.approx(0.02)
    assert berm.placed_volume_m3 == 0.0
    assert berm.spilled_volume_m3 == pytest.approx(0.02)


def test_inside_fraction_is_the_gaussian_cdf_product(config):
    berm = BermField(config)
    sigma = config.sand.dump.spread_m
    x, y = config.berm_target.center
    fx = _normal_cdf((config.berm_target.x1 - x) / sigma) - _normal_cdf((config.berm_target.x0 - x) / sigma)
    fy = _normal_cdf((config.berm_target.y1 - y) / sigma) - _normal_cdf((config.berm_target.y0 - y) / sigma)
    result = berm.deposit(x, y, 1.0)
    assert result.inside_fraction == pytest.approx(fx * fy)


def test_zero_deposit_changes_nothing(config):
    berm = BermField(config)
    result = berm.deposit(*config.berm_target.center, 0.0)
    assert result.placed_m3 == 0.0
    assert berm.placed_volume_m3 == 0.0
    assert berm.dumps == 0


def test_kernel_is_truncated_at_three_sigmas(config):
    berm = BermField(config)
    berm.deposit(*config.berm_target.center, 0.05)
    sigma = config.sand.dump.spread_m
    radius = KERNEL_SIGMAS * sigma
    heights = np.asarray([h for _, _, h in berm.cells()], dtype=float).reshape(berm.ny, berm.nx)
    xs = np.asarray([x for x, _, _ in berm.cells()], dtype=float).reshape(berm.ny, berm.nx)
    ys = np.asarray([y for _, y, _ in berm.cells()], dtype=float).reshape(berm.ny, berm.nx)
    cx, cy = berm.rect.center
    far = (xs - cx) ** 2 + (ys - cy) ** 2 > radius * radius
    assert heights[far].max(initial=0.0) == 0.0
    assert heights[~far].max() > 0.0


def test_peak_is_in_the_middle_of_a_centred_deposit(config):
    berm = BermField(config)
    berm.deposit(*config.berm_target.center, 0.05)
    heights = np.asarray([h for _, _, h in berm.cells()], dtype=float).reshape(berm.ny, berm.nx)
    peak_index = np.unravel_index(int(np.argmax(heights)), heights.shape)
    assert peak_index[0] == pytest.approx(berm.ny // 2, abs=1)
    assert peak_index[1] == pytest.approx(berm.nx // 2, abs=1)


def test_coverage_counts_cells_at_target_height(config):
    berm = BermField(config)
    # A flat fill of twice the target height must cover the whole footprint.
    berm.height[:, :] = 2.0 * berm.target_height_m
    assert berm.coverage == pytest.approx(1.0)
    assert berm.cells_filled == berm.cells_total
    assert berm.mean_height_m == pytest.approx(2.0 * berm.target_height_m)
    assert berm.completion == 1.0
    # Just under the target height is not counted as filled.
    berm.height[:, :] = 0.99 * berm.target_height_m
    assert berm.coverage == 0.0
    assert berm.cells_filled == 0


def test_completion_is_volume_fraction_and_is_capped(config):
    berm = BermField(config)
    # Well past a berm's worth, to be sure the tails cannot keep it under 1.
    berm.deposit(*config.berm_target.center, 2.0 * config.berm_target_volume_m3)
    assert berm.completion == pytest.approx(1.0)
    berm.deposit(*config.berm_target.center, config.berm_target_volume_m3)
    assert berm.completion == 1.0  # never more than done
    assert berm.volume_fraction > 1.0


def test_reset_clears_everything(config):
    berm = BermField(config)
    berm.deposit(*config.berm_target.center, 0.05)
    berm.deposit(8.0, 4.0, 0.05)
    berm.reset()
    assert berm.placed_volume_m3 == 0.0
    assert berm.spilled_volume_m3 == 0.0
    assert berm.dumps == 0
    assert berm.peak_height_m == 0.0


def test_cells_view_is_the_whole_grid(config):
    berm = BermField(config)
    cells = berm.cells()
    assert len(cells) == berm.cells_total
    for x, y, height in cells:
        assert berm.rect.contains(x, y)
        assert height == 0.0


def test_summary_mentions_the_numbers(config):
    berm = BermField(config)
    berm.deposit(*config.berm_target.center, 0.05)
    summary = berm.summary()
    assert "m3" in summary
    assert "coverage" in summary


def test_six_realistic_dumps_nearly_fill_the_berm(config):
    """The design target: module 5's six cycles must just about do it.

    This is the one sand test that encodes an engineering expectation rather than
    a mechanism rule.  If it fails after a tuning change, the change made the
    competition's own cycle budget stop being enough to build the berm.
    """
    mechanism, tub, berm = rig(config)
    total_placed = 0.0
    for index in range(6):
        # A tub's worth of sand, dumped from module 5's pose for this cycle.
        x, y, yaw = dump_pose(config, index)
        lx, ly = config.landing_point(x, y, yaw)
        volume = config.mechanism.tub.capacity_m3 * config.cycle.dig.target_fill
        berm.begin_dump()
        result = berm.deposit(lx, ly, volume)
        total_placed += result.placed_m3
    assert berm.dumps == 6
    assert berm.completion > 0.90
    assert berm.spilled_volume_m3 < 0.25 * total_placed
    assert 0.0 < berm.coverage < 1.0  # a ridge, not a flat prism: see the docs
    assert mechanism is not None and tub is not None and math.isfinite(berm.mean_height_m)
