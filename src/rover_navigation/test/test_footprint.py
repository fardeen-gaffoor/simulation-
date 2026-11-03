"""The exact footprint tests: a rotated rectangle, not a disc.

Module 5 plans on a map inflated by the rover's *inscribed* radius, because a
circumscribed-radius map does not fit this arena.  The cost of that choice is
paid here: a goal pose and an in-place turn are both checked against the actual
1.3 x 0.7 m rectangle, so the part the inflation does not cover - the rover's
length as it turns - is still accounted for.

The cases that matter are the ones a disc gets wrong:

* a **squeeze** the rectangle can fit into but the disc cannot (that is what
  makes the arena passable at all), and
* a **turn** into a boulder the nose would sweep through even though the body is
  clear at the current heading (that is what the disc over-approximates).
"""

from __future__ import annotations

import math

import pytest

from rover_navigation.footprint import (
    corner_offsets,
    footprint_block,
    rect_corners,
    rect_disc_clearance,
    rotation_block,
)
from rover_navigation.geometry import Pose3

from support import boulder, crater, pose


def _length(config) -> float:
    return config.robot.length_m


def _width(config) -> float:
    return config.robot.width_m


# =========================================================================== #
# the rectangle itself
# =========================================================================== #
def test_corner_offsets_are_the_four_corners():
    corners = corner_offsets(0.65, 0.35)
    assert set(corners) == {(0.65, 0.35), (0.65, -0.35), (-0.65, -0.35), (-0.65, 0.35)}
    assert len(corners) == 4


def test_at_zero_yaw_the_corners_are_axis_aligned(config):
    corners = rect_corners(2.0, 3.0, 0.0, _length(config) / 2, _width(config) / 2)
    xs = sorted({round(x, 6) for x, _ in corners})
    ys = sorted({round(y, 6) for _, y in corners})
    assert xs == pytest.approx([2.0 - _length(config) / 2, 2.0 + _length(config) / 2])
    assert ys == pytest.approx([3.0 - _width(config) / 2, 3.0 + _width(config) / 2])


def test_a_quarter_turn_swaps_length_and_width(config):
    corners = rect_corners(2.0, 3.0, math.pi / 2, _length(config) / 2, _width(config) / 2)
    xs = [x for x, _ in corners]
    ys = [y for _, y in corners]
    assert max(xs) - min(xs) == pytest.approx(_width(config), abs=1e-9)
    assert max(ys) - min(ys) == pytest.approx(_length(config), abs=1e-9)


def test_the_rectangle_stays_centred_on_the_pose(config):
    corners = rect_corners(4.0, 1.5, 0.7, _length(config) / 2, _width(config) / 2)
    assert sum(x for x, _ in corners) / 4 == pytest.approx(4.0)
    assert sum(y for _, y in corners) / 4 == pytest.approx(1.5)


# =========================================================================== #
# distance from a disc centre to the rectangle
# =========================================================================== #
def test_a_disc_off_the_side_is_its_own_distance_away():
    # Half a metre to the +y side of a rectangle half a metre wide.
    assert rect_disc_clearance(0.0, 0.0, 0.0, 0.65, 0.35, 0.0, 1.0) == pytest.approx(0.65)


def test_a_disc_off_the_nose_is_its_own_distance_away():
    assert rect_disc_clearance(0.0, 0.0, 0.0, 0.65, 0.35, 1.0, 0.0) == pytest.approx(0.35)


def test_a_disc_in_a_corner_measures_diagonally():
    assert rect_disc_clearance(0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0) == pytest.approx(math.hypot(0.5, 0.5))


def test_a_disc_on_the_boundary_has_zero_clearance():
    assert rect_disc_clearance(0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0) == pytest.approx(0.0)


def test_a_disc_inside_reports_a_negative_clearance():
    assert rect_disc_clearance(0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0) == pytest.approx(-0.5)
    assert rect_disc_clearance(0.0, 0.0, 0.0, 0.5, 0.5, 0.4, 0.0) == pytest.approx(-0.1)


def test_the_clearance_rotates_with_the_rectangle():
    # A rectangle turned 90 deg: the point that was off the nose is now off the side.
    assert rect_disc_clearance(0.0, 0.0, math.pi / 2, 0.65, 0.35, 1.0, 0.0) == pytest.approx(0.65)


# =========================================================================== #
# footprint_block: the arena walls
# =========================================================================== #
def test_an_open_pose_is_allowed(config):
    assert footprint_block(config, pose(4.5, 2.5, 0.0)) is None


def test_a_pose_with_its_nose_through_the_west_wall_is_refused(config):
    reason = footprint_block(config, pose(0.4, 2.5, math.pi))
    assert reason is not None
    assert "wall" in reason


def test_the_wall_margin_is_the_half_extent_of_the_body(config):
    # Facing +x, the nose reaches half_length ahead of the centre.
    half = _length(config) / 2
    margin = config.arena.wall_margin_m
    allowed = footprint_block(config, pose(margin + half + 0.01, 2.5, 0.0))
    refused = footprint_block(config, pose(margin + half - 0.01, 2.5, 0.0))
    assert allowed is None
    assert refused is not None


def test_a_pose_turned_sideways_fits_closer_to_the_wall(config):
    """The point of a rectangle: facing along the wall, the body is narrower."""
    half_length = _length(config) / 2
    half_width = _width(config) / 2
    margin = config.arena.wall_margin_m
    assert half_width < half_length
    assert footprint_block(config, pose(margin + half_width + 0.01, 2.5, math.pi / 2)) is None


def test_the_start_pose_shipped_in_the_config_is_legal(config):
    park = config.mission.park
    assert footprint_block(config, Pose3.from_xy_yaw(park.x, park.y, park.yaw)) is None


# =========================================================================== #
# footprint_block: obstacles
# =========================================================================== #
def test_a_pose_inside_a_boulder_is_refused(config):
    record = boulder(4.5, 2.5, 0.15)
    reason = footprint_block(config, pose(4.5, 2.5, 0.0), [record])
    assert reason is not None
    assert "boulder" in reason


def test_a_pose_clear_of_a_boulder_is_allowed(config):
    record = boulder(4.5, 2.5, 0.15)
    # 1.5 m away: outside even the swept circle.
    assert footprint_block(config, pose(6.0, 2.5, 0.0), [record]) is None


def test_a_boulder_behind_the_tail_blocks(config):
    record = boulder(4.5, 2.5, 0.15)
    reason = footprint_block(config, pose(4.5 + 0.6, 2.5, 0.0), [record])
    assert reason is not None
    assert "boulder" in reason


def test_the_clearance_parameter_widens_the_test(config):
    record = boulder(4.5, 2.5, 0.15)
    just_clear = pose(4.5, 2.5 + 0.35 + 0.15 + 0.05, 0.0)
    assert footprint_block(config, just_clear, [record]) is None
    assert footprint_block(config, just_clear, [record], clearance=0.20) is not None


def test_a_crater_is_treated_as_a_larger_disc(config):
    record = crater(4.5, 2.5, 0.20)
    # A boulder of the same radius would not reach this far.
    probe = pose(4.5, 2.5 + 0.35 + 0.20 * config.costmap.crater_radius_scale * 0.5, 0.0)
    assert footprint_block(config, probe, [record]) is not None


def test_no_obstacles_means_no_obstacle_reason(config):
    assert footprint_block(config, pose(4.5, 2.5, 0.0), []) is None


# =========================================================================== #
# footprint_block: keep-outs
# =========================================================================== #
def test_a_pose_inside_a_keep_out_is_refused(config):
    from rover_navigation.config import Rect

    keep_out = Rect(4.0, 2.0, 5.0, 3.0)
    reason = footprint_block(config, pose(4.5, 2.5, 0.0), [], [keep_out])
    assert reason is not None
    assert "keep-out" in reason


def test_a_pose_just_outside_a_keep_out_is_allowed(config):
    from rover_navigation.config import Rect

    keep_out = Rect(4.0, 2.0, 5.0, 3.0)
    assert footprint_block(config, pose(4.5, 3.0 + _width(config) / 2 + 0.05, 0.0), [], [keep_out]) is None


def test_a_turned_body_can_reach_a_keep_out_a_straight_one_would_miss(config):
    from rover_navigation.config import Rect

    keep_out = Rect(4.4, 2.0, 5.0, 3.0)
    # Facing +y (nose along the keep-out's +x edge) the body is narrow in x...
    facing_north = pose(4.4 - _width(config) / 2 - 0.02, 2.5, math.pi / 2)
    assert footprint_block(config, facing_north, [], [keep_out]) is None
    # ...but turned to face +x its nose reaches into it.
    facing_east = pose(4.4 - _width(config) / 2 - 0.02, 2.5, 0.0)
    assert footprint_block(config, facing_east, [], [keep_out]) is not None


# =========================================================================== #
# rotation_block
# =========================================================================== #
def test_rotating_in_open_space_is_allowed(config):
    assert rotation_block(config, pose(4.5, 2.5, 0.0), math.pi) is None


def test_a_small_heading_correction_next_to_a_boulder_is_allowed(config):
    record = boulder(4.5, 2.5 + 0.35 + 0.15 + 0.05, 0.15)
    assert rotation_block(config, pose(4.5, 2.5, 0.0), math.radians(4.0), [record]) is None


def test_a_half_turn_into_the_same_boulder_is_refused(config):
    record = boulder(4.5, 2.5 + 0.35 + 0.15 + 0.05, 0.15)
    reason = rotation_block(config, pose(4.5, 2.5, 0.0), math.pi, [record])
    assert reason is not None
    assert "boulder" in reason


def test_a_boulder_off_the_nose_allows_the_turn_that_clears_it(config):
    """A boulder directly ahead blocks *some* headings, not all of them."""
    record = boulder(4.5 + 0.5, 2.5, 0.15)
    # At yaw 0 the boulder is 0.5 m ahead, inside the 0.65 m nose: blocked.
    assert rotation_block(config, pose(4.5, 2.5, 0.0), 0.0, [record]) is not None
    # Turning 90 deg sideways pulls the nose off it.
    assert rotation_block(config, pose(4.5, 2.5, math.pi / 2), math.pi / 2, [record]) is None


def test_a_turn_is_refused_when_a_keep_out_is_in_the_swept_arc(config):
    from rover_navigation.config import Rect

    keep_out = Rect(4.5 + 0.4, 2.3, 5.6, 2.7)
    assert rotation_block(config, pose(4.5, 2.5, 0.0), 0.0, [], [keep_out]) is not None
    assert rotation_block(config, pose(4.5, 2.5, math.pi / 2), math.pi / 2, [], [keep_out]) is None


def test_a_full_turn_between_two_flank_boulders_is_decided_by_the_widest_heading(config):
    """The test samples the whole arc, so it sees the worst heading, not the end one.

    Two boulders leave a gap 10 cm wider than the rover: driving through facing
    +x fits, but turning the 1.3 m long axis across the gap does not.
    """
    gap = _width(config) + 0.10
    records = [
        boulder(4.5, 2.5 - gap / 2 - 0.15, 0.15),
        boulder(4.5, 2.5 + gap / 2 + 0.15, 0.15),
    ]
    # Facing +x the body is 0.70 m across a 0.80 m gap: legal.
    assert rotation_block(config, pose(4.5, 2.5, 0.0), 0.0, records) is None
    # Turning to face +y sweeps the nose and tail into both boulders.
    assert rotation_block(config, pose(4.5, 2.5, 0.0), math.pi / 2, records) is not None


def test_rotation_block_returns_the_wall_reason_near_a_corner(config):
    reason = rotation_block(config, pose(0.4, 0.4, 0.0), math.pi / 2)
    assert reason is not None
    assert "wall" in reason


def test_a_zero_length_turn_is_allowed(config):
    assert rotation_block(config, pose(4.5, 2.5, 0.3), 0.3) is None


def test_the_turn_wraps_the_short_way(config):
    """A turn that would only sweep the boulder the *long* way round is allowed.

    The boulder sits on the +y flank (bearing 90 deg).  From +3.0 rad to -3.0 rad
    the short way is +0.28 rad, which moves the nose *away* from 90 deg; taking
    the long way would sweep straight over it.
    """
    record = boulder(4.5, 2.5 + 0.35 + 0.15 + 0.05, 0.15)
    assert rotation_block(config, pose(4.5, 2.5, 3.0), -3.0, [record]) is None


def test_a_turn_across_the_flank_boulder_is_refused(config):
    """...and the arc that does sweep it is refused, which is the other half."""
    record = boulder(4.5, 2.5 + 0.35 + 0.15 + 0.05, 0.15)
    # +0.5 rad to +2.0 rad: the arc contains 90 deg, where the boulder is.
    assert rotation_block(config, pose(4.5, 2.5, 0.5), 2.0, [record]) is not None


def test_a_target_written_unwrapped_behaves_the_same(config):
    """``2.0`` and ``2.0 - 2*pi`` are the same heading, so the same decision."""
    record = boulder(4.5, 2.5 + 0.35 + 0.15 + 0.05, 0.15)
    start = pose(4.5, 2.5, 0.5)
    assert rotation_block(config, start, 2.0, [record]) == rotation_block(
        config, start, 2.0 - 2.0 * math.pi, [record]
    )
