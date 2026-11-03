"""Start pose tests.

The challenge requires a *random* start pose **and heading** inside the start
zone, drawn per run.  What matters for the rest of the stack is:

* the whole stowed footprint (1.5 x 0.75 m at an arbitrary heading) is inside
  the start zone with the configured margin, for every seed,
* the spawn point sits on the sand surface plus the configured offset,
* the pose is a pure function of the run seed (so the launch file, the world
  file and the pose node can never disagree),
* the draw is actually random (positions and headings are spread out) and
  never lands in the construction/obstacle zones.
"""

from __future__ import annotations

import math

import pytest

from cat_arena.start_pose import (
    footprint_corners,
    pose_acceptance_rate,
    rotation_matrix,
    sample_start_pose,
    yaw_bucket,
)
from cat_arena.terrain import TerrainField

SEEDS = list(range(40))


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def test_rotation_matrix_matches_math():
    for yaw in (0.0, 0.3, math.pi / 2, -1.2, math.pi):
        c, s = math.cos(yaw), math.sin(yaw)
        rot = rotation_matrix(yaw)
        assert rot[0][0] == pytest.approx(c)
        assert rot[0][1] == pytest.approx(-s)
        assert rot[1][0] == pytest.approx(s)
        assert rot[1][1] == pytest.approx(c)


def test_footprint_corners_are_the_rotated_rectangle():
    length, width = 1.5, 0.75
    corners = footprint_corners(2.0, 3.0, 0.0, length, width)
    assert corners == pytest.approx(
        [(2.75, 3.375), (2.75, 2.625), (1.25, 2.625), (1.25, 3.375)]
    )
    for yaw in (0.25, 1.0, -2.0, math.pi):
        pts = footprint_corners(2.0, 3.0, yaw, length, width)
        for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
            distance = math.hypot(bx - ax, by - ay)
            assert distance == pytest.approx(length) or distance == pytest.approx(width)


# --------------------------------------------------------------------------- #
# per-seed guarantees
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_pose_fits_inside_the_start_zone(cfg, seed):
    pose = sample_start_pose(cfg, seed)
    rect = cfg.zone("start").rect
    margin = cfg.start_pose.margin
    for cx, cy in pose.corners:
        assert rect.contains(cx, cy, margin=margin - 1e-9), (
            f"seed {seed}: corner ({cx:.3f}, {cy:.3f}) escapes the start zone"
        )
    assert pose.zone == "start"
    assert pose.x == pytest.approx(sum(c[0] for c in pose.corners) / 4.0)
    assert pose.y == pytest.approx(sum(c[1] for c in pose.corners) / 4.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_pose_never_lands_in_another_zone(cfg, seed):
    pose = sample_start_pose(cfg, seed)
    for name in ("construction", "obstacle", "excavation", "berm_target"):
        assert not cfg.zone(name).rect.contains(pose.x, pose.y), (
            f"seed {seed}: spawn point inside the {name} zone"
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_spawn_height_sits_on_the_sand(cfg, seed):
    field = TerrainField(cfg, [])
    pose = sample_start_pose(cfg, seed, surface=field.height)
    assert pose.z == pytest.approx(field.height(pose.x, pose.y) + cfg.spawn_z)
    pose_flat = sample_start_pose(cfg, seed)
    assert pose_flat.z == pytest.approx(cfg.spawn_z)


@pytest.mark.parametrize("seed", SEEDS)
def test_pose_is_a_pure_function_of_the_seed(cfg, seed):
    a = sample_start_pose(cfg, seed)
    b = sample_start_pose(cfg, seed)
    assert a.to_dict() == b.to_dict()


def test_pose_is_never_a_fixed_heading(cfg):
    yaws = [sample_start_pose(cfg, seed).yaw for seed in SEEDS]
    assert len(set(round(y, 6) for y in yaws)) == len(SEEDS)
    buckets = {yaw_bucket(y) for y in yaws}
    assert len(buckets) == 4, "headings must be drawn over the full circle"
    assert max(yaws) - min(yaws) > math.pi


def test_pose_is_never_a_fixed_position(cfg):
    xs = [sample_start_pose(cfg, seed).x for seed in SEEDS]
    ys = [sample_start_pose(cfg, seed).y for seed in SEEDS]
    assert max(xs) - min(xs) > 0.5
    assert max(ys) - min(ys) > 0.5
    rect = cfg.zone("start").rect
    for x, y in zip(xs, ys):
        assert rect.x0 <= x <= rect.x1 and rect.y0 <= y <= rect.y1


def test_declared_forward_is_recorded(cfg):
    pose = sample_start_pose(cfg, 1)
    assert pose.declared_forward_deg == cfg.start_pose.declared_forward_deg
    assert pose.footprint_length == cfg.start_pose.footprint_length
    assert pose.footprint_width == cfg.start_pose.footprint_width


def test_yaw_override(cfg):
    pose = sample_start_pose(cfg, 3, yaw_override_deg=90.0)
    assert pose.yaw == pytest.approx(math.pi / 2)
    assert pose.yaw_deg == pytest.approx(90.0)
    # the override still has to fit the footprint inside the zone
    rect = cfg.zone("start").rect
    assert all(rect.contains(cx, cy) for cx, cy in pose.corners)


def test_acceptance_rate_is_healthy(cfg):
    """The start zone is legitimately tight: a 1.5 x 0.75 m footprint at a
    random heading needs the zone's smallest dimension (2.0 m) to accommodate a
    1.677 m diagonal, so only a fraction of raw draws fit.  What matters is that
    it is not vanishingly small - ``max_tries`` is 200000."""
    rate = pose_acceptance_rate(cfg, seed=99, samples=400)
    assert rate > 0.03, f"only {rate:.1%} of draws fit - the start zone is too tight"
