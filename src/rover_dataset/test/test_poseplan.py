"""Capture-pose planning tests."""

from __future__ import annotations

import math

import pytest

from rover_dataset.poseplan import footprint_inside, plan_poses

from conftest import make_ground_truth

BOULDERS = [(4.0, 2.5, 0.15, 0.18), (6.0, 1.5, 0.15, 0.17)]
CRATERS = [(5.0, 3.5, 0.22)]


@pytest.fixture
def ground_truth():
    return make_ground_truth(boulders=BOULDERS, craters=CRATERS)


def test_plans_the_requested_number_of_poses(cfg, ground_truth):
    poses = plan_poses(ground_truth, cfg.capture, 1234, "sim_0000", poses_per_scene=25)
    assert len(poses) == 25
    assert [pose.index for pose in poses] == list(range(25))


def test_planning_is_deterministic(cfg, ground_truth):
    first = plan_poses(ground_truth, cfg.capture, 1234, "sim_0000", poses_per_scene=10)
    second = plan_poses(ground_truth, cfg.capture, 1234, "sim_0000", poses_per_scene=10)
    assert [p.as_dict() for p in first] == [p.as_dict() for p in second]


def test_different_scenes_get_different_poses(cfg, ground_truth):
    a = plan_poses(ground_truth, cfg.capture, 1234, "sim_0000", poses_per_scene=10)
    b = plan_poses(ground_truth, cfg.capture, 1234, "sim_0001", poses_per_scene=10)
    assert [p.as_dict() for p in a] != [p.as_dict() for p in b]


def test_poses_stay_inside_their_region(cfg, ground_truth):
    plan = cfg.capture.pose_plan
    regions = {region.name: region for region in plan.regions}
    for pose in plan_poses(ground_truth, cfg.capture, 7, "sim_0000", poses_per_scene=40):
        region = regions[pose.region]
        assert region.x0 + plan.margin - 1e-6 <= pose.x <= region.x1 - plan.margin + 1e-6
        assert region.y0 + plan.margin - 1e-6 <= pose.y <= region.y1 - plan.margin + 1e-6


def test_poses_stay_inside_the_arena(cfg, ground_truth):
    for pose in plan_poses(ground_truth, cfg.capture, 7, "sim_0000", poses_per_scene=60):
        assert 0.0 <= pose.x <= ground_truth.length_x
        assert 0.0 <= pose.y <= ground_truth.width_y


def test_poses_clear_the_obstacles(cfg, ground_truth):
    plan = cfg.capture.pose_plan
    for pose in plan_poses(ground_truth, cfg.capture, 7, "sim_0000", poses_per_scene=50):
        for boulder in ground_truth.boulders:
            distance = math.hypot(pose.x - boulder.center[0], pose.y - boulder.center[1])
            assert distance >= plan.min_obstacle_clearance + boulder.radius
        for crater in ground_truth.craters:
            distance = math.hypot(pose.x - crater.center[0], pose.y - crater.center[1])
            assert distance >= plan.min_crater_clearance + crater.radius


def test_headings_are_wrapped_and_finite(cfg, ground_truth):
    for pose in plan_poses(ground_truth, cfg.capture, 7, "sim_0000", poses_per_scene=30):
        assert -math.pi <= pose.yaw <= math.pi
        assert math.isfinite(pose.yaw)


def test_footprint_check_accepts_the_centre_and_rejects_the_corner():
    assert footprint_inside(4.5, 2.5, 0.0, 1.5, 0.75, 9.0, 5.0)
    assert not footprint_inside(0.05, 2.5, 0.0, 1.5, 0.75, 9.0, 5.0)
    assert not footprint_inside(2.5, 0.05, 0.0, 1.5, 0.75, 9.0, 5.0)


def test_footprint_check_accounts_for_rotation():
    # at 45 deg the footprint reaches further diagonally
    assert footprint_inside(1.0, 1.0, 0.0, 1.5, 0.75, 9.0, 5.0)
    assert not footprint_inside(0.6, 0.6, math.pi / 4, 1.5, 0.75, 9.0, 5.0)


def test_planning_survives_a_degenerate_region(cfg, ground_truth):
    from dataclasses import replace

    big_margin = replace(
        cfg, capture=replace(cfg.capture, pose_plan=replace(cfg.capture.pose_plan, margin=5.0))
    )
    poses = plan_poses(ground_truth, big_margin.capture, 7, "sim_0000", poses_per_scene=5)
    assert poses == []  # every region vanishes; must not crash
