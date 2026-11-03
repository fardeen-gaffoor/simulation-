"""``map -> odom`` from one tag sighting: the arithmetic that anchors the map.

The localiser is the one place a *measured* pose (tag in the camera) meets a
*known* pose (tag in the map).  The property worth testing hardest is the round
trip: construct a rover pose, a camera extrinsics, and a catalogue tag, compute
the sighting that geometry *would* produce, and check the solver returns the
rover's true pose back.  That catches a wrong frame convention - the bug that
silently rotates every fix by 180 degrees - which a hand-built observation would
happily hide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rover_perception.calibration_ref import load_calibration
from rover_perception.config import LocalizationCfg, load_config
from rover_perception.geometry import Pose3, rpy_matrix
from rover_perception.localizer import (
    FixSolution,
    Localizer,
    OdomBuffer,
    TagFix,
    TagObservation,
    _fix_weight,
    map_to_odom_from_tag,
    planar_delta,
    planarize,
    smooth_planar,
    solve_fixes,
    tag_pose_in_odom,
)

_DEG = math.pi / 180.0


@pytest.fixture(scope="module")
def camera(calibration_path):
    return load_calibration(calibration_path).camera("cam_tag")


def _camera_pose(camera):
    return Pose3(camera.r_base_optical_matrix, camera.t_base_cam_vector)


def sighting(camera, tag, base_map_to_odom: Pose3) -> TagObservation:
    """The ``T_cam_tag`` that this exact geometry would produce.

    ``base_map_to_odom`` is the rover's *true* pose in the map; the returned
    observation is what a perfect detector looking at ``tag`` from that pose
    would report.  Solving it back must give the identity correction.
    """
    camera_map = base_map_to_odom.compose(_camera_pose(camera))
    measurement = camera_map.inverse().compose(tag.pose)
    return TagObservation(tag_id=tag.id, pose_camera=measurement, size_m=tag.size_m, decision_margin=90.0)


# --------------------------------------------------------------------------- #
# planar helpers
# --------------------------------------------------------------------------- #
def test_planarize_keeps_only_x_y_yaw():
    pose = Pose3(rpy_matrix(0.3, -0.2, 0.1), (1.0, 2.0, 0.5))
    flat = planarize(pose)
    x, y, z, roll, pitch, yaw = flat.as_rpy()
    assert (x, y) == pytest.approx((1.0, 2.0))
    assert z == pytest.approx(0.0)
    assert (roll, pitch) == pytest.approx((0.0, 0.0))
    assert yaw == pytest.approx(0.1)


def test_planar_delta_wraps_yaw():
    current = Pose3.from_xy_yaw(0.0, 0.0, 179.0 * _DEG)
    proposed = Pose3.from_xy_yaw(0.2, -0.3, -179.0 * _DEG)
    dx, dy, dyaw = planar_delta(current, proposed)
    assert (dx, dy) == pytest.approx((0.2, -0.3))
    assert dyaw == pytest.approx(2.0 * _DEG, abs=1e-9), "179 -> -179 is +2 degrees, not -358"


def test_smooth_planar_gains():
    current = Pose3.from_xy_yaw(0.0, 0.0, 0.0)
    proposed = Pose3.from_xy_yaw(1.0, 2.0, 0.5)
    assert smooth_planar(current, proposed, 1.0, 1.0).x == pytest.approx(1.0)
    assert smooth_planar(current, proposed, 0.5, 0.5).x == pytest.approx(0.5)
    half = smooth_planar(current, proposed, 0.0, 0.0)
    assert (half.x, half.y, half.yaw) == pytest.approx((0.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# the round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "x,y,yaw_deg,tag_id",
    [
        (1.0, 3.0, 90.0, 0),
        (2.5, 4.4, 70.0, 0),
        (1.0, 3.5, 0.0, 1),
        (5.5, 4.0, 120.0, 2),
    ],
)
def test_sighting_round_trips_to_the_true_pose(camera, catalogue, x, y, yaw_deg, tag_id):
    true_base = Pose3.from_xy_yaw(x, y, yaw_deg * _DEG)
    observation = sighting(camera, catalogue.tag(tag_id), true_base)
    solution = solve_fixes(
        [observation], catalogue, camera, true_base, load_config(_cfg_path()).localization
    )
    assert solution.accepted, solution.reason
    assert solution.used_tags == (tag_id,)
    # identity correction: the solver recovers exactly the pose the sighting came from
    assert solution.proposed_map_to_odom.x == pytest.approx(0.0, abs=1e-9)
    assert solution.proposed_map_to_odom.y == pytest.approx(0.0, abs=1e-9)
    assert solution.proposed_map_to_odom.yaw == pytest.approx(0.0, abs=1e-9)


def test_map_to_odom_from_tag_returns_the_tag_frame(camera, catalogue):
    true_base = Pose3.from_xy_yaw(1.5, 4.2, 80.0 * _DEG)
    tag = catalogue.tag(2)
    observation = sighting(camera, tag, true_base)
    tag_in_odom = tag_pose_in_odom(observation, camera, true_base)
    assert tag_in_odom.translation == pytest.approx(tag.pose.translation, abs=1e-9)
    correction = map_to_odom_from_tag(observation, tag, camera, true_base)
    assert correction.translation == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert correction.rotation == pytest.approx(np.eye(3), abs=1e-9)


def test_a_wrong_starting_correction_is_recovered(camera, catalogue, cfg):
    """A dead-reckoned offset must be corrected, not compounded."""
    true_base = Pose3.from_xy_yaw(1.0, 3.0, 85.0 * _DEG)
    observation = sighting(camera, catalogue.tag(0), true_base)
    wrong = Pose3.from_xy_yaw(0.4, -0.2, 5.0 * _DEG)
    solution = solve_fixes([observation], catalogue, camera, true_base, cfg.localization, wrong)
    assert solution.accepted
    assert solution.proposed_map_to_odom.x == pytest.approx(0.0, abs=1e-9)
    assert solution.proposed_map_to_odom.y == pytest.approx(0.0, abs=1e-9)
    assert solution.proposed_map_to_odom.yaw == pytest.approx(0.0, abs=1e-9)
    assert solution.correction_xy_m == pytest.approx(math.hypot(0.4, 0.2), abs=1e-9)
    assert solution.correction_yaw_rad == pytest.approx(-5.0 * _DEG, abs=1e-9)


def test_the_first_fix_establishes_the_transform(camera, catalogue, cfg):
    """``/odom`` starts at the rover's boot pose, not at the arena corner.

    So the very first sighting usually implies a map->odom the incremental guard
    would call implausible.  ``initialized=False`` marks that sighting as an
    *establishment*: it is accepted at its true value, while the range/tag guards
    still apply and every later sighting is guarded as before.
    """
    true_pose = Pose3.from_xy_yaw(1.0, 3.4, 90.0 * _DEG)  # nowhere near the origin
    observation = sighting(camera, catalogue.tag(0), true_pose)

    guarded = solve_fixes(
        [observation], catalogue, camera, Pose3.identity(), cfg.localization, Pose3.identity()
    )
    assert not guarded.accepted
    assert guarded.reason == "correction_too_large"

    first = solve_fixes(
        [observation],
        catalogue,
        camera,
        Pose3.identity(),
        cfg.localization,
        Pose3.identity(),
        initialized=False,
    )
    assert first.accepted, first.reason
    assert (first.proposed_map_to_odom.x, first.proposed_map_to_odom.y) == pytest.approx(
        (1.0, 3.4), abs=1e-6
    )
    assert first.proposed_map_to_odom.yaw == pytest.approx(math.pi / 2, abs=1e-6)
    assert first.correction_xy_m == 0.0, "establishing the map is not a correction"
    assert first.correction_yaw_rad == 0.0


def test_localizer_initializes_on_its_first_update(camera, catalogue, cfg):
    """One sighting must take the estimator from un-localized to localized."""
    localizer = Localizer(cfg.localization, catalogue, camera)
    true_pose = Pose3.from_xy_yaw(1.0, 3.4, 90.0 * _DEG)
    # the rover sits at true_pose; its odometry starts there and so reads (0,0,0)
    observation = sighting(camera, catalogue.tag(0), true_pose)
    assert not localizer.localized
    assert localizer.sigma_xy_m == pytest.approx(cfg.localization.max_sigma_xy_m)

    solution = localizer.update([observation], Pose3.identity(), now=1.0)
    assert solution.accepted, solution.reason
    assert localizer.localized and localizer.fixes_total == 1
    assert (localizer.map_to_odom.x, localizer.map_to_odom.y) == pytest.approx((1.0, 3.4), abs=1e-6)
    assert localizer.map_to_odom.yaw == pytest.approx(math.pi / 2, abs=1e-6)
    assert localizer.sigma_xy_m == pytest.approx(cfg.localization.fix_sigma_xy_m)

    # ... and the guard is live again for the next, implausible sighting
    bad = sighting(camera, catalogue.tag(0), Pose3.from_xy_yaw(5.0, 1.0, 0.0))
    rejected = localizer.update([bad], Pose3.identity(), now=2.0)
    assert not rejected.accepted
    assert rejected.reason == "correction_too_large"


# --------------------------------------------------------------------------- #
# every rejection reason
# --------------------------------------------------------------------------- #
def _cfg_path():
    import os

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "perception.yaml")


def test_no_sightings_is_not_a_fix(catalogue, camera, cfg):
    solution = solve_fixes([], catalogue, camera, Pose3.identity(), cfg.localization)
    assert not solution.accepted
    assert solution.reason == "no_tags"
    assert solution.fixes == ()


def test_unknown_tag_is_rejected(catalogue, camera, cfg):
    observation = TagObservation(999, Pose3.identity(), decision_margin=90.0)
    solution = solve_fixes([observation], catalogue, camera, Pose3.identity(), cfg.localization)
    assert not solution.accepted
    assert solution.reason == "unknown_tag"
    assert solution.rejected[0].tag_id == 999


def test_tag_too_close_is_rejected(camera, catalogue, cfg):
    true_base = Pose3.from_xy_yaw(1.0, 3.0, 90.0 * _DEG)
    observation = sighting(camera, catalogue.tag(0), true_base)
    strict = _with(cfg.localization, min_tag_range_m=10.0)
    solution = solve_fixes([observation], catalogue, camera, true_base, strict)
    assert not solution.accepted
    assert solution.reason == "tag_too_close"


def test_tag_too_far_is_rejected(camera, catalogue, cfg):
    true_base = Pose3.from_xy_yaw(1.0, 3.0, 90.0 * _DEG)
    observation = sighting(camera, catalogue.tag(0), true_base)
    strict = _with(cfg.localization, max_tag_range_m=0.5)
    solution = solve_fixes([observation], catalogue, camera, true_base, strict)
    assert not solution.accepted
    assert solution.reason == "tag_too_far"


def test_a_huge_jump_is_rejected(camera, catalogue, cfg):
    true_base = Pose3.from_xy_yaw(1.0, 3.0, 90.0 * _DEG)
    observation = sighting(camera, catalogue.tag(0), true_base)
    wrong = Pose3.from_xy_yaw(5.0, 0.0, 0.0)
    solution = solve_fixes([observation], catalogue, camera, true_base, cfg.localization, wrong)
    assert not solution.accepted
    assert solution.reason == "correction_too_large"
    assert solution.rejected[0].reason == "correction_too_large"


def test_a_huge_yaw_jump_is_rejected(camera, catalogue, cfg):
    true_base = Pose3.from_xy_yaw(1.0, 3.0, 90.0 * _DEG)
    observation = sighting(camera, catalogue.tag(0), true_base)
    # the odometry frame is yawed relative to the proposal without moving it, so
    # only the yaw gate can catch the disagreement
    wrong = Pose3.from_xy_yaw(0.0, 0.0, 0.33)
    strict = _with(cfg.localization, max_correction_m=99.0, max_correction_yaw_rad=0.05)
    solution = solve_fixes([observation], catalogue, camera, true_base, strict, wrong)
    assert not solution.accepted
    assert solution.reason == "correction_yaw_too_large"


def _with(cfg: LocalizationCfg, **changes) -> LocalizationCfg:
    import dataclasses

    return dataclasses.replace(cfg, **changes)


# --------------------------------------------------------------------------- #
# weighting and combination
# --------------------------------------------------------------------------- #
def test_closer_tags_weigh_more(camera, catalogue, cfg):
    near = sighting(camera, catalogue.tag(0), Pose3.from_xy_yaw(1.0, 3.5, 90.0 * _DEG))
    far = sighting(camera, catalogue.tag(2), Pose3.from_xy_yaw(1.0, 3.5, 90.0 * _DEG))
    assert near.range_m < far.range_m
    assert _fix_weight(near, cfg.localization) > _fix_weight(far, cfg.localization)
    # a zero margin is clamped rather than dividing by zero
    assert _fix_weight(_with_margin(near, 0.0), cfg.localization) > 0.0


def _with_margin(observation: TagObservation, margin: float) -> TagObservation:
    import dataclasses

    return dataclasses.replace(observation, decision_margin=margin)


def test_two_tags_are_averaged_by_weight(camera, catalogue, cfg):
    """The far tag's disagreement must be outvoted, but not ignored."""
    true_base = Pose3.from_xy_yaw(1.0, 3.5, 90.0 * _DEG)
    near_obs = sighting(camera, catalogue.tag(0), true_base)
    # the far tag is measured from a slightly wrong odometry pose, so it proposes
    # a 0.3 m correction that the near tag says is unnecessary
    offset = Pose3.from_xy_yaw(-0.3, 0.0, 0.0)
    far_obs = sighting(camera, catalogue.tag(2), true_base.compose(offset))
    assert near_obs.range_m < far_obs.range_m
    solution = solve_fixes([near_obs, far_obs], catalogue, camera, true_base, cfg.localization)
    assert solution.accepted
    assert set(solution.used_tags) == {0, 2}
    weight_near = _fix_weight(near_obs, cfg.localization)
    weight_far = _fix_weight(far_obs, cfg.localization)
    expected = 0.3 * weight_far / (weight_near + weight_far)
    assert solution.correction_xy_m == pytest.approx(expected, abs=1e-9)
    assert solution.correction_xy_m < 0.05, "the near tag dominates"
    assert solution.fixes[0].accepted or solution.fixes[1].accepted


def test_mixed_acceptance_still_fixes(camera, catalogue, cfg):
    true_base = Pose3.from_xy_yaw(1.0, 3.5, 90.0 * _DEG)
    good = sighting(camera, catalogue.tag(0), true_base)
    bad = TagObservation(999, Pose3.identity(), decision_margin=90.0)
    solution = solve_fixes([bad, good], catalogue, camera, true_base, cfg.localization)
    assert solution.accepted
    assert solution.used_tags == (0,)
    assert [fix.reason for fix in solution.fixes] == ["unknown_tag", "ok"]


# --------------------------------------------------------------------------- #
# the odometry buffer
# --------------------------------------------------------------------------- #
def test_odom_buffer_keeps_a_bounded_history():
    buffer = OdomBuffer(capacity=3)
    for index in range(6):
        buffer.add(float(index), Pose3.from_xy_yaw(float(index), 0.0, 0.0))
    assert buffer.count == 3
    assert buffer.latest().x == pytest.approx(5.0)


def test_odom_buffer_picks_the_nearest_sample():
    buffer = OdomBuffer()
    buffer.add(0.00, Pose3.from_xy_yaw(1.0, 0.0, 0.0))
    buffer.add(0.10, Pose3.from_xy_yaw(2.0, 0.0, 0.0))
    buffer.add(0.20, Pose3.from_xy_yaw(3.0, 0.0, 0.0))
    assert buffer.nearest(0.03).x == pytest.approx(1.0)
    assert buffer.nearest(0.12).x == pytest.approx(2.0)
    assert buffer.nearest(-5.0).x == pytest.approx(1.0), "an old stamp uses the oldest sample"
    assert buffer.nearest(99.0).x == pytest.approx(3.0)


def test_odom_buffer_is_planar_and_clearable():
    buffer = OdomBuffer()
    buffer.add(1.0, Pose3(rpy_matrix(0.4, 0.2, 0.7), (1.0, 2.0, 9.0)))
    sample = buffer.latest()
    _x, _y, z, roll, pitch, _yaw = sample.as_rpy()
    assert (z, roll, pitch) == pytest.approx((0.0, 0.0, 0.0))
    buffer.clear()
    assert buffer.count == 0
    assert buffer.latest() is None
    assert buffer.nearest(0.0) is None


# --------------------------------------------------------------------------- #
# the stateful estimator
# --------------------------------------------------------------------------- #
def test_localizer_starts_unlocalised_and_pessimistic(camera, catalogue, cfg):
    localizer = Localizer(cfg.localization, catalogue, camera)
    assert not localizer.localized
    assert localizer.sigma_xy_m == pytest.approx(cfg.localization.max_sigma_xy_m)
    assert math.isinf(localizer.time_since_fix_s(0.0))
    assert localizer.map_to_odom.x == pytest.approx(0.0)


def test_propagate_grows_sigma_with_distance(camera, catalogue, cfg):
    localizer = Localizer(cfg.localization, catalogue, camera)
    localizer.sigma_xy_m = 0.0
    localizer.propagate(Pose3.from_xy_yaw(0.0, 0.0, 0.0), 0.0)
    assert localizer.sigma_xy_m == pytest.approx(0.0), "the first sample sets the origin"
    localizer.propagate(Pose3.from_xy_yaw(2.0, 0.0, 0.0), 1.0)
    assert localizer.sigma_xy_m == pytest.approx(2.0 * cfg.localization.growth_m_per_m)
    localizer.propagate(Pose3.from_xy_yaw(2.0, 0.0, math.pi / 2.0), 2.0)
    assert localizer.sigma_yaw_rad == pytest.approx(0.5 * math.pi * cfg.localization.growth_rad_per_rad)


def test_sigma_is_capped(camera, catalogue, cfg):
    localizer = Localizer(cfg.localization, catalogue, camera)
    localizer.propagate(Pose3.from_xy_yaw(0.0, 0.0, 0.0), 0.0)
    localizer.propagate(Pose3.from_xy_yaw(1000.0, 0.0, 0.0), 1.0)
    assert localizer.sigma_xy_m == pytest.approx(cfg.localization.max_sigma_xy_m)
    # yaw is normalised into (-pi, pi], so one step can never exceed the cap:
    # a run of quarter turns is what actually reaches it
    for step in range(1, 40):
        localizer.propagate(Pose3.from_xy_yaw(1000.0, 0.0, step * math.pi / 2.0), 1.0 + step)
    assert localizer.sigma_yaw_rad == pytest.approx(cfg.localization.max_sigma_yaw_rad)


def test_update_accepts_and_shrinks(camera, catalogue, cfg):
    localizer = Localizer(cfg.localization, catalogue, camera)
    true_base = Pose3.from_xy_yaw(1.0, 3.0, 90.0 * _DEG)
    solution = localizer.update([sighting(camera, catalogue.tag(0), true_base)], true_base, 5.0)
    assert isinstance(solution, FixSolution)
    assert solution.accepted
    assert localizer.localized
    assert localizer.fixes_total == 1
    assert localizer.last_fix_at == 5.0
    assert localizer.time_since_fix_s(6.5) == pytest.approx(1.5)
    assert localizer.sigma_xy_m == pytest.approx(cfg.localization.fix_sigma_xy_m)
    assert localizer.sigma_yaw_rad == pytest.approx(cfg.localization.fix_sigma_yaw_rad)
    assert localizer.last_tag_id == 0


def test_two_tags_shrink_sigma_by_root_n(camera, catalogue, cfg):
    localizer = Localizer(cfg.localization, catalogue, camera)
    true_base = Pose3.from_xy_yaw(1.0, 3.5, 90.0 * _DEG)
    observations = [
        sighting(camera, catalogue.tag(0), true_base),
        sighting(camera, catalogue.tag(2), true_base),
    ]
    assert localizer.update(observations, true_base, 1.0).accepted
    assert localizer.sigma_xy_m == pytest.approx(cfg.localization.fix_sigma_xy_m / math.sqrt(2.0))


def test_rejected_update_keeps_the_previous_fix(camera, catalogue, cfg):
    localizer = Localizer(cfg.localization, catalogue, camera)
    true_base = Pose3.from_xy_yaw(1.0, 3.0, 90.0 * _DEG)
    localizer.update([sighting(camera, catalogue.tag(0), true_base)], true_base, 1.0)
    before = localizer.map_to_odom
    solution = localizer.update([], true_base, 2.0)
    assert not solution.accepted
    assert localizer.rejected_total == 1
    assert localizer.last_reason == "no_tags"
    assert (localizer.map_to_odom.x, localizer.map_to_odom.y) == pytest.approx((before.x, before.y))


def test_state_fields_match_the_message(camera, catalogue, cfg):
    localizer = Localizer(cfg.localization, catalogue, camera)
    true_base = Pose3.from_xy_yaw(1.0, 3.0, 90.0 * _DEG)
    localizer.update([sighting(camera, catalogue.tag(0), true_base)], true_base, 2.0)
    fields = localizer.state_fields(2.5)
    assert set(fields) == {
        "localized",
        "fixes_total",
        "fix_age_frames",
        "time_since_fix_s",
        "sigma_xy_m",
        "sigma_yaw_rad",
        "map_to_odom",
        "last_tag_id",
        "last_fix_xy_m",
        "last_fix_yaw_rad",
        "reason",
    }
    assert fields["localized"] is True
    assert fields["time_since_fix_s"] == pytest.approx(0.5)
    assert isinstance(fields["map_to_odom"], Pose3)
    assert fields["reason"] == "ok"


def test_tag_fix_pose_property():
    pose = Pose3.from_xy_yaw(0.1, 0.2, 0.3)
    assert TagFix(0, True, "ok", 1.0, pose, 2.0).pose is pose
    assert TagFix(0, False, "tag_too_far", 9.0, Pose3.identity()).pose is None
