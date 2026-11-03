"""Multi-frame tracking: what turns single-frame noise into obstacles.

The competition publishes obstacles to a planner, so the tracker has one job
above all: **never invent an obstacle because of a flicker, and never lose a
crater because it looked like a boulder once.**  Confirmation needs repeated
hits; a class disagreement always resolves towards the more dangerous class and
marks the track conservative; and a missed track's uncertainty grows instead of
its position being trusted.
"""

from __future__ import annotations

import math

import pytest

from rover_perception.config import TrackingCfg, load_config
from rover_perception.tracking import SIGMA_GROWTH_PER_MISS, Track, Tracker
from support import observation, track

import os


@pytest.fixture(scope="module")
def cfg():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_config(os.path.join(here, "..", "config", "perception.yaml")).tracking


def _tracker(cfg: TrackingCfg, **changes) -> Tracker:
    import dataclasses

    return Tracker(dataclasses.replace(cfg, **changes))


# --------------------------------------------------------------------------- #
# the track
# --------------------------------------------------------------------------- #
def test_track_reports_age_and_seen_age(cfg):
    one = track(1.0, 2.0, now=5.0, confidence=0.9, observations=3)
    one.created_at = 2.0
    one.last_seen_at = 4.5
    assert one.age_s(5.0) == pytest.approx(3.0)
    assert one.last_seen_age_s(5.0) == pytest.approx(0.5)
    assert one.position == pytest.approx((1.0, 2.0))
    one.created_at, one.last_seen_at = 9.0, 9.0
    assert one.age_s(5.0) == 0.0, "a clock going backwards must not make an age negative"


def test_confirmation_needs_hits_and_confidence(cfg):
    assert not track(1.0, 1.0, observations=1).is_confirmed(cfg)
    assert not track(1.0, 1.0, observations=3, confidence=0.1).is_confirmed(cfg)
    assert track(1.0, 1.0, observations=2, confidence=0.3).is_confirmed(cfg)


def test_track_serialises_every_field(cfg):
    one = track(1.5, 2.5, kind="crater", depth=0.0, height=0.0, now=10.0)
    entry = one.as_dict(10.0)
    assert set(entry) == {
        "id",
        "kind",
        "class_id",
        "x",
        "y",
        "radius",
        "height",
        "depth",
        "score",
        "confidence",
        "observations",
        "age_s",
        "last_seen_age_s",
        "misses",
        "conservative",
        "source",
        "sigma_xy_m",
    }
    assert entry["kind"] == "crater"
    assert entry["class_id"] == 1
    assert entry["x"] == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# association and creation
# --------------------------------------------------------------------------- #
def test_one_frame_is_not_enough_to_publish(cfg):
    tracker = _tracker(cfg)
    confirmed = tracker.update([observation(3.0, 2.0)], now=0.0, source="cam_nav")
    assert confirmed == []
    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].id == "obstacle_0"
    assert tracker.tracks[0].observations == 1
    assert tracker.tracks[0].source == "cam_nav"


def test_repeated_sightings_confirm_and_smooth(cfg):
    tracker = _tracker(cfg)
    tracker.update([observation(3.0, 2.0)], now=0.0)
    confirmed = tracker.update([observation(3.2, 2.0)], now=0.1)
    assert len(confirmed) == 1
    one = confirmed[0]
    assert one.observations == 2
    # position_gain 0.5: halfway between the first and the second sighting
    assert one.x == pytest.approx(3.1)
    assert one.sigma_xy_m == pytest.approx(0.1), "both observations have the same sigma"


def test_a_distant_sighting_is_a_new_track(cfg):
    tracker = _tracker(cfg)
    tracker.update([observation(3.0, 2.0)], now=0.0)
    tracker.update([observation(3.0, 2.0)], now=0.0)
    tracker.update([observation(5.0, 2.0)], now=0.0)
    assert len(tracker.tracks) == 2
    ids = sorted(one.id for one in tracker.tracks)
    assert ids == ["obstacle_0", "obstacle_1"]


def test_an_out_of_radius_sighting_does_not_steal_a_track(cfg):
    near = _tracker(cfg, associate_radius_m=0.30)
    near.update([observation(3.0, 2.0)], now=0.0)
    near.update([observation(3.25, 2.0)], now=0.0)
    assert len(near.tracks) == 1, "0.25 m is inside the gate"
    far = _tracker(cfg, associate_radius_m=0.30)
    far.update([observation(3.0, 2.0)], now=0.0)
    far.update([observation(3.35, 2.0)], now=0.0)
    assert len(far.tracks) == 2, "0.35 m is outside it"


# --------------------------------------------------------------------------- #
# ageing out
# --------------------------------------------------------------------------- #
def test_misses_grow_the_sigma_and_then_drop_the_track(cfg):
    tracker = _tracker(cfg, max_misses=3)
    tracker.update([observation(3.0, 2.0)], now=0.0)
    tracker.update([observation(3.0, 2.0)], now=0.0)
    before = tracker.tracks[0].sigma_xy_m
    # same timestamp, so decay cannot confound the miss count
    tracker.update([], now=0.0)
    assert tracker.tracks[0].misses == 1
    assert tracker.tracks[0].sigma_xy_m == pytest.approx(before * SIGMA_GROWTH_PER_MISS)
    tracker.update([], now=0.0)
    tracker.update([], now=0.0)
    assert len(tracker.tracks) == 1, "max_misses = 3 still keeps the track"
    tracker.update([], now=0.0)
    assert tracker.tracks == []


def test_a_reappearance_resets_the_miss_count(cfg):
    tracker = _tracker(cfg, max_misses=3)
    tracker.update([observation(3.0, 2.0)], now=0.0)
    tracker.update([], now=0.0)
    tracker.update([observation(3.05, 2.0)], now=0.0)
    assert tracker.tracks[0].misses == 0


def test_confidence_decays_with_the_half_life(cfg):
    tracker = _tracker(cfg, decay_half_life_s=1.0, min_confidence=0.05)
    tracker.update([observation(3.0, 2.0)], now=0.0)
    first = tracker.tracks[0].confidence
    tracker.update([], now=1.0)
    assert tracker.tracks[0].confidence == pytest.approx(first * 0.5, rel=1e-6)
    tracker.update([], now=2.0)
    assert tracker.tracks[0].confidence == pytest.approx(first * 0.25, rel=1e-6)


def test_low_confidence_tracks_are_pruned(cfg):
    tracker = _tracker(cfg, min_confidence=0.4, decay_half_life_s=0.1)
    tracker.update([observation(3.0, 2.0, score=1.0)], now=0.0)
    assert tracker.tracks, "the track starts above the floor"
    tracker.update([], now=5.0)
    assert tracker.tracks == []


def test_decay_is_a_function_of_wall_clock_not_of_update_rate(cfg):
    """A half-life means *time*, so 10 updates must decay the same as 2.

    Iterative decay (``confidence *= ...`` on every call) would make the node's
    publish rate decide how long a track survives, which is not what the config
    promises.
    """
    half_life = 2.0
    results = []
    for steps in (2, 5, 20):
        tracker = _tracker(cfg, decay_half_life_s=half_life, min_confidence=0.01, max_misses=1000)
        tracker.update([observation(3.0, 2.0)], now=0.0)
        start = tracker.tracks[0].confidence
        for step in range(steps):
            tracker.update([], now=(step + 1) * half_life / steps)
        results.append((start, tracker.tracks[0].confidence))
    for start, end in results:
        assert end == pytest.approx(start * 0.5, rel=1e-9)
    assert len({round(end / start, 9) for start, end in results}) == 1


# --------------------------------------------------------------------------- #
# class handling
# --------------------------------------------------------------------------- #
def test_kind_escalates_from_boulder_to_crater(cfg):
    tracker = _tracker(cfg)
    tracker.update([observation(3.0, 2.0, kind="boulder")], now=0.0)
    tracker.update([observation(3.0, 2.0, kind="crater")], now=0.0)
    one = tracker.tracks[0]
    assert one.kind == "crater"
    assert one.class_id == 1
    assert one.conservative is True, "a disagreement is recorded, not hidden"
    assert len(tracker.tracks) == 1, "the disagreement does not fork the track"


def test_kind_never_downgrades(cfg):
    tracker = _tracker(cfg)
    tracker.update([observation(3.0, 2.0, kind="crater")], now=0.0)
    tracker.update([observation(3.0, 2.0, kind="boulder")], now=0.0)
    one = tracker.tracks[0]
    assert one.kind == "crater", "a crater can never be reclassified as drivable"
    assert one.conservative is True


def test_conservative_observations_mark_the_track(cfg):
    tracker = _tracker(cfg)
    tracker.update([observation(3.0, 2.0, conservative=True)], now=0.0)
    assert tracker.tracks[0].conservative is True


def test_craters_get_a_confidence_bias(cfg):
    """A crater false negative is a wheel in a hole; a bias is the cheap insurance."""
    assert cfg.crater_bias > 0.0
    boulder = _tracker(cfg)
    crater = _tracker(cfg)
    boulder.update([observation(3.0, 2.0, kind="boulder", score=0.9)], now=0.0)
    crater.update([observation(3.0, 2.0, kind="crater", score=0.9)], now=0.0)
    assert crater.tracks[0].confidence > boulder.tracks[0].confidence


# --------------------------------------------------------------------------- #
# the published set
# --------------------------------------------------------------------------- #
def test_confirmed_tracks_are_sorted_and_capped(cfg):
    tracker = _tracker(cfg, max_obstacles=2)
    # three obstacles, each seen twice so all three confirm
    for x, score in ((1.0, 0.7), (3.0, 0.9), (5.0, 0.6)):
        tracker.update([observation(x, 1.0, score=score)], now=0.0)
    confirmed = tracker.update(
        [observation(1.0, 1.0, score=0.7), observation(3.0, 1.0, score=0.9), observation(5.0, 1.0, score=0.6)],
        now=0.0,
    )
    assert len(confirmed) == 2, "max_obstacles caps the published list"
    assert confirmed[0].x == pytest.approx(3.0), "highest confidence first"


def test_reset_clears_everything(cfg):
    tracker = _tracker(cfg)
    tracker.update([observation(3.0, 2.0)], now=0.0)
    tracker.reset()
    assert tracker.tracks == []
    assert tracker.confirmed == []
    tracker.update([observation(3.0, 2.0)], now=0.0)
    assert tracker.tracks[0].id == "obstacle_0", "ids restart from a clean slate"


def test_empty_update_is_harmless(cfg):
    tracker = _tracker(cfg)
    assert tracker.update([], now=0.0) == []
    assert tracker.tracks == []
