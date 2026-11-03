"""The 4000 kbps rule: the meter that decides whether comms is legal.

The competition measures an *average over the link*, so the meter is tested as a
wall-clock window, not as a frame counter: what matters is that a burst of first
frames cannot trigger a spurious de-rate (the settle floor), that old samples
age out, and that the total is the sum over every camera.
"""

from __future__ import annotations

import pytest

from rover_perception.bandwidth import ByteRateMeter, CameraTraffic

_KIB = 1024


def _meter(**changes) -> ByteRateMeter:
    options = dict(names=("cam_nav", "cam_tag"), window_s=10.0, limit_kbps=4000.0, settle_s=1.0)
    options.update(changes)
    return ByteRateMeter(**options)


def test_a_constructor_rejects_a_degenerate_window():
    with pytest.raises(ValueError):
        ByteRateMeter(["a"], window_s=0.0)
    with pytest.raises(ValueError):
        ByteRateMeter(["a"], settle_s=0.0)


def test_an_unknown_camera_is_a_key_error():
    meter = _meter()
    with pytest.raises(KeyError):
        meter.observe("cam_dump", 10, now=0.0)
    with pytest.raises(KeyError):
        meter.rate_kbps("cam_dump", now=0.0)


def test_a_single_frame_is_measured_over_the_settle_window():
    meter = _meter()
    meter.observe("cam_nav", 125_000, now=0.0)
    # 125000 bytes * 8 = 1,000,000 bits; the denominator is floored at settle_s
    assert meter.rate_kbps("cam_nav", 0.0) == pytest.approx(1000.0)
    assert meter.rate_hz("cam_nav", 0.0) == pytest.approx(1.0)
    assert meter.rate_kbps("cam_tag", 0.0) == pytest.approx(0.0)
    assert meter.total_kbps(0.0) == pytest.approx(1000.0)


def test_the_average_spreads_over_the_elapsed_window():
    meter = _meter()
    for index in range(4):
        meter.observe("cam_nav", 125_000, now=index * 1.0)
    # 4 frames over 3 s = 4,000,000 bits / 3 s
    assert meter.rate_kbps("cam_nav", 3.0) == pytest.approx(4_000_000 / 3.0 / 1000.0)
    assert meter.rate_hz("cam_nav", 3.0) == pytest.approx(4.0 / 3.0)


def test_the_window_is_capped_so_an_old_burst_stops_counting():
    meter = _meter(window_s=10.0, settle_s=1.0)
    meter.observe("cam_nav", 1_250_000, now=0.0)
    # a full window later the numerator is the same but the denominator is 10 s
    meter.tick(now=11.0)
    assert meter.rate_kbps("cam_nav", 11.0) == pytest.approx(0.0), "the sample aged out"
    meter.observe("cam_nav", 125_000, now=11.0)
    assert meter.rate_kbps("cam_nav", 11.0) == pytest.approx(100.0), "1e6 bits / 10 s"


def test_counters_are_cumulative_but_the_rate_is_a_window():
    meter = _meter()
    meter.observe("cam_nav", 1000, now=0.0)
    meter.observe("cam_nav", 2000, now=1.0)
    assert meter.frames("cam_nav") == 2
    assert meter.bytes("cam_nav") == 3000
    assert meter.last_frame_at("cam_nav") == pytest.approx(1.0)
    assert meter.rate_kbps("cam_nav", 1.0) == pytest.approx(3000 * 8 / 1000.0 / 1.0)


def test_total_sums_every_camera():
    meter = _meter()
    meter.observe("cam_nav", 125_000, now=0.0)
    meter.observe("cam_tag", 62_500, now=0.0)
    assert meter.total_kbps(0.0) == pytest.approx(1500.0)


def test_four_thousand_kbps_is_the_limit_the_manager_compares_against():
    meter = _meter()
    # 4,000,000 bits in 1 s = 4000 kbps, exactly the cap
    meter.observe("cam_nav", 500_000, now=0.0)
    assert meter.rate_kbps("cam_nav", 0.0) == pytest.approx(4000.0)
    assert meter.limit_kbps == pytest.approx(4000.0)


def test_streaming_is_a_staleness_question():
    meter = _meter()
    assert not meter.is_streaming("cam_nav", now=0.0, stale_after_s=2.0), "never seen is not streaming"
    meter.observe("cam_nav", 10, now=1.0)
    assert meter.is_streaming("cam_nav", now=3.0, stale_after_s=2.0)
    assert not meter.is_streaming("cam_nav", now=3.5, stale_after_s=2.0)


def test_traffic_returns_the_live_counters():
    meter = _meter()
    meter.observe("cam_nav", 42, now=0.0)
    traffic = meter.traffic("cam_nav")
    assert isinstance(traffic, CameraTraffic)
    assert traffic.name == "cam_nav"
    assert traffic.frames == 1 and traffic.bytes == 42


def test_reset_clears_the_history():
    meter = _meter()
    meter.observe("cam_nav", 125_000, now=0.0)
    meter.reset()
    assert meter.frames("cam_nav") == 0
    assert meter.bytes("cam_nav") == 0
    assert meter.last_frame_at("cam_nav") is None
    assert meter.rate_kbps("cam_nav", 0.0) == pytest.approx(0.0)
    assert meter.elapsed(0.0) == pytest.approx(0.0)


def test_elapsed_counts_from_the_first_frame():
    meter = _meter()
    assert meter.elapsed(5.0) == pytest.approx(0.0), "nothing observed yet"
    meter.observe("cam_nav", 10, now=2.0)
    assert meter.elapsed(5.0) == pytest.approx(3.0)
    assert meter.elapsed(1.0) == pytest.approx(0.0), "a backwards clock does not go negative"
