"""fps, CPU and latency: the numbers in the health log.

Everything is driven by a synthetic clock and a synthetic ``/proc`` reading, so
the arithmetic (and its guards: a first sample has no delta, a backwards clock,
a malformed stat line) is checked without depending on the machine's load.
"""

from __future__ import annotations

import os

import pytest

from rover_perception.cpu import CpuMeter, FpsMeter, LatencyMeter, clock_ticks, read_proc_stat


# --------------------------------------------------------------------------- #
# /proc/self/stat
# --------------------------------------------------------------------------- #
def _stat_line(comm: str, utime: int = 5000, stime: int = 1000) -> str:
    # pid (comm) state ppid ... utime stime: 13+ fields after the state
    tail = f"R 1 2 3 4 5 6 7 8 9 10 {utime} {stime} 20 21 22"
    return f"4321 ({comm}) {tail}\n"


def test_read_proc_stat_parses_a_command_with_spaces_and_parens(tmp_path):
    path = os.path.join(str(tmp_path), "stat")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_stat_line("my rover (nav) node"))
    ticks = clock_ticks()
    utime, stime = read_proc_stat(path)
    assert utime == pytest.approx(5000 / ticks)
    assert stime == pytest.approx(1000 / ticks)


def test_read_proc_stat_on_a_truncated_line_is_none(tmp_path):
    path = os.path.join(str(tmp_path), "stat")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("1 (x) R 2 3\n")
    assert read_proc_stat(path) is None


def test_read_proc_stat_on_a_missing_file_is_none():
    assert read_proc_stat("/nonexistent/stat") is None


def test_clock_ticks_is_positive():
    assert clock_ticks() > 0.0


# --------------------------------------------------------------------------- #
# CPU
# --------------------------------------------------------------------------- #
class _Proc:
    """A mutable stand-in for ``/proc/<pid>/stat``."""

    def __init__(self, utime: float, stime: float):
        self.utime = utime
        self.stime = stime

    def __call__(self, _path: str):
        return (self.utime, self.stime)


def test_cpu_needs_two_samples_for_a_percentage():
    proc = _Proc(1.0, 0.0)
    meter = CpuMeter(reader=proc)
    assert meter.sample(0.0) == pytest.approx(0.0), "no baseline yet"
    assert meter.percent == pytest.approx(0.0)
    proc.utime = 1.5  # +0.5 s of CPU over 1 s of wall clock
    assert meter.sample(1.0) == pytest.approx(50.0)
    assert meter.percent == pytest.approx(50.0)


def test_cpu_can_exceed_one_hundred_percent_on_several_cores():
    proc = _Proc(0.0, 0.0)
    meter = CpuMeter(reader=proc)
    meter.sample(0.0)
    proc.utime = 2.0
    assert meter.sample(1.0) == pytest.approx(200.0)


def test_cpu_ignores_a_zero_time_delta():
    proc = _Proc(0.0, 0.0)
    meter = CpuMeter(reader=proc)
    meter.sample(0.0)
    proc.utime = 1.0
    assert meter.sample(1.0) == pytest.approx(100.0)
    assert meter.sample(1.0) == pytest.approx(100.0), "a repeated timestamp keeps the last value"


def test_cpu_treats_a_vanishing_counter_as_zero_not_negative():
    proc = _Proc(5.0, 0.0)
    meter = CpuMeter(reader=proc)
    meter.sample(0.0)
    proc.utime = 1.0  # the counter went backwards
    assert meter.sample(1.0) == pytest.approx(0.0)


def test_cpu_survives_an_unreadable_proc():
    meter = CpuMeter(reader=lambda _path: None)
    assert meter.sample(0.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# fps
# --------------------------------------------------------------------------- #
def test_fps_of_a_single_tick_is_zero():
    meter = FpsMeter(window_s=2.0)
    assert meter.fps(0.0) == pytest.approx(0.0)
    meter.tick(0.0)
    assert meter.fps(0.0) == pytest.approx(0.0)
    assert meter.frames == 1


def test_fps_counts_intervals_not_frames():
    meter = FpsMeter(window_s=10.0)
    for step in range(5):
        meter.tick(step * 0.5)
    assert meter.fps(2.0) == pytest.approx(2.0), "4 intervals over 2 s"


def test_fps_forgets_frames_outside_the_window():
    meter = FpsMeter(window_s=1.0)
    meter.tick(0.0)
    meter.tick(0.5)
    meter.tick(1.0)
    assert meter.fps(1.0) == pytest.approx(2.0)
    meter.tick(3.0)
    assert meter.frames == 1, "everything older than 2 s is dropped"
    assert meter.fps(3.0) == pytest.approx(0.0), "one frame is not a rate"
    meter.tick(3.5)
    assert meter.frames == 2
    assert meter.fps(3.5) == pytest.approx(2.0)


def test_fps_rejects_a_bad_window_and_resets():
    with pytest.raises(ValueError):
        FpsMeter(window_s=0.0)
    meter = FpsMeter()
    meter.tick(0.0)
    meter.reset()
    assert meter.frames == 0
    assert meter.fps(1.0) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# latency
# --------------------------------------------------------------------------- #
def test_latency_reports_mean_and_worst_in_milliseconds():
    meter = LatencyMeter(window_s=5.0)
    meter.add(0.010, now=0.0)
    meter.add(0.030, now=0.5)
    assert meter.mean_ms() == pytest.approx(20.0)
    assert meter.max_ms() == pytest.approx(30.0)
    assert meter.samples == 2


def test_latency_prunes_samples_older_than_the_window():
    meter = LatencyMeter(window_s=1.0)
    meter.add(0.010, now=0.0)
    meter.add(0.020, now=0.5)
    meter.add(0.500, now=2.0)
    assert meter.samples == 1
    assert meter.mean_ms() == pytest.approx(500.0)
    assert meter.max_ms() == pytest.approx(500.0)


def test_empty_latency_is_zero():
    meter = LatencyMeter()
    assert meter.mean_ms() == pytest.approx(0.0)
    assert meter.max_ms() == pytest.approx(0.0)
    assert meter.samples == 0
