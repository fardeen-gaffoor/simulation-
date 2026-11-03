"""Thermal zones: the Pi throttles, and the camera manager must know.

The tests write *real files* in a temporary directory and point the monitor at
them, because the interesting behaviour is in the parsing (millidegrees vs
degrees, implausible values, missing zones) rather than in a mock.  A machine
with no zones at all is a normal, reported state: nominal, zone ``"none"``.
"""

from __future__ import annotations

import os

import pytest

from rover_perception.config import load_config
from rover_perception.thermal import (
    THERMAL_CRITICAL,
    THERMAL_NOMINAL,
    THERMAL_THROTTLING,
    THERMAL_WARNING,
    ThermalMonitor,
    ThermalReading,
    discover_zones,
    read_zone_temperature,
)


@pytest.fixture(scope="module")
def tcfg():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_config(os.path.join(here, "..", "config", "perception.yaml")).camera_manager.thermal


def _zone(directory, index: int, text: str) -> str:
    folder = os.path.join(directory, f"thermal_zone{index}")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "temp")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


# --------------------------------------------------------------------------- #
# reading one zone
# --------------------------------------------------------------------------- #
def test_millidegrees_are_converted(tmp_path):
    assert read_zone_temperature(_zone(str(tmp_path), 0, "45000\n")) == pytest.approx(45.0)


def test_whole_degrees_are_accepted(tmp_path):
    assert read_zone_temperature(_zone(str(tmp_path), 0, "45\n")) == pytest.approx(45.0)


def test_a_missing_file_is_none():
    assert read_zone_temperature("/nonexistent/thermal_zone0/temp") is None


def test_garbage_is_none(tmp_path):
    assert read_zone_temperature(_zone(str(tmp_path), 0, "not a number")) is None


def test_an_implausible_value_is_refused(tmp_path):
    assert read_zone_temperature(_zone(str(tmp_path), 0, "999999")) is None, "999.999 C is not real"
    assert read_zone_temperature(_zone(str(tmp_path), 1, "-40000")) is None, "far below any real zone"
    assert read_zone_temperature(_zone(str(tmp_path), 2, "-20")) == pytest.approx(-20.0), "cold is fine"


def test_discover_zones_finds_what_exists(tmp_path):
    _zone(str(tmp_path), 2, "40000")
    _zone(str(tmp_path), 0, "40000")
    found = discover_zones(os.path.join(str(tmp_path), "thermal_zone*", "temp"))
    assert [os.path.basename(os.path.dirname(path)) for path in found] == [
        "thermal_zone0",
        "thermal_zone2",
    ], "sorted for determinism"


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def test_every_state_boundary(tcfg):
    monitor = ThermalMonitor(tcfg, paths=[])
    assert monitor.classify(None) == THERMAL_NOMINAL, "no reading is not a fault"
    assert monitor.classify(tcfg.warn_c - 1.0) == THERMAL_NOMINAL
    assert monitor.classify(tcfg.warn_c) == THERMAL_WARNING
    assert monitor.classify(tcfg.throttle_c - 0.1) == THERMAL_WARNING
    assert monitor.classify(tcfg.throttle_c) == THERMAL_THROTTLING
    assert monitor.classify(tcfg.critical_c - 0.1) == THERMAL_THROTTLING
    assert monitor.classify(tcfg.critical_c) == THERMAL_CRITICAL


def test_the_thresholds_are_ordered(tcfg):
    assert 0.0 < tcfg.warn_c < tcfg.throttle_c < tcfg.critical_c
    assert tcfg.throttle_steps >= 1


# --------------------------------------------------------------------------- #
# the monitor
# --------------------------------------------------------------------------- #
def test_the_hottest_zone_wins(tcfg, tmp_path):
    paths = [
        _zone(str(tmp_path), 0, "60000"),
        _zone(str(tmp_path), 1, "88000"),
        _zone(str(tmp_path), 2, "71000"),
    ]
    reading = ThermalMonitor(tcfg, paths=paths).read()
    assert isinstance(reading, ThermalReading)
    assert reading.zone == "thermal_zone1"
    assert reading.temperature_c == pytest.approx(88.0)
    assert reading.state == THERMAL_THROTTLING
    assert reading.state_name == "throttling"
    assert reading.available
    assert set(reading.zones) == set(paths)


def test_no_zones_is_nominal_and_reported(tcfg):
    reading = ThermalMonitor(tcfg, paths=[]).read()
    assert reading.zone == "none"
    assert reading.temperature_c is None
    assert reading.state == THERMAL_NOMINAL
    assert not reading.available
    assert reading.state_name == "nominal"


def test_an_unreadable_zone_is_skipped(tcfg, tmp_path):
    good = _zone(str(tmp_path), 0, "79000")
    missing = os.path.join(str(tmp_path), "thermal_zone9", "temp")
    reading = ThermalMonitor(tcfg, paths=[missing, good]).read()
    assert reading.temperature_c == pytest.approx(79.0)
    assert list(reading.zones) == [good]


def test_disabled_monitoring_reports_nothing(tcfg):
    import dataclasses

    disabled = dataclasses.replace(tcfg, enabled=False)
    reading = ThermalMonitor(disabled, paths=["/dev/null"]).read()
    assert not reading.available
    assert reading.zones == {}


def test_paths_prefers_the_explicit_list(tcfg, tmp_path):
    explicit = _zone(str(tmp_path), 0, "40000")
    monitor = ThermalMonitor(tcfg, paths=[explicit])
    assert monitor.paths() == [explicit]


def test_temperatures_are_a_path_to_value_map(tcfg, tmp_path):
    paths = [_zone(str(tmp_path), 0, "50000"), _zone(str(tmp_path), 1, "55000")]
    values = ThermalMonitor(tcfg, paths=paths).temperatures()
    assert values == pytest.approx({paths[0]: 50.0, paths[1]: 55.0})
