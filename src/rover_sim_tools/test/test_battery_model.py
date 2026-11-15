"""Tests for the battery model (shared with sandsim)."""

from __future__ import annotations

import pytest

from rover_sim_tools.battery_model import BatteryModel


@pytest.fixture
def battery() -> BatteryModel:
    return BatteryModel(
        voltage_nominal_v=12.0,
        capacity_wh=360.0,
        voltage_cutoff_v=10.0,
        r_internal_ohm=0.05,
    )


def test_initial_state(battery: BatteryModel) -> None:
    assert battery.energy_wh == 360.0
    assert battery.soc() == pytest.approx(1.0)
    assert battery.terminal_voltage_v(0.0) == pytest.approx(12.0)


def test_voltage_sag(battery: BatteryModel) -> None:
    v = battery.terminal_voltage_v(10.0)
    assert v == pytest.approx(12.0 - 10.0 * 0.05)


def test_energy_integration(battery: BatteryModel) -> None:
    # Draw 10 A for 1 hour at ~11.5 V terminal voltage
    battery.step(10.0, 3600.0)
    assert battery.energy_wh < 360.0
    assert battery.energy_wh > 0.0
    # Energy removed = V_terminal * I * t = ~11.5 V * 10 A * 1 h = 115 Wh
    energy_removed = 360.0 - battery.energy_wh
    assert energy_removed == pytest.approx(115.0, abs=5.0)


def test_depleted_at_cutoff(battery: BatteryModel) -> None:
    # Drain the battery: at 20 A, V_terminal = 11 V (> 10 V cutoff)
    # Energy = 360 Wh, power = 11 V * 20 A = 220 W, runtime = 360/220 h = 5891 s
    battery.step(20.0, 6000.0)
    assert battery.energy_wh == pytest.approx(0.0, abs=0.1)
    assert battery.is_depleted(10.0)


def test_remaining_runtime(battery: BatteryModel) -> None:
    runtime = battery.remaining_runtime_s(10.0)
    assert runtime > 0.0
    # ~360 Wh / (11.5 V * 10 A) = ~3.1 h = ~11200 s
    assert runtime == pytest.approx(11200.0, abs=500.0)

    runtime_zero = battery.remaining_runtime_s(0.0)
    assert runtime_zero == float("inf")


def test_soc_range(battery: BatteryModel) -> None:
    assert 0.0 <= battery.soc() <= 1.0
    # At 30 A, V_terminal = 10.5 V (> 10 V cutoff), enough to drain fully
    battery.step(30.0, 4114.0)
    assert battery.soc() == pytest.approx(0.0, abs=0.01)
