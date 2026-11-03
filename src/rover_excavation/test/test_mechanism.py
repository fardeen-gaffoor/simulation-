"""The mechanism: slew-limited actuators, the tub gate, and the two interlocks."""

from __future__ import annotations

import pytest

from rover_excavation.config import default_config_path
from rover_excavation.mechanism import (
    BELT_TOL_RADPS,
    STATE_DEPLOYING,
    STATE_FAULT,
    STATE_NAMES,
    STATE_STOWED,
    Mechanism,
    MechanismCommand,
    state_name,
)

from support import config_with


def _step(mechanism, seconds, dt=0.01):
    steps = int(round(seconds / dt))
    for _ in range(steps):
        mechanism.step(dt)


def test_starts_stowed_and_travel_ready(config):
    mechanism = Mechanism(config.mechanism)
    assert mechanism.pitch_rad == pytest.approx(config.mechanism.ladder.stowed_pitch_rad)
    assert mechanism.belt_velocity_radps == 0.0
    assert mechanism.tub_gate == 0.0
    assert mechanism.travel_ready
    assert mechanism.fault is None
    assert mechanism.snapshot().state == STATE_STOWED


def test_ladder_slews_at_its_rate(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.command(config.mechanism.ladder.dig_pitch_rad, 0.0, 0.0)
    rate = config.mechanism.ladder.max_rate_radps
    mechanism.step(0.1)
    moved = mechanism.pitch_rad - config.mechanism.ladder.stowed_pitch_rad
    assert moved == pytest.approx(rate * 0.1)
    # time_to_pitch predicts exactly that: remaining travel / rate.
    expected = mechanism.time_to_pitch(config.mechanism.ladder.dig_pitch_rad)
    assert expected == pytest.approx((config.mechanism.ladder.dig_pitch_rad - mechanism.pitch_rad) / rate)
    _step(mechanism, expected + 0.05)
    assert mechanism.at_pitch(config.mechanism.ladder.dig_pitch_rad)
    assert not mechanism.travel_ready


def test_ladder_clamps_to_its_limits(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.command(10.0, 0.0, 0.0)
    assert mechanism.target.ladder_pitch_rad == pytest.approx(config.mechanism.ladder.max_pitch_rad)
    mechanism.command(-10.0, 0.0, 0.0)
    assert mechanism.target.ladder_pitch_rad == pytest.approx(config.mechanism.ladder.min_pitch_rad)


def test_belt_accelerates_within_its_limit(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.command(mechanism.stowed_pitch_rad, config.mechanism.belt.max_velocity_radps, 0.0)
    mechanism.step(0.05)
    assert mechanism.belt_velocity_radps == pytest.approx(config.mechanism.belt.max_accel_radps2 * 0.05)


def test_belt_clamps_both_ways(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.command(mechanism.stowed_pitch_rad, 99.0, 0.0)
    assert mechanism.target.belt_velocity_radps == pytest.approx(config.mechanism.belt.max_velocity_radps)
    mechanism.command(mechanism.stowed_pitch_rad, -99.0, 0.0)
    assert mechanism.target.belt_velocity_radps == pytest.approx(-config.mechanism.belt.max_velocity_radps)


def test_belt_surface_speed_follows_the_roller(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.belt_velocity_radps = config.mechanism.belt.dig_velocity_radps
    assert mechanism.belt_linear_mps == pytest.approx(
        config.mechanism.belt.dig_velocity_radps * config.mechanism.belt.roller_radius_m
    )


def test_gate_opens_over_its_time(config):
    mechanism = Mechanism(config.mechanism)
    open_time = config.mechanism.tub.gate_open_time_s
    mechanism.command(mechanism.stowed_pitch_rad, 0.0, 1.0)
    mechanism.step(0.5 * open_time)
    assert mechanism.tub_gate == pytest.approx(0.5, abs=0.02)
    _step(mechanism, open_time)
    assert mechanism.tub_gate == pytest.approx(1.0)
    mechanism.command(mechanism.stowed_pitch_rad, 0.0, 0.0)
    _step(mechanism, open_time + 0.05)
    assert mechanism.tub_gate == pytest.approx(0.0, abs=1e-9)


def test_depth_is_zero_above_the_sand(config):
    mechanism = Mechanism(config.mechanism)
    assert mechanism.depth_m == 0.0
    mechanism.pitch_rad = config.mechanism.ladder.dig_pitch_rad
    assert mechanism.depth_m == pytest.approx(
        config.mechanism.ladder.tip_depth_m(config.mechanism.ladder.dig_pitch_rad)
    )


def test_travel_ready_requires_all_three(config):
    mechanism = Mechanism(config.mechanism)
    assert mechanism.travel_ready
    mechanism.belt_velocity_radps = 1.0
    assert not mechanism.travel_ready
    mechanism.belt_velocity_radps = 0.0
    mechanism.tub_gate = 1.0
    assert not mechanism.travel_ready
    mechanism.tub_gate = 0.0
    mechanism.pitch_rad = 0.0
    assert not mechanism.travel_ready
    mechanism.pitch_rad = mechanism.stowed_pitch_rad
    assert mechanism.travel_ready


def test_travel_ready_tolerance(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.belt_velocity_radps = 0.5 * BELT_TOL_RADPS
    mechanism.tub_gate = 0.01
    assert mechanism.travel_ready


def test_depth_interlock_trips_and_stows(config):
    mechanism = Mechanism(config.mechanism)
    # Reach past the interlock by stepping the pitch beyond it directly.  The
    # guard sits just under the deepest the ladder can reach, so the extreme
    # pitch is what trips it.
    mechanism.pitch_rad = config.mechanism.ladder.max_pitch_rad
    mechanism.step(0.01)
    assert mechanism.fault is not None
    assert "interlock" in mechanism.fault
    assert mechanism.state == STATE_FAULT
    assert mechanism.travel_ready is False
    # The fault asked it to stow, so it will raise itself out.
    _step(mechanism, 5.0)
    assert mechanism.pitch_rad == pytest.approx(config.mechanism.ladder.stowed_pitch_rad)


def test_fault_is_sticky_until_cleared(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.raise_fault("test fault")
    assert mechanism.fault == "test fault"
    mechanism.raise_fault("second fault")
    assert mechanism.fault == "test fault"  # the first one is the cause
    mechanism.clear_fault()
    assert mechanism.fault is None


def test_hold_freezes_the_target(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.command(0.0, 3.0, 1.0)
    mechanism.step(0.4)
    mechanism.hold()
    assert mechanism.target.ladder_pitch_rad == pytest.approx(mechanism.pitch_rad)
    assert mechanism.target.belt_velocity_radps == pytest.approx(mechanism.belt_velocity_radps)
    assert mechanism.target.tub_gate == pytest.approx(mechanism.tub_gate)


def test_stow_commands_the_travel_attitude(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.command(mechanism.dig_pitch_rad, 4.0, 1.0)
    _step(mechanism, 3.0)
    mechanism.stow()
    _step(mechanism, 6.0)
    assert mechanism.travel_ready


def test_no_motion_without_a_positive_timestep(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.command(mechanism.dig_pitch_rad, 0.0, 0.0)
    before = mechanism.pitch_rad
    mechanism.step(0.0)
    mechanism.step(-1.0)
    assert mechanism.pitch_rad == before


def test_snapshot_carries_everything_the_message_needs(config):
    mechanism = Mechanism(config.mechanism)
    snapshot = mechanism.snapshot()
    assert snapshot.state == STATE_STOWED
    assert snapshot.state_name == "STOWED"
    assert snapshot.travel_ready is True
    assert snapshot.ladder_pitch_rad == pytest.approx(mechanism.pitch_rad)
    assert snapshot.ladder_depth_m == 0.0
    assert snapshot.belt_velocity_radps == 0.0
    assert snapshot.belt_linear_mps == 0.0
    assert snapshot.tub_gate == 0.0


def test_state_names_cover_the_message_constants():
    assert STATE_NAMES[0] == "STOWED"
    assert len(STATE_NAMES) == 6
    assert state_name(2) == "DIGGING"
    assert state_name(99).startswith("UNKNOWN")


def test_dig_pitch_is_deeper_than_the_sand_surface(config):
    mechanism = Mechanism(config.mechanism)
    assert mechanism.dig_pitch_rad > config.mechanism.ladder.min_dig_pitch_rad
    assert mechanism.stowed_pitch_rad < config.mechanism.ladder.min_dig_pitch_rad


def test_command_is_immutable():
    command = MechanismCommand(0.1, 2.0, 0.5)
    with pytest.raises(Exception):
        command.ladder_pitch_rad = 0.2  # type: ignore[misc]
    assert command.as_tuple() == (0.1, 2.0, 0.5)


def test_deploying_state_is_set_by_the_caller(config):
    mechanism = Mechanism(config.mechanism)
    mechanism.state = STATE_DEPLOYING
    assert mechanism.snapshot().state_name == "DEPLOYING"


def test_interlock_never_trips_from_the_dig_pitch(config_file, tmp_path):
    # Even with the shallowest legal dig pitch, the tip stays inside the limit.
    patched = config_with(config_file, tmp_path, {})
    mechanism = Mechanism(patched.mechanism)
    mechanism.command(patched.mechanism.ladder.dig_pitch_rad, 0.0, 0.0)
    _step(mechanism, 10.0)
    assert mechanism.fault is None


def test_time_to_pitch_is_zero_when_there(config):
    mechanism = Mechanism(config.mechanism)
    assert mechanism.time_to_pitch(mechanism.stowed_pitch_rad) == pytest.approx(0.0)
    assert mechanism.time_to_pitch(mechanism.stowed_pitch_rad) == 0.0


def test_loading_with_a_different_stowed_pitch(config_file, tmp_path):
    patched = config_with(config_file, tmp_path, {"mechanism.ladder.stowed_pitch_rad": -0.50})
    mechanism = Mechanism(patched.mechanism, pitch_rad=-0.10)
    assert mechanism.pitch_rad == pytest.approx(-0.10)
    assert not mechanism.travel_ready
    assert default_config_path().endswith("excavation.yaml")
