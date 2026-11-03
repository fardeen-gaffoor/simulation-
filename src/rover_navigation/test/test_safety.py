"""The velocity gate: the single writer of ``/cmd_vel``.

Every reason the rover can be stopped lives here, in one order, and the order is
part of the contract:

1. **e-stop** - the hardware chain has the last word;
2. **staleness** - Gazebo's diff-drive holds the last velocity *forever*, so
   silence from the navigator must mean stop, never "keep going at the last
   speed you heard";
3. **localisation** - the map frame is only meaningful while the tags anchor it,
   so the gate scales with ``sigma`` and stops before the map is badly wrong;
4. **the reactive brake** - a boulder perception has just found inside the
   bumper distance, without waiting for a re-plan.

The tests are grouped the same way, and each one asserts the *reason* as well as
the zero command: a stop with no explanation is indistinguishable from a bug.
"""

from __future__ import annotations

import math

import pytest

from rover_navigation.config import NavigationConfigError
from rover_navigation.geometry import Pose3
from rover_navigation.safety import GateDecision, LocalizationSnapshot, VelocityGate

from support import boulder, config_with, pose

_SHIPPED = None  # the `config_file` fixture supplies the path


def _gate(config) -> VelocityGate:
    return VelocityGate(config)


def _good_snapshot(**overrides) -> LocalizationSnapshot:
    values = dict(has_state=True, localized=True, sigma_xy_m=0.03, sigma_yaw_rad=0.02, time_since_fix_s=0.1)
    values.update(overrides)
    return LocalizationSnapshot(**values)


def _decide(config, gate, **overrides):
    """A gate tick with sane defaults, overridden per test."""
    arguments = dict(
        command=(0.3, 0.0),
        now=10.0,
        last_command_time=10.0,
        estop=False,
        snapshot=_good_snapshot(),
        pose=pose(4.5, 2.5, 0.0),
        records=[],
        dt=0.05,
    )
    arguments.update(overrides)
    return gate.decide(**arguments)


# =========================================================================== #
# LocalizationSnapshot
# =========================================================================== #
def test_an_absent_snapshot_says_so():
    assert "no localization state" in LocalizationSnapshot().summary


def test_a_snapshot_summarises_itself():
    summary = _good_snapshot().summary
    assert "localized=True" in summary
    assert "0.030" in summary


# =========================================================================== #
# the order of the checks: e-stop first
# =========================================================================== #
def test_an_estop_stops_the_rover(config):
    decision = _decide(config, _gate(config), estop=True)
    assert decision.stopped
    assert decision.estop is True
    assert decision.allowed is False
    assert "e-stop" in decision.reason


def test_the_estop_wins_over_every_other_problem(config):
    """Even with a stale, unlocalised command and a boulder ahead, it says e-stop."""
    decision = _decide(
        config,
        _gate(config),
        estop=True,
        last_command_time=0.0,
        snapshot=_good_snapshot(localized=False),
        records=[boulder(5.0, 2.5, 0.2)],
    )
    assert "e-stop" in decision.reason


def test_an_estop_clears_when_it_is_released(config):
    gate = _gate(config)
    assert _decide(config, gate, estop=True).stopped
    decision = _decide(config, gate, estop=False)
    assert decision.allowed is True
    assert decision.linear > 0.0


# =========================================================================== #
# staleness
# =========================================================================== #
def test_no_command_yet_holds_with_a_stale_reason(config):
    decision = _decide(config, _gate(config), command=None, last_command_time=None)
    assert decision.stopped
    assert decision.stale is True
    assert "no command" in decision.reason


def test_a_command_older_than_the_timeout_is_refused(config):
    decision = _decide(
        config, _gate(config), now=10.0, last_command_time=10.0 - config.safety.command_timeout_s - 0.01
    )
    assert decision.stopped
    assert decision.stale is True
    assert "no fresh command" in decision.reason


def test_a_command_exactly_at_the_timeout_is_still_allowed(config):
    decision = _decide(
        config, _gate(config), now=10.0, last_command_time=10.0 - config.safety.command_timeout_s
    )
    assert decision.allowed is True


def test_a_fresh_command_is_forwarded(config):
    decision = _decide(config, _gate(config), command=(0.3, 0.0), now=10.0, last_command_time=9.9)
    assert decision.allowed is True
    assert decision.stale is False


# =========================================================================== #
# localisation gating
# =========================================================================== #
def test_no_localization_state_at_all_holds_when_a_fix_is_required(config):
    decision = _decide(config, _gate(config), snapshot=LocalizationSnapshot())
    assert decision.stopped
    assert "localization" in decision.reason


def test_an_unlocalised_map_holds(config):
    decision = _decide(config, _gate(config), snapshot=_good_snapshot(localized=False))
    assert decision.stopped
    assert "tag fix" in decision.reason


def test_a_stale_fix_holds(config):
    decision = _decide(
        config,
        _gate(config),
        snapshot=_good_snapshot(time_since_fix_s=config.safety.max_time_since_fix_s + 1.0),
    )
    assert decision.stopped
    assert "no tag fix" in decision.reason


def test_a_good_fix_runs_at_full_speed(config):
    """No scale is applied: the command arrives at the speed that was asked for."""
    gate = _gate(config)
    for _ in range(50):                     # let the acceleration ramp settle
        decision = _decide(config, gate, command=(0.3, 0.0))
    assert decision.speed_scale == pytest.approx(1.0)
    assert decision.linear == pytest.approx(0.3, abs=1e-6)
    assert decision.reason == ""


def test_speed_scales_down_as_xy_uncertainty_grows(config):
    mid = 0.5 * (config.safety.sigma_xy_slow_m + config.safety.sigma_xy_stop_m)
    decision = _decide(config, _gate(config), snapshot=_good_snapshot(sigma_xy_m=mid))
    assert 0.0 < decision.speed_scale < 1.0
    assert decision.allowed is True
    assert decision.linear < 0.3
    assert "scaled" in decision.reason


def test_speed_stops_at_the_xy_stop_threshold(config):
    decision = _decide(
        config, _gate(config), snapshot=_good_snapshot(sigma_xy_m=config.safety.sigma_xy_stop_m)
    )
    assert decision.stopped
    assert "uncertainty" in decision.reason


def test_yaw_uncertainty_slows_the_rover_too(config):
    mid = 0.5 * (config.safety.sigma_yaw_slow_rad + config.safety.sigma_yaw_stop_rad)
    decision = _decide(config, _gate(config), snapshot=_good_snapshot(sigma_yaw_rad=mid))
    assert 0.0 < decision.speed_scale < 1.0


def test_the_worse_of_the_two_uncertainties_decides(config):
    """A tight yaw must not rescue a loose xy: the minimum of the two is taken."""
    decision = _decide(
        config,
        _gate(config),
        snapshot=_good_snapshot(sigma_yaw_rad=0.0, sigma_xy_m=config.safety.sigma_xy_stop_m),
    )
    assert decision.stopped


def test_the_ramp_is_one_below_slow_zero_at_stop_and_linear_between(config):
    gate = _gate(config)
    slow = config.safety.sigma_xy_slow_m
    stop = config.safety.sigma_xy_stop_m
    assert gate.localization_scale(_good_snapshot(sigma_xy_m=slow))[0] == pytest.approx(1.0)
    assert gate.localization_scale(_good_snapshot(sigma_xy_m=stop))[0] == pytest.approx(0.0)
    middle = gate.localization_scale(_good_snapshot(sigma_xy_m=0.5 * (slow + stop)))[0]
    assert middle == pytest.approx(0.5)


def test_scaling_applies_to_the_yaw_rate_too(config):
    mid = 0.5 * (config.safety.sigma_xy_slow_m + config.safety.sigma_xy_stop_m)
    gate = _gate(config)
    for _ in range(200):
        decision = _decide(config, gate, command=(0.0, 0.5), snapshot=_good_snapshot(sigma_xy_m=mid))
    assert decision.allowed
    assert decision.speed_scale == pytest.approx(0.5)
    assert decision.angular == pytest.approx(0.5 * decision.speed_scale)


def test_the_gate_reports_the_limits_in_force(config):
    decision = _decide(config, _gate(config))
    assert decision.max_speed == pytest.approx(config.robot.max_speed_mps * decision.speed_scale)
    assert decision.max_yaw_rate == pytest.approx(config.robot.max_yaw_rate_radps * decision.speed_scale)


def test_requiring_a_localization_fix_can_be_turned_off(config_file, tmp_path):
    cfg = config_with(config_file, tmp_path, {"safety.require_localized": False})
    decision = _decide(cfg, VelocityGate(cfg), snapshot=LocalizationSnapshot())
    assert decision.allowed is True
    assert decision.reason == ""


# =========================================================================== #
# the reactive brake
# =========================================================================== #
def test_an_obstacle_inside_the_bumper_distance_stops_the_rover(config):
    # The nose is 0.65 m ahead of the centre; the boulder centre 0.75 m ahead
    # leaves 0.75 - 0.65 - 0.15 = -0.05 m of clearance.
    decision = _decide(config, _gate(config), records=[boulder(4.5 + 0.75, 2.5, 0.15)])
    assert decision.stopped
    assert decision.emergency is True
    assert "ahead" in decision.reason


def test_an_obstacle_beyond_the_bumper_distance_does_not_stop_it(config):
    limit = config.safety.emergency_stop_m
    # Clearance a comfortable margin above the limit.
    record = boulder(4.5 + 0.65 + limit + 0.15 + 0.10, 2.5, 0.15)
    decision = _decide(config, _gate(config), records=[record])
    assert decision.allowed is True
    assert decision.emergency is False


def test_the_brake_uses_the_obstacle_radius_not_the_centre(config):
    """A big boulder is dangerous from further away, exactly like a real one."""
    limit = config.safety.emergency_stop_m
    centre_distance = 0.65 + limit + 0.40
    # radius 0.35 -> clearance = 0.40 > limit: allowed...
    allowed = _decide(config, _gate(config), records=[boulder(4.5 + centre_distance, 2.5, 0.35)])
    assert allowed.allowed is True
    # ...the same centre with a 0.55 m radius is not.
    stopped = _decide(config, _gate(config), records=[boulder(4.5 + centre_distance, 2.5, 0.55)])
    assert stopped.stopped


def test_an_obstacle_beside_the_rover_does_not_brake_a_forward_command(config):
    """The test is made at the bumper, so a flank obstacle must not stall a turn."""
    record = boulder(4.5, 2.5 + 0.35, 0.15)
    decision = _decide(config, _gate(config), command=(0.3, 0.0), records=[record])
    assert decision.allowed is True


def test_the_brake_only_looks_forward(config):
    """A zero-linear command (a pure turn) is never braked by the bumper test."""
    record = boulder(4.5 + 0.65, 2.5, 0.15)     # solidly inside the bumper
    decision = _decide(config, _gate(config), command=(0.0, 0.4), records=[record])
    assert decision.allowed is True


def test_the_brake_reason_names_the_obstacle_kind(config):
    decision = _decide(config, _gate(config), records=[boulder(4.5 + 0.7, 2.5, 0.15)])
    assert "boulder" in decision.reason


def test_the_brake_needs_a_pose(config):
    """With no map pose there is nothing to measure the clearance against."""
    decision = _decide(config, _gate(config), pose=None, records=[boulder(4.5 + 0.7, 2.5, 0.15)])
    assert decision.allowed is True


# =========================================================================== #
# clamping and rate limiting
# =========================================================================== #
def test_a_command_above_the_speed_limit_is_clamped(config):
    gate = _gate(config)
    # Ramp up over several ticks so the rate limit is not what limits the test.
    for _ in range(100):
        decision = _decide(config, gate, command=(99.0, 0.0))
    assert decision.linear == pytest.approx(config.robot.max_speed_mps)


def test_a_command_above_the_yaw_limit_is_clamped(config):
    gate = _gate(config)
    for _ in range(200):
        decision = _decide(config, gate, command=(0.0, 99.0))
    assert decision.angular == pytest.approx(config.robot.max_yaw_rate_radps)


def test_the_first_tick_is_rate_limited_from_rest(config):
    """The wheels never see a step: the first tick ramps from zero."""
    decision = _decide(config, _gate(config), command=(0.45, 1.2), dt=0.05)
    assert decision.linear <= config.robot.max_accel_mps2 * 0.05 + 1e-9
    assert abs(decision.angular) <= config.robot.max_yaw_accel_radps2 * 0.05 + 1e-9


def test_the_speed_ramps_up_over_successive_ticks(config):
    gate = _gate(config)
    speeds = []
    for _ in range(10):
        speeds.append(_decide(config, gate, command=(0.45, 0.0), dt=0.05).linear)
    assert speeds == sorted(speeds)
    assert speeds[-1] > speeds[0]


def test_the_ramp_is_limited_by_the_configured_acceleration(config):
    gate = _gate(config)
    first = _decide(config, gate, command=(0.45, 0.0), dt=0.05)
    second = _decide(config, gate, command=(0.45, 0.0), dt=0.05)
    assert first.linear == pytest.approx(config.robot.max_accel_mps2 * 0.05)
    assert second.linear - first.linear <= config.robot.max_accel_mps2 * 0.05 + 1e-9
    assert second.linear > first.linear
    assert second.linear < 0.45


def test_a_zero_dt_skips_the_rate_limit(config):
    decision = _decide(config, _gate(config), command=(0.45, 0.0), dt=0.0)
    assert decision.linear == pytest.approx(0.45)


def test_a_negative_command_is_clamped_to_the_negative_limit(config):
    gate = _gate(config)
    for _ in range(200):
        decision = _decide(config, gate, command=(-99.0, 0.0))
    assert decision.linear == pytest.approx(-config.robot.max_speed_mps)


def test_reset_forgets_the_last_command(config):
    gate = _gate(config)
    for _ in range(20):
        _decide(config, gate, command=(0.45, 0.0))
    gate.reset()
    decision = _decide(config, gate, command=(0.45, 0.0), dt=0.05)
    assert decision.linear <= config.robot.max_accel_mps2 * 0.05 + 1e-9


def test_a_hold_snaps_to_zero_and_restarts_the_ramp(config):
    gate = _gate(config)
    for _ in range(20):
        _decide(config, gate, command=(0.45, 0.0))
    assert _decide(config, gate, estop=True).stopped
    decision = _decide(config, gate, command=(0.45, 0.0), dt=0.05)
    assert decision.linear <= config.robot.max_accel_mps2 * 0.05 + 1e-9


# =========================================================================== #
# GateDecision
# =========================================================================== #
def test_a_stopped_decision_knows_it(config):
    assert GateDecision(allowed=False, linear=0.0, angular=0.0).stopped is True
    assert GateDecision(allowed=True, linear=1e-6, angular=0.0).stopped is False


def test_the_decision_reports_whether_it_was_allowed(config):
    allowed = _decide(config, _gate(config))
    assert allowed.allowed is True
    held = _decide(config, _gate(config), estop=True)
    assert held.allowed is False


def test_the_gate_is_stateless_apart_from_the_last_command(config):
    """Two gates fed the same tick produce the same decision."""
    first = _decide(config, _gate(config))
    second = _decide(config, _gate(config))
    assert (first.linear, first.angular, first.allowed) == (second.linear, second.angular, second.allowed)
