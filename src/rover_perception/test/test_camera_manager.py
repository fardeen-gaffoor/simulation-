"""The camera manager: which camera runs at what rate, and why.

This is the node that keeps the rover inside the 4000 kbps rule without losing
the feed it currently needs.  It is tested as a pure state machine: given a mode,
a measured rate and a thermal state, the command it produces is fully determined,
so every rung of the ladder and every thermal floor is pinned here.
"""

from __future__ import annotations

import os

import pytest

from rover_perception.camera_manager import (
    CameraCommand,
    PhaseCommand,
    PhaseController,
    THERMAL_CRITICAL,
    THERMAL_NOMINAL,
    THERMAL_THROTTLING,
    THERMAL_WARNING,
)
from rover_perception.config import load_config


@pytest.fixture(scope="module")
def cfg():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_config(os.path.join(here, "..", "config", "perception.yaml"))


@pytest.fixture()
def controller(cfg):
    return PhaseController(cfg.camera_manager, cfg.camera_names, cfg.active_camera_for_mode)


def _by_name(command: PhaseCommand) -> dict:
    return {entry.name: entry for entry in command.commands}


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def test_no_cameras_is_an_error(cfg):
    with pytest.raises(ValueError):
        PhaseController(cfg.camera_manager, [], cfg.active_camera_for_mode)


def test_the_phase_command_carries_the_limits(controller, cfg):
    command = controller.update(0.0)
    assert isinstance(command, PhaseCommand)
    assert command.limit_kbps == pytest.approx(cfg.camera_manager.bandwidth.limit_kbps)
    assert command.measured_kbps == pytest.approx(0.0)
    assert command.bandwidth_ok
    assert not command.throttled
    assert command.derate_index == 0
    assert command.derate_factor == pytest.approx(1.0)
    assert command.thermal_state == THERMAL_NOMINAL
    assert command.thermal_state_name == "nominal"


def test_an_unknown_mode_is_rejected(controller):
    with pytest.raises(ValueError):
        controller.set_mode("sprint")
    controller.set_mode("tag")
    assert controller.mode == "tag"


# --------------------------------------------------------------------------- #
# phase rates
# --------------------------------------------------------------------------- #
def test_nav_runs_the_nav_camera_fast_and_the_rest_slow(controller, cfg):
    command = controller.update(0.0)
    entries = _by_name(command)
    assert command.active_camera == "cam_nav"
    assert entries["cam_nav"].rate_hz == pytest.approx(cfg.camera_manager.active_rate_hz)
    assert entries["cam_nav"].active
    for idle in ("cam_tag", "cam_excavation", "cam_dump"):
        assert entries[idle].rate_hz == pytest.approx(cfg.camera_manager.idle_rate_hz)
        assert not entries[idle].active
        assert entries[idle].enabled


def test_each_phase_activates_its_own_camera(controller):
    for mode, camera in (("tag", "cam_tag"), ("excavate", "cam_excavation"), ("dump", "cam_dump")):
        controller.set_mode(mode)
        command = controller.update(0.0)
        assert command.active_camera == camera
        assert _by_name(command)[camera].active
        assert command.mode == mode


def test_idle_switches_every_camera_off(controller):
    controller.set_mode("idle")
    command = controller.update(0.0)
    assert command.active_camera == ""
    for entry in command.commands:
        assert entry.rate_hz == pytest.approx(0.0)
        assert not entry.enabled
        assert not entry.active
    assert command.enabled_cameras == ()


def test_degraded_streams_everything_at_the_floor_rate(controller, cfg):
    controller.set_mode("degraded")
    command = controller.update(0.0)
    assert command.active_camera == ""
    for entry in command.commands:
        assert entry.rate_hz == pytest.approx(cfg.camera_manager.min_rate_hz)
        assert entry.enabled
        assert not entry.active, "no camera is 'active' in the degraded phase"


# --------------------------------------------------------------------------- #
# the bandwidth ladder
# --------------------------------------------------------------------------- #
def test_over_the_limit_steps_the_ladder_up_once(controller, cfg):
    command = controller.update(4500.0)
    assert command.derate_index == 1
    assert command.derate_factor == pytest.approx(cfg.camera_manager.bandwidth.derate_factors[1])
    assert command.throttled
    assert not command.bandwidth_ok
    assert command.reasons, "the reason is recorded for the log"
    assert _by_name(command)["cam_nav"].rate_hz == pytest.approx(
        cfg.camera_manager.active_rate_hz * command.derate_factor
    )


def test_the_ladder_walks_all_the_way_to_off(cfg):
    import dataclasses

    manager = dataclasses.replace(cfg.camera_manager, bandwidth=dataclasses.replace(cfg.camera_manager.bandwidth, hard_limit_kbps=10.0))
    controller = PhaseController(manager, cfg.camera_names, cfg.active_camera_for_mode)
    steps = len(manager.bandwidth.derate_factors) - 1
    for _ in range(steps):
        command = controller.update(999.0)
    assert command.derate_index == steps
    assert command.derate_factor == pytest.approx(0.0)
    for entry in command.commands:
        assert not entry.enabled, "the last rung switches everything off"
    assert command.enabled_cameras == ()


def test_hysteresis_keeps_a_rate_that_is_merely_close_to_the_limit(controller, cfg):
    controller.update(4500.0)
    assert controller.derate_index == 1
    band = cfg.camera_manager.bandwidth
    # inside the hysteresis band: over the limit-minus-hysteresis, so no step down
    command = controller.update(band.limit_kbps - 100.0)
    assert command.derate_index == 1


def test_comfortably_under_the_limit_steps_back_down(controller, cfg):
    controller.update(4500.0)
    controller.update(4500.0)
    assert controller.derate_index == 2
    band = cfg.camera_manager.bandwidth
    command = controller.update(band.limit_kbps - band.hysteresis_kbps - 1.0)
    assert command.derate_index == 1, "one rung at a time, not a jump"


def test_the_ladder_never_goes_below_zero(controller):
    for _ in range(5):
        command = controller.update(0.0)
    assert command.derate_index == 0


def test_parked_at_the_floor_it_stops_repeating_itself(cfg):
    """In the sim the cameras keep publishing no matter what the phase says.

    The meter therefore stays over budget at the floor forever, so the ladder has
    to record the reason once and then keep quiet - otherwise the same line is
    logged at camera rate for the whole run.
    """
    import dataclasses

    manager = dataclasses.replace(
        cfg.camera_manager,
        bandwidth=dataclasses.replace(cfg.camera_manager.bandwidth, hard_limit_kbps=10.0),
    )
    controller = PhaseController(manager, cfg.camera_names, cfg.active_camera_for_mode)
    steps = len(manager.bandwidth.derate_factors) - 1
    for _ in range(steps):
        controller.update(999.0)
    assert controller.derate_index == steps
    command = controller.update(999.0)
    assert command.derate_index == steps
    assert command.reasons == (), "already at the floor: nothing changed, nothing to say"


def test_a_low_derate_switches_off_the_idle_cameras(controller, cfg):
    """At factor 0.2 the idle rate drops under min_rate_hz and the camera stops."""
    band = cfg.camera_manager.bandwidth
    index = band.derate_factors.index(0.2)
    for _ in range(index):
        command = controller.update(4500.0)
    assert command.derate_factor == pytest.approx(0.2)
    entries = _by_name(command)
    assert entries["cam_nav"].enabled, "the active camera is still worth streaming"
    for idle in ("cam_tag", "cam_excavation", "cam_dump"):
        assert not entries[idle].enabled
        assert entries[idle].rate_hz == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# the thermal floor
# --------------------------------------------------------------------------- #
def test_thermal_warning_does_not_derate(controller):
    command = controller.update(0.0, THERMAL_WARNING)
    assert command.derate_index == 0
    assert command.thermal_state_name == "warning"


def test_thermal_throttling_imposes_a_floor(controller, cfg):
    command = controller.update(0.0, THERMAL_THROTTLING)
    assert command.derate_index == cfg.camera_manager.thermal.throttle_steps
    assert command.throttled
    assert any("thermal" in reason for reason in command.reasons)


def test_thermal_critical_switches_everything_off(controller):
    command = controller.update(0.0, THERMAL_CRITICAL)
    assert command.derate_factor == pytest.approx(0.0)
    for entry in command.commands:
        assert not entry.enabled


def test_the_thermal_floor_can_only_raise_the_derate(controller):
    controller.update(4500.0)
    controller.update(4500.0)
    controller.update(4500.0)
    assert controller.derate_index == 3
    # hold the measured rate inside the hysteresis band so the *bandwidth* logic
    # is neutral and only the thermal term can move the index
    command = controller.update(4000.0, THERMAL_WARNING)
    assert command.derate_index == 3, "cooling down does not undo the bandwidth ladder"


# --------------------------------------------------------------------------- #
# accessors
# --------------------------------------------------------------------------- #
def test_command_for_and_enabled_cameras(controller):
    command = controller.update(0.0)
    assert isinstance(command.command_for("cam_nav"), CameraCommand)
    with pytest.raises(KeyError):
        command.command_for("cam_side")
    assert set(command.enabled_cameras) == set(_by_name(command))
    assert command.enabled_cameras == ("cam_nav", "cam_tag", "cam_excavation", "cam_dump")


def test_snapshot_is_a_reportable_dict(controller):
    controller.update(0.0)
    snapshot = controller.snapshot()
    assert snapshot["mode"] == "nav"
    assert snapshot["thermal_state"] == "nominal"
    assert snapshot["active_camera"] == "cam_nav"
    assert snapshot["derate_index"] == 0
    assert "measured_kbps" in snapshot


def test_disabled_idle_camera_reports_no_active_camera(controller):
    controller.set_mode("tag")
    # at the last rung the active camera is off, so there is no active feed
    for _ in range(5):
        command = controller.update(99999.0)
    assert command.active_camera == ""
    assert command.active_rate_hz == pytest.approx(0.0)
