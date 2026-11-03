"""One work cycle: the stage sequence, the timing, and every way it can fail."""

from __future__ import annotations

import pytest

from rover_excavation.cycle import (
    STAGE_DONE,
    STAGE_MOVING,
    STAGE_NAMES,
    STAGE_RETRACTING,
    STAGE_STARTING,
    STAGE_WORKING,
    WorkCycle,
)
from rover_excavation.mechanism import STATE_DIGGING, STATE_DUMPING

from support import DIG_POSE, dump_pose, play, rig


def _step(mechanism, seconds, dt=0.01):
    for _ in range(int(round(seconds / dt))):
        mechanism.step(dt)


# --------------------------------------------------------------------------- #
# digging
# --------------------------------------------------------------------------- #
def test_dig_fills_the_tub_and_stows(config):
    mechanism, tub, berm = rig(config)
    cycle, _ = play(config, "excavate", DIG_POSE, mechanism, tub, berm)
    assert cycle.success
    assert cycle.stage == STAGE_DONE
    assert cycle.stage_name == "DONE"
    assert tub.fraction >= config.cycle.dig.target_fill
    assert cycle.amount_moved == pytest.approx(tub.volume_m3)
    assert mechanism.travel_ready
    assert berm.dumps == 0  # a dig deposits nothing


def test_dig_visits_every_stage(config):
    mechanism, tub, berm = rig(config)
    cycle = WorkCycle(config, "excavate", DIG_POSE, mechanism, tub, berm, now=0.0)
    seen = {cycle.stage}
    now = 0.0
    while not cycle.done:
        mechanism.step(0.01)
        cycle.step(now)
        seen.add(cycle.stage)
        now += 0.01
    assert seen == {STAGE_STARTING, STAGE_MOVING, STAGE_WORKING, STAGE_RETRACTING, STAGE_DONE}


def test_dig_waits_for_the_ladder_before_running_the_belt(config):
    mechanism, tub, berm = rig(config)
    cycle = WorkCycle(config, "excavate", DIG_POSE, mechanism, tub, berm)
    now = 0.0
    saw_digging = False
    while not cycle.done:
        mechanism.step(0.01)
        cycle.step(now)
        if cycle.stage == STAGE_WORKING:
            saw_digging = True
            assert mechanism.at_pitch(config.mechanism.ladder.dig_pitch_rad)
        now += 0.01
    assert saw_digging
    assert cycle.success


def test_dig_reports_digging_state(config):
    mechanism, tub, berm = rig(config)
    cycle = WorkCycle(config, "excavate", DIG_POSE, mechanism, tub, berm)
    now = 0.0
    states = set()
    while not cycle.done:
        mechanism.step(0.01)
        cycle.step(now)
        states.add(mechanism.state)
        now += 0.01
    assert STATE_DIGGING in states


def test_dig_honours_min_s(config_file, tmp_path):
    from support import config_with

    patched = config_with(config_file, tmp_path, {"cycle.dig.min_s": 0.0, "mechanism.tub.capacity_m3": 0.0346})
    mechanism, tub, berm = rig(patched)
    tub.fill()  # already full: the dig has nothing to do but must still run
    cycle, _ = play(patched, "excavate", DIG_POSE, mechanism, tub, berm)
    assert cycle.success
    assert cycle.amount_moved == pytest.approx(0.0, abs=1e-9)


def test_dig_takes_the_expected_time(config):
    mechanism, tub, berm = rig(config)
    cycle, _ = play(config, "excavate", DIG_POSE, mechanism, tub, berm)
    # Two ladder slews plus the tub filling at the modelled rate.
    expected = (
        2.0 * abs(config.mechanism.ladder.dig_pitch_rad - config.mechanism.ladder.stowed_pitch_rad)
        / config.mechanism.ladder.max_rate_radps
        + config.mechanism.tub.capacity_m3
        * config.cycle.dig.target_fill
        / (config.sand.dig.coefficient
           * config.mechanism.ladder.engaged_width_m
           * config.mechanism.ladder.tip_depth_m(config.mechanism.ladder.dig_pitch_rad)
           * config.mechanism.belt.surface_speed_mps(config.mechanism.belt.dig_velocity_radps))
    )
    assert cycle.elapsed_s == pytest.approx(expected, rel=0.15)


def test_dig_stops_early_at_target_amount(config):
    mechanism, tub, berm = rig(config)
    cycle, _ = play(config, "excavate", DIG_POSE, mechanism, tub, berm, target_amount=0.01)
    assert cycle.success
    assert cycle.amount_moved == pytest.approx(0.01, rel=0.05)
    assert tub.volume_m3 == pytest.approx(0.01, rel=0.05)


# --------------------------------------------------------------------------- #
# dumping
# --------------------------------------------------------------------------- #
def test_dump_empties_the_tub_onto_the_berm(config):
    mechanism, tub, berm = rig(config)
    tub.fill()
    before = tub.volume_m3
    cycle, _ = play(config, "dump", dump_pose(config, 2), mechanism, tub, berm)
    assert cycle.success
    assert cycle.stage == STAGE_DONE
    assert tub.fraction <= config.cycle.dump.target_empty + 1e-9
    assert cycle.amount_moved == pytest.approx(before * (1.0 - config.cycle.dump.target_empty), rel=0.02)
    assert berm.dumps == 1
    assert berm.placed_volume_m3 > 0.0
    assert mechanism.travel_ready


def test_dump_opens_the_gate_before_reversing(config):
    mechanism, tub, berm = rig(config)
    tub.fill()
    cycle = WorkCycle(config, "dump", dump_pose(config, 0), mechanism, tub, berm)
    now = 0.0
    reversed_early = False
    while not cycle.done:
        mechanism.step(0.01)
        cycle.step(now)
        if cycle.stage == STAGE_MOVING and mechanism.belt_velocity_radps < -0.1:
            reversed_early = True
        now += 0.01
    assert not reversed_early
    assert cycle.success


def test_dump_reports_dumping_state(config):
    mechanism, tub, berm = rig(config)
    tub.fill()
    cycle = WorkCycle(config, "dump", dump_pose(config, 0), mechanism, tub, berm)
    now = 0.0
    states = set()
    while not cycle.done:
        mechanism.step(0.01)
        cycle.step(now)
        states.add(mechanism.state)
        now += 0.01
    assert STATE_DUMPING in states


def test_dump_of_an_empty_tub_succeeds_and_moves_nothing(config):
    mechanism, tub, berm = rig(config)
    cycle, _ = play(config, "dump", dump_pose(config, 0), mechanism, tub, berm)
    assert cycle.success
    assert cycle.amount_moved == 0.0
    assert berm.dumps == 0
    assert "empty" in cycle.message


def test_dump_lands_inside_the_target_for_every_pose(config):
    mechanism, tub, berm = rig(config)
    for index in range(config.nav_dump.points):
        tub.fill()
        cycle, _ = play(config, "dump", dump_pose(config, index), mechanism, tub, berm)
        assert cycle.success
    assert berm.dumps == config.nav_dump.points
    # Most of what left the tub landed in the target, not around it.
    assert berm.placed_volume_m3 > 0.7 * config.berm_target_volume_m3


# --------------------------------------------------------------------------- #
# a whole dig/dump pair
# --------------------------------------------------------------------------- #
def test_six_cycles_fill_the_berm(config):
    mechanism, tub, berm = rig(config)
    now = 0.0
    for index in range(6):
        dig, now = play(config, "excavate", DIG_POSE, mechanism, tub, berm, now=now, cycle_id=index)
        assert dig.success
        dump, now = play(config, "dump", dump_pose(config, index), mechanism, tub, berm, now=now, cycle_id=index)
        assert dump.success
    assert berm.dumps == 6
    assert berm.completion > 0.90
    assert mechanism.travel_ready


def test_cycles_are_independent_of_the_starting_pitch(config):
    """A cycle must cope with the ladder being anywhere legal when it starts."""
    mechanism, tub, berm = rig(config)
    # Fully retracted: the ladder has further to travel than usual.
    mechanism.pitch_rad = config.mechanism.ladder.min_pitch_rad
    cycle, _ = play(config, "excavate", DIG_POSE, mechanism, tub, berm)
    assert cycle.success
    assert tub.fraction >= config.cycle.dig.target_fill


def test_an_illegal_cut_depth_trips_the_depth_interlock(config_file, tmp_path):
    """A cut past the machine's depth guard is a fault, not a cycle that copes."""
    from support import config_with

    # The shipped guard is deeper than the ladder can reach; tighten it so the
    # dig pitch itself is past the limit.
    patched = config_with(config_file, tmp_path, {"mechanism.interlocks.max_depth_m": 0.02})
    mechanism, tub, berm = rig(patched)
    cycle, _ = play(patched, "excavate", DIG_POSE, mechanism, tub, berm)
    assert not cycle.success
    assert mechanism.fault is not None
    assert "interlock" in cycle.message


# --------------------------------------------------------------------------- #
# failures
# --------------------------------------------------------------------------- #
def test_timeout_fails_the_cycle_and_stows(config_file, tmp_path):
    from support import config_with

    # A timeout so short that the ladder cannot even deploy.
    patched = config_with(config_file, tmp_path, {"cycle.dig.timeout_s": 0.30, "cycle.dig.min_s": 0.0})
    mechanism, tub, berm = rig(patched)
    cycle, _ = play(patched, "excavate", DIG_POSE, mechanism, tub, berm)
    assert not cycle.success
    assert cycle.stage == STAGE_DONE
    assert "timed out" in cycle.message
    # The ladder is *commanded* to stow; on the node, the next idle tick is what
    # actually gets it there, and travel_ready only turns true when it arrives.
    assert mechanism.target.ladder_pitch_rad == pytest.approx(
        patched.mechanism.ladder.stowed_pitch_rad
    )
    _step(mechanism, 3.0)
    assert mechanism.travel_ready


def test_mechanism_fault_fails_the_cycle(config_file, tmp_path):
    from support import config_with

    patched = config_with(config_file, tmp_path, {"sand.dig.max_rate_m3_s": 0.012})
    mechanism, tub, berm = rig(patched)
    cycle = WorkCycle(patched, "excavate", DIG_POSE, mechanism, tub, berm)
    now = 0.0
    while not cycle.done and now < 30.0:
        mechanism.step(0.01)
        cycle.step(now)
        if cycle.stage == STAGE_WORKING:
            mechanism.raise_fault("injected fault")
        now += 0.01
    assert not cycle.success
    assert "injected fault" in cycle.message


def test_unknown_phase_is_rejected(config):
    mechanism, tub, berm = rig(config)
    with pytest.raises(ValueError, match="phase"):
        WorkCycle(config, "teleport", DIG_POSE, mechanism, tub, berm)


def test_goal_timeout_overrides_the_config(config):
    mechanism, tub, berm = rig(config)
    cycle = WorkCycle(config, "excavate", DIG_POSE, mechanism, tub, berm, timeout_s=12.5)
    assert cycle.timeout_s == pytest.approx(12.5)
    other = WorkCycle(config, "excavate", DIG_POSE, mechanism, tub, berm, timeout_s=0.0)
    assert other.timeout_s == pytest.approx(config.cycle.dig.timeout_s)


def test_timeout_longer_than_the_config_is_honoured(config):
    mechanism, tub, berm = rig(config)
    cycle = WorkCycle(config, "excavate", DIG_POSE, mechanism, tub, berm, timeout_s=999.0)
    assert cycle.timeout_s == pytest.approx(999.0)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def test_progress_is_monotone_and_ends_at_one(config):
    mechanism, tub, berm = rig(config)
    cycle = WorkCycle(config, "excavate", DIG_POSE, mechanism, tub, berm)
    previous = 0.0
    now = 0.0
    while not cycle.done:
        mechanism.step(0.01)
        cycle.step(now)
        assert cycle.progress >= previous - 1e-9
        assert 0.0 <= cycle.progress <= 1.0
        previous = cycle.progress
        now += 0.01
    assert cycle.progress == pytest.approx(1.0)


def test_detail_mentions_the_current_activity(config):
    mechanism, tub, berm = rig(config)
    cycle = WorkCycle(config, "excavate", DIG_POSE, mechanism, tub, berm)
    assert "starting" in cycle.detail
    mechanism.step(0.01)
    cycle.step(0.0)
    assert cycle.stage == STAGE_MOVING
    assert "ladder" in cycle.detail


def test_stage_names_cover_the_action_constants():
    from rover_navigation_msgs.action import WorkCycle as WorkCycleAction

    assert STAGE_NAMES == ("STARTING", "MOVING", "WORKING", "RETRACTING", "DONE")
    assert WorkCycleAction.Feedback.STAGE_STARTING == STAGE_STARTING
    assert WorkCycleAction.Feedback.STAGE_DONE == STAGE_DONE


def test_a_step_after_done_does_nothing(config):
    mechanism, tub, berm = rig(config)
    cycle, _ = play(config, "excavate", DIG_POSE, mechanism, tub, berm)
    amount = cycle.amount_moved
    cycle.step(1e6)
    assert cycle.amount_moved == amount
    assert cycle.success
