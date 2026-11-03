"""The behaviour sequencer: the order of the run, and the camera phases.

The sequencer is ROS-free on purpose, so the whole competition run - acquire,
dig, haul, dump, repeat, park - is exercised here by driving the state machine
directly, including the paths that only happen when something fails.

Two things are checked beyond "it reaches the end":

* **the generated poses are legal.**  The dig and dump poses come out of the
  arena geometry, so each one is run through the same exact footprint test the
  navigator uses before it will accept a goal.  A pose the rover cannot legally
  occupy is a run-ending bug that no amount of controller tuning fixes.
* **``navigate_only`` cannot be mistaken for a real run.**  It never counts a
  cycle it did not move sand for, and both the phase detail and the mode say so.
"""

from __future__ import annotations

import math

import pytest

from rover_navigation.config import PHASE_NAMES, with_work_mode
from rover_navigation.footprint import footprint_block
from rover_navigation.geometry import Pose3, wrap_angle
from rover_navigation.mission import (
    MAX_CONSECUTIVE_FAILURES,
    PERCEPTION_MODES,
    PHASE_ABORTED,
    PHASE_ACQUIRE,
    PHASE_COMPLETE,
    PHASE_DUMP,
    PHASE_EXCAVATE,
    PHASE_IDLE,
    PHASE_PARK,
    PHASE_RETURN,
    PHASE_TRAVEL_TO_DIG,
    PHASE_TRAVEL_TO_DUMP,
    MissionCommand,
    MissionSequencer,
    Waypoint,
    dig_poses,
    dump_poses,
)

from support import config_with, pose

_START = Pose3.from_xy_yaw(1.0, 3.8, math.pi)


def _tick(seq: MissionSequencer, now: float, **overrides) -> MissionCommand:
    arguments = dict(pose=_START, localized=True, work_available=True)
    arguments.update(overrides)
    return seq.tick(now, **arguments)


def _started(config, now: float = 0.0) -> MissionSequencer:
    seq = MissionSequencer(config)
    seq.start(now)
    return seq


def _at(seq: MissionSequencer, now: float, **overrides) -> MissionCommand:
    """Tick, then report whatever was requested as successful.

    Returns the command that the tick produced, after the request has been
    acknowledged.
    """
    command = _tick(seq, now, **overrides)
    if command.goal is not None:
        seq.nav_finished(now, True, "arrived")
    elif command.work is not None:
        seq.work_finished(now, True, "done", 0.01)
    return command


# =========================================================================== #
# the generated poses
# =========================================================================== #
def test_dig_poses_sweep_the_strip_in_two_passes(config):
    poses = dig_poses(config)
    assert len(poses) == 2 * config.mission.dig.points_per_pass
    xs = sorted({round(waypoint.x, 6) for waypoint in poses})
    assert len(xs) == 2
    # Both passes are inside the excavation strip, inset from its x edges.
    zone = config.arena.zone("excavation")
    for x in xs:
        assert zone.x0 < x < zone.x1


def test_dig_poses_alternate_heading_between_passes(config):
    poses = dig_poses(config)
    per_pass = config.mission.dig.points_per_pass
    assert poses[0].yaw == pytest.approx(config.mission.dig.heading_rad)
    assert poses[per_pass].yaw == pytest.approx(-config.mission.dig.heading_rad)


def test_the_second_pass_runs_the_other_way(config):
    poses = dig_poses(config)
    per_pass = config.mission.dig.points_per_pass
    first = [waypoint.y for waypoint in poses[:per_pass]]
    second = [waypoint.y for waypoint in poses[per_pass:]]
    assert first == sorted(first)
    assert second == sorted(second, reverse=True)


def test_dig_poses_stay_inside_the_free_rectangle(config):
    free = config.free_rect
    for waypoint in dig_poses(config):
        assert free.contains(waypoint.x, waypoint.y), waypoint


def test_dig_poses_are_labelled_and_unique(config):
    poses = dig_poses(config)
    labels = [waypoint.label for waypoint in poses]
    assert len(set(labels)) == len(labels)
    assert labels[0] == "dig_p0_s0"


def test_every_dig_pose_is_drivable(config):
    for waypoint in dig_poses(config):
        reason = footprint_block(config, Pose3.from_xy_yaw(waypoint.x, waypoint.y, waypoint.yaw))
        assert reason is None, f"{waypoint.label}: {reason}"


def test_dig_poses_are_empty_when_the_strip_is_inset_away(tmp_path, config_file):
    cfg = config_with(config_file, tmp_path, {"mission.dig.y_inset_m": 20.0})
    assert dig_poses(cfg) == []


def test_dump_poses_lie_along_the_berm_target(config):
    poses = dump_poses(config)
    target = config.arena.zone("berm_target")
    assert len(poses) == config.mission.dump.points
    inset = config.mission.dump.inset_m
    for waypoint in poses:
        assert target.x0 + inset - 1e-9 <= waypoint.x <= target.x1 - inset + 1e-9
        assert waypoint.yaw == pytest.approx(config.mission.dump.heading_rad)


def test_dump_poses_stand_off_by_the_swept_circle(config):
    poses = dump_poses(config)
    target = config.arena.zone("berm_target")
    expected = target.y1 + config.robot.circumscribed_radius_m + config.mission.dump.standoff_m
    for waypoint in poses:
        assert waypoint.y == pytest.approx(expected)


def test_the_whole_body_clears_the_berm_target_at_a_dump_pose(config):
    """The navigator keeps the target as a keep-out once the berm exists."""
    target = config.arena.zone("berm_target")
    for waypoint in dump_poses(config):
        reason = footprint_block(
            config, Pose3.from_xy_yaw(waypoint.x, waypoint.y, waypoint.yaw), [], [target]
        )
        assert reason is None, f"{waypoint.label}: {reason}"


def test_dump_poses_are_inside_the_arena(config):
    for waypoint in dump_poses(config):
        reason = footprint_block(config, Pose3.from_xy_yaw(waypoint.x, waypoint.y, waypoint.yaw))
        assert reason is None, f"{waypoint.label}: {reason}"


def test_dump_poses_are_ordered_across_the_target(config):
    xs = [waypoint.x for waypoint in dump_poses(config)]
    assert xs == sorted(xs)
    assert len(set(round(x, 6) for x in xs)) == len(xs)


def test_a_single_dump_point_lands_in_the_middle_of_the_target(config_file, tmp_path):
    cfg = config_with(config_file, tmp_path, {"mission.dump.points": 1})
    poses = dump_poses(cfg)
    target = cfg.arena.zone("berm_target")
    assert len(poses) == 1
    assert poses[0].x == pytest.approx(0.5 * (target.x0 + target.x1))


def test_a_waypoint_carries_its_label():
    waypoint = Waypoint(1.0, 2.0, 0.5, "somewhere")
    assert (waypoint.x, waypoint.y, waypoint.yaw) == (1.0, 2.0, 0.5)
    assert waypoint.label == "somewhere"
    assert Waypoint(1.0, 2.0, 0.0).label == ""


# =========================================================================== #
# lifecycle
# =========================================================================== #
def test_a_fresh_sequencer_is_not_running(config):
    command = _tick(MissionSequencer(config), 0.0)
    assert command.phase == PHASE_IDLE
    assert command.running is False
    assert command.detail == "not started"
    assert command.finished is False


def test_start_enters_acquire(config):
    seq = _started(config)
    assert seq.phase == PHASE_ACQUIRE
    assert seq.running is True
    assert "tag" in seq.detail


def test_a_disabled_mission_does_not_run(tmp_path, config_file):
    cfg = config_with(config_file, tmp_path, {"mission.enabled": False})
    seq = _started(cfg)
    assert seq.phase == PHASE_IDLE
    assert seq.running is False
    assert "disabled" in seq.detail


def test_the_terminal_property_covers_only_the_end_states(config):
    seq = MissionSequencer(config)
    assert seq.terminal is False
    seq.phase = PHASE_COMPLETE
    assert seq.terminal is True
    seq.phase = PHASE_ABORTED
    assert seq.terminal is True


def test_the_status_reports_the_cycle_target(config):
    seq = _started(config)
    command = _tick(seq, 0.0)
    assert command.cycles_target == config.mission.cycles_target
    assert command.cycles_completed == 0


# =========================================================================== #
# acquire
# =========================================================================== #
def test_acquire_waits_for_odometry_before_doing_anything(config):
    command = _tick(_started(config), 0.0, pose=None, localized=False)
    assert command.goal is None
    assert "odometry" in command.detail


def test_acquire_drives_to_the_park_pose_first(config):
    """A rover in a corner of the start zone needs somewhere legal to spin."""
    command = _tick(_started(config), 0.0, localized=False)
    assert command.goal is not None
    assert "staging" in command.goal.label
    assert command.goal.x == pytest.approx(config.mission.park.x)
    assert command.goal.y == pytest.approx(config.mission.park.y)


def test_acquire_spins_on_the_spot_after_staging(config):
    seq = _started(config)
    first = _tick(seq, 0.0, localized=False)
    seq.nav_finished(0.0, True, "arrived")
    second = _tick(seq, 1.0, localized=False)
    assert second.goal is not None
    assert "search turn" in second.goal.label
    assert second.goal.x == pytest.approx(_START.x)
    assert second.goal.y == pytest.approx(_START.y)
    assert second.goal.yaw == pytest.approx(wrap_angle(_START.yaw + config.mission.acquire_spin_rad))


def test_acquire_stages_even_if_the_staging_pose_is_unreachable(config):
    seq = _started(config)
    _tick(seq, 0.0, localized=False)
    seq.nav_finished(0.0, False, "no path")
    command = _tick(seq, 1.0, localized=False)
    assert "search turn" in command.goal.label


def test_a_localised_map_leaves_acquire_immediately(config):
    command = _tick(_started(config), 0.0, localized=True)
    assert command.phase == PHASE_TRAVEL_TO_DIG
    assert command.goal is not None
    assert command.goal.label.startswith("dig_")


def test_acquire_gives_up_when_no_tag_is_found_in_time(config):
    seq = _started(config)
    command = _tick(seq, config.mission.acquire_timeout_s + 1.0, localized=False)
    assert command.finished is True
    assert seq.phase == PHASE_ABORTED
    assert "tag fix" in seq.detail


def test_acquire_gives_up_after_repeated_spin_failures(config):
    """The message must point at the arena, not at the route."""
    seq = _started(config)
    now = 0.0
    for _ in range(MAX_CONSECUTIVE_FAILURES + 2):
        now += 0.1
        command = _tick(seq, now, localized=False)
        if command.goal is not None:
            seq.nav_finished(now, False, "blocked")
    assert seq.phase == PHASE_ABORTED
    assert "turn" in seq.detail
    assert "tag" in seq.detail


# =========================================================================== #
# travel to dig
# =========================================================================== #
def test_travel_to_dig_requests_the_first_dig_pose(config):
    """A tag fix in ACQUIRE hands straight over to the first dig leg."""
    seq = _started(config)
    command = _tick(seq, 0.0, localized=True)
    assert command.phase == PHASE_TRAVEL_TO_DIG
    assert command.goal is not None
    assert command.goal.label == dig_poses(config)[0].label


def test_arriving_at_a_dig_pose_enters_excavate(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)
    assert seq.phase == PHASE_EXCAVATE


def test_a_failed_dig_leg_moves_to_the_next_pose(config):
    seq = _started(config)
    _tick(seq, 0.0, localized=True)
    _tick(seq, 0.1, localized=True)
    seq.nav_finished(0.1, False, "no path")
    command = _tick(seq, 0.2, localized=True)
    assert command.goal.label == dig_poses(config)[1].label


def test_the_run_gives_up_after_enough_consecutive_navigation_failures(config):
    seq = _started(config)
    now = 0.0
    for _ in range(MAX_CONSECUTIVE_FAILURES + 2):
        now += 0.1
        command = _tick(seq, now, localized=True)
        if command.goal is not None:
            seq.nav_finished(now, False, "no path")
    assert seq.phase == PHASE_ABORTED
    assert "giving up" in seq.detail


def test_a_success_resets_the_failure_counter(config):
    seq = _started(config)
    _tick(seq, 0.0, localized=True)
    _tick(seq, 0.1, localized=True)
    seq.nav_finished(0.1, False, "no path")
    assert seq._consecutive_failures == 1
    _tick(seq, 0.2, localized=True)
    seq.nav_finished(0.2, True, "arrived")
    assert seq._consecutive_failures == 0


def test_travel_to_dig_fails_when_no_pose_exists(tmp_path, config_file):
    cfg = config_with(config_file, tmp_path, {"mission.dig.y_inset_m": 20.0})
    seq = _started(cfg)
    command = _tick(seq, 0.0, localized=True)
    assert command.finished is True
    assert seq.phase == PHASE_ABORTED
    assert "excavation pose" in seq.detail


# =========================================================================== #
# excavate
# =========================================================================== #
def test_excavate_in_delegate_mode_asks_module_6(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)                 # -> EXCAVATE
    command = _tick(seq, 0.1, work_available=True)
    assert command.work is not None
    phase, x, y, yaw = command.work
    assert phase == "excavate"
    waypoint = dig_poses(config)[0]
    assert (x, y, yaw) == pytest.approx((waypoint.x, waypoint.y, waypoint.yaw))


def test_excavate_in_delegate_mode_waits_for_the_server(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)
    command = _tick(seq, 0.1, work_available=False)
    assert command.work is None
    assert "waiting" in command.detail
    assert "WorkCycle" in command.detail or "excavation" in command.detail


def test_excavate_aborts_when_the_server_never_appears(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)
    command = _tick(seq, 0.1 + config.mission.excavation_wait_s + 1.0, work_available=False)
    assert command.finished is True
    assert seq.phase == PHASE_ABORTED
    assert "WorkCycle" in seq.detail


def test_a_completed_dig_cycle_hauls_to_the_berm(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)
    _tick(seq, 0.1, work_available=True)
    seq.work_finished(0.1, True, "dug", 0.02)
    assert seq.phase == PHASE_TRAVEL_TO_DUMP
    assert seq.dig_index == 1


def test_a_failed_dig_cycle_is_retried(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)
    _tick(seq, 0.1, work_available=True)
    seq.work_finished(0.1, False, "ladder jammed", 0.0)
    assert seq.phase == PHASE_EXCAVATE
    command = _tick(seq, 0.2, work_available=True)
    assert command.work is not None
    assert "ladder jammed" not in seq.detail


def test_excavate_says_which_mode_it_is_in(config_file, tmp_path):
    """Every status detail carries the mode, so a dry run cannot look real."""
    cfg = with_work_mode(config_with(config_file, tmp_path, {}), "navigate_only")
    seq = _started(cfg)
    first = _tick(seq, 0.0, localized=True)
    assert "navigate_only" in first.detail
    seq.nav_finished(0.0, True, "arrived")       # the dig leg: -> EXCAVATE
    command = _tick(seq, 0.1)
    assert command.phase == PHASE_TRAVEL_TO_DUMP
    assert "navigate_only" in command.detail


# =========================================================================== #
# dump
# =========================================================================== #
def test_dump_in_delegate_mode_asks_module_6(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)                 # -> EXCAVATE
    _at(seq, 0.1)                                 # -> TRAVEL_TO_DUMP
    command = _tick(seq, 0.2)
    assert command.goal.label.startswith("dump_")
    seq.nav_finished(0.2, True, "arrived")
    assert seq.phase == PHASE_DUMP
    command = _tick(seq, 0.3, work_available=True)
    assert command.work is not None
    phase, x, y, yaw = command.work
    assert phase == "dump"
    assert (x, y, yaw) == pytest.approx(
        (dump_poses(config)[0].x, dump_poses(config)[0].y, dump_poses(config)[0].yaw)
    )


def test_a_completed_dump_counts_a_cycle(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)
    _at(seq, 0.1)
    _at(seq, 0.2)
    assert seq.phase == PHASE_DUMP
    _tick(seq, 0.3, work_available=True)
    seq.work_finished(0.3, True, "dumped", 0.02)
    assert seq.cycles_completed == 1
    assert seq.phase == PHASE_TRAVEL_TO_DIG


def test_the_last_cycle_returns_to_park(config):
    seq = _started(config)
    seq.cycles_completed = config.mission.cycles_target - 1
    _at(seq, 0.0, localized=True)
    _at(seq, 0.1)
    _at(seq, 0.2)
    _tick(seq, 0.3, work_available=True)
    seq.work_finished(0.3, True, "dumped", 0.02)
    assert seq.cycles_completed == config.mission.cycles_target
    assert seq.phase == PHASE_RETURN


def test_a_failed_dump_is_retried(config):
    seq = _started(config)
    _at(seq, 0.0, localized=True)
    _at(seq, 0.1)
    _at(seq, 0.2)
    _tick(seq, 0.3, work_available=True)
    seq.work_finished(0.3, False, "tub empty", 0.0)
    assert seq.cycles_completed == 0
    assert seq.phase == PHASE_DUMP
    command = _tick(seq, 0.4, work_available=True)
    assert command.work is not None


# =========================================================================== #
# navigate_only
# =========================================================================== #
def test_navigate_only_never_claims_a_cycle_before_the_dump(config_file, tmp_path):
    cfg = with_work_mode(config_with(config_file, tmp_path, {}), "navigate_only")
    seq = _started(cfg)
    command = _at(seq, 0.0, localized=True)      # dig leg, reported as arrived
    assert command.work is None
    assert seq.cycles_completed == 0
    assert seq.phase == PHASE_EXCAVATE


def test_navigate_only_counts_a_cycle_at_the_dump(config_file, tmp_path):
    cfg = with_work_mode(config_with(config_file, tmp_path, {}), "navigate_only")
    seq = _started(cfg)
    _at(seq, 0.0, localized=True)                # travel to dig
    _at(seq, 0.1)                                # travel to dump
    _at(seq, 0.2)                                # the dump itself
    assert seq.cycles_completed == 1
    assert seq.phase == PHASE_EXCAVATE           # ... and the next cycle is under way


# =========================================================================== #
# return and park
# =========================================================================== #
def test_return_drives_to_the_park_pose(config):
    seq = _started(config)
    seq._enter(PHASE_RETURN, 0.0)
    command = _tick(seq, 0.1)
    assert command.goal.label == "park"
    assert (command.goal.x, command.goal.y) == pytest.approx((config.mission.park.x, config.mission.park.y))


def test_arriving_at_park_enters_the_park_phase(config):
    seq = _started(config)
    seq._enter(PHASE_RETURN, 0.0)
    _tick(seq, 0.1)
    seq.nav_finished(0.1, True, "arrived")
    assert seq.phase == PHASE_PARK


def test_park_takes_two_ticks_to_complete(config):
    seq = _started(config)
    seq._enter(PHASE_PARK, 0.0)
    first = _tick(seq, 0.1)
    assert first.finished is False
    assert seq.detail == "parked"
    second = _tick(seq, 0.2)
    assert second.finished is True
    assert seq.phase == PHASE_COMPLETE
    assert seq.running is False


def test_a_failed_return_leg_still_parks(config):
    """There is nowhere left to go, so trying again is the only option."""
    seq = _started(config)
    seq._enter(PHASE_RETURN, 0.0)
    _tick(seq, 0.1)
    seq.nav_finished(0.1, False, "no path")
    command = _tick(seq, 0.2)
    assert command.goal.label == "park"


# =========================================================================== #
# the whole run
# =========================================================================== #
def _drive_run(seq: MissionSequencer, *, work_available: bool, localized: bool = True):
    """Drive the sequencer to its end, as the node would, and count the ticks."""
    now = 0.0
    seq.start(now)
    pose = _START
    ticks = 0
    while not seq.terminal and ticks < 4000:
        ticks += 1
        now += 0.1
        command = seq.tick(now, pose=pose, localized=localized, work_available=work_available)
        if command.goal is not None:
            pose = Pose3.from_xy_yaw(command.goal.x, command.goal.y, command.goal.yaw)
            seq.nav_finished(now, True, "arrived")
        elif command.work is not None:
            seq.work_finished(now, True, "done", 0.02)
    return seq, ticks


def test_a_full_delegated_run_reaches_completion(config):
    seq, ticks = _drive_run(MissionSequencer(config), work_available=True)
    assert seq.phase == PHASE_COMPLETE
    assert seq.cycles_completed == config.mission.cycles_target
    assert ticks < 4000


def test_a_full_navigate_only_run_reaches_completion(config_file, tmp_path):
    cfg = with_work_mode(config_with(config_file, tmp_path, {}), "navigate_only")
    seq, _ = _drive_run(MissionSequencer(cfg), work_available=False)
    assert seq.phase == PHASE_COMPLETE
    assert seq.cycles_completed == cfg.mission.cycles_target


def test_a_run_where_every_leg_fails_aborts(config):
    seq = MissionSequencer(config)
    now = 0.0
    seq.start(now)
    pose = _START
    while not seq.terminal and now < 500.0:
        now += 0.1
        command = seq.tick(now, pose=pose, localized=True, work_available=False)
        if command.goal is not None:
            seq.nav_finished(now, False, "no path")
    assert seq.phase == PHASE_ABORTED
    assert seq.cycles_completed == 0


def test_progress_reaches_one_at_the_end(config):
    seq, _ = _drive_run(MissionSequencer(config), work_available=True)
    command = seq.tick(9999.0, pose=_START, localized=True)
    assert command.progress == pytest.approx(1.0)
    assert command.finished is True


def test_progress_is_monotone_within_a_cycle(config):
    seq = _started(config)
    seen = []
    now = 0.0
    for _ in range(4):
        now += 0.1
        command = _tick(seq, now, localized=True)
        if command.goal is not None:
            seq.nav_finished(now, True, "arrived")
        elif command.work is not None:
            seq.work_finished(now, True, "done", 0.02)
        seen.append(command.progress)
    assert seen == sorted(seen)


# =========================================================================== #
# the perception mode contract with module 4
# =========================================================================== #
def test_every_phase_has_a_perception_mode():
    """The status message's `mode` is `_PHASE_MODES[phase]`, so every phase needs one."""
    modes = {_mode_for(phase) for phase in range(len(PHASE_NAMES))}
    assert len(modes) > 1
    assert modes <= set(PERCEPTION_MODES)


def _mode_for(phase: int) -> str:
    from rover_navigation.mission import _PHASE_MODES

    return _PHASE_MODES[phase]


def test_the_mode_vocabulary_is_what_module_4_understands():
    assert set(PERCEPTION_MODES) == {"idle", "nav", "tag", "excavate", "dump", "degraded"}


def test_the_phase_names_are_unique_and_in_status_order():
    assert len(set(PHASE_NAMES)) == len(PHASE_NAMES)
    assert PHASE_NAMES[PHASE_IDLE] == "idle"
    assert PHASE_NAMES[PHASE_COMPLETE] == "complete"
    assert PHASE_NAMES[PHASE_ABORTED] == "aborted"


def test_the_phase_numbers_match_the_status_message():
    assert (PHASE_IDLE, PHASE_ACQUIRE, PHASE_TRAVEL_TO_DIG, PHASE_EXCAVATE) == (0, 1, 2, 3)
    assert (PHASE_TRAVEL_TO_DUMP, PHASE_DUMP, PHASE_RETURN, PHASE_PARK) == (4, 5, 6, 7)
    assert (PHASE_COMPLETE, PHASE_ABORTED) == (8, 9)


def test_each_phase_reports_the_mode_module_4_should_be_in(config):
    seq = _started(config)
    assert _tick(seq, 0.0, localized=False).mode == "tag"
    seq._enter(PHASE_TRAVEL_TO_DIG, 1.0)
    assert _tick(seq, 1.1).mode == "nav"
    seq._pending = "none"
    seq._enter(PHASE_EXCAVATE, 2.0)
    assert _tick(seq, 2.1, work_available=False).mode == "excavate"
    seq._pending = "none"
    seq._enter(PHASE_DUMP, 3.0)
    assert _tick(seq, 3.1, work_available=False).mode == "dump"
    seq._pending = "none"
    seq._enter(PHASE_PARK, 4.0)
    assert _tick(seq, 4.1).mode == "idle"


def test_a_terminal_run_reports_the_idle_mode(config):
    seq, _ = _drive_run(MissionSequencer(config), work_available=True)
    command = _tick(seq, 9999.0)
    assert command.mode == "idle"
    assert command.phase == PHASE_COMPLETE


# =========================================================================== #
# bookkeeping
# =========================================================================== #
def test_pending_starts_as_none_and_is_cleared_by_feedback(config):
    seq = _started(config)
    assert seq.pending == "none"
    _tick(seq, 0.0, localized=True)
    assert seq.pending == "nav"
    seq.nav_finished(0.0, True, "arrived")
    assert seq.pending == "none"


def test_feedback_for_the_wrong_action_is_ignored(config):
    seq = _started(config)
    _tick(seq, 0.0, localized=True)
    seq.work_finished(0.0, True, "done", 1.0)
    assert seq.pending == "nav"


def test_a_mission_command_defaults_to_idle():
    command = MissionCommand()
    assert command.phase == PHASE_IDLE
    assert command.phase_name == "idle"
    assert command.running is False
    assert command.goal is None
    assert command.work is None
    assert command.finished is False


def test_start_clears_previous_progress(config):
    seq = _started(config)
    seq.cycles_completed = 5
    seq.dig_index = 9
    seq.start(0.0)
    assert seq.cycles_completed == 0
    assert seq.dig_index == 0


def test_fail_is_terminal_and_says_why(config):
    seq = _started(config)
    seq.fail("because")
    assert seq.phase == PHASE_ABORTED
    assert seq.terminal is True
    assert seq.pending == "none"
    command = _tick(seq, 0.0)
    assert command.finished is True
    assert command.detail == "because"
