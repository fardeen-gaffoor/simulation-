"""The wire contract in ``rover_navigation_msgs``.

The messages and actions are a *contract*, not a convenience: module 4 publishes
the obstacle map and the localisation state module 5 consumes, and module 6 will
implement the ``WorkCycle`` server module 5 calls.  A field renamed here and not
there is a runtime no-op that no amount of node testing catches, so the constants
are pinned against the ones the module actually uses.

``rover_navigation_msgs`` is the only place ``NavigateTo`` and ``WorkCycle`` are
defined - module 4's messages live in ``rover_perception_msgs`` and are imported,
not duplicated.
"""

from __future__ import annotations

import pytest

from rover_navigation import mission as mission_module

from rover_navigation_msgs.action import NavigateTo, WorkCycle
from rover_navigation_msgs.msg import GateStatus, MissionStatus, NavigationStatus


def _fields(message_or_action) -> set:
    """The field names of a generated message class."""
    return set(message_or_action.get_fields_and_field_types())


# =========================================================================== #
# NavigationStatus
# =========================================================================== #
def test_the_navigation_states_are_the_documented_numbers():
    assert NavigationStatus.STATE_IDLE == 0
    assert NavigationStatus.STATE_PLANNING == 1
    assert NavigationStatus.STATE_TURNING == 2
    assert NavigationStatus.STATE_DRIVING == 3
    assert NavigationStatus.STATE_ARRIVED == 4
    assert NavigationStatus.STATE_HOLDING == 5
    assert NavigationStatus.STATE_BLOCKED == 6
    assert NavigationStatus.STATE_FAILED == 7


def test_the_navigation_states_are_distinct():
    codes = [
        NavigationStatus.STATE_IDLE,
        NavigationStatus.STATE_PLANNING,
        NavigationStatus.STATE_TURNING,
        NavigationStatus.STATE_DRIVING,
        NavigationStatus.STATE_ARRIVED,
        NavigationStatus.STATE_HOLDING,
        NavigationStatus.STATE_BLOCKED,
        NavigationStatus.STATE_FAILED,
    ]
    assert sorted(codes) == list(range(len(codes)))


def test_navigation_status_carries_the_fields_the_node_fills():
    expected = {
        "header",
        "state",
        "state_name",
        "active",
        "localized",
        "map_ready",
        "goal_id",
        "goal_pose",
        "current_pose",
        "distance_remaining_m",
        "path_length_m",
        "speed_mps",
        "yaw_rate_radps",
        "sigma_xy_m",
        "sigma_yaw_rad",
        "replans",
        "obstacles_in_map",
        "blocked_count",
        "goal_rejections",
        "message",
    }
    assert expected <= _fields(NavigationStatus)


def test_every_state_name_the_node_reports_has_a_code():
    """The node maps ``state_name`` -> code; a missing key silently reports IDLE."""
    from rover_navigation.nodes.navigation_node import _STATE_CODES

    for name, code in _STATE_CODES.items():
        assert isinstance(code, int)
        assert 0 <= code <= NavigationStatus.STATE_FAILED, name
    # Every constant has a name, and the names are the controller's states.
    assert len(set(_STATE_CODES.values())) == len(_STATE_CODES)
    assert {"idle", "turning", "driving", "arrived", "blocked", "failed"} <= set(_STATE_CODES)


def test_a_navigation_status_defaults_to_idle():
    message = NavigationStatus()
    assert message.state == NavigationStatus.STATE_IDLE
    assert message.active is False
    assert message.message == ""


# =========================================================================== #
# GateStatus
# =========================================================================== #
def test_gate_status_carries_the_fields_the_node_fills():
    expected = {
        "header",
        "estop",
        "allowed",
        "stale_command",
        "emergency_brake",
        "localized",
        "speed_scale",
        "sigma_xy_m",
        "reason",
        "command",
        "max_speed_mps",
        "max_yaw_rate_radps",
    }
    assert expected <= _fields(GateStatus)


def test_a_gate_status_defaults_to_not_allowed():
    message = GateStatus()
    assert message.allowed is False
    assert message.estop is False
    assert message.speed_scale == pytest.approx(0.0)


# =========================================================================== #
# MissionStatus
# =========================================================================== #
def test_the_mission_phases_are_the_documented_numbers():
    assert MissionStatus.PHASE_IDLE == 0
    assert MissionStatus.PHASE_ACQUIRE == 1
    assert MissionStatus.PHASE_TRAVEL_TO_DIG == 2
    assert MissionStatus.PHASE_EXCAVATE == 3
    assert MissionStatus.PHASE_TRAVEL_TO_DUMP == 4
    assert MissionStatus.PHASE_DUMP == 5
    assert MissionStatus.PHASE_RETURN == 6
    assert MissionStatus.PHASE_PARK == 7
    assert MissionStatus.PHASE_COMPLETE == 8
    assert MissionStatus.PHASE_ABORTED == 9


def test_the_phase_constants_match_the_sequencer():
    """The sequencer's numbers are what the message carries; they must line up."""
    assert mission_module.PHASE_IDLE == MissionStatus.PHASE_IDLE
    assert mission_module.PHASE_ACQUIRE == MissionStatus.PHASE_ACQUIRE
    assert mission_module.PHASE_TRAVEL_TO_DIG == MissionStatus.PHASE_TRAVEL_TO_DIG
    assert mission_module.PHASE_EXCAVATE == MissionStatus.PHASE_EXCAVATE
    assert mission_module.PHASE_TRAVEL_TO_DUMP == MissionStatus.PHASE_TRAVEL_TO_DUMP
    assert mission_module.PHASE_DUMP == MissionStatus.PHASE_DUMP
    assert mission_module.PHASE_RETURN == MissionStatus.PHASE_RETURN
    assert mission_module.PHASE_PARK == MissionStatus.PHASE_PARK
    assert mission_module.PHASE_COMPLETE == MissionStatus.PHASE_COMPLETE
    assert mission_module.PHASE_ABORTED == MissionStatus.PHASE_ABORTED


def test_the_phase_names_are_the_status_phase_names():
    """``phase_name`` is ``PHASE_NAMES[phase]``; the vocabulary is shared."""
    codes = list(range(len(mission_module.PHASE_NAMES)))
    assert codes == list(range(10))
    assert mission_module.PHASE_NAMES[MissionStatus.PHASE_COMPLETE] == "complete"
    assert mission_module.PHASE_NAMES[MissionStatus.PHASE_ABORTED] == "aborted"


def test_mission_status_carries_the_fields_the_node_fills():
    expected = {
        "header",
        "phase",
        "phase_name",
        "running",
        "cycles_completed",
        "cycles_target",
        "cycle_progress",
        "detail",
        "excavation_action",
        "excavation_available",
        "perception_mode",
    }
    assert expected <= _fields(MissionStatus)


def test_a_mission_status_defaults_to_idle_and_not_running():
    message = MissionStatus()
    assert message.phase == MissionStatus.PHASE_IDLE
    assert message.running is False
    assert message.cycles_completed == 0


# =========================================================================== #
# NavigateTo
# =========================================================================== #
def test_navigate_to_goal_fields():
    assert {"goal", "position_tolerance_m", "yaw_tolerance_rad", "max_speed_mps", "timeout_s"} <= _fields(
        NavigateTo.Goal
    )


def test_navigate_to_result_fields():
    expected = {
        "success",
        "message",
        "final_distance_m",
        "final_yaw_error_rad",
        "elapsed_s",
        "path_length_m",
        "replans",
    }
    assert expected <= _fields(NavigateTo.Result)


def test_navigate_to_feedback_fields():
    assert {"current_pose", "distance_remaining_m", "speed_mps", "state"} <= _fields(NavigateTo.Feedback)


def test_a_navigate_to_goal_defaults_are_all_zero():
    """A zero means 'use the config', so the defaults are part of the contract."""
    goal = NavigateTo.Goal()
    assert goal.position_tolerance_m == pytest.approx(0.0)
    assert goal.yaw_tolerance_rad == pytest.approx(0.0)
    assert goal.max_speed_mps == pytest.approx(0.0)
    assert goal.timeout_s == pytest.approx(0.0)


def test_a_navigate_to_result_defaults_to_failure():
    result = NavigateTo.Result()
    assert result.success is False
    assert result.replans == 0


# =========================================================================== #
# WorkCycle - the module 6 delegation contract
# =========================================================================== #
def test_work_cycle_goal_fields():
    assert {"phase", "work_pose", "timeout_s", "target_amount"} <= _fields(WorkCycle.Goal)


def test_work_cycle_result_fields():
    assert {"success", "message", "amount_moved", "elapsed_s"} <= _fields(WorkCycle.Result)


def test_work_cycle_feedback_stages_are_the_documented_numbers():
    assert WorkCycle.Feedback.STAGE_STARTING == 0
    assert WorkCycle.Feedback.STAGE_MOVING == 1
    assert WorkCycle.Feedback.STAGE_WORKING == 2
    assert WorkCycle.Feedback.STAGE_RETRACTING == 3
    assert WorkCycle.Feedback.STAGE_DONE == 4


def test_work_cycle_feedback_fields():
    assert {"stage", "progress", "detail"} <= _fields(WorkCycle.Feedback)


def test_the_work_cycle_phases_are_the_perception_modes():
    """Module 5 sends ``phase = 'excavate' | 'dump'``; module 4's mode vocabulary
    is what the sequencer switches on, so the two must not drift apart."""
    assert {"excavate", "dump"} <= set(mission_module.PERCEPTION_MODES)


def test_a_work_cycle_goal_defaults_to_no_target():
    goal = WorkCycle.Goal()
    assert goal.phase == ""
    assert goal.target_amount == pytest.approx(0.0)
    assert goal.timeout_s == pytest.approx(0.0)
