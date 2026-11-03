"""The wire contract in ``rover_excavation_msgs``.

The messages are a *contract*, not a convenience: module 5 calls the
``WorkCycle`` action and anyone watching the run reads ``MechanismStatus`` and
``BermStatus``.  A field renamed here and not in the publisher is a runtime no-op
that no amount of library testing catches, so the constants are pinned against
the ones the module actually uses and the builders are checked for completeness.

The action itself lives in ``rover_navigation_msgs`` because module 5 is the
caller and had to exist first; its ``STAGE_*`` constants are pinned against
:mod:`rover_excavation.cycle` here, from the server's side.
"""

from __future__ import annotations

import pytest

from rover_excavation import nodeutil
from rover_excavation.cycle import (
    STAGE_DONE,
    STAGE_MOVING,
    STAGE_NAMES,
    STAGE_RETRACTING,
    STAGE_STARTING,
    STAGE_WORKING,
)
from rover_excavation.mechanism import (
    STATE_DEPLOYING,
    STATE_DIGGING,
    STATE_DUMPING,
    STATE_FAULT,
    STATE_RETRACTING,
    STATE_STOWED,
)

from rover_excavation_msgs.msg import BermStatus, MechanismStatus
from rover_navigation_msgs.action import WorkCycle


def _fields(message_or_action) -> dict:
    return dict(message_or_action.get_fields_and_field_types())


def _names(message_or_action) -> set:
    return set(_fields(message_or_action))


class _Node:
    """The two things the message builders need from a node: a clock."""

    class _Now:
        def __init__(self, nanoseconds: int):
            self.nanoseconds = nanoseconds

    class _Clock:
        def __init__(self, seconds: float):
            self._seconds = float(seconds)

        def now(self):
            return _Node._Now(int(self._seconds * 1e9))

    def __init__(self, seconds: float = 0.0):
        self._clock = _Node._Clock(seconds)

    def get_clock(self):
        return self._clock


# =========================================================================== #
# MechanismStatus
# =========================================================================== #
def test_the_mechanism_states_are_the_numbers_the_model_uses():
    assert MechanismStatus.STATE_STOWED == STATE_STOWED == 0
    assert MechanismStatus.STATE_DEPLOYING == STATE_DEPLOYING == 1
    assert MechanismStatus.STATE_DIGGING == STATE_DIGGING == 2
    assert MechanismStatus.STATE_DUMPING == STATE_DUMPING == 3
    assert MechanismStatus.STATE_RETRACTING == STATE_RETRACTING == 4
    assert MechanismStatus.STATE_FAULT == STATE_FAULT == 5


def test_the_mechanism_states_are_distinct_and_start_at_zero():
    codes = [
        MechanismStatus.STATE_STOWED,
        MechanismStatus.STATE_DEPLOYING,
        MechanismStatus.STATE_DIGGING,
        MechanismStatus.STATE_DUMPING,
        MechanismStatus.STATE_RETRACTING,
        MechanismStatus.STATE_FAULT,
    ]
    assert sorted(codes) == list(range(len(codes)))


def test_mechanism_status_carries_the_fields_the_node_fills():
    expected = {
        "header",
        "state",
        "state_name",
        "travel_ready",
        "ladder_pitch_rad",
        "ladder_depth_m",
        "belt_velocity_radps",
        "belt_linear_mps",
        "tub_gate",
        "tub_fill_m3",
        "tub_capacity_m3",
        "tub_fraction",
        "message",
    }
    assert _names(MechanismStatus) == expected


def test_mechanism_status_is_constructible_and_has_the_documented_types():
    message = MechanismStatus()
    message.travel_ready = True
    message.state_name = "STOWED"
    assert message.travel_ready is True
    assert message.state_name == "STOWED"
    assert _fields(MechanismStatus)["travel_ready"] == "boolean"
    assert _fields(MechanismStatus)["ladder_pitch_rad"] == "float"
    assert _fields(MechanismStatus)["header"] == "std_msgs/Header"


# =========================================================================== #
# BermStatus
# =========================================================================== #
def test_berm_status_carries_the_fields_the_node_fills():
    expected = {
        "header",
        "target_volume_m3",
        "placed_volume_m3",
        "spilled_volume_m3",
        "completion",
        "mean_height_m",
        "peak_height_m",
        "coverage",
        "cells_total",
        "cells_filled",
        "target_height_m",
        "dumps",
        "message",
    }
    assert _names(BermStatus) == expected


def test_the_counts_are_unsigned_integers_for_a_reason():
    """A negative cell count means the message was built from nonsense."""
    assert _fields(BermStatus)["cells_total"] == "uint32"
    assert _fields(BermStatus)["cells_filled"] == "uint32"
    assert _fields(BermStatus)["dumps"] == "uint32"


# =========================================================================== #
# the builders
# =========================================================================== #
def test_the_mechanism_builder_fills_every_field(config):
    from rover_excavation.mechanism import Mechanism
    from rover_excavation.sand import Tub

    mechanism = Mechanism(config.mechanism)
    tub = Tub(config.mechanism.tub)
    tub.fill()
    message = nodeutil.mechanism_status_msg(_Node(12.5), config, mechanism.snapshot(), tub.volume_m3, tub.fraction)
    assert message.header.frame_id == config.frames.base
    assert message.header.stamp.sec == 12
    assert message.state == STATE_STOWED
    assert message.state_name == "STOWED"
    assert message.travel_ready is True
    assert message.ladder_pitch_rad == pytest.approx(config.mechanism.ladder.stowed_pitch_rad)
    assert message.ladder_depth_m == 0.0
    assert message.tub_fill_m3 == pytest.approx(config.mechanism.tub.capacity_m3)
    assert message.tub_capacity_m3 == pytest.approx(config.mechanism.tub.capacity_m3)
    assert message.tub_fraction == pytest.approx(1.0)
    assert message.message


def test_the_berm_builder_fills_every_field(config):
    from rover_excavation.sand import BermField

    berm = BermField(config)
    berm.begin_dump()
    berm.deposit(*config.berm_target.center, 0.02)
    message = nodeutil.berm_status_msg(_Node(), config, berm)
    assert message.header.frame_id == config.frames.map
    assert message.target_volume_m3 == pytest.approx(berm.target_volume_m3)
    assert message.placed_volume_m3 > 0.0
    assert message.completion > 0.0
    assert message.cells_total == berm.cells_total
    assert message.dumps == 1
    assert message.target_height_m == pytest.approx(config.sand.berm.target_height_m)
    assert message.message


def test_the_marker_builder_makes_one_cube_per_cell(config):
    from visualization_msgs.msg import Marker

    from rover_excavation.sand import BermField

    berm = BermField(config)
    array = nodeutil.berm_marker_array(_Node(), config, berm)
    assert len(array.markers) == berm.cells_total
    assert {marker.type for marker in array.markers} == {Marker.CUBE}
    assert {marker.ns for marker in array.markers} == {"berm"}
    assert [marker.id for marker in array.markers] == list(range(berm.cells_total))
    assert all(marker.action == Marker.ADD for marker in array.markers)
    assert all(marker.header.frame_id == config.frames.map for marker in array.markers)


def test_a_tall_cell_is_more_opaque_than_a_bare_one(config):
    from rover_excavation.sand import BermField

    berm = BermField(config)
    bare = nodeutil.berm_marker_array(_Node(), config, berm).markers
    berm.deposit(*config.berm_target.center, 0.1)
    tall = nodeutil.berm_marker_array(_Node(), config, berm).markers
    assert max(m.color.a for m in tall) > max(m.color.a for m in bare)
    assert max(m.pose.position.z for m in tall) > 0.0


# =========================================================================== #
# the action module 6 serves
# =========================================================================== #
def test_the_action_stages_are_the_ones_the_cycle_uses():
    assert WorkCycle.Feedback.STAGE_STARTING == STAGE_STARTING
    assert WorkCycle.Feedback.STAGE_MOVING == STAGE_MOVING
    assert WorkCycle.Feedback.STAGE_WORKING == STAGE_WORKING
    assert WorkCycle.Feedback.STAGE_RETRACTING == STAGE_RETRACTING
    assert WorkCycle.Feedback.STAGE_DONE == STAGE_DONE
    assert len(STAGE_NAMES) == 5


def test_the_action_goal_is_what_the_node_reads():
    assert _names(WorkCycle.Goal) == {"phase", "work_pose", "timeout_s", "target_amount"}
    assert _fields(WorkCycle.Goal)["work_pose"] == "geometry_msgs/PoseStamped"


def test_the_action_result_is_what_the_node_writes():
    assert _names(WorkCycle.Result) == {"success", "message", "amount_moved", "elapsed_s"}


def test_the_action_feedback_is_what_the_node_publishes():
    assert _names(WorkCycle.Feedback) == {"stage", "progress", "detail"}
