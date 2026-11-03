"""Pure pursuit with an in-place turn phase.

The controller is the only piece of module 5 that decides *how* to move, and it
has two properties the rest of the stack leans on:

* **no reverse.**  ``cam_nav`` is forward-facing with a 100 degree field of
  view, so every command it emits has ``linear >= 0``; a test that did not check
  this would let a regression drive the rover blind.
* **it turns before it drives.**  The rover is a skid-steer, so a large heading
  error is met by rotating on the spot rather than by arcing - which is both
  cheaper and what keeps the body inside the corridor the planner certified.

Speed and yaw rate are clamped to the robot's limits and tapered near the goal,
and ``force_drive`` is the documented escape hatch for the one case where an
in-place turn is impossible: the caller has already asked :mod:`..footprint`
whether there is room to spin.
"""

from __future__ import annotations

import math

import pytest

from rover_navigation.controller import ControlCommand, PurePursuitController
from rover_navigation.geometry import Pose3

from support import pose


def _controller(config) -> PurePursuitController:
    return PurePursuitController(config)


def _straight_path(x0: float, x1: float, y: float, count: int = 40):
    return [(x0 + (x1 - x0) * i / (count - 1), y) for i in range(count)]


# =========================================================================== #
# ControlCommand
# =========================================================================== #
def test_a_stop_command_is_not_moving():
    command = ControlCommand(0.0, 0.0, "idle", 0.0, 0.0, 0.0, (0.0, 0.0), False)
    assert command.moving is False


def test_a_driving_command_is_moving():
    command = ControlCommand(0.2, 0.0, "driving", 1.0, 0.0, 0.0, (1.0, 0.0), False)
    assert command.moving is True


def test_a_turn_in_place_counts_as_moving():
    command = ControlCommand(0.0, 0.5, "turning", 1.0, 0.0, 0.5, (1.0, 0.0), False)
    assert command.moving is True


def test_the_controller_stop_helper_is_all_zeros(config):
    command = _controller(config).stop()
    assert (command.linear, command.angular) == (0.0, 0.0)
    assert command.state == "idle"
    assert command.goal_reached is False


# =========================================================================== #
# arrival
# =========================================================================== #
def test_arriving_on_the_goal_with_the_right_heading_reports_arrival(config):
    controller = _controller(config)
    path = _straight_path(1.0, 3.0, 2.0)
    command = controller.compute(pose(2.999, 2.0, 0.0), path, goal_yaw=0.0)
    assert command.goal_reached is True
    assert command.state == "arrived"
    assert command.linear == 0.0
    assert command.angular == 0.0


def test_arriving_with_the_wrong_heading_turns_before_stopping(config):
    controller = _controller(config)
    path = _straight_path(1.0, 3.0, 2.0)
    command = controller.compute(pose(3.0, 2.0, 0.0), path, goal_yaw=math.pi / 2)
    assert command.goal_reached is False
    assert command.state == "turning"
    assert command.linear == 0.0
    assert command.angular > 0.0            # +pi/2 is to the left
    assert command.yaw_error_rad == pytest.approx(math.pi / 2)


def test_the_final_turn_goes_the_short_way(config):
    controller = _controller(config)
    path = _straight_path(1.0, 3.0, 2.0)
    command = controller.compute(pose(3.0, 2.0, 0.0), path, goal_yaw=-math.pi / 2)
    assert command.angular < 0.0


def test_the_final_turn_is_eased_as_it_closes(config):
    controller = _controller(config)
    path = _straight_path(1.0, 3.0, 2.0)
    close = controller.compute(pose(3.0, 2.0, 0.0), path, goal_yaw=math.radians(6.0))
    far = controller.compute(pose(3.0, 2.0, 0.0), path, goal_yaw=math.radians(60.0))
    assert abs(close.angular) < abs(far.angular)


def test_force_drive_accepts_the_goal_heading_as_it_stands(config):
    """The escape hatch: the caller has already proved there is no room to spin."""
    controller = _controller(config)
    path = _straight_path(1.0, 3.0, 2.0)
    command = controller.compute(pose(3.0, 2.0, 0.0), path, goal_yaw=math.pi, force_drive=True)
    assert command.goal_reached is True


def test_a_custom_position_tolerance_widens_arrival(config):
    controller = _controller(config)
    path = _straight_path(1.0, 3.0, 2.0)
    strict = controller.compute(pose(3.0 - 2 * config.robot.goal_tolerance_m, 2.0, 0.0), path, 0.0)
    loose = controller.compute(
        pose(3.0 - 2 * config.robot.goal_tolerance_m, 2.0, 0.0), path, 0.0, position_tolerance_m=0.5
    )
    assert strict.goal_reached is False
    assert loose.goal_reached is True


def test_the_distance_to_the_goal_is_reported(config):
    command = _controller(config).compute(pose(1.0, 2.0, 0.0), _straight_path(1.0, 4.0, 2.0), 0.0)
    assert command.distance_remaining_m == pytest.approx(3.0)


# =========================================================================== #
# turning before driving
# =========================================================================== #
def test_a_big_heading_error_turns_on_the_spot(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    # Pointing away from the path by far more than in_place_turn_rad.
    command = controller.compute(pose(1.0, 2.0, math.pi), path, 0.0)
    assert command.state == "turning"
    assert command.linear == 0.0
    assert abs(command.angular) > 0.0


def test_a_small_heading_error_drives_and_steers(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    command = controller.compute(pose(1.0, 2.0, math.radians(10.0)), path, 0.0)
    assert command.state == "driving"
    assert command.linear > 0.0
    # Pointing above the path (+10 deg) needs a right-hand correction.
    assert command.angular < 0.0


def test_the_turn_flips_sign_with_the_error(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    left = controller.compute(pose(1.0, 2.0, -math.pi / 2), path, 0.0)
    right = controller.compute(pose(1.0, 2.0, math.pi / 2), path, 0.0)
    assert left.angular > 0.0
    assert right.angular < 0.0


def test_turning_in_place_never_translates(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    for heading in (math.pi, -math.pi / 2, math.pi / 2, 2.5):
        command = controller.compute(pose(1.0, 2.0, heading), path, 0.0)
        if command.state == "turning":
            assert command.linear == 0.0


# =========================================================================== #
# no reverse
# =========================================================================== #
def test_the_controller_never_asks_for_reverse(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    for heading in [index * math.pi / 6 for index in range(12)]:
        command = controller.compute(pose(1.0, 2.0, heading), path, 0.0)
        assert command.linear >= 0.0, heading


def test_a_path_behind_the_rover_is_turned_into_not_reversed_into(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    # Facing the path backwards, 50 cm from its start.
    command = controller.compute(pose(1.0, 2.0, math.pi), path, 0.0)
    assert command.linear == 0.0
    assert command.state == "turning"


# =========================================================================== #
# limits and shaping
# =========================================================================== #
def test_speed_is_clamped_to_the_robot_limit(config):
    controller = _controller(config)
    command = controller.compute(
        pose(1.0, 2.0, 0.0), _straight_path(1.0, 20.0, 2.0), 0.0, max_speed=10.0
    )
    assert command.linear <= config.robot.max_speed_mps + 1e-9


def test_the_max_speed_argument_can_lower_the_ceiling(config):
    controller = _controller(config)
    slow = controller.compute(pose(1.0, 2.0, 0.0), _straight_path(1.0, 8.0, 2.0), 0.0, max_speed=0.1)
    fast = controller.compute(pose(1.0, 2.0, 0.0), _straight_path(1.0, 8.0, 2.0), 0.0)
    assert slow.linear <= 0.1 + 1e-9
    assert fast.linear > slow.linear


def test_yaw_rate_is_clamped_to_the_robot_limit(config):
    controller = _controller(config)
    for heading in (math.radians(1.0), math.radians(25.0), math.radians(-25.0)):
        command = controller.compute(pose(2.0, 2.0, heading), _straight_path(1.0, 4.0, 2.0), 0.0)
        assert abs(command.angular) <= config.robot.max_yaw_rate_radps + 1e-9, heading


def test_speed_never_falls_below_the_minimum_while_driving(config):
    controller = _controller(config)
    command = controller.compute(pose(1.0, 2.0, math.radians(20.0)), _straight_path(1.0, 4.0, 2.0), 0.0)
    if command.state == "driving":
        assert command.linear >= config.controller.min_speed_mps - 1e-9


def test_speed_tapers_over_the_last_stretch(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    far = controller.compute(pose(1.2, 2.0, 0.0), path, 0.0)
    near = controller.compute(pose(4.0 - 0.4 * config.controller.slowdown_distance_m, 2.0, 0.0), path, 0.0)
    assert far.state == "driving" and near.state == "driving"
    assert near.linear < far.linear


def test_a_crooked_approach_near_the_goal_turns_instead_of_arcing(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    # Close to the goal, still carrying a large heading error.
    command = controller.compute(
        pose(3.9, 2.0, math.radians(40.0)), path, 0.0, position_tolerance_m=0.05
    )
    assert command.state == "turning"
    assert command.linear == 0.0


def test_a_larger_heading_error_means_less_speed(config):
    controller = _controller(config)
    path = _straight_path(1.0, 8.0, 2.0)
    straight = controller.compute(pose(1.0, 2.0, 0.0), path, 0.0)
    crooked = controller.compute(pose(1.0, 2.0, math.radians(25.0)), path, 0.0)
    assert crooked.linear < straight.linear


# =========================================================================== #
# the lookahead point
# =========================================================================== #
def test_the_lookahead_point_is_ahead_on_the_path(config):
    controller = _controller(config)
    path = _straight_path(0.5, 5.0, 2.0, count=100)
    command = controller.compute(pose(1.0, 2.0, 0.0), path, 0.0)
    assert command.lookahead[0] > 1.0
    assert command.lookahead[1] == pytest.approx(2.0)
    assert command.lookahead[0] - 1.0 <= config.controller.max_lookahead_m + 1e-9


def test_the_lookahead_never_reaches_past_the_goal(config):
    controller = _controller(config)
    path = _straight_path(1.0, 1.4, 2.0, count=20)
    command = controller.compute(pose(1.35, 2.0, 0.0), path, 0.0, position_tolerance_m=0.01)
    assert command.lookahead[0] <= 1.4 + 1e-9


def test_the_lookahead_is_at_least_the_minimum(config):
    controller = _controller(config)
    path = _straight_path(1.0, 6.0, 2.0, count=100)
    command = controller.compute(pose(1.0, 2.0, 0.0), path, 0.0, max_speed=0.01)
    assert command.lookahead[0] - 1.0 >= config.controller.min_lookahead_m - 1e-9


def test_a_path_behind_the_pose_still_yields_a_lookahead(config):
    """The path is the authority: the projection snaps onto it, not off it."""
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0, count=40)
    command = controller.compute(pose(2.5, 2.4, math.radians(45.0)), path, 0.0)
    assert command.lookahead[0] >= 2.5


def test_force_drive_keeps_the_lookahead_tight(config):
    controller = _controller(config)
    path = _straight_path(1.0, 6.0, 2.0, count=200)
    command = controller.compute(pose(1.0, 2.0, math.radians(80.0)), path, 0.0, force_drive=True)
    assert command.state == "driving"
    assert command.lookahead[0] - 1.0 <= config.controller.min_lookahead_m + 1e-9


def test_force_drive_translates_despite_a_large_heading_error(config):
    controller = _controller(config)
    path = _straight_path(1.0, 6.0, 2.0, count=200)
    turned = controller.compute(pose(1.0, 2.0, math.pi), path, 0.0)
    forced = controller.compute(pose(1.0, 2.0, math.pi), path, 0.0, force_drive=True)
    assert turned.linear == 0.0
    assert forced.linear > 0.0


# =========================================================================== #
# degenerate inputs
# =========================================================================== #
def test_an_empty_path_is_the_pose_itself(config):
    controller = _controller(config)
    command = controller.compute(pose(3.0, 2.0, 0.0), [], goal_yaw=0.0)
    assert command.goal_reached is True


def test_a_single_point_path_arrives_at_it(config):
    controller = _controller(config)
    command = controller.compute(pose(3.0, 2.0, 0.0), [(3.01, 2.0)], 0.0)
    assert command.goal_reached is True


def test_a_path_with_a_repeated_point_does_not_divide_by_zero(config):
    controller = _controller(config)
    command = controller.compute(pose(1.0, 2.0, 0.0), [(1.0, 2.0), (1.0, 2.0), (4.0, 2.0)], 0.0)
    assert command.state in ("driving", "turning", "arrived")
    assert math.isfinite(command.linear) and math.isfinite(command.angular)


def test_compute_is_stateless(config):
    controller = _controller(config)
    path = _straight_path(1.0, 4.0, 2.0)
    first = controller.compute(pose(1.0, 2.0, math.radians(20.0)), path, 0.0)
    second = controller.compute(pose(1.0, 2.0, math.radians(20.0)), path, 0.0)
    assert (first.linear, first.angular, first.state) == (second.linear, second.angular, second.state)


# =========================================================================== #
# the projection helpers, used directly because they carry the geometry
# =========================================================================== #
def test_project_arc_measures_along_the_polyline(config):
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    assert PurePursuitController._project_arc(path, 1.0, 0.0) == pytest.approx(1.0)
    assert PurePursuitController._project_arc(path, 2.0, 1.0) == pytest.approx(3.0)


def test_project_arc_clamps_before_the_start_and_after_the_end(config):
    path = [(1.0, 1.0), (3.0, 1.0)]
    assert PurePursuitController._project_arc(path, -5.0, 1.0) == pytest.approx(0.0)
    assert PurePursuitController._project_arc(path, 99.0, 1.0) == pytest.approx(2.0)


def test_point_at_walks_the_polyline(config):
    path = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0)]
    assert PurePursuitController._point_at(path, 1.0) == pytest.approx((1.0, 0.0))
    assert PurePursuitController._point_at(path, 3.0) == pytest.approx((2.0, 1.0))
    assert PurePursuitController._point_at(path, 99.0) == pytest.approx((2.0, 2.0))
    assert PurePursuitController._point_at(path, -1.0) == pytest.approx((0.0, 0.0))


def test_point_at_handles_a_single_point_path(config):
    assert PurePursuitController._point_at([(1.0, 2.0)], 5.0) == (1.0, 2.0)
    assert PurePursuitController._point_at([], 5.0) == (0.0, 0.0)


def test_project_reports_the_arc_still_to_run(config):
    path = [(0.0, 0.0), (4.0, 0.0)]
    remaining, closest = PurePursuitController._project(path, 1.0, 0.0)
    assert remaining == pytest.approx(3.0)
    assert closest == pytest.approx((1.0, 0.0))


def test_shape_turn_eases_and_otherwise_passes_through(config):
    assert PurePursuitController._shape_turn(0.0, 1.0) == pytest.approx(0.0)
    assert PurePursuitController._shape_turn(0.125, 1.0) == pytest.approx(0.5)
    assert PurePursuitController._shape_turn(0.25, 1.0) == pytest.approx(1.0)
    assert PurePursuitController._shape_turn(1.5, 1.0) == pytest.approx(1.0)
    # Monotone in the error; the sign follows the yaw rate it is given.
    assert abs(PurePursuitController._shape_turn(0.2, 1.0)) > abs(PurePursuitController._shape_turn(0.1, 1.0))
    assert PurePursuitController._shape_turn(-0.125, -1.0) == pytest.approx(-0.5)
