"""``navigation``: obstacle memory -> costmap -> plan -> twist.

Started by ``navigation.launch.py`` as ``rover_navigation_navigator``.  This node
is the only place in the stack that turns a destination into wheel speeds, and
its output never reaches the motors directly: it publishes a *raw* command on
``/nav/cmd_vel`` which the velocity gate then vouches for (or vetoes) before it
becomes ``/cmd_vel``.  That split is deliberate - it keeps "what would be a good
thing to do" and "what is safe to do" in different nodes, so the safety rules are
not something the planner can be tempted to optimise around.

The pipeline per control tick is one clear line:

    obstacles -> ObstacleMemory -> Costmap -> Planner -> PurePursuit -> /nav/cmd_vel

with three things worth calling out:

* **Module 4's map is partial and forgetful; ours is not.**  Perception sees only
  what is in front of ``cam_nav`` and decays a track as soon as it leaves the
  field of view.  The memory here accumulates those observations, because the
  competition fixes the obstacles in place and a boulder the rover has driven
  past is still a boulder.
* **The goal has to be somewhere the rover can legally *be*, not just reach.**
  ``footprint_block`` checks the whole body rectangle at the goal heading, so a
  goal that would leave the rover's nose in a wall is refused in the goal
  callback rather than discovered by the bumper.
* **A blocked in-place turn falls back to arcing.**  ``rotation_block`` samples
  the swept rectangle; when spinning on the spot would hit something, the rover
  follows the plan with a short lookahead instead of holding forever.
"""

from __future__ import annotations

import argparse
import math
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.task import Future
from std_msgs.msg import Bool

from rover_navigation_msgs.action import NavigateTo
from rover_navigation_msgs.msg import NavigationStatus
from rover_perception_msgs.msg import LocalizationState, ObstacleArray

from .. import footprint, nodeutil
from ..config import NavigationConfig
from ..controller import ControlCommand, PurePursuitController
from ..costmap import Costmap, ObstacleMemory, ObstacleRecord
from ..geometry import Pose3
from ..planner import Planner

__all__ = ["NavigationNode", "main", "add_arguments"]

NODE_NAME = "rover_navigation_navigator"

#: State name -> the ``NavigationStatus.STATE_*`` constant.
_STATE_CODES: Dict[str, int] = {
    "idle": NavigationStatus.STATE_IDLE,
    "planning": NavigationStatus.STATE_PLANNING,
    "turning": NavigationStatus.STATE_TURNING,
    "driving": NavigationStatus.STATE_DRIVING,
    "arrived": NavigationStatus.STATE_ARRIVED,
    "holding": NavigationStatus.STATE_HOLDING,
    "blocked": NavigationStatus.STATE_BLOCKED,
    "failed": NavigationStatus.STATE_FAILED,
}


class NavigationNode(Node):
    """Obstacle map, planner, controller and the ``NavigateTo`` action server."""

    def __init__(self, config_override: str = "", publish_hz_override: float = 0.0) -> None:
        super().__init__(NODE_NAME)
        self.config: NavigationConfig = nodeutil.load_node_config(self, override=config_override)
        self.publish_hz = float(
            nodeutil.declare(
                self, "publish_hz", publish_hz_override or 10.0, "control loop and /nav/cmd_vel rate"
            )
        )
        self.costmap_hz = float(nodeutil.declare(self, "costmap_hz", 1.0, "OccupancyGrid preview rate"))
        self.publish_costmap = bool(nodeutil.declare(self, "publish_costmap", True, "publish /nav/costmap"))

        cfg = self.config
        self.memory = ObstacleMemory(cfg)
        self.planner = Planner(cfg)
        self.controller = PurePursuitController(cfg)

        self.records: List[ObstacleRecord] = []
        self.costmap: Optional[Costmap] = None
        self.tracker = nodeutil.PoseTracker(preferred_topic=cfg.topics.odometry_topics[0], node=self)
        self.localized = False
        self.sigma_xy_m = 0.0
        self.sigma_yaw_rad = 0.0
        self.time_since_fix_s = 0.0
        self.berm_started = False

        self.state = "idle"
        self.message = "waiting for a goal"
        self.command = ControlCommand(0.0, 0.0, "idle", 0.0, 0.0, 0.0, (0.0, 0.0), False)
        self.path: List[Tuple[float, float]] = []
        self.path_length_m = 0.0
        self.replans = 0
        self.blocked_count = 0
        self.goal_rejections = 0
        self.frames_seen = 0
        self._last_plan_time = 0.0
        self._last_plan_obstacles = -1

        # Active goal state.
        self._goal_handle: Any = None
        self._goal_pose: Optional[Pose3] = None
        self._goal_label = ""
        self._goal_tolerance = 0.0
        self._goal_yaw_tolerance = 0.0
        self._goal_max_speed = 0.0
        self._goal_timeout = 0.0
        self._goal_started = 0.0
        self._goal_replans = 0
        self._goal_seq = 0
        self._goal_path_length = 0.0
        self._outcome: Optional[Tuple[str, str]] = None  # (terminal state, message)
        self._finish: Optional[Future] = None

        self._group = ReentrantCallbackGroup()
        self._estop = False

        topics = cfg.topics
        self.cmd_publisher = self.create_publisher(Twist, topics.nav_cmd_vel, 10)
        self.path_publisher = self.create_publisher(Path, topics.path, 10)
        self.status_publisher = self.create_publisher(NavigationStatus, topics.nav_status, 10)
        self.costmap_publisher = (
            self.create_publisher(OccupancyGrid, topics.costmap, 1) if self.publish_costmap else None
        )

        # The obstacle map is module 4's wire contract: depending on the type is
        # the point of rover_perception_msgs existing, not a coupling to its
        # implementation.
        self.obstacle_subscription = self.create_subscription(
            ObstacleArray, topics.obstacles, self._on_obstacles, nodeutil.sensor_qos(5)
        )
        self.localization_subscription = self.create_subscription(
            LocalizationState, topics.localization_state, self._on_localization, nodeutil.sensor_qos(5)
        )
        self.odom_subscriptions = [
            self.create_subscription(
                Odometry, topic, partial(self._on_odometry, topic=topic), nodeutil.sensor_qos(20)
            )
            for topic in topics.odometry_topics
        ]
        self.berm_subscription = self.create_subscription(
            Bool, topics.berm_started, self._on_berm, nodeutil.latched_qos(1)
        )

        self.action_server = ActionServer(
            self,
            NavigateTo,
            topics.navigate_action,
            execute_callback=self._execute,
            goal_callback=self._on_goal_request,
            cancel_callback=self._on_cancel,
            callback_group=self._group,
        )

        self.timer = self.create_timer(1.0 / max(1.0, self.publish_hz), self._tick)
        self._costmap_timer = self.create_timer(1.0 / max(0.1, self.costmap_hz), self._publish_costmap)
        self.get_logger().info(
            f"{NODE_NAME} up: action {topics.navigate_action}, raw command {topics.nav_cmd_vel}, "
            f"{self.publish_hz:g} Hz, footprint {cfg.robot.length_m:g}x{cfg.robot.width_m:g} m, "
            f"inflation {cfg.robot.obstacle_inflation_m:.3f} m"
        )

    # ------------------------------------------------------------------ #
    # subscriptions
    # ------------------------------------------------------------------ #
    def _on_obstacles(self, message: Any) -> None:
        frame = message.header.frame_id or self.config.frames.map
        if frame != self.config.frames.map:
            self.get_logger().warn(
                f"obstacles are in frame '{frame}', expected '{self.config.frames.map}'; "
                "module 4 publishes the map frame, so this is a configuration mismatch",
            )
            return
        now = nodeutil.now_seconds(self)
        incoming: List[ObstacleRecord] = []
        for obstacle in message.obstacles:
            incoming.append(
                ObstacleRecord(
                    identifier=obstacle.id or "unnamed",
                    kind=int(obstacle.kind),
                    x=float(obstacle.pose.pose.position.x),
                    y=float(obstacle.pose.pose.position.y),
                    radius=float(obstacle.radius),
                    confidence=float(obstacle.confidence),
                    observations=int(obstacle.observations),
                    conservative=bool(obstacle.conservative),
                )
            )
        self.memory.update(incoming, now)
        self.records = self.memory.records
        self.frames_seen += 1

    def _on_localization(self, message: LocalizationState) -> None:
        # `map_to_odom` is the correction module 4 applies; the rover's map pose
        # is that composed with odometry, exactly as module 4's detector does it.
        self.localized = bool(message.localized)
        self.sigma_xy_m = float(message.sigma_xy_m)
        self.sigma_yaw_rad = float(message.sigma_yaw_rad)
        self.time_since_fix_s = float(message.time_since_fix_s)
        self.tracker.on_localization(message)

    def _on_odometry(self, message: Odometry, topic: str) -> None:
        self.tracker.on_odometry(message, topic)

    @property
    def pose(self) -> Optional[Pose3]:
        return self.tracker.pose

    def _on_berm(self, message: Bool) -> None:
        if bool(message.data) and not self.berm_started:
            self.get_logger().info("berm reported present: its footprint is now a keep-out for planning")
        self.berm_started = bool(message.data)

    def _keep_outs(self) -> List[Any]:
        """Everything the planner must treat as blocked that is not an obstacle."""
        if not self.berm_started:
            return []
        return [self.config.arena.zone("berm_target")]

    # ------------------------------------------------------------------ #
    # action server plumbing
    # ------------------------------------------------------------------ #
    def _on_goal_request(self, request: Any) -> Any:
        """Cheap synchronisation test: is the request even well formed?"""
        goal = request.goal
        frame = goal.header.frame_id or self.config.frames.map
        if frame != self.config.frames.map:
            self.goal_rejections += 1
            self.get_logger().warn(
                f"refusing a goal in frame '{frame}'; module 5 navigates in '{self.config.frames.map}'"
            )
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, goal_handle: Any) -> Any:
        return CancelResponse.ACCEPT

    async def _execute(self, goal_handle: Any) -> Any:
        """Validate, supersede, then hold the goal open while the timer drives it.

        Every terminal transition happens here and nowhere else.  rclpy calls this
        once per accepted goal and, *after* it returns, aborts the handle itself if
        it is still active - so a goal finalised anywhere else would be transitioned
        twice, and rclpy rejects that with "invalid transition from state ABORTED".
        Two goals can be accepted in the same control tick and their callbacks
        interleave, so a goal's identity is carried in ``sequence`` instead of being
        read back out of the node's single-goal fields, and the check after the wait
        is what stops a superseded goal from clobbering the one that replaced it.
        """
        request = goal_handle.request

        # Validation that needs the live map.  A refusal must not disturb a goal
        # that is already running, so it happens before anything is superseded.
        pose = nodeutil.pose_stamped_to_pose3(request.goal)
        blocked = footprint.footprint_block(
            self.config, pose, self.records, self._keep_outs(), clearance=0.0
        )
        if blocked is not None:
            self.goal_rejections += 1
            self.get_logger().warn(
                f"refusing goal ({pose.x:.2f}, {pose.y:.2f}, yaw {pose.yaw:.2f}): {blocked}"
            )
            goal_handle.abort()
            return self._build_result(False, f"goal refused: {blocked}", 0.0, 0.0, 0.0, 0.0)

        # One goal at a time: the newest wins.  The superseded goal's callback is
        # waiting on its own future, so resolving it lets that callback return; the
        # sequence check there tells it not to touch the state it no longer owns.
        previous, previous_finish = self._goal_handle, self._finish
        if previous is not None:
            try:
                previous.abort()
            except Exception:  # pragma: no cover - already terminal
                pass
        if previous_finish is not None and not previous_finish.done():
            previous_finish.set_result(True)
        self._goal_seq += 1
        sequence = self._goal_seq

        self._goal_handle = goal_handle
        self._goal_pose = pose
        self._goal_label = f"({pose.x:.2f}, {pose.y:.2f})"
        self._goal_tolerance = float(request.position_tolerance_m)
        self._goal_yaw_tolerance = float(request.yaw_tolerance_rad)
        self._goal_max_speed = float(request.max_speed_mps)
        self._goal_timeout = float(request.timeout_s) or self.config.controller.goal_timeout_s
        self._goal_started = nodeutil.now_seconds(self)
        self._goal_replans = 0
        self._goal_path_length = 0.0
        self._outcome = None
        finish = self._finish = Future()
        self.path = []
        self._last_plan_obstacles = -1
        self.get_logger().info(f"goal accepted: {self._goal_label}")

        await finish

        if sequence != self._goal_seq:
            # A newer goal took over while this one was running.
            return self._build_result(False, "superseded by a newer goal", 0.0, 0.0, 0.0, 0.0)

        outcome, message = self._outcome or ("failed", "goal ended without an outcome")
        result = self._build_result(
            outcome == "succeeded",
            message,
            self.command.distance_remaining_m,
            self.command.yaw_error_rad,
            max(0.0, nodeutil.now_seconds(self) - self._goal_started),
            self._goal_path_length,
        )
        self._goal_pose = None
        self._goal_handle = None
        self._outcome = None
        if outcome == "succeeded":
            goal_handle.succeed()
        elif outcome == "canceled":
            goal_handle.canceled()
        else:
            goal_handle.abort()
        self.get_logger().info(f"goal {outcome}: {message}")
        return result

    def _build_result(
        self,
        success: bool,
        message: str,
        final_distance: float,
        final_yaw_error: float,
        elapsed: float,
        path_length: float,
    ) -> Any:
        result = NavigateTo.Result()
        result.success = bool(success)
        result.message = str(message)
        result.final_distance_m = float(final_distance)
        result.final_yaw_error_rad = float(final_yaw_error)
        result.elapsed_s = float(elapsed)
        result.path_length_m = float(path_length)
        result.replans = int(self._goal_replans)
        return result

    def _finish_goal(self, outcome: str, message: str) -> None:
        """Resolve the outstanding execute callback (called from the timer)."""
        if self._finish is None or self._finish.done():
            return
        self._outcome = (outcome, message)
        self._finish.set_result(True)

    # ------------------------------------------------------------------ #
    # the control loop
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        now = nodeutil.now_seconds(self)
        self.records = self.memory.records
        self.costmap = Costmap(self.config, self.records, self._keep_outs())
        self._publish_path()
        self._publish_status()

        if self._goal_handle is None or self._goal_pose is None:
            self.state = "idle"
            self.message = "waiting for a goal"
            self.command = self.controller.stop()
            self._publish_command()
            return

        if self._goal_handle.is_cancel_requested:
            self.state = "holding"
            self.message = "goal canceled by the client"
            self.command = self.controller.stop()
            self._publish_command()
            self._finish_goal("canceled", "canceled by the client")
            return

        if self.pose is None:
            self.state = "holding"
            self.message = "no map pose yet: waiting for odometry and a tag fix"
            self.command = self.controller.stop()
            self._publish_command()
            return

        if now - self._goal_started > self._goal_timeout:
            self.state = "failed"
            self.message = f"goal timed out after {self._goal_timeout:g}s"
            self.command = self.controller.stop()
            self._publish_command()
            self._finish_goal("failed", self.message)
            return

        if not self._have_plan(now):
            result = self.planner.plan(
                self.costmap, (self.pose.x, self.pose.y), (self._goal_pose.x, self._goal_pose.y)
            )
            self.replans += 1
            self._goal_replans += 1
            self._last_plan_time = now
            self._last_plan_obstacles = len(self.records)
            if not result.success:
                self.state = "blocked"
                self.blocked_count += 1
                self.message = result.message
                self.path = []
                self.path_length_m = 0.0
                self.command = self.controller.stop()
                self._publish_command()
                # Not fatal: the map changes every frame, so keep trying until
                # the goal times out.
                return
            self.path = list(result.waypoints)
            self.path_length_m = result.length_m
            # The result reports how far the rover was *told* to go over the whole
            # goal, not the length of whichever plan happened to be current when it
            # arrived; every replan's route is added, so a goal that was replanned
            # twenty times says so instead of reporting the last 0.13 m.
            self._goal_path_length += result.length_m
            if result.goal_snapped:
                # Perception found something on the goal; work at the nearest
                # legal point instead of failing the whole leg.
                self.get_logger().warn(
                    f"goal {self._goal_label} is inside an obstacle; "
                    f"working at ({self.path[-1][0]:.2f}, {self.path[-1][1]:.2f}) instead"
                )

        command = self.controller.compute(
            self.pose,
            self.path,
            self._goal_pose.yaw,
            max_speed=self._goal_max_speed,
            position_tolerance_m=self._goal_tolerance,
            yaw_tolerance_rad=self._goal_yaw_tolerance,
        )
        self.message = ""
        if command.state == "turning" and abs(command.angular) > 1e-6:
            blocked = footprint.rotation_block(
                self.config, self.pose, self._lookahead_yaw(command), self.records, self._keep_outs()
            )
            if blocked is not None:
                command = self.controller.compute(
                    self.pose,
                    self.path,
                    self._goal_pose.yaw,
                    max_speed=self._goal_max_speed,
                    position_tolerance_m=self._goal_tolerance,
                    yaw_tolerance_rad=self._goal_yaw_tolerance,
                    force_drive=True,
                )
                self.message = f"no room to turn ({blocked}); arcing instead"
        self.command = command
        self.state = command.state
        self._publish_command()

        if command.state == "arrived":
            self.state = "arrived"
            self.message = f"arrived at {self._goal_label}"
            self.path = []
            self._finish_goal("succeeded", self.message)
            return
        self._publish_feedback()

    def _have_plan(self, now: float) -> bool:
        if not self.path:
            return False
        if now - self._last_plan_time >= self.config.controller.replan_period_s:
            return False
        if self._last_plan_obstacles != len(self.records):
            return False
        return not self._path_blocked()

    def _path_blocked(self) -> bool:
        """Is the near part of the plan now inside an obstacle?"""
        if self.costmap is None or self.pose is None or len(self.path) < 2:
            return False
        lookahead = self.config.controller.lookahead_m * 2.0
        travelled = 0.0
        previous = (self.pose.x, self.pose.y)
        for point in self.path:
            travelled += ((point[0] - previous[0]) ** 2 + (point[1] - previous[1]) ** 2) ** 0.5
            previous = point
            if travelled > lookahead:
                break
            if not self.costmap.is_free(point[0], point[1]):
                return True
        return False

    def _lookahead_yaw(self, command: ControlCommand) -> float:
        """The heading pure pursuit is aiming for (what a turn would rotate to)."""
        if self.pose is None:
            return 0.0
        return math.atan2(command.lookahead[1] - self.pose.y, command.lookahead[0] - self.pose.x)

    def _publish_command(self) -> None:
        twist = Twist()
        twist.linear.x = float(self.command.linear)
        twist.angular.z = float(self.command.angular)
        self.cmd_publisher.publish(twist)

    def _publish_path(self) -> None:
        message = Path()
        message.header = nodeutil.make_header(self, self.config.frames.map)
        for x, y in self.path:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        self.path_publisher.publish(message)

    def _publish_feedback(self) -> None:
        if self._goal_handle is None:
            return
        feedback = NavigateTo.Feedback()
        feedback.current_pose = nodeutil.pose_to_pose_stamped(
            self, self.pose if self.pose is not None else Pose3.identity(), self.config.frames.map
        )
        feedback.distance_remaining_m = float(self.command.distance_remaining_m)
        feedback.speed_mps = float(self.command.linear)
        feedback.state = _STATE_CODES.get(self.state, NavigationStatus.STATE_IDLE)
        self._goal_handle.publish_feedback(feedback)

    def _publish_status(self) -> None:
        message = NavigationStatus()
        message.header = nodeutil.make_header(self, self.config.frames.map)
        message.state = _STATE_CODES.get(self.state, NavigationStatus.STATE_IDLE)
        message.state_name = self.state
        message.active = self._goal_handle is not None
        message.localized = bool(self.localized)
        message.map_ready = self.frames_seen > 0
        message.goal_id = self._goal_label
        if self._goal_pose is not None:
            message.goal_pose = nodeutil.pose_to_pose_stamped(self, self._goal_pose, self.config.frames.map)
        if self.pose is not None:
            message.current_pose = nodeutil.pose_to_pose_stamped(self, self.pose, self.config.frames.map)
        message.distance_remaining_m = float(self.command.distance_remaining_m)
        message.path_length_m = float(self.path_length_m)
        message.speed_mps = float(self.command.linear)
        message.yaw_rate_radps = float(self.command.angular)
        message.sigma_xy_m = float(self.sigma_xy_m)
        message.sigma_yaw_rad = float(self.sigma_yaw_rad)
        message.replans = int(self._goal_replans)
        message.obstacles_in_map = int(len(self.records))
        message.blocked_count = int(self.blocked_count)
        message.goal_rejections = int(self.goal_rejections)
        message.message = self.message
        self.status_publisher.publish(message)

    def _publish_costmap(self) -> None:
        if self.costmap_publisher is None or self.costmap is None:
            return
        data, origin = self.costmap.to_occupancy_grid()
        message = OccupancyGrid()
        message.header = nodeutil.make_header(self, self.config.frames.map)
        message.info.resolution = float(self.costmap.resolution)
        message.info.width = int(self.costmap.nx)
        message.info.height = int(self.costmap.ny)
        message.info.origin.position.x = float(origin[0])
        message.info.origin.position.y = float(origin[1])
        message.info.origin.orientation.w = 1.0
        message.data = data
        self.costmap_publisher.publish(message)

    # ------------------------------------------------------------------ #
    # introspection (tests and the info tool use these)
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "message": self.message,
            "active": self._goal_handle is not None,
            "goal": None
            if self._goal_pose is None
            else (round(self._goal_pose.x, 3), round(self._goal_pose.y, 3), round(self._goal_pose.yaw, 3)),
            "pose": None if self.pose is None else (round(self.pose.x, 3), round(self.pose.y, 3), round(self.pose.yaw, 3)),
            "localized": self.localized,
            "sigma_xy_m": round(self.sigma_xy_m, 4),
            "obstacles": len(self.records),
            "memory_accepted": self.memory.accepted,
            "memory_rejected": self.memory.rejected,
            "memory_merged": self.memory.merged,
            "frames_seen": self.frames_seen,
            "path_points": len(self.path),
            "path_length_m": round(self.path_length_m, 3),
            "replans": self.replans,
            "blocked_count": self.blocked_count,
            "goal_rejections": self.goal_rejections,
            "berm_keep_out": self.berm_started,
            "command": (round(self.command.linear, 4), round(self.command.angular, 4)),
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--publish-hz", type=float, default=0.0, help="control loop rate (0 = 10 Hz)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return nodeutil.node_main(
        argv,
        "module 5 navigator: obstacle memory, costmap, planner, controller, NavigateTo action",
        build=lambda args: NavigationNode(publish_hz_override=args.publish_hz),
        add_arguments=add_arguments,
        epilog=(
            "Subscribes to the module 4 obstacle map and localisation state, plans with A* on an "
            "inflated costmap, follows the path with pure pursuit and publishes the raw command on "
            "/nav/cmd_vel.  The velocity gate is what makes it /cmd_vel."
        ),
    )
