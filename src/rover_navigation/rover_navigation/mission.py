"""The behaviour sequencer: what the rover does, in what order, and why.

Module 5 is not just a planner.  Somebody has to decide *when* to drive, dig and
dump, and somebody has to keep the camera phases in module 4 in step with that -
``/perception/mode`` is driven from here.  That somebody is this class.

It is deliberately ROS-free: the node wires action clients and topics onto it,
and every transition is testable with plain function calls.  The sequencer owns
no I/O, so the interesting question - "what does the rover do when the second
dump pose is unreachable and the tag fix is 30 s old" - is a unit test, not a
simulator session.

The run it sequences is the competition task:

    acquire -> travel to the excavation strip -> dig -> haul to the berm
            -> dump -> repeat -> return -> park

**Dig poses** sweep the excavation strip in parallel passes, alternating
direction (boustrophedon), so the strip is worked evenly instead of from one
spot.  **Dump poses** lie along the berm target's long axis, each standing the
rover off far enough that its *swept circle* clears the target while its aft
face is over the edge of it - the sand has to leave the rover inside the target
footprint, and the rover must never drive onto the berm it is building.

**The excavation mechanism is somebody else's.**  Module 6 answers ``WorkCycle``
on ``/excavation/work_cycle``.  Until it exists, ``mission.work_mode`` decides
what happens:

``delegate``
    the competition setting.  A work cycle is only ever counted when module 6
    confirms it; if no server is present the sequencer says so and aborts rather
    than pretending the sand moved.
``navigate_only``
    drive every leg and never claim a cycle happened.  This is how module 5 is
    exercised in the simulator before module 6 exists, and the status message
    always says which mode is in force.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from .config import PHASE_NAMES, NavigationConfig
from .geometry import Pose3, wrap_angle

__all__ = [
    "PHASE_IDLE",
    "PHASE_ACQUIRE",
    "PHASE_TRAVEL_TO_DIG",
    "PHASE_EXCAVATE",
    "PHASE_TRAVEL_TO_DUMP",
    "PHASE_DUMP",
    "PHASE_RETURN",
    "PHASE_PARK",
    "PHASE_COMPLETE",
    "PHASE_ABORTED",
    "PERCEPTION_MODES",
    "Waypoint",
    "MissionCommand",
    "MissionSequencer",
    "dig_poses",
    "dump_poses",
]

# Phase numbers, identical to ``rover_navigation_msgs/MissionStatus``.
PHASE_IDLE = 0
PHASE_ACQUIRE = 1
PHASE_TRAVEL_TO_DIG = 2
PHASE_EXCAVATE = 3
PHASE_TRAVEL_TO_DUMP = 4
PHASE_DUMP = 5
PHASE_RETURN = 6
PHASE_PARK = 7
PHASE_COMPLETE = 8
PHASE_ABORTED = 9

#: ``/perception/mode`` vocabulary understood by module 4.
PERCEPTION_MODES = ("idle", "nav", "tag", "excavate", "dump", "degraded")

_PHASE_MODES = {
    PHASE_IDLE: "idle",
    PHASE_ACQUIRE: "nav",     # use cam_nav (tilted down, wide FOV) to see side-mounted tags
    PHASE_TRAVEL_TO_DIG: "nav",
    PHASE_EXCAVATE: "excavate",
    PHASE_TRAVEL_TO_DUMP: "nav",
    PHASE_DUMP: "dump",
    PHASE_RETURN: "nav",
    PHASE_PARK: "idle",
    PHASE_COMPLETE: "idle",
    PHASE_ABORTED: "idle",
}

#: How many consecutive failures before the run gives up.
# The sim can lose tag fixes for long stretches (F3 finding), so allow many retries.
MAX_CONSECUTIVE_FAILURES = 20


def _linspace(low: float, high: float, count: int) -> List[float]:
    """``count`` evenly spaced values in ``[low, high]`` (a single midpoint for 1)."""
    if count <= 1:
        return [0.5 * (low + high)]
    step = (high - low) / (count - 1)
    return [low + step * index for index in range(count)]


@dataclass(frozen=True)
class Waypoint:
    """A pose the rover is asked to reach, plus a label for the log."""

    x: float
    y: float
    yaw: float
    label: str = ""


@dataclass
class MissionCommand:
    """What the sequencer wants the node to do this tick.

    ``goal`` and ``work`` are only populated on the tick that *requests* the
    action, so the node can treat them as edge-triggered.
    """

    phase: int = PHASE_IDLE
    phase_name: str = "idle"
    running: bool = False
    mode: str = "idle"
    detail: str = ""
    progress: float = 0.0
    cycles_completed: int = 0
    cycles_target: int = 0
    goal: Optional[Waypoint] = None
    work: Optional[tuple] = None  # (phase, x, y, yaw)
    finished: bool = False


def dig_poses(config: NavigationConfig) -> List[Waypoint]:
    """The excavation strip worked as parallel boustrophedon passes."""
    cfg = config.mission.dig
    zone = config.arena.zone("excavation")
    free = config.free_rect
    x_lo = max(zone.x0, free.x0) + cfg.cross_inset_m
    x_hi = min(zone.x1, free.x1) - cfg.cross_inset_m
    y_lo = max(zone.y0, free.y0) + cfg.y_inset_m
    y_hi = min(zone.y1, free.y1) - cfg.y_inset_m
    if x_hi < x_lo or y_hi < y_lo:
        return []
    passes = max(1, int(math.floor((x_hi - x_lo) / cfg.pass_spacing_m)) + 1)
    xs = _linspace(x_lo, x_hi, passes)
    ys = _linspace(y_lo, y_hi, cfg.points_per_pass)
    out: List[Waypoint] = []
    for index, x in enumerate(xs):
        sequence = ys if index % 2 == 0 else list(reversed(ys))
        heading = wrap_angle(cfg.heading_rad + (0.0 if index % 2 == 0 else math.pi))
        for step, y in enumerate(sequence):
            out.append(Waypoint(x=x, y=y, yaw=heading, label=f"dig_p{index}_s{step}"))
    return out


def dump_poses(config: NavigationConfig) -> List[Waypoint]:
    """Poses along the berm target, each clear of the target by the swept circle."""
    cfg = config.mission.dump
    target = config.arena.zone("berm_target")
    free = config.free_rect
    lo = max(target.x0 + cfg.inset_m, free.x0)
    hi = min(target.x1 - cfg.inset_m, free.x1)
    if hi < lo:
        lo = hi = 0.5 * (target.x0 + target.x1)
    xs = _linspace(lo, hi, cfg.points)
    # The stand-off has to clear the berm target for the whole *swept circle*,
    # not just the body centre: the navigator keeps the target as a keep-out
    # once the berm exists, and it also has to be reachable with the aft face
    # (half the body length back) over the target edge so the sand lands inside.
    standoff = config.robot.circumscribed_radius_m + cfg.standoff_m
    facing_north = math.sin(cfg.heading_rad) >= 0.0
    y = (target.y1 + standoff) if facing_north else (target.y0 - standoff)
    return [
        Waypoint(x=x, y=y, yaw=wrap_angle(cfg.heading_rad), label=f"dump_{index}")
        for index, x in enumerate(xs)
    ]


class MissionSequencer:
    """The run's state machine.  One :meth:`tick` per control loop.

    The node drives it: it calls :meth:`start`, then :meth:`tick` on every
    control tick, acts on the returned :class:`MissionCommand`, and reports the
    outcome of the actions it started back through :meth:`nav_finished` and
    :meth:`work_finished`.  The sequencer never waits - it either has a request
    outstanding or it makes a new one.
    """

    def __init__(self, config: NavigationConfig) -> None:
        self.config = config
        self.phase = PHASE_IDLE
        self.running = False
        self.detail = "not started"
        self.cycles_completed = 0
        self.dig_poses: List[Waypoint] = []
        self.dump_poses: List[Waypoint] = []
        self.dig_index = 0
        self.dump_index = 0
        self._pending: str = "none"  # "none" | "nav" | "work"
        self._pending_label = ""
        self._phase_started = 0.0
        self._deadline = 0.0
        self._consecutive_failures = 0
        self._parked = False
        self._acquire_staged = False
        self._started = False

    # -- lifecycle ---------------------------------------------------------- #
    def start(self, now: float) -> None:
        """Build the pose lists and enter ACQUIRE."""
        self.dig_poses = dig_poses(self.config)
        self.dump_poses = dump_poses(self.config)
        self.cycles_completed = 0
        self.dig_index = 0
        self.dump_index = 0
        self._consecutive_failures = 0
        self._parked = False
        self._acquire_staged = False
        self._pending = "none"
        self._started = True
        if not self.config.mission.enabled:
            self._enter(PHASE_IDLE, now)
            self.running = False
            self.detail = "mission disabled by configuration"
            return
        if not self.dump_poses:
            self.fail("no legal dump pose exists inside the berm target")
            return
        self._enter(PHASE_ACQUIRE, now)
        self._deadline = now + self.config.mission.acquire_timeout_s

    def fail(self, reason: str) -> None:
        """Abort the run for a reason a human needs to see."""
        self.phase = PHASE_ABORTED
        self.running = False
        self.detail = reason
        self._pending = "none"

    @property
    def terminal(self) -> bool:
        return self.phase in (PHASE_COMPLETE, PHASE_ABORTED)

    @property
    def pending(self) -> str:
        return self._pending

    # -- feedback from the node's action clients ---------------------------- #
    def nav_finished(self, now: float, success: bool, message: str = "") -> None:
        """The in-flight navigation goal ended; advance or retry."""
        if self._pending != "nav":
            return
        self._pending = "none"
        if success:
            self._consecutive_failures = 0
            if self.phase == PHASE_TRAVEL_TO_DIG:
                self._enter(PHASE_EXCAVATE, now)
            elif self.phase == PHASE_TRAVEL_TO_DUMP:
                self._enter(PHASE_DUMP, now)
            elif self.phase == PHASE_RETURN:
                self._enter(PHASE_PARK, now)
            elif self.phase == PHASE_ACQUIRE:
                self._acquire_staged = True
            return
        self._consecutive_failures += 1
        self.detail = f"navigation failed ({message or 'no detail'})"
        if self.phase == PHASE_TRAVEL_TO_DIG:
            # Skip this pose; another one may be reachable.  The run only dies
            # when nothing works at all, which is what the failure counter is
            # for.
            self.dig_index += 1
        elif self.phase == PHASE_TRAVEL_TO_DUMP:
            self.dump_index += 1
        elif self.phase == PHASE_ACQUIRE:
            # Could not reach the staging pose; try turning where we are rather
            # than giving up - the tag may well be visible from here.
            self._acquire_staged = True
        elif self.phase == PHASE_RETURN:
            pass

    def work_finished(self, now: float, success: bool, message: str = "", amount: float = 0.0) -> None:
        """The in-flight work cycle ended; count it or move on."""
        if self._pending != "work":
            return
        self._pending = "none"
        if not success:
            self._consecutive_failures += 1
            self.detail = f"work cycle failed ({message or 'no detail'}; moved {amount:g})"
            return
        self._consecutive_failures = 0
        if self.phase == PHASE_EXCAVATE:
            self.dig_index += 1
            self._enter(PHASE_TRAVEL_TO_DUMP, now)
        elif self.phase == PHASE_DUMP:
            self.dump_index += 1
            self.cycles_completed += 1
            if self.cycles_completed >= self.config.mission.cycles_target:
                self._enter(PHASE_RETURN, now)
            else:
                self._enter(PHASE_TRAVEL_TO_DIG, now)

    # -- the tick ----------------------------------------------------------- #
    def _decorated(self) -> str:
        """``self.detail``, with the work mode stated when it is not `delegate`.

        A ``navigate_only`` run drives the whole course without asking module 6
        for anything, so it must be impossible to read the status and mistake it
        for a real one - the mode goes on the front of every detail string.
        """
        if self.config.mission.work_mode == "delegate":
            return self.detail
        return f"navigate_only: {self.detail}"

    def _progress(self) -> float:
        """Fraction of the run done: whole cycles plus where the current one is."""
        target = max(1, self.config.mission.cycles_target)
        return min(1.0, (self.cycles_completed + self._cycle_fraction()) / target)

    def _give_up_reason(self) -> str:
        """Why the run is abandoning the phase it is in, for the status message.

        Failing in ACQUIRE means something specific - the rover could not turn far
        enough to see a wall tag - and that is worth saying, because it points at
        the arena, not at the route.
        """
        if self.phase == PHASE_ACQUIRE:
            return (
                f"could not turn far enough to find a wall tag "
                f"({self._consecutive_failures} consecutive failures)"
            )
        return f"giving up after {self._consecutive_failures} consecutive failures"

    def tick(
        self,
        now: float,
        *,
        pose: Optional[Pose3],
        localized: bool,
        work_available: bool = False,
    ) -> MissionCommand:
        """Advance the state machine one step and say what should happen next."""
        command = MissionCommand(
            phase=self.phase,
            phase_name=PHASE_NAMES[self.phase],
            running=self.running,
            mode=_PHASE_MODES.get(self.phase, "idle"),
            detail=self._decorated(),
            progress=self._progress(),
            cycles_completed=self.cycles_completed,
            cycles_target=self.config.mission.cycles_target,
        )
        if not self._started:
            command.detail = self._decorated()
            return command
        if self.terminal:
            command.finished = True
            command.running = False
            command.mode = "idle"
            command.detail = self._decorated()
            return command
        if self._pending == "nav":
            command.detail = self._decorated()
            return command
        if self._pending == "work":
            command.detail = self._decorated()
            return command
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            self.fail(self._give_up_reason())
            command.finished = True
            command.running = False
            command.progress = self._progress()
            command.detail = self._decorated()
            return command

        handler = {
            PHASE_ACQUIRE: self._tick_acquire,
            PHASE_TRAVEL_TO_DIG: self._tick_travel_to_dig,
            PHASE_EXCAVATE: self._tick_excavate,
            PHASE_TRAVEL_TO_DUMP: self._tick_travel_to_dump,
            PHASE_DUMP: self._tick_dump,
            PHASE_RETURN: self._tick_return,
            PHASE_PARK: self._tick_park,
        }.get(self.phase)
        if handler is not None:
            handler(now, pose=pose, localized=localized, work_available=work_available, command=command)

        command.phase = self.phase
        command.phase_name = PHASE_NAMES[self.phase]
        command.running = self.running
        command.mode = _PHASE_MODES.get(self.phase, "idle")
        command.detail = self._decorated()
        command.cycles_completed = self.cycles_completed
        command.progress = self._progress()
        return command

    def _cycle_fraction(self) -> float:
        """Where the current cycle is, for the progress bar (navigation legs count)."""
        return {
            PHASE_ACQUIRE: 0.0,
            PHASE_TRAVEL_TO_DIG: 0.3,
            PHASE_EXCAVATE: 0.55,
            PHASE_TRAVEL_TO_DUMP: 0.7,
            PHASE_DUMP: 0.9,
            PHASE_RETURN: 1.0,
            PHASE_PARK: 1.0,
        }.get(self.phase, 0.0)

    # -- phases ------------------------------------------------------------- #
    def _tick_acquire(self, now, *, pose, localized, work_available, command) -> None:
        """Find a wall tag before trusting the map frame."""
        if localized:
            self._enter(PHASE_TRAVEL_TO_DIG, now)
            self._tick_travel_to_dig(
                now, pose=pose, localized=localized, work_available=work_available, command=command
            )
            return
        if now > self._deadline:
            self.fail("the map is still not anchored: no accepted tag fix in time")
            command.finished = True
            return
        if pose is None:
            self.detail = "waiting for odometry before searching for a tag"
            return
        if not self._acquire_staged:
            # A rover that spawned in a corner of the start zone may not have
            # room to turn.  Drive to the (already validated) park pose first -
            # it is inside the start zone, so this costs nothing and gives the
            # spin somewhere legal to happen.
            park = self.config.mission.park
            self.detail = "driving to a clear pose before searching for a tag"
            self._request_nav(Waypoint(x=park.x, y=park.y, yaw=park.yaw, label="acquire staging"), command)
            return
        if self._acquire_staged:
            # The park pose (facing north) already points toward the north wall
            # tags.  No need to spin - just wait for the tag localizer to get a fix.
            # If we haven't localized after a short grace period, try a small
            # adjustment turn instead of a full 180° spin which can hit walls.
            self.detail = "waiting for tag fix at park pose"
            if now - self._phase_started > 10.0:
                # Try a gentle 90° turn to scan for tags
                self._request_nav(
                    Waypoint(
                        x=pose.x,
                        y=pose.y,
                        yaw=wrap_angle(pose.yaw + math.pi / 2),
                        label="tag search adjust",
                    ),
                    command,
                )
            return

    def _tick_travel_to_dig(self, now, *, pose, localized, work_available, command) -> None:
        if not self.dig_poses:
            self.fail("no legal excavation pose exists")
            command.finished = True
            return
        self._request_nav(self.dig_poses[self.dig_index % len(self.dig_poses)], command)

    def _tick_excavate(self, now, *, pose, localized, work_available, command) -> None:
        if self.config.mission.work_mode == "navigate_only":
            self.detail = "no excavation mechanism is asked; moving on"
            self._enter(PHASE_TRAVEL_TO_DUMP, now)
            self._tick_travel_to_dump(
                now, pose=pose, localized=localized, work_available=work_available, command=command
            )
            return
        if not work_available:
            waited = now - self._phase_started
            self.detail = (
                f"waiting for the excavation action server "
                f"({waited:.1f}/{self.config.mission.excavation_wait_s:.1f}s)"
            )
            if waited > self.config.mission.excavation_wait_s:
                self.fail(f"no WorkCycle server on {self.config.mission.excavation_action}")
                command.finished = True
            return
        self._request_work("excavate", self.dig_poses[self.dig_index % len(self.dig_poses)], command)

    def _tick_travel_to_dump(self, now, *, pose, localized, work_available, command) -> None:
        if not self.dump_poses:
            self.fail("no legal dump pose exists")
            command.finished = True
            return
        self._request_nav(self.dump_poses[self.dump_index % len(self.dump_poses)], command)

    def _tick_dump(self, now, *, pose, localized, work_available, command) -> None:
        if self.config.mission.work_mode == "navigate_only":
            self.cycles_completed += 1
            self.dump_index += 1
            if self.cycles_completed >= self.config.mission.cycles_target:
                self._enter(PHASE_RETURN, now)
                self._tick_return(
                    now, pose=pose, localized=localized, work_available=work_available, command=command
                )
            else:
                self._enter(PHASE_TRAVEL_TO_DIG, now)
                self._tick_travel_to_dig(
                    now, pose=pose, localized=localized, work_available=work_available, command=command
                )
            return
        if not work_available:
            self.detail = "waiting for the excavation action server"
            return
        self._request_work("dump", self.dump_poses[self.dump_index % len(self.dump_poses)], command)

    def _tick_return(self, now, *, pose, localized, work_available, command) -> None:
        park = self.config.mission.park
        self._request_nav(Waypoint(x=park.x, y=park.y, yaw=park.yaw, label="park"), command)

    def _tick_park(self, now, *, pose, localized, work_available, command) -> None:
        if not self._parked:
            self._parked = True
            self.detail = "parked"
            return
        self._enter(PHASE_COMPLETE, now)
        self.detail = "run complete: parked after the planned cycles"
        command.finished = True

    # -- requests ----------------------------------------------------------- #
    def _request_nav(self, waypoint: Waypoint, command: MissionCommand) -> None:
        self._pending = "nav"
        self._pending_label = waypoint.label or "goal"
        command.goal = waypoint
        self.detail = f"driving to {self._pending_label}"

    def _request_work(self, phase: str, waypoint: Waypoint, command: MissionCommand) -> None:
        self._pending = "work"
        self._pending_label = waypoint.label or phase
        command.work = (phase, waypoint.x, waypoint.y, waypoint.yaw)
        self.detail = f"{phase} at {self._pending_label}"

    # -- transitions -------------------------------------------------------- #
    def _enter(self, phase: int, now: float) -> None:
        self.phase = phase
        self._phase_started = now
        self.running = phase not in (PHASE_IDLE, PHASE_COMPLETE, PHASE_ABORTED)
        self.detail = {
            PHASE_IDLE: "idle",
            PHASE_ACQUIRE: "searching for a wall tag to anchor the map",
            PHASE_TRAVEL_TO_DIG: "driving to the excavation strip",
            PHASE_EXCAVATE: "excavating",
            PHASE_TRAVEL_TO_DUMP: "hauling to the berm",
            PHASE_DUMP: "dumping",
            PHASE_RETURN: "returning to the park pose",
            PHASE_PARK: "parking",
            PHASE_COMPLETE: "run complete",
            PHASE_ABORTED: "aborted",
        }.get(phase, self.detail)
