"""The rover's actual footprint: a rotated rectangle, not a disc.

Module 5 originally planned with a single disc of the rover's *circumscribed*
radius - the circle a rover that can turn on the spot sweeps - because that
makes the planner trivially point-like.  That is correct but it does not fit
this arena: a 0.84 m circle around every 0.35 m boulder seals the 5.5 m obstacle
zone, and a live check confirmed the west column becomes unreachable from the
start zone on the shipped seed.  So the costmap inflates by the *inscribed*
radius (the half-width, which is what a body translating along a path actually
sweeps) and the awkward half of the problem is solved exactly here instead:

* a **goal pose** is accepted only if the rectangle at that pose clears every
  wall, remembered obstacle and keep-out;
* an **in-place turn** is accepted only if no sample of the rectangle between
  the current heading and the target heading touches anything.

Both tests are exact: a rotated rectangle against a disc reduces to "closest
point of the rectangle in its own frame", and against the arena it reduces to
"all four corners inside the inset rectangle".  Neither needs a library, and
both are cheap enough to run every control tick.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .config import NavigationConfig, Rect
from .costmap import KIND_CRATER, ObstacleRecord
from .geometry import Pose3, wrap_angle

__all__ = [
    "OP_FOOTPRINT_LENGTH_M",
    "OP_FOOTPRINT_WIDTH_M",
    "corner_offsets",
    "rect_corners",
    "rect_disc_clearance",
    "footprint_block",
    "rotation_block",
]

#: The footprint module 5 plans for, in metres.
#:
#: This is the rover as *built* (module 2 measures 1.2396 x 0.655 m) padded by
#: about 5%, NOT the competition's 1.5 x 0.75 m build limit.  Planning for the
#: build limit is safe but useless in this arena - see the module docstring -
#: so the limit is kept as a hard ceiling (checked against rover.yaml in the
#: tests) while the planner uses the real body.  Both numbers live in
#: ``navigation.yaml`` under ``robot``; these are only the fallback defaults.
OP_FOOTPRINT_LENGTH_M = 1.30
OP_FOOTPRINT_WIDTH_M = 0.70

#: Angular step used when sampling a rotation (radians).  0.15 rad is 8.6 deg,
#: well inside the margin each sample is required to keep.
_ROTATION_SAMPLE_RAD = 0.15


def corner_offsets(half_length: float, half_width: float) -> List[Tuple[float, float]]:
    """The four corners of the body rectangle in the body frame."""
    return [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]


def rect_corners(x: float, y: float, yaw: float, half_length: float, half_width: float) -> List[Tuple[float, float]]:
    """The four corners of the body rectangle placed at ``(x, y, yaw)``."""
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    out: List[Tuple[float, float]] = []
    for dx, dy in corner_offsets(half_length, half_width):
        out.append((x + cos_yaw * dx - sin_yaw * dy, y + sin_yaw * dx + cos_yaw * dy))
    return out


def rect_disc_clearance(
    x: float, y: float, yaw: float, half_length: float, half_width: float, cx: float, cy: float
) -> float:
    """Distance from the disc centre ``(cx, cy)`` to the rectangle's boundary.

    Negative inside.  Computed by rotating the point into the rectangle's own
    frame - where the rectangle is axis-aligned - clamping to the half extents
    to get the closest point, and measuring back.
    """
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    dx, dy = cx - x, cy - y
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    clamped_x = max(-half_length, min(half_length, local_x))
    clamped_y = max(-half_width, min(half_width, local_y))
    if clamped_x == local_x and clamped_y == local_y:
        # The centre is inside: report the (negative) distance to the nearest
        # edge, so a caller can still rank two intrusions.
        return -min(half_length - abs(local_x), half_width - abs(local_y))
    return math.hypot(local_x - clamped_x, local_y - clamped_y)


def _obstacle_radius(config: NavigationConfig, record: ObstacleRecord) -> float:
    scale = (
        config.costmap.crater_radius_scale
        if record.kind == KIND_CRATER
        else config.costmap.boulder_radius_scale
    )
    return max(0.0, record.radius) * scale


def _rect_blocks(
    config: NavigationConfig,
    rect: Rect,
    x: float,
    y: float,
    yaw: float,
    records: Sequence[ObstacleRecord],
    keep_outs: Sequence[Rect],
    clearance: float,
) -> Optional[str]:
    half_length = 0.5 * config.robot.length_m
    half_width = 0.5 * config.robot.width_m
    wall_inset = config.arena.wall_margin_m
    for corner_x, corner_y in rect_corners(x, y, yaw, half_length, half_width):
        if not (
            rect.x0 + wall_inset <= corner_x <= rect.x1 - wall_inset
            and rect.y0 + wall_inset <= corner_y <= rect.y1 - wall_inset
        ):
            return "the body would touch an arena wall"
    for record in records:
        distance = rect_disc_clearance(x, y, yaw, half_length, half_width, record.x, record.y)
        if distance < _obstacle_radius(config, record) + clearance:
            return f"the body would touch {record.kind_name} '{record.identifier}'"
    for keep_out in keep_outs:
        grown = keep_out.inset(-clearance)
        # Rectangle against axis-aligned rectangle: exact via the separating
        # axis test in the rotated frame (2 axes) and the world frame (2 axes).
        if _rects_overlap(x, y, yaw, half_length, half_width, grown):
            return "the body would enter the berm keep-out area"
    return None


def _rects_overlap(
    x: float, y: float, yaw: float, half_length: float, half_width: float, other: Rect
) -> bool:
    """Separating-axis test between the rotated body rectangle and an AABB."""
    axes = [(math.cos(yaw), math.sin(yaw)), (-math.sin(yaw), math.cos(yaw)), (1.0, 0.0), (0.0, 1.0)]
    corners = rect_corners(x, y, yaw, half_length, half_width)
    other_corners = [
        (other.x0, other.y0),
        (other.x1, other.y0),
        (other.x1, other.y1),
        (other.x0, other.y1),
    ]
    for axis_x, axis_y in axes:
        own = [axis_x * cx + axis_y * cy for cx, cy in corners]
        theirs = [axis_x * cx + axis_y * cy for cx, cy in other_corners]
        if max(own) < min(theirs) or max(theirs) < min(own):
            return False
    return True


def footprint_block(
    config: NavigationConfig,
    pose: Pose3,
    records: Sequence[ObstacleRecord] = (),
    keep_outs: Sequence[Rect] = (),
    clearance: float = 0.0,
) -> Optional[str]:
    """Why the rover may not *be* at ``pose``, or ``None`` when it may.

    Used to validate goals: a goal the rover cannot legally occupy is refused
    before a plan is wasted on it.
    """
    return _rect_blocks(
        config, config.arena.rect, pose.x, pose.y, pose.yaw, records, keep_outs, clearance
    )


def rotation_block(
    config: NavigationConfig,
    pose: Pose3,
    target_yaw: float,
    records: Sequence[ObstacleRecord] = (),
    keep_outs: Sequence[Rect] = (),
    clearance: float = 0.0,
) -> Optional[str]:
    """Why the rover may not rotate from its current heading to ``target_yaw``.

    The rectangle is sampled along the rotation rather than swept as a disc, so
    a small correction next to a boulder is allowed while a half-turn into the
    same boulder is not.
    """
    delta = wrap_angle(target_yaw - pose.yaw)
    steps = max(1, int(math.ceil(abs(delta) / _ROTATION_SAMPLE_RAD)))
    for step in range(1, steps + 1):
        yaw = wrap_angle(pose.yaw + delta * step / steps)
        blocked = _rect_blocks(
            config, config.arena.rect, pose.x, pose.y, yaw, records, keep_outs, clearance
        )
        if blocked is not None:
            return blocked
    return None
