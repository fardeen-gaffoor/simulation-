"""(c) ``map -> odom`` from an AprilTag sighting.  No AMCL, no particle filter.

The arena frame is anchored by the tags on the walls, which is the only absolute
reference the competition allows (no GPS, no compass, no magnetometer).  A tag
sighting is a *measurement of where the rover is*, so one observation is enough
to solve for the transform exactly:

    T_map_tag   known  (tags.yaml: the generated catalogue)
    T_cam_tag   measured by the detector, tag pose in the camera optical frame
    T_base_cam  known  (module 2's calibration extrinsics)
    T_odom_base known  (the EKF's /odometry/filtered, or raw /odom)

    T_base_tag = T_base_cam @ T_cam_tag
    T_odom_tag = T_odom_base @ T_base_tag
    T_map_odom = T_map_tag  @ inv(T_odom_tag)

Everything is planar in the sense that matters: the rover drives on flat sand, so
only ``x``, ``y`` and ``yaw`` of the correction are ever applied, and the z /
roll / pitch components are deliberately discarded rather than allowed to tilt
the map frame.

Between fixes the estimate is pure dead reckoning, so its uncertainty grows with
how far the rover has actually driven (``sigma.growth_m_per_m``) and how much it
has turned (``sigma.growth_rad_per_rad``), and collapses back to the fix sigma on
every accepted sighting.  That uncertainty is published
(``LocalizationState``) precisely so module 5 can slow down or stop and look
around when it gets large - there is no other safety net without GPS.

Fixes are rejected, never silently applied, when they are implausible: a jump
bigger than ``max_correction_m`` / ``max_correction_yaw_rad``, a tag read at an
absurd range, or a non-finite solve.  Each rejection carries a reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .calibration_ref import CameraModel
from .config import LocalizationCfg
from .geometry import Pose3, normalize_angle, rpy_matrix
from .tags import TagCatalogue, TagSpec

__all__ = [
    "TagObservation",
    "TagFix",
    "FixSolution",
    "OdomBuffer",
    "planarize",
    "planar_delta",
    "tag_pose_in_odom",
    "map_to_odom_from_tag",
    "smooth_planar",
    "solve_fixes",
    "Localizer",
]


@dataclass(frozen=True)
class TagObservation:
    """One tag sighting, in the terms the solver needs."""

    tag_id: int
    #: pose of the tag in the camera *optical* frame (T_cam_tag)
    pose_camera: Pose3
    size_m: float = 0.0
    decision_margin: float = 0.0
    hamming: float = 0.0
    stamp: float = 0.0

    @property
    def range_m(self) -> float:
        return float(np.linalg.norm(self.pose_camera.translation))


@dataclass(frozen=True)
class TagFix:
    """What one tag said about the map -> odom transform."""

    tag_id: int
    accepted: bool
    reason: str
    range_m: float
    map_to_odom: Pose3
    weight: float = 0.0

    @property
    def pose(self) -> Optional[Pose3]:
        return self.map_to_odom if self.accepted else None


@dataclass(frozen=True)
class FixSolution:
    """The combined answer of one frame's tag sightings."""

    accepted: bool
    reason: str
    proposed_map_to_odom: Pose3
    fixes: Tuple[TagFix, ...] = ()
    correction_xy_m: float = 0.0
    correction_yaw_rad: float = 0.0
    used_tags: Tuple[int, ...] = ()

    @property
    def rejected(self) -> Tuple[TagFix, ...]:
        return tuple(fix for fix in self.fixes if not fix.accepted)


def planarize(pose: Pose3) -> Pose3:
    """Keep only ``x``, ``y``, ``yaw``: the rover drives on flat sand."""
    return Pose3.from_xy_yaw(pose.x, pose.y, pose.yaw)


def planar_delta(current: Pose3, proposed: Pose3) -> Tuple[float, float, float]:
    """``(dx, dy, dyaw)`` that takes ``current`` to ``proposed``, in the map frame.

    The translation part is the difference of the correction's own translations
    (both are map -> odom transforms, so subtracting them is meaningful), and the
    yaw part is wrapped into ``(-pi, pi]`` so a correction of -179 degrees is not
    mistaken for +181.
    """
    return (
        float(proposed.x - current.x),
        float(proposed.y - current.y),
        float(normalize_angle(proposed.yaw - current.yaw)),
    )


def tag_pose_in_odom(
    observation: TagObservation,
    camera: CameraModel,
    odom_base: Pose3,
) -> Pose3:
    """``T_odom_tag``: the measurement expressed in the odometry frame."""
    base_from_camera = Pose3(camera.r_base_optical_matrix, camera.t_base_cam_vector)
    return odom_base.compose(base_from_camera).compose(observation.pose_camera)


def map_to_odom_from_tag(
    observation: TagObservation,
    tag: TagSpec,
    camera: CameraModel,
    odom_base: Pose3,
) -> Pose3:
    """``T_map_odom`` implied by this single tag sighting."""
    return tag.pose.compose(tag_pose_in_odom(observation, camera, odom_base).inverse())


def _fix_weight(observation: TagObservation, cfg: LocalizationCfg) -> float:
    """Confidence weight for one tag: better margin and shorter range wins.

    Range is the dominant error term for a *monocular* pose estimate - a tag at
    5 m has an order of magnitude more position error than one at 1 m - so the
    weight falls off with range squared, and the detector's decision margin
    (0..~100 in apriltag terms) scales it.
    """
    margin = max(1.0, float(observation.decision_margin)) if observation.decision_margin > 0.0 else 1.0
    range_m = max(observation.range_m, cfg.min_tag_range_m)
    return float(margin / (range_m * range_m))


def smooth_planar(
    current: Pose3,
    proposed: Pose3,
    xy_gain: float,
    yaw_gain: float,
) -> Pose3:
    """Blend the correction towards ``proposed`` (gains of 1.0 apply it fully)."""
    dx, dy, dyaw = planar_delta(current, proposed)
    return Pose3.from_xy_yaw(
        current.x + xy_gain * dx,
        current.y + xy_gain * dy,
        normalize_angle(current.yaw + yaw_gain * dyaw),
    )


def solve_fixes(
    observations: Sequence[TagObservation],
    catalogue: TagCatalogue,
    camera: CameraModel,
    odom_base: Pose3,
    cfg: LocalizationCfg,
    current_map_to_odom: Optional[Pose3] = None,
    initialized: bool = True,
) -> FixSolution:
    """Solve one frame's tag sightings into a single ``map -> odom`` proposal.

    Tags are solved independently and then combined with the weights from
    :func:`_fix_weight`; this matters because two wall tags seen at once disagree
    slightly (pose noise, tag size tolerance, mounting error) and averaging is
    both more accurate and a usable cross-check.

    ``initialized=False`` means there is no ``map -> odom`` estimate yet: the
    odometry frame's origin is wherever the rover booted, not the arena corner,
    so the *first* sighting does not correct a transform - it establishes one.
    The incremental ``max_correction_*`` guards are skipped on that first
    sighting (they would reject a whole-arena-sized jump that is in fact
    correct); the range/tag/geometry guards still apply, and every later sighting
    is guarded normally.
    """
    current = planarize(current_map_to_odom) if current_map_to_odom is not None else Pose3.identity()
    if not observations:
        return FixSolution(accepted=False, reason="no_tags", proposed_map_to_odom=current)

    fixes: List[TagFix] = []
    proposals: List[Tuple[float, Pose3, int]] = []
    for observation in observations:
        if not catalogue.has(observation.tag_id):
            fixes.append(
                TagFix(observation.tag_id, False, "unknown_tag", observation.range_m, current)
            )
            continue
        near = cfg.min_tag_range_m
        far = cfg.max_tag_range_m
        if observation.range_m < near:
            fixes.append(
                TagFix(observation.tag_id, False, "tag_too_close", observation.range_m, current)
            )
            continue
        if observation.range_m > far:
            fixes.append(
                TagFix(observation.tag_id, False, "tag_too_far", observation.range_m, current)
            )
            continue
        tag = catalogue.tag(observation.tag_id)
        proposal = planarize(
            map_to_odom_from_tag(observation, tag, camera, planarize(odom_base))
        )
        if not proposal.is_finite():
            fixes.append(
                TagFix(observation.tag_id, False, "non_finite", observation.range_m, current)
            )
            continue
        if initialized:
            dx, dy, dyaw = planar_delta(current, proposal)
            if math.hypot(dx, dy) > cfg.max_correction_m:
                fixes.append(
                    TagFix(observation.tag_id, False, "correction_too_large", observation.range_m, proposal)
                )
                continue
            if abs(dyaw) > cfg.max_correction_yaw_rad:
                fixes.append(
                    TagFix(observation.tag_id, False, "correction_yaw_too_large", observation.range_m, proposal)
                )
                continue
        weight = _fix_weight(observation, cfg)
        fixes.append(TagFix(observation.tag_id, True, "ok", observation.range_m, proposal, weight))
        proposals.append((weight, proposal, observation.tag_id))

    if not proposals:
        reason = fixes[0].reason if fixes else "no_tags"
        return FixSolution(
            accepted=False, reason=reason, proposed_map_to_odom=current, fixes=tuple(fixes)
        )

    total_weight = sum(weight for weight, _pose, _tag_id in proposals)
    x = sum(weight * pose.x for weight, pose, _tag_id in proposals) / total_weight
    y = sum(weight * pose.y for weight, pose, _tag_id in proposals) / total_weight
    # circular mean of the yaw proposals, so 179 and -179 average to 180
    sin_sum = sum(weight * math.sin(pose.yaw) for weight, pose, _tag_id in proposals)
    cos_sum = sum(weight * math.cos(pose.yaw) for weight, pose, _tag_id in proposals)
    proposed = Pose3.from_xy_yaw(x, y, math.atan2(sin_sum, cos_sum))
    if initialized:
        blended = smooth_planar(current, proposed, cfg.xy_gain, cfg.yaw_gain)
    else:
        # nothing to blend from: the first sighting *is* the estimate
        blended = proposed
    correction_xy, correction_yaw = math.hypot(
        float(blended.x - current.x), float(blended.y - current.y)
    ), float(normalize_angle(blended.yaw - current.yaw))
    return FixSolution(
        accepted=True,
        reason="ok",
        proposed_map_to_odom=blended,
        fixes=tuple(fixes),
        correction_xy_m=float(correction_xy) if initialized else 0.0,
        correction_yaw_rad=float(correction_yaw) if initialized else 0.0,
        used_tags=tuple(tag_id for _weight, _pose, tag_id in proposals),
    )


@dataclass
class OdomBuffer:
    """A short history of odometry poses, queried by timestamp.

    A tag frame and the odometry sample used with it must describe the *same
    instant*: at 0.3 m/s a 20 ms mismatch is ~6 mm, which is the same order as
    the fix itself, so the robot's own odometry is kept in a tiny ring buffer and
    the sample nearest the frame's stamp is used.  Bounded by ``capacity`` so a
    long run cannot grow memory, and it tolerates a stamp that predates the
    buffer (the oldest sample is returned rather than nothing).
    """

    capacity: int = 64
    _samples: List[Tuple[float, Pose3]] = field(default_factory=list)

    def add(self, stamp: float, pose: Pose3) -> None:
        self._samples.append((float(stamp), planarize(pose)))
        if len(self._samples) > max(1, int(self.capacity)):
            del self._samples[: len(self._samples) - max(1, int(self.capacity))]

    def latest(self) -> Optional[Pose3]:
        return self._samples[-1][1] if self._samples else None

    def nearest(self, stamp: float) -> Optional[Pose3]:
        if not self._samples:
            return None
        best = min(self._samples, key=lambda sample: abs(sample[0] - float(stamp)))
        return best[1]

    @property
    def count(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()


@dataclass
class Localizer:
    """Stateful ``map -> odom`` estimator: what ``tag_localizer`` publishes.

    Holds the transform, the uncertainty that grows while dead reckoning, and the
    fix statistics.  It is deliberately separate from the node so the whole
    behaviour (including "what happens when no tag has ever been seen") is
    unit-testable.
    """

    cfg: LocalizationCfg
    catalogue: TagCatalogue
    camera: CameraModel
    map_to_odom: Pose3 = field(default_factory=Pose3.identity)
    localized: bool = False
    fixes_total: int = 0
    rejected_total: int = 0
    last_fix_at: Optional[float] = None
    last_tag_id: int = -1
    last_correction_xy_m: float = 0.0
    last_correction_yaw_rad: float = 0.0
    last_reason: str = "no_tags"
    sigma_xy_m: float = 0.0
    sigma_yaw_rad: float = 0.0
    _last_odom: Optional[Pose3] = None
    _last_odom_at: Optional[float] = None

    def __post_init__(self) -> None:
        self.sigma_xy_m = self.cfg.max_sigma_xy_m if not self.localized else self.cfg.fix_sigma_xy_m

    # -- dead reckoning ----------------------------------------------------- #
    def propagate(self, odom_base: Pose3, now: float) -> None:
        """Grow the uncertainty by how far the rover has driven since last time.

        This is the honest cost of camera-only localisation: with no fix the map
        position drifts, and the only defence is knowing *how much* it has
        drifted so module 5 can act before it matters.  The 1.25x odometry slip
        measured on tilled sand (README, verified facts) is folded into
        ``growth_m_per_m`` rather than pretended away.
        """
        current = planarize(odom_base)
        if self._last_odom is not None:
            distance = math.hypot(float(current.x - self._last_odom.x), float(current.y - self._last_odom.y))
            dyaw = abs(float(normalize_angle(current.yaw - self._last_odom.yaw)))
            self.sigma_xy_m = min(
                self.cfg.max_sigma_xy_m,
                self.sigma_xy_m + self.cfg.growth_m_per_m * distance,
            )
            self.sigma_yaw_rad = min(
                self.cfg.max_sigma_yaw_rad,
                self.sigma_yaw_rad + self.cfg.growth_rad_per_rad * dyaw,
            )
        self._last_odom = current
        self._last_odom_at = now

    # -- fixes -------------------------------------------------------------- #
    def update(
        self,
        observations: Sequence[TagObservation],
        odom_base: Pose3,
        now: float,
    ) -> FixSolution:
        """Solve and (if accepted) apply one frame's tag sightings.

        ``self.localized`` doubles as "a ``map -> odom`` estimate exists": before
        the first accepted fix the odometry frame's origin is arbitrary, so that
        first fix *establishes* the transform instead of correcting it and is
        deliberately not held to ``max_correction_m`` (see :func:`solve_fixes`).
        """
        self.propagate(odom_base, now)
        solution = solve_fixes(
            observations,
            self.catalogue,
            self.camera,
            odom_base,
            self.cfg,
            self.map_to_odom,
            initialized=self.localized,
        )
        self.last_reason = solution.reason
        if not solution.accepted:
            self.rejected_total += len(solution.fixes) or 1
            return solution
        self.map_to_odom = solution.proposed_map_to_odom
        self.localized = True
        self.fixes_total += 1
        self.last_fix_at = now
        self.last_correction_xy_m = solution.correction_xy_m
        self.last_correction_yaw_rad = solution.correction_yaw_rad
        used = solution.used_tags
        self.last_tag_id = int(used[0]) if used else -1
        # n independent tags shrink the fix sigma by sqrt(n)
        scale = 1.0 / max(1.0, math.sqrt(float(len(used))))
        self.sigma_xy_m = min(self.cfg.max_sigma_xy_m, self.cfg.fix_sigma_xy_m * scale)
        self.sigma_yaw_rad = min(self.cfg.max_sigma_yaw_rad, self.cfg.fix_sigma_yaw_rad * scale)
        return solution

    # -- reporting ---------------------------------------------------------- #
    def time_since_fix_s(self, now: float) -> float:
        if self.last_fix_at is None:
            return float("inf")
        return max(0.0, now - self.last_fix_at)

    def state_fields(self, now: float) -> Dict[str, Any]:
        """Every field of ``rover_perception_msgs/LocalizationState``."""
        return {
            "localized": bool(self.localized),
            "fixes_total": int(self.fixes_total),
            "fix_age_frames": int(self.rejected_total),
            "time_since_fix_s": self.time_since_fix_s(now),
            "sigma_xy_m": float(self.sigma_xy_m),
            "sigma_yaw_rad": float(self.sigma_yaw_rad),
            "map_to_odom": self.map_to_odom,
            "last_tag_id": int(self.last_tag_id),
            "last_fix_xy_m": float(self.last_correction_xy_m),
            "last_fix_yaw_rad": float(self.last_correction_yaw_rad),
            "reason": self.last_reason,
        }
