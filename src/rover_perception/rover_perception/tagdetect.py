"""AprilTag detection on ``cam_tag`` - a thin, testable wrapper over dt-apriltags.

``apriltag_ros`` is not installable on this machine (it needs a sudo apt
install, and the apt packages cannot see this numpy), so module 4 detects tags
with ``dt-apriltags`` - the same AprilTag 3 C library, wrapped for Python - and
keeps the ROS side (topics, messages, launch) in the package where it belongs.
The interface is interchangeable: if ``apriltag_ros`` later arrives, only
:mod:`rover_perception.tagdetect` and the node change.

Two things this wrapper does that a bare ``Detector.detect`` does not:

* **one size per call.**  apriltag's pose solve takes a single ``tag_size``, so
  the tags are grouped by physical size and detected once per distinct size;
  the arena uses one size (0.18 m) but a mixed board would silently get wrong
  poses otherwise.  The size used is recorded on every result.
* **a decodable, bounded result set.**  Hamming distance, decision margin and a
  detection cap are applied here, in the config, not scattered through the node,
  and the pose is validated before it is called valid - a degenerate quad can
  make apriltag return a ``None``-ish pose and it must not reach the localizer.

The tag's frame convention matters for localisation: apriltag report the tag
pose with ``+z`` out of the printed face and ``+y`` pointing down the image, and
module 4 draws the boards with that same ``+z`` (= module 1's board normal, see
:mod:`rover_perception.tags`).  ``test/test_tagdetect.py`` renders a tag, detects
it and checks the id round-trips; the convention itself is verified live in the
simulator (docs/tags.md) because a mirror error decodes to no id at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .calibration_ref import CameraModel
from .config import TagDetectionCfg
from .geometry import Pose3

__all__ = [
    "TagDetectorError",
    "TagDetectionResult",
    "TagDetector",
    "available",
    "install_hint",
    "corner_array",
]

_INSTALL_HINT = "pip install --user dt-apriltags   (see docs/tags.md)"


class TagDetectorError(RuntimeError):
    """Raised when the detector cannot be constructed or run."""


def available() -> bool:
    try:
        import dt_apriltags  # noqa: F401

        return True
    except Exception:
        return False


def install_hint() -> str:
    return _INSTALL_HINT


@dataclass(frozen=True)
class TagDetectionResult:
    """One AprilTag seen in one frame, in the optical frame."""

    tag_id: int
    family: str
    size_m: float
    hamming: float
    decision_margin: float
    #: T_cam_tag (optical frame), or ``None`` when the pose was degenerate
    pose_camera: Optional[Pose3]
    #: flat ``x0 y0 x1 y1 x2 y2 x3 y3`` in image pixels, top-left origin
    corners: Tuple[float, ...] = ()
    center: Tuple[float, float] = (0.0, 0.0)

    @property
    def pose_valid(self) -> bool:
        return self.pose_camera is not None

    @property
    def range_m(self) -> float:
        if self.pose_camera is None:
            return float("nan")
        return float(np.linalg.norm(self.pose_camera.translation))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.tag_id),
            "family": self.family,
            "size_m": round(self.size_m, 6),
            "hamming": float(self.hamming),
            "decision_margin": round(self.decision_margin, 4),
            "pose_valid": self.pose_valid,
            "range_m": None if self.pose_camera is None else round(self.range_m, 4),
            "center": [round(self.center[0], 2), round(self.center[1], 2)],
            "corners": [round(v, 2) for v in self.corners],
        }


def corner_array(corners: Sequence[Any]) -> Tuple[float, ...]:
    """Flatten apriltag's ``[[x, y], ...]`` into eight floats, safely."""
    if corners is None:
        return ()
    try:
        flat = np.asarray(corners, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return ()
    if flat.size < 8:
        return ()
    return tuple(float(value) for value in flat[:8])


def _pose_from_result(result: Any) -> Optional[Pose3]:
    """``pose_R``/``pose_t`` -> :class:`Pose3`, or ``None`` if degenerate."""
    rotation = getattr(result, "pose_R", None)
    translation = getattr(result, "pose_t", None)
    if rotation is None or translation is None:
        return None
    try:
        matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
        vector = np.asarray(translation, dtype=float).reshape(-1)[:3]
    except (TypeError, ValueError):
        return None
    if matrix.shape != (3, 3) or vector.shape != (3,):
        return None
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(vector))):
        return None
    # a valid rotation satisfies R R^T = I; apriltag occasionally returns a
    # garbage matrix for a near-degenerate quad, and feeding that to the
    # localizer would poison the map frame.
    if float(np.max(np.abs(matrix @ matrix.T - np.eye(3)))) > 1e-3:
        return None
    return Pose3(matrix, vector)


class TagDetector:
    """Config-driven wrapper over ``dt_apriltags.Detector``."""

    def __init__(
        self,
        cfg: TagDetectionCfg,
        camera: CameraModel,
        sizes: Mapping[int, float],
        family: str = "tag36h11",
        default_size_m: float = 0.18,
    ):
        if not available():
            raise TagDetectorError(
                "dt-apriltags is not installed; " + _INSTALL_HINT
            )
        from dt_apriltags import Detector

        self.cfg = cfg
        self.camera = camera
        self.family = family
        self.sizes: Dict[int, float] = {int(k): float(v) for k, v in (sizes or {}).items()}
        self.default_size_m = float(default_size_m)
        if self.default_size_m <= 0.0:
            raise TagDetectorError(f"default_size_m must be > 0, got {default_size_m}")
        self._detector = Detector(
            families=family,
            nthreads=max(1, int(cfg.nthreads)),
            quad_decimate=float(cfg.quad_decimate),
            quad_sigma=float(cfg.quad_sigma),
            refine_edges=1 if cfg.refine_edges else 0,
            debug=0,
        )

    # -- introspection ------------------------------------------------------ #
    def size_for(self, tag_id: int) -> float:
        """Physical black-square edge for a tag id (catalogue, else the default)."""
        return self.sizes.get(int(tag_id), self.default_size_m)

    def distinct_sizes(self) -> Tuple[float, ...]:
        values = set(self.sizes.values())
        values.add(self.default_size_m)
        return tuple(sorted(values))

    def describe(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "nthreads": int(self.cfg.nthreads),
            "quad_decimate": float(self.cfg.quad_decimate),
            "refine_edges": bool(self.cfg.refine_edges),
            "decision_margin_min": float(self.cfg.decision_margin_min),
            "hamming_max": int(self.cfg.hamming_max),
            "max_detections": int(self.cfg.max_detections),
            "sizes_m": {str(k): v for k, v in sorted(self.sizes.items())},
            "default_size_m": self.default_size_m,
            "camera": self.camera.name,
        }

    # -- detection ---------------------------------------------------------- #
    def detect(self, gray: np.ndarray) -> List[TagDetectionResult]:
        """Detect tags in a greyscale image and return them best-first."""
        array = np.ascontiguousarray(gray, dtype=np.uint8)
        if array.ndim != 2:
            raise TagDetectorError(f"detect expects a 2-D greyscale image, got {array.shape}")
        camera_params = [self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy]

        found: Dict[int, TagDetectionResult] = {}
        for size_m in self.distinct_sizes():
            try:
                raw = self._detector.detect(
                    array,
                    estimate_tag_pose=True,
                    camera_params=camera_params,
                    tag_size=size_m,
                )
            except Exception as exc:  # pragma: no cover - native library failure
                raise TagDetectorError(f"apriltag detection failed: {exc}") from exc
            for result in raw or ():
                detection = self._convert(result, size_m)
                if detection is None:
                    continue
                previous = found.get(detection.tag_id)
                if previous is None or detection.decision_margin > previous.decision_margin:
                    found[detection.tag_id] = detection

        ordered = sorted(found.values(), key=lambda item: -item.decision_margin)
        return ordered[: max(1, int(self.cfg.max_detections))]

    def _convert(self, result: Any, size_m: float) -> Optional[TagDetectionResult]:
        """Apply the configured acceptance rules to one raw apriltag result."""
        try:
            tag_id = int(getattr(result, "tag_id"))
        except (TypeError, ValueError):
            return None
        hamming = float(getattr(result, "hamming", 0.0) or 0.0)
        margin = float(getattr(result, "decision_margin", 0.0) or 0.0)
        if hamming > float(self.cfg.hamming_max):
            return None
        if margin < float(self.cfg.decision_margin_min):
            return None
        corners = corner_array(getattr(result, "corners", ()))
        center_raw = getattr(result, "center", None)
        if center_raw is not None:
            center = tuple(float(v) for v in np.asarray(center_raw, dtype=float).reshape(-1)[:2])
        elif corners:
            xs = corners[0::2]
            ys = corners[1::2]
            center = (sum(xs) / len(xs), sum(ys) / len(ys))
        else:
            center = (0.0, 0.0)
        # use the tag's own size where the catalogue knows it, so a mixed-size
        # board still yields a correct pose
        nominal = self.size_for(tag_id)
        pose = _pose_from_result(result)
        if pose is not None and abs(nominal - size_m) > 1e-9:
            pose = _rescale_pose(result, nominal, size_m)
        return TagDetectionResult(
            tag_id=tag_id,
            family=self.family,
            size_m=nominal,
            hamming=hamming,
            decision_margin=margin,
            pose_camera=pose,
            corners=corners,
            center=(float(center[0]), float(center[1])),
        )


def _rescale_pose(result: Any, nominal: float, used: float) -> Optional[Pose3]:
    """Fix a pose that was solved with the wrong tag size.

    A tag pose estimate scales linearly with the assumed tag size (the solve is
    a homography decomposition, so translation is proportional and rotation is
    unchanged) - so a corrected size is a scale factor on ``pose_t``, not a
    re-detection.  Only ever needed for a board with more than one tag size.
    """
    pose = _pose_from_result(result)
    if pose is None or used <= 0.0:
        return None
    scale = float(nominal) / float(used)
    return Pose3(pose.rotation, pose.translation * scale)
