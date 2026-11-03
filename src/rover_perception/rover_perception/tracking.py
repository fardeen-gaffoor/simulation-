"""Multi-frame tracking: one track per physical obstacle, in the map frame.

A single camera frame is weak evidence: a shadow can look like a rock, a rock
can be half-occluded, and the classical backend can flip its mind about whether
a blob is convex or concave.  Tracking turns a stream of noisy observations into
a small set of objects with an explicit, decaying confidence, which is what
module 5 wants to route around.

Rules (deterministic, no RNG, so a replay is reproducible):

* an observation is matched to the nearest track within ``associate_radius_m``,
  regardless of class, so classification jitter cannot clone an obstacle,
* a track's position and radius are exponential moving averages with
  ``position_gain``, so one bad frame cannot teleport an obstacle,
* confidence rises towards 1 with every hit (``hit_gain`` scaled by the detector
  score, plus ``crater_bias`` for craters) and decays with a half-life
  (``decay_half_life_s``) while the object is not seen,
* a track is published once it has ``min_hits`` observations and is retired
  after ``max_misses`` misses or when its confidence falls below
  ``min_confidence``,
* **kind only ever escalates**: a track that has ever been called a crater stays
  a crater (``conservative`` is set when the two classes disagreed).  Missing a
  hole is much worse than a needless detour, so nothing is ever downgraded from
  crater to boulder.

Track ids are ``obstacle_<n>``: deliberately class-neutral, because a track may
escalate its class after it was created.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import TrackingCfg
from .groundproj import GroundObservation

__all__ = ["Track", "Tracker", "SIGMA_GROWTH_PER_MISS"]

#: A missed frame multiplies a track's position sigma by this much.
SIGMA_GROWTH_PER_MISS = 1.5


@dataclass
class Track:
    """One tracked obstacle in the map frame."""

    id: str
    kind: str
    class_id: int
    x: float
    y: float
    radius: float
    height: float
    depth: float
    score: float
    confidence: float
    observations: int
    created_at: float
    last_seen_at: float
    misses: int = 0
    conservative: bool = False
    source: str = "unknown"
    range_m: float = 0.0
    sigma_xy_m: float = 0.12

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    def age_s(self, now: float) -> float:
        return max(0.0, now - self.created_at)

    def last_seen_age_s(self, now: float) -> float:
        return max(0.0, now - self.last_seen_at)

    def is_confirmed(self, cfg: TrackingCfg) -> bool:
        return self.observations >= cfg.min_hits and self.confidence >= cfg.min_confidence

    def as_dict(self, now: float) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "class_id": self.class_id,
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "radius": round(self.radius, 6),
            "height": round(self.height, 6),
            "depth": round(self.depth, 6),
            "score": round(self.score, 4),
            "confidence": round(self.confidence, 4),
            "observations": self.observations,
            "age_s": round(self.age_s(now), 3),
            "last_seen_age_s": round(self.last_seen_age_s(now), 3),
            "misses": self.misses,
            "conservative": self.conservative,
            "source": self.source,
            "sigma_xy_m": round(self.sigma_xy_m, 6),
        }


class Tracker:
    """Deterministic multi-target tracker (greedy nearest-neighbour)."""

    def __init__(self, cfg: TrackingCfg):
        self.cfg = cfg
        self._tracks: List[Track] = []
        self._created = 0
        self._last_update: Optional[float] = None
        #: confidence as of each track's last sighting, so decay stays a pure
        #: function of "time since last seen" instead of compounding per update
        self._seen_confidence: Dict[str, float] = {}

    # -- introspection ------------------------------------------------------ #
    @property
    def tracks(self) -> List[Track]:
        return list(self._tracks)

    @property
    def confirmed(self) -> List[Track]:
        return [track for track in self._tracks if track.is_confirmed(self.cfg)]

    def reset(self) -> None:
        self._tracks.clear()
        self._created = 0
        self._last_update = None
        self._seen_confidence.clear()

    # -- internals ---------------------------------------------------------- #
    def _next_id(self) -> str:
        index = self._created
        self._created += 1
        return f"obstacle_{index}"

    def _decay(self, now: float) -> None:
        """Re-derive every track's confidence from its last sighting.

        Deliberately *not* iterative: ``confidence *= 0.5 ** (elapsed / half_life)``
        on every update would make the effective half-life depend on how often the
        node calls :meth:`update` (15 Hz would prune a track far sooner than 5 Hz).
        Keeping the confidence-at-last-sighting makes the decay a function of
        wall-clock time since the sighting, which is what a half-life means.
        """
        half_life = self.cfg.decay_half_life_s
        for track in self._tracks:
            base = float(self._seen_confidence.get(track.id, track.confidence))
            elapsed = track.last_seen_age_s(now)
            if elapsed > 0.0:
                track.confidence = float(base * (0.5 ** (elapsed / half_life)))
            else:
                track.confidence = base
            self._seen_confidence[track.id] = base

    def _score_with_bias(self, observation: GroundObservation) -> float:
        score = max(0.0, min(1.0, observation.score))
        if observation.kind == "crater":
            score = min(1.0, score + self.cfg.crater_bias)
        return score

    def _nearest(self, observation: GroundObservation, taken: set) -> Optional[int]:
        best_index: Optional[int] = None
        best_distance = self.cfg.associate_radius_m
        for index, track in enumerate(self._tracks):
            if index in taken:
                continue
            distance = math.hypot(track.x - observation.x, track.y - observation.y)
            if distance <= best_distance:
                best_index, best_distance = index, distance
        return best_index

    def _create(self, observation: GroundObservation, now: float, source: str) -> int:
        weight = self._score_with_bias(observation)
        track = Track(
            id=self._next_id(),
            kind=observation.kind,
            class_id=observation.class_id,
            x=observation.x,
            y=observation.y,
            radius=observation.radius,
            height=observation.height,
            depth=observation.depth,
            score=observation.score,
            confidence=min(1.0, self.cfg.hit_gain * weight),
            observations=1,
            created_at=now,
            last_seen_at=now,
            conservative=observation.conservative,
            source=source,
            range_m=observation.range_m,
            sigma_xy_m=max(observation.range_sigma_m, 1e-3),
        )
        self._tracks.append(track)
        self._seen_confidence[track.id] = track.confidence
        return len(self._tracks) - 1

    # -- the update --------------------------------------------------------- #
    def update(
        self,
        observations: Sequence[GroundObservation],
        now: float,
        source: str = "unknown",
    ) -> List[Track]:
        """Fold one frame of observations in and return the confirmed tracks."""
        self._decay(now)
        ordered = sorted(observations, key=lambda obs: (-obs.score, obs.kind, obs.x, obs.y))
        taken: set = set()
        gain = self.cfg.position_gain

        for observation in ordered:
            index = self._nearest(observation, taken)
            if index is None:
                taken.add(self._create(observation, now, source))
                continue
            track = self._tracks[index]
            taken.add(index)

            track.x = (1.0 - gain) * track.x + gain * observation.x
            track.y = (1.0 - gain) * track.y + gain * observation.y
            track.radius = (1.0 - gain) * track.radius + gain * observation.radius
            track.height = (1.0 - gain) * track.height + gain * observation.height
            track.depth = (1.0 - gain) * track.depth + gain * observation.depth
            track.score = max(track.score, observation.score)
            track.observations += 1
            track.last_seen_at = now
            track.misses = 0
            track.range_m = observation.range_m
            track.source = source
            track.sigma_xy_m = max(
                1e-3, (1.0 - gain) * track.sigma_xy_m + gain * observation.range_sigma_m
            )
            if observation.conservative:
                track.conservative = True
            if observation.kind != track.kind:
                # kind only escalates; a disagreement is recorded, never ignored
                track.conservative = True
                if observation.kind == "crater":
                    track.kind = "crater"
                    track.class_id = observation.class_id
            weight = self._score_with_bias(observation)
            track.confidence = float(
                min(1.0, track.confidence + (1.0 - track.confidence) * self.cfg.hit_gain * weight)
            )
            self._seen_confidence[track.id] = track.confidence

        for index, track in enumerate(self._tracks):
            if index not in taken:
                track.misses += 1
                track.sigma_xy_m *= SIGMA_GROWTH_PER_MISS

        self._tracks = [
            track
            for track in self._tracks
            if track.misses <= self.cfg.max_misses and track.confidence >= self.cfg.min_confidence
        ]
        live = {track.id for track in self._tracks}
        self._seen_confidence = {
            key: value for key, value in self._seen_confidence.items() if key in live
        }
        self._last_update = now
        confirmed = [track for track in self._tracks if track.is_confirmed(self.cfg)]
        confirmed.sort(key=lambda track: (-track.confidence, track.id))
        return confirmed[: self.cfg.max_obstacles]
