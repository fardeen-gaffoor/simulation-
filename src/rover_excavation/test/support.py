"""Small builders shared by the ``rover_excavation`` tests.

Everything here produces *inputs* - a patched config file, a fresh mechanism/tub/
berm rig, a played-out work cycle - so a test can state its scenario in one line
and still be working with the real, validated objects the node uses.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import yaml

from rover_excavation.cycle import WorkCycle
from rover_excavation.mechanism import Mechanism
from rover_excavation.sand import BermField, Tub

__all__ = [
    "config_with",
    "rig",
    "play",
    "play_cycle",
    "dump_pose_xs",
    "dump_pose",
    "DIG_POSE",
    "STEP_S",
]

#: The dry run's model timestep: the control loop is 10 Hz, but the models are
#: integrated at 100 Hz so the timing assertions are not tick-quantised.
STEP_S = 0.01

#: Any legal pose in the excavation zone; a dig cycle deposits nothing, so the
#: pose itself does not matter to the sand model.
DIG_POSE: Tuple[float, float, float] = (8.0, 2.5, 1.570796)


def _apply(document: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = document
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def config_with(base_path: str, tmp_path: Any, overrides: Optional[Dict[str, Any]] = None) -> Any:
    """Load the shipped config with dotted-path overrides applied.

    ``config_with(path, tmp, {"cycle.dig.timeout_s": 3.0})`` writes a patched copy
    of ``excavation.yaml`` into ``tmp_path`` and returns the validated config.
    The shipped file is never modified, so the override cannot leak into another
    test.
    """
    from rover_excavation.config import load_config

    with open(base_path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    document = copy.deepcopy(document)
    for key, value in (overrides or {}).items():
        _apply(document, key, value)
    target = os.path.join(str(tmp_path), "excavation_override.yaml")
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(document, handle, sort_keys=False)
    return load_config(target)


def rig(config: Any) -> Tuple[Mechanism, Tub, BermField]:
    """A fresh mechanism + tub + berm, i.e. the physical state of one run."""
    return Mechanism(config.mechanism), Tub(config.mechanism.tub), BermField(config)


def play(
    config: Any,
    phase: str,
    pose: Tuple[float, float, float],
    mechanism: Mechanism,
    tub: Tub,
    berm: BermField,
    *,
    now: float = 0.0,
    cycle_id: int = 0,
    timeout_s: float = 0.0,
    target_amount: float = 0.0,
    dt: float = STEP_S,
    max_s: float = 600.0,
) -> Tuple[WorkCycle, float]:
    """Run one work cycle to completion, exactly the way the node drives it.

    The node steps the mechanism once per tick and then the cycle; this does the
    same, so a test exercises the same interleaving.
    """
    cycle = WorkCycle(
        config,
        phase,
        pose,
        mechanism,
        tub,
        berm,
        cycle_id=cycle_id,
        timeout_s=timeout_s,
        target_amount=target_amount,
        now=now,
    )
    started = now
    while not cycle.done and now - started < max_s:
        mechanism.step(dt)
        cycle.step(now)
        now += dt
    return cycle, now


play_cycle = play


def dump_pose_xs(config: Any) -> List[float]:
    """Module 5's dump x positions, derived from the copy in ``excavation.yaml``."""
    target = config.berm_target
    inset = config.nav_dump.inset_m
    lo = target.x0 + inset
    hi = target.x1 - inset
    points = max(1, config.nav_dump.points)
    if points == 1:
        return [0.5 * (lo + hi)]
    return [lo + (hi - lo) * index / (points - 1) for index in range(points)]


def dump_pose(config: Any, index: int = 0) -> Tuple[float, float, float]:
    """One of module 5's dump poses: along the target's long axis, facing +y."""
    xs = dump_pose_xs(config)
    return (xs[index % len(xs)], config.berm_dump_pose_y, config.nav_dump.heading_rad)
