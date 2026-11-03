"""Deterministic seed derivation for the dataset factory.

Same contract as ``cat_arena.seeding``: every stochastic sub-system (scene
randomization, pose plan, augmentation, split) draws from a namespace derived
from one master seed, so a dataset run is reproducible byte-for-byte and adding
a new sub-system cannot shift the random stream of the existing ones.

The prefix differs from cat_arena's (``rover_dataset|`` vs ``cat_arena|``) so the
two packages never share a stream even when given the same seed.
"""

from __future__ import annotations

import hashlib
import random
from typing import Optional

import numpy as np

__all__ = ["resolve_master_seed", "derive_seed", "rng_python", "rng_numpy", "seed_from"]

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF


def resolve_master_seed(seed: Optional[int]) -> int:
    """A usable master seed; ``None``/negative draws a fresh 64-bit seed."""
    if seed is None or seed < 0:
        return random.SystemRandom().getrandbits(63)
    return int(seed)


def derive_seed(master: int, namespace: str, *parts: object) -> int:
    """Stable 64-bit sub-seed from ``master``, a namespace and extra parts."""
    key = "rover_dataset|{}|{}".format(int(master), namespace)
    for part in parts:
        key += f"|{part}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _MASK64


def seed_from(master: int, namespace: str, *parts: object) -> int:
    """Convenience alias used where a plain integer seed is wanted."""
    return derive_seed(master, namespace, *parts)


def rng_python(master: int, namespace: str, *parts: object) -> random.Random:
    """A :class:`random.Random` for a namespace (stable across Python versions)."""
    return random.Random(derive_seed(master, namespace, *parts))


def rng_numpy(master: int, namespace: str, *parts: object) -> np.random.Generator:
    """A :class:`numpy.random.Generator` (PCG64) for a namespace."""
    return np.random.default_rng(derive_seed(master, namespace, *parts) & _MASK32)
