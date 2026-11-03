"""Deterministic seed derivation.

Every stochastic sub-system draws from a namespace derived from the single
run seed, so that:

* the same seed always produces byte-identical output (hard requirement for
  batch tests in later modules),
* adding or reordering one sub-system does not shift the random stream of the
  others (a classic source of "my seed no longer reproduces" pain),
* each sub-system can be inspected / re-generated in isolation.
"""

from __future__ import annotations

import hashlib
import random
from typing import Optional

import numpy as np

__all__ = ["derive_seed", "rng_python", "rng_numpy", "resolve_master_seed"]

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF


def resolve_master_seed(seed: Optional[int], namespace: str = "master") -> int:
    """Return a usable master seed.

    ``None`` or a negative value means "unseeded": a random 64-bit seed is
    drawn (the generator logs it so a run can always be reproduced).
    """
    if seed is None or seed < 0:
        return random.SystemRandom().getrandbits(63)
    return int(seed)


def derive_seed(master: int, namespace: str) -> int:
    """Derive a stable 64-bit sub-seed from ``master`` and a namespace string."""
    digest = hashlib.sha256(f"cat_arena|{int(master)}|{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _MASK64


def rng_python(master: int, namespace: str) -> random.Random:
    """:class:`random.Random` for a namespace (stable across Python versions)."""
    return random.Random(derive_seed(master, namespace))


def rng_numpy(master: int, namespace: str) -> np.random.Generator:
    """:class:`numpy.random.Generator` (PCG64) for a namespace."""
    return np.random.default_rng(derive_seed(master, namespace) & _MASK32)
