"""Shared pytest fixtures for the cat_arena tests.

The tests must run in three places:

* ``colcon test`` (install space exists and is on ``PYTHONPATH``),
* ``pytest test/`` straight from the source tree (no install space),
* ``pytest`` from an IDE.

This file makes ``import cat_arena`` work in all of them by putting the source
tree root on ``sys.path``, and exposes a couple of session-scoped fixtures for
the expensive (world-generating) tests.
"""

from __future__ import annotations

import os
import shutil
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from cat_arena.config import load_config  # noqa: E402  (path fiddling above)

CONFIG_PATH = os.path.join(_ROOT, "config", "arena.yaml")


def gz_available() -> bool:
    return shutil.which("gz") is not None


@pytest.fixture(scope="session")
def config_path() -> str:
    assert os.path.isfile(CONFIG_PATH), f"missing test config: {CONFIG_PATH}"
    return CONFIG_PATH


@pytest.fixture(scope="session")
def cfg(config_path):
    return load_config(config_path)


@pytest.fixture(scope="session")
def generated(tmp_path_factory, config_path):
    """One full arena generation shared by every test module that needs it."""
    from cat_arena.generate import generate

    out = tmp_path_factory.mktemp("arena_seed_11")
    return generate(
        seed=11,
        config_path=config_path,
        output_dir=str(out),
        validate=True,
        verbose=False,
        include_timestamp=False,
    )
