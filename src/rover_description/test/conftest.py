"""Shared pytest fixtures for the rover_description tests.

The tests must run in three places:

* ``colcon test`` (install space exists and is on ``PYTHONPATH``),
* ``pytest test/`` straight from the source tree (no install space),
* ``pytest`` from an IDE.

This file makes ``import rover_description`` work in all of them and exposes the
fixtures for the (xacro-expanding) generation tests.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any, Callable, Dict, Optional

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CONFIG_PATH = os.path.join(_ROOT, "config", "rover.yaml")
BRIDGES_PATH = os.path.join(_ROOT, "config", "bridges.yaml")


def gz_available() -> bool:
    return shutil.which("gz") is not None


def xacro_available() -> bool:
    try:
        import xacro  # noqa: F401

        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def config_path() -> str:
    assert os.path.isfile(CONFIG_PATH), f"missing test config: {CONFIG_PATH}"
    return CONFIG_PATH


@pytest.fixture(scope="session")
def bridges_path() -> str:
    assert os.path.isfile(BRIDGES_PATH), f"missing bridges config: {BRIDGES_PATH}"
    return BRIDGES_PATH


@pytest.fixture(scope="session")
def cfg(config_path):
    from rover_description.config import load_config

    return load_config(config_path)


@pytest.fixture(scope="session")
def generated(tmp_path_factory, config_path):
    """One full rover generation shared by every test module that needs it."""
    from rover_description.generate import generate

    if not xacro_available():
        pytest.skip("xacro is not importable; skipping model generation")
    out = tmp_path_factory.mktemp("rover_generated")
    return generate(config_path=config_path, output_dir=str(out), write_sdf=False, validate=True)


@pytest.fixture
def make_config(tmp_path) -> Callable[[Optional[Callable[[Dict[str, Any]], None]]], str]:
    """Write a (possibly mutated) copy of rover.yaml to a temp path."""

    def _make(mutate: Optional[Callable[[Dict[str, Any]], None]] = None) -> str:
        import yaml

        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle)
        if mutate is not None:
            mutate(doc)
        target = tmp_path / "rover.yaml"
        with open(target, "w", encoding="utf-8") as handle:
            yaml.safe_dump(doc, handle)
        return str(target)

    return _make
