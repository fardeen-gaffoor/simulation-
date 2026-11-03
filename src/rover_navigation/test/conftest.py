"""Shared pytest fixtures for the ``rover_navigation`` tests.

The tests have to run in three places, exactly like modules 1-4:

* ``colcon test`` (the install space exists and is on ``PYTHONPATH``),
* ``pytest test/`` straight from the source tree (nothing installed),
* an IDE.

so the source root goes on ``sys.path`` here, and the session fixtures are built
from the *shipped* ``config/navigation.yaml`` - the same file the nodes load -
rather than a hand-written copy of it.  A test that mutates the config writes a
patched copy into pytest's temporary directory, so the shipped file is never
touched and the other tests keep seeing the real defaults.

**No test here reads ``ground_truth.yaml``.**  Module 5's whole premise is that
obstacles arrive from live perception and nothing else, so the obstacle
scenarios in these tests are synthetic and stated in the test that needs them.
The one place the real world appears is the live simulator acceptance run.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CONFIG_PATH = os.path.join(_ROOT, "config", "navigation.yaml")


@pytest.fixture(scope="session")
def config_path() -> str:
    return CONFIG_PATH


@pytest.fixture(scope="session")
def config_file() -> str:
    """The shipped ``navigation.yaml`` path, for tests that need to patch it."""
    return CONFIG_PATH


@pytest.fixture(scope="session")
def config():
    from rover_navigation.config import load_config

    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def rclpy_session():
    """One rclpy context for every node test (nodes are built and destroyed per test).

    ``colcon test`` and a plain ``pytest test/`` both land here; if something else
    already initialised rclpy (an IDE runner, say) it is reused rather than
    re-initialised, because a second ``rclpy.init`` in one process raises.
    """
    rclpy = pytest.importorskip("rclpy")
    if not rclpy.ok():
        rclpy.init(args=None)
    yield rclpy
    if rclpy.ok():
        rclpy.shutdown()
