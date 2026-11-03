"""Shared pytest fixtures for the ``rover_perception`` tests.

The tests have to run in three places, exactly like modules 1-3:

* ``colcon test`` (the install space exists and is on ``PYTHONPATH``),
* ``pytest test/`` straight from the source tree (nothing installed),
* an IDE.

So the source root goes on ``sys.path`` here, and every generated artefact the
algorithm tests need is produced *by the module that owns it* rather than
committed:

* ``calibration_path`` comes from module 2's generator, so the tests exercise the
  real schema, not a hand-written approximation of it,
* ``markers_path`` is a synthetic board file (module 1's is seed-dependent, and a
  test that changes meaning with the seed is not a test),
* ``tags.yaml`` is built by module 4 itself from that marker file.

Nothing here touches the network, Gazebo or the real filesystem outside pytest's
temporary directories.
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

from support import write_config, write_markers  # noqa: E402  (path fiddling above)

from rover_perception.config import load_config  # noqa: E402

CONFIG_PATH = os.path.join(_ROOT, "config", "perception.yaml")

#: The three boards of the reference arena (seed 1 of module 1).  Kept here so a
#: test can name ``markers_path``'s contents without reloading the file.
REFERENCE_BOARDS = [
    {
        "id": 0,
        "mount": "north_wall",
        "note": "north wall, inside the start zone",
        "position": (1.0, 4.983, 0.4),
        "rpy": (0.0, 1.570796327, -1.570796327),
        "normal": (0.0, -1.0, 0.0),
    },
    {
        "id": 1,
        "mount": "west_wall",
        "note": "west wall, inside the start zone",
        "position": (0.017, 3.85, 0.4),
        "rpy": (0.0, 1.570796327, 0.0),
        "normal": (1.0, 0.0, 0.0),
    },
    {
        "id": 2,
        "mount": "north_wall",
        "note": "mid-arena north wall",
        "position": (6.0, 4.983, 0.4),
        "rpy": (0.0, 1.570796327, -1.570796327),
        "normal": (0.0, -1.0, 0.0),
    },
]


def _module2_available() -> bool:
    try:
        import rover_description  # noqa: F401

        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def source_root() -> str:
    return _ROOT


@pytest.fixture(scope="session")
def shipped_config_path() -> str:
    assert os.path.isfile(CONFIG_PATH), f"missing module 4 config: {CONFIG_PATH}"
    return CONFIG_PATH


@pytest.fixture(scope="session")
def calibration_path(tmp_path_factory) -> str:
    """A real ``calibration.yaml`` from module 2 (falls back to a synthetic one)."""
    out = tmp_path_factory.mktemp("rover_description")
    if _module2_available():
        from rover_description.generate import generate

        result = generate(output_dir=str(out), write_sdf=False, validate=True)
        return result.calibration_path
    from support import synthetic_calibration

    return synthetic_calibration(str(out / "calibration.yaml"))


@pytest.fixture(scope="session")
def markers_path(tmp_path_factory) -> str:
    """A deterministic, seed-free ``markers.yaml`` (the reference three boards)."""
    out = tmp_path_factory.mktemp("markers")
    return write_markers(str(out / "markers.yaml"), REFERENCE_BOARDS)


@pytest.fixture(scope="session")
def tags_dir(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("perception_artifacts"))


@pytest.fixture(scope="session")
def config_path(tmp_path_factory, shipped_config_path, calibration_path, markers_path, tags_dir) -> str:
    """The shipped ``perception.yaml`` with its artefact paths pointed at the tmp dir."""
    out = tmp_path_factory.mktemp("perception_config")
    return write_config(
        str(out / "perception.yaml"),
        shipped_config_path,
        {
            "artifacts.calibration": calibration_path,
            "artifacts.markers": markers_path,
            "artifacts.output_dir": tags_dir,
        },
    )


@pytest.fixture(scope="session")
def cfg(config_path):
    return load_config(config_path)


@pytest.fixture(scope="session")
def catalogue(markers_path, tags_dir, cfg):
    """The runtime tag catalogue, written and reloaded (the runtime path)."""
    from rover_perception import tags

    document = tags.build_document(
        tags_mod_markers(markers_path), cfg.tags.board, frame_id=cfg.arena.frame_id
    )
    path = os.path.join(tags_dir, "tags.yaml")
    tags.write_catalogue(path, document)
    return tags.load_tags(path)


def tags_mod_markers(markers_path: str):
    from rover_perception.markers_ref import load_markers

    return load_markers(markers_path)


@pytest.fixture(scope="session")
def rclpy_session():
    """One rclpy context for every node test (nodes are created and destroyed per test)."""
    rclpy = pytest.importorskip("rclpy")
    if not rclpy.ok():
        rclpy.init(args=None)
    yield rclpy
    if rclpy.ok():
        rclpy.shutdown()
