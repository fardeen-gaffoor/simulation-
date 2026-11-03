"""Shared pytest fixtures for the rover_dataset tests.

The tests run in three places: ``colcon test`` (install space on
``PYTHONPATH``), ``pytest test/`` straight from the source tree, and an IDE.
This file makes ``import rover_dataset`` work in all of them, and exposes the
small synthetic fixtures the projection/augmentation tests build on.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rover_dataset.calibration_ref import CalibrationDoc, CameraModel, load_calibration  # noqa: E402
from rover_dataset.config import DatasetConfig, load_config  # noqa: E402
from rover_dataset.groundtruth import Boulder, Crater, GroundTruth  # noqa: E402

CONFIG_PATH = os.path.join(_ROOT, "config", "dataset.yaml")


@pytest.fixture(scope="session")
def cfg() -> DatasetConfig:
    assert os.path.isfile(CONFIG_PATH), f"missing {CONFIG_PATH}"
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def calibration(tmp_path_factory) -> CalibrationDoc:
    """Module 2's generated calibration.

    ``calibration.yaml`` is a *generated* artefact of module 2, so if the
    workspace has not generated one yet this fixture generates it (module 2 is a
    declared dependency of this package).
    """
    try:
        return load_calibration()
    except Exception:
        pass
    try:
        from rover_description.config import default_config_path as rover_config_path
        from rover_description.generate import generate
    except Exception as exc:  # pragma: no cover - only when module 2 is absent
        pytest.skip(f"rover_description is not importable: {exc}")
    out = tmp_path_factory.mktemp("rover_calibration")
    generate(config_path=rover_config_path(), output_dir=str(out), write_sdf=False, validate=False)
    return load_calibration(os.path.join(str(out), "calibration.yaml"))


def make_ground_truth(
    boulders: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    craters: Optional[Sequence[Tuple[float, float, float]]] = None,
    *,
    length_x: float = 9.0,
    width_y: float = 5.0,
    path: str = "/dev/null",
) -> GroundTruth:
    """Build a :class:`GroundTruth` from ``(x, y, z, radius)`` / ``(x, y, radius)``."""
    boulder_objects: List[Boulder] = []
    for index, (x, y, z, radius) in enumerate(boulders or []):
        boulder_objects.append(
            Boulder(
                id=f"boulder_{index}",
                center=(x, y, z),
                radius=radius,
                diameter=2.0 * radius,
                exposed_height=max(0.0, z + radius),
                top_z=z + radius,
                zone="obstacle",
            )
        )
    crater_objects: List[Crater] = []
    for index, (x, y, radius) in enumerate(craters or []):
        crater_objects.append(
            Crater(
                id=f"crater_{index}",
                center=(x, y),
                radius=radius,
                width=2.0 * radius,
                depth=0.1,
                zone="obstacle",
            )
        )
    return GroundTruth(
        path=path,
        seed=1,
        world_name="cat_arena",
        length_x=length_x,
        width_y=width_y,
        spawn_z_offset=0.08,
        boulders=boulder_objects,
        craters=crater_objects,
        zones={},
        start_pose=(1.0, 3.5, 0.08, 0.0),
        raw={},
    )


@pytest.fixture
def image() -> np.ndarray:
    """A deterministic 64x48 RGB test image with some structure."""
    rng = np.random.default_rng(1234)
    gradient = np.linspace(0.0, 1.0, 64 * 48, dtype=np.float64).reshape(48, 64, 1)
    base = np.concatenate([gradient * 200, gradient * 160, gradient * 120], axis=2)
    noise = rng.normal(0.0, 6.0, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def small_camera() -> CameraModel:
    """A tiny, easy-to-reason-about front camera looking straight down +x."""
    return CameraModel(
        name="test_cam",
        topic="/cam_test",
        camera_info_topic="/cam_test/camera_info",
        frame_id="test_optical_frame",
        width=64,
        height=48,
        fx=60.0,
        fy=60.0,
        cx=32.0,
        cy=24.0,
        t_base_cam=(0.30, 0.0, 0.50),
        # columns are the optical axes in base_link:
        #   right = -y   down = -z   forward = +x
        r_base_optical=(0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0),
    )
