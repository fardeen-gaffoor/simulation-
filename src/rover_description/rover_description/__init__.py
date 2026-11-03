"""``rover_description``: model, calibration and interface stubs for the
Caterpillar Autonomy Challenge berm-construction rover.

The package is an *asset* package (ament_cmake): the xacro model, the generated
URDF/SDF, the calibration file and the mass/envelope artifacts all live here.
The small runtime node (:mod:`rover_description.stub_topics`) is Python, as the
rest of the stack is.
"""

from __future__ import annotations

from .config import (
    CAMERA_TOPICS,
    RoverConfig,
    RoverConfigError,
    default_config_path,
    load_config,
)

__version__ = "0.2.0"

__all__ = [
    "CAMERA_TOPICS",
    "RoverConfig",
    "RoverConfigError",
    "default_config_path",
    "load_config",
    "__version__",
]
