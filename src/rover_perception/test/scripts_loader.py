"""Import helpers for the ``scripts/*.py`` command line tools.

``scripts/`` is not an importable package (the files are installed as ``ros2
run`` executables), so tests load them by path - the same pattern modules 1-3
use.  Keeping this in one place means the tests exercise the *exact* file that
gets installed, not a copy.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_ROOT, "scripts")


def load_script(name: str) -> Any:
    """Import ``scripts/<name>.py`` (without the ``.py``) as a module."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    path = os.path.join(_SCRIPTS, f"{name}.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"script not found: {path}")
    module_name = f"rover_perception_script_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def launch_path(name: str) -> str:
    """Absolute path of a launch file, given ``"perception"`` or ``"perception.launch.py"``."""
    filename = name if name.endswith(".py") else f"{name}.launch.py"
    return os.path.join(_ROOT, "launch", filename)


def load_launch(name: str) -> Any:
    """Import ``launch/<name>.py`` as a module (it is not an importable package)."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    path = launch_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"launch file not found: {path}")
    module_name = f"rover_perception_launch_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
