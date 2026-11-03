"""Import helpers for ``scripts/*.py`` and ``launch/*.py``.

Neither ``scripts/`` nor ``launch/`` is an importable package (the scripts are
installed as ``ros2 run`` executables and the launch files are loaded by
``ros2 launch``), so tests load them by path.  Keeping this in one place means
the tests exercise the *exact* files that get installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load(path: str, module_name: str) -> Any:
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_script(name: str) -> Any:
    """Import ``scripts/<name>.py`` (without the ``.py``)."""
    return _load(os.path.join(_ROOT, "scripts", f"{name}.py"), f"rover_script_{name}")


def script_path(name: str) -> str:
    return os.path.join(_ROOT, "scripts", f"{name}.py")


def load_launch(name: str) -> Any:
    """Import ``launch/<name>.launch.py`` (without ``.launch.py``)."""
    return _load(os.path.join(_ROOT, "launch", f"{name}.launch.py"), f"rover_launch_{name}")


def launch_path(name: str) -> str:
    return os.path.join(_ROOT, "launch", f"{name}.launch.py")


def declared_argument_names(launch_description) -> set:
    """The ``DeclareLaunchArgument`` names in a LaunchDescription."""
    names = set()
    for entity in launch_description.entities:
        name = getattr(entity, "name", None)
        if name is not None:
            names.add(name)
    return names
