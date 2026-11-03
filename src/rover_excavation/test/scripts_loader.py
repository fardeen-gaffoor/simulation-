"""Import helpers for ``scripts/*.py`` and ``launch/*.py``.

Neither directory is an importable package (the scripts are installed as ``ros2
run`` executables, the launch files are loaded by ``launch``), so tests load them
by path - the same pattern modules 1-5 use.  Keeping it in one place means the
tests exercise the *exact* file that gets installed, not a copy.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SCRIPTS = os.path.join(_ROOT, "scripts")
_LAUNCH = os.path.join(_ROOT, "launch")

__all__ = ["load_script", "load_launch", "launch_path", "script_path"]


def script_path(name: str) -> str:
    filename = name if name.endswith(".py") else f"{name}.py"
    return os.path.join(_SCRIPTS, filename)


def launch_path(name: str) -> str:
    filename = name if name.endswith(".py") else f"{name}.launch.py"
    return os.path.join(_LAUNCH, filename)


def _load(path: str, module_name: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_script(name: str) -> Any:
    """Import ``scripts/<name>.py`` (without the ``.py``) as a module."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    return _load(script_path(name), f"rover_excavation_script_{name}")


def load_launch(name: str) -> Any:
    """Import ``launch/<name>.launch.py`` as a module."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    return _load(launch_path(name), f"rover_excavation_launch_{name}")
