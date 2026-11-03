"""Documentation, launch file and script hygiene tests.

These are cheap and catch the class of mistake that only shows up as "the tool
did not run" during the competition.
"""

from __future__ import annotations

import importlib.util
import os
import py_compile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_labeling_checklist_exists_and_covers_the_essentials():
    path = os.path.join(_ROOT, "docs", "labeling_checklist.md")
    assert os.path.isfile(path)
    text = open(path, encoding="utf-8").read()
    assert len(text) > 2000
    for topic in (
        "mounting",
        "YOLO format",
        "boulder",
        "crater",
        "truncat",
        "occlus",
        "verify_dataset.py",
        "training artefact",
    ):
        assert topic.lower() in text.lower(), f"checklist does not mention {topic!r}"


def test_print_checklist_finds_the_document():
    path = os.path.join(_ROOT, "scripts", "print_checklist.py")
    module = _load_module(path, "print_checklist_probe")
    resolved = module.checklist_path()
    assert os.path.isfile(resolved)
    assert module.main([]) == 0


def test_readme_exists():
    path = os.path.join(_ROOT, "README.md")
    assert os.path.isfile(path)
    assert len(open(path, encoding="utf-8").read()) > 1000


def test_launch_description_builds():
    pytest.importorskip("launch")
    pytest.importorskip("launch_ros")
    path = os.path.join(_ROOT, "launch", "collect.launch.py")
    module = _load_module(path, "collect_launch_probe")
    description = module.generate_launch_description()
    assert description is not None
    entities = description.entities
    assert entities
    # the arguments the orchestrator passes must all be declared
    names = {
        entity.name
        for entity in entities
        if hasattr(entity, "name") and isinstance(getattr(entity, "name"), str)
    }
    for required in ("scene_dir", "scene_id", "seed", "dataset_root", "cameras", "report"):
        assert required in names


def test_every_script_compiles():
    scripts = os.path.join(_ROOT, "scripts")
    assert os.path.isdir(scripts)
    names = [name for name in sorted(os.listdir(scripts)) if name.endswith(".py")]
    assert len(names) >= 7
    for name in names:
        py_compile.compile(os.path.join(scripts, name), doraise=True)


def test_every_module_imports():
    """A syntax or import error in any module is a bug, not a surprise."""
    package = os.path.join(_ROOT, "rover_dataset")
    for name in sorted(os.listdir(package)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        module_name = f"rover_dataset.{name[:-3]}"
        __import__(module_name)


def test_version_is_declared():
    import rover_dataset

    assert rover_dataset.__version__ == "0.3.0"
