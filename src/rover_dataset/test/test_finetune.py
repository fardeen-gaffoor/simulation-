"""Fine-tuning plan tests: the recipe must come from dataset.yaml and be audited."""

from __future__ import annotations

import json
import os

import pytest

from rover_dataset.finetune import (
    build_training_plan,
    python_snippet,
    training_arguments,
    yolo_cli_command,
)


def _dataset_dir(tmp_path, sampling=None, written=None):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 2\n", encoding="utf-8")
    payload = {"scene_split": {"sim_0000": "train", "sim_0001": "val"}}
    if sampling is not None:
        payload["sampling"] = sampling
    if written is not None:
        payload["written"] = written
    (root / "split.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(root / "data.yaml")


def test_arguments_come_from_the_config(cfg, tmp_path):
    args = training_arguments(cfg, _dataset_dir(tmp_path))
    assert args["imgsz"] == cfg.finetune.imgsz
    assert args["epochs"] == cfg.finetune.epochs
    assert args["batch"] == cfg.finetune.batch
    assert args["seed"] == cfg.finetune.seed
    assert args["model"] == cfg.finetune.base_model
    assert os.path.isabs(args["data"])


def test_cli_command_is_complete(cfg, tmp_path):
    dataset_yaml = _dataset_dir(tmp_path)
    command = yolo_cli_command(cfg, dataset_yaml)
    assert command[:3] == ["yolo", "detect", "train"]
    assert any(part.startswith("data=") and part.endswith("data.yaml") for part in command)
    assert any(part.startswith("model=") for part in command)


def test_python_snippet_is_runnable(cfg, tmp_path):
    snippet = python_snippet(cfg, _dataset_dir(tmp_path))
    assert "from ultralytics import YOLO" in snippet
    assert "results = model.train(**train_args)" in snippet
    assert cfg.finetune.base_model in snippet
    # it must be valid Python: json's `true`/`false` would NameError on paste
    compile(snippet, "<finetune snippet>", "exec")
    assert "'exist_ok': True," in snippet
    assert "true" not in snippet and "false" not in snippet


def test_missing_dataset_yaml_is_reported(cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="split_dataset"):
        build_training_plan(cfg, str(tmp_path / "nope" / "data.yaml"))


def test_plan_warns_when_real_photos_are_absent(cfg, tmp_path):
    dataset_yaml = _dataset_dir(
        tmp_path,
        sampling={"real_images": 0, "sim_images": 500, "reachable": False,
                  "target_real_fraction": 0.6, "real_fraction_after": 0.0},
        written={"train": 400, "val": 100},
    )
    plan = build_training_plan(cfg, dataset_yaml)
    assert any("no real photographs" in warning for warning in plan.warnings)


def test_plan_warns_when_the_target_is_unreachable(cfg, tmp_path):
    dataset_yaml = _dataset_dir(
        tmp_path,
        sampling={"real_images": 20, "sim_images": 500, "reachable": False,
                  "target_real_fraction": 0.6, "real_fraction_after": 0.25},
        written={"train": 500, "val": 100},
    )
    plan = build_training_plan(cfg, dataset_yaml)
    assert any("below the target" in warning for warning in plan.warnings)


def test_plan_warns_when_val_is_empty(cfg, tmp_path):
    dataset_yaml = _dataset_dir(
        tmp_path,
        sampling={"real_images": 100, "sim_images": 100, "reachable": True,
                  "target_real_fraction": 0.6, "real_fraction_after": 0.62},
        written={"train": 200, "val": 0},
    )
    plan = build_training_plan(cfg, dataset_yaml)
    assert any("validation split is empty" in warning for warning in plan.warnings)


def test_plan_is_clean_when_everything_is_in_order(cfg, tmp_path):
    dataset_yaml = _dataset_dir(
        tmp_path,
        sampling={"real_images": 200, "sim_images": 100, "reachable": True,
                  "target_real_fraction": 0.6, "real_fraction_after": 0.62},
        written={"train": 300, "val": 80},
    )
    plan = build_training_plan(cfg, dataset_yaml)
    assert plan.warnings == []
    assert plan.sampling["real_fraction_after"] == pytest.approx(0.62)
