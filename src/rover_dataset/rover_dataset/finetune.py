"""Fine-tuning recipe: turn ``dataset.yaml`` + the built dataset into a command.

The script never guesses: it reads the split plan next to the dataset, warns
loudly when the real-photo weighting target was not reached, and refuses to run
if ``ultralytics`` is not installed.  ``--print-only`` (the default in the
script wrapper) emits the exact command and a Python snippet without touching
the GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .config import DatasetConfig

__all__ = [
    "TrainingPlan",
    "training_arguments",
    "yolo_cli_command",
    "python_snippet",
    "build_training_plan",
    "ultralytics_available",
    "run_training",
]


@dataclass
class TrainingPlan:
    arguments: Dict[str, Any] = field(default_factory=dict)
    command: List[str] = field(default_factory=list)
    snippet: str = ""
    dataset_yaml: str = ""
    sampling: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "arguments": self.arguments,
            "command": self.command,
            "dataset_yaml": self.dataset_yaml,
            "sampling": self.sampling,
            "warnings": self.warnings,
        }


def training_arguments(cfg: DatasetConfig, dataset_yaml: str) -> Dict[str, Any]:
    """The ultralytics ``train`` keyword arguments in one place."""
    ft = cfg.finetune
    args: Dict[str, Any] = {
        "data": os.path.abspath(dataset_yaml),
        "model": ft.base_model,
        "imgsz": ft.imgsz,
        "epochs": ft.epochs,
        "batch": ft.batch,
        "patience": ft.patience,
        "workers": ft.workers,
        "project": ft.project,
        "name": ft.name,
        "seed": ft.seed,
        "exist_ok": True,
    }
    if ft.device:
        args["device"] = ft.device
    if ft.freeze:
        args["freeze"] = ft.freeze
    return args


def yolo_cli_command(cfg: DatasetConfig, dataset_yaml: str) -> List[str]:
    args = training_arguments(cfg, dataset_yaml)
    command = ["yolo", "detect", "train"]
    for key, value in args.items():
        command.append(f"{key}={value}")
    return command


def _python_literal(value: Any) -> str:
    """``repr`` is enough for the scalar types a training config can hold."""
    return repr(value)


def python_snippet(cfg: DatasetConfig, dataset_yaml: str) -> str:
    """A runnable Python equivalent of :func:`yolo_cli_command`.

    Hand-formatted (not ``json``/``pprint``) so ``True``/``False`` come out as
    Python literals and the whole thing is valid, pasteable Python.
    """
    args = training_arguments(cfg, dataset_yaml)
    body = "\n".join(
        f"    {key!r}: {_python_literal(args[key])}," for key in sorted(args)
    )
    lines = [
        "from ultralytics import YOLO",
        "",
        f"model = YOLO({cfg.finetune.base_model!r})",
        "train_args = {",
        body,
        "}",
        "results = model.train(**train_args)",
        "print(results.save_dir)",
    ]
    return "\n".join(lines) + "\n"


def _load_split_plan(dataset_yaml: str) -> Dict[str, Any]:
    root = os.path.dirname(os.path.abspath(dataset_yaml))
    path = os.path.join(root, "split.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_training_plan(cfg: DatasetConfig, dataset_yaml: str) -> TrainingPlan:
    """Assemble the plan and audit the real-photo weighting."""
    if not os.path.isfile(dataset_yaml):
        raise FileNotFoundError(
            f"{dataset_yaml} not found; run 'ros2 run rover_dataset split_dataset.py' first"
        )
    plan = TrainingPlan(
        arguments=training_arguments(cfg, dataset_yaml),
        command=yolo_cli_command(cfg, dataset_yaml),
        snippet=python_snippet(cfg, dataset_yaml),
        dataset_yaml=os.path.abspath(dataset_yaml),
    )
    split = _load_split_plan(dataset_yaml)
    plan.sampling = dict(split.get("sampling") or {})
    if not plan.sampling:
        plan.warnings.append("no split.json next to the dataset: run split_dataset.py")
        return plan
    if plan.sampling.get("real_images", 0) == 0:
        plan.warnings.append(
            "no real photographs in the training set: import them with import_real_photos.py "
            "before trusting the model (the brief requires real photos to dominate)"
        )
    elif not plan.sampling.get("reachable", True):
        plan.warnings.append(
            "real-photo fraction "
            f"{plan.sampling.get('real_fraction_after', 0.0):.2f} is below the target "
            f"{plan.sampling.get('target_real_fraction', 0.0):.2f} even at max_repeat="
            f"{cfg.sampling.max_repeat}; add more real photos"
        )
    written = dict(split.get("written") or {})
    if written.get("val", 0) == 0:
        plan.warnings.append("the validation split is empty; lower split.val_fraction or add scenes")
    return plan


def ultralytics_available() -> Tuple[bool, str]:
    """``(available, version_or_reason)`` without importing torch eagerly."""
    try:
        import ultralytics  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(ultralytics, "__version__", "unknown"))


def run_training(cfg: DatasetConfig, dataset_yaml: str, stream=None) -> int:
    """Run the fine-tune.  Requires ``ultralytics`` (pip install ultralytics)."""
    available, detail = ultralytics_available()
    if not available:
        print(
            "ultralytics is not available "
            f"({detail}).\nInstall it first:  pip install --user ultralytics\n"
            "Then re-run this command.",
            file=stream or sys.stderr,
        )
        return 2
    plan = build_training_plan(cfg, dataset_yaml)
    for warning in plan.warnings:
        print(f"[finetune] WARNING: {warning}", file=stream or sys.stderr)
    from ultralytics import YOLO

    model = YOLO(cfg.finetune.base_model)
    results = model.train(**plan.arguments)
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        print(f"[finetune] results: {save_dir}")
    return 0


def yolo_binary() -> Optional[str]:
    return shutil.which("yolo")
