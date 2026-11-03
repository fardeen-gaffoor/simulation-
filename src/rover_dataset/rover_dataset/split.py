"""Train/val split by scene, and real-photo weighting.

**Why by scene.**  Frames from one scene share the lighting, the sand colour, the
camera-mount jitter and (very often) the same obstacles seen from nearby poses.
A frame-level split would put near-duplicates in both halves and report a
validation score that means nothing.  A scene is therefore the atomic unit of
the split, and the split is deterministic from ``split.seed``.

**Why weighting.**  The brief is explicit: real photographs must carry most of
the weight in the final model.  Ultralytics has no per-sample loss weights, so
extra copies (symlinks by default) of the real training images are materialised
until the real images are at least ``sampling.target_real_fraction`` of the
effective training set.  The plan is computed, reported and capped -- never
guessed at training time.
"""

from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .config import DatasetConfig, SplitCfg
from .dataset import (
    SPLITS,
    DatasetLayout,
    RawLayout,
    finalize_dataset,
    raw_stems,
    scene_of_stem,
)
from .seeding import rng_python

__all__ = [
    "SplitResult",
    "split_scenes",
    "sampling_plan",
    "build_split",
    "clear_split_dirs",
]


@dataclass
class SplitResult:
    scene_split: Dict[str, str] = field(default_factory=dict)
    sampling: Dict[str, Any] = field(default_factory=dict)
    written: Dict[str, int] = field(default_factory=dict)
    scenes: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scene_split": self.scene_split,
            "sampling": self.sampling,
            "written": self.written,
        }


def split_scenes(
    scene_ids: Sequence[str],
    sources: Dict[str, str],
    cfg: SplitCfg,
) -> Dict[str, str]:
    """Assign whole scenes to train/val, deterministically."""
    unique = sorted(set(scene_ids))
    if not unique:
        return {}
    rng = rng_python(cfg.seed, cfg.seed_namespace)
    if cfg.stratify_by_source:
        groups: Dict[str, List[str]] = {}
        for scene_id in unique:
            groups.setdefault(sources.get(scene_id, "sim"), []).append(scene_id)
    else:
        groups = {"all": list(unique)}

    result: Dict[str, str] = {}
    for source in sorted(groups):
        ids = sorted(groups[source])
        shuffled = ids[:]
        rng.shuffle(shuffled)
        n_val = int(round(len(ids) * cfg.val_fraction))
        if len(ids) >= 2:
            n_val = max(1, min(len(ids) - 1, n_val))
        else:
            n_val = 0
        for index, scene_id in enumerate(shuffled):
            result[scene_id] = "val" if index < n_val else "train"
    return result


def sampling_plan(train_counts: Dict[str, int], cfg: Any) -> Dict[str, Any]:
    """How many copies of each real training image are needed."""
    sim = int(train_counts.get("sim", 0))
    real = int(train_counts.get("real", 0))
    sim_repeat = int(cfg.sim_repeat)
    target = float(cfg.target_real_fraction)
    sim_effective = sim * sim_repeat
    if real <= 0 or target <= 0.0:
        return {
            "target_real_fraction": target,
            "sim_repeat": sim_repeat,
            "real_repeat": 1,
            "real_fraction_before": 0.0 if real <= 0 else 1.0,
            "real_fraction_after": 0.0 if real <= 0 else 1.0,
            "reachable": real > 0 or target <= 0.0,
            "sim_images": sim,
            "real_images": real,
        }
    before = real / (sim_effective + real) if (sim_effective + real) else 0.0
    needed = (target * sim_effective) / (real * (1.0 - target))
    real_repeat = max(1, int(math.ceil(needed - 1e-9)))
    capped = real_repeat > int(cfg.max_repeat)
    real_repeat = min(real_repeat, int(cfg.max_repeat))
    after = (real * real_repeat) / (sim_effective + real * real_repeat)
    return {
        "target_real_fraction": target,
        "sim_repeat": sim_repeat,
        "real_repeat": real_repeat,
        "real_fraction_before": before,
        "real_fraction_after": after,
        "reachable": after + 1e-9 >= target,
        "capped_by_max_repeat": capped,
        "sim_images": sim,
        "real_images": real,
    }


def clear_split_dirs(layout: DatasetLayout) -> None:
    """Remove the materialised split dirs so a rebuild is idempotent."""
    for directory in (layout.images_dir, layout.labels_dir, layout.meta_dir):
        for split in SPLITS:
            target = os.path.join(directory, split)
            if os.path.isdir(target):
                shutil.rmtree(target)
    layout.ensure_dirs()


def _link_or_copy(source: str, destination: str, use_links: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    if os.path.lexists(destination):
        os.remove(destination)
    if use_links:
        os.symlink(os.path.relpath(source, os.path.dirname(os.path.abspath(destination))), destination)
    else:
        shutil.copy2(source, destination)


def _stem_variants(stem: str, repeat: int) -> List[str]:
    if repeat <= 1:
        return [stem]
    return [stem] + [f"{stem}__r{k}" for k in range(2, repeat + 1)]


def build_split(
    cfg: DatasetConfig,
    dataset_root: str,
    scenes: Sequence[Dict[str, Any]],
    *,
    use_links: bool = True,
    classes_file: Optional[str] = None,
) -> SplitResult:
    """Materialise ``images/{train,val}`` from the raw staging area.

    ``scenes`` is the list of scene records produced by collection/import; each
    needs ``scene_id``, ``source`` and ``images``.
    """
    layout = DatasetLayout(
        root=dataset_root,
        image_format=cfg.paths.image_format,
        image_quality=cfg.paths.image_quality,
        write_sidecar=cfg.paths.write_sidecar,
        classes_file=classes_file or cfg.paths.classes_file,
        data_yaml=cfg.paths.data_yaml,
    )
    raw = RawLayout(
        root=dataset_root,
        image_format=cfg.paths.image_format,
        image_quality=cfg.paths.image_quality,
        write_sidecar=cfg.paths.write_sidecar,
    )
    stems = raw_stems(raw)
    sources = {str(s["scene_id"]): str(s.get("source", "sim")) for s in scenes}
    scene_ids = sorted({scene_of_stem(stem) for stem in stems} | set(sources))
    scene_split = split_scenes(scene_ids, sources, cfg.split)

    per_scene_stems: Dict[str, List[str]] = {}
    for stem in stems:
        per_scene_stems.setdefault(scene_of_stem(stem), []).append(stem)

    train_counts: Dict[str, int] = {}
    for scene_id, split in scene_split.items():
        if split != "train":
            continue
        source = sources.get(scene_id, "sim")
        train_counts[source] = train_counts.get(source, 0) + len(per_scene_stems.get(scene_id, []))
    plan = sampling_plan(train_counts, cfg.sampling)

    clear_split_dirs(layout)
    written = {"train": 0, "val": 0}
    per_scene_written: Dict[str, int] = {}
    for scene_id in scene_ids:
        split = scene_split.get(scene_id, "train")
        source = sources.get(scene_id, "sim")
        repeat = 1
        if split == "train":
            repeat = int(plan["real_repeat"]) if source == "real" else int(plan["sim_repeat"])
        for stem in per_scene_stems.get(scene_id, []):
            for variant in _stem_variants(stem, repeat):
                _link_or_copy(
                    raw.image_path(split, stem), layout.image_path(split, variant), use_links
                )
                _link_or_copy(
                    raw.label_path(split, stem), layout.label_path(split, variant), use_links
                )
                if layout.write_sidecar and os.path.isfile(raw.meta_path(split, stem)):
                    _link_or_copy(
                        raw.meta_path(split, stem), layout.meta_path(split, variant), use_links
                    )
                written[split] += 1
                per_scene_written[scene_id] = per_scene_written.get(scene_id, 0) + 1

    scene_records: List[Dict[str, Any]] = []
    for scene in scenes:
        record = dict(scene)
        record["split"] = scene_split.get(str(record["scene_id"]), "train")
        record["images_written"] = per_scene_written.get(str(record["scene_id"]), 0)
        scene_records.append(record)

    split_plan = {
        "schema_version": 1,
        "dataset_config_sha256": cfg.config_sha256(),
        "val_fraction": cfg.split.val_fraction,
        "split_seed": cfg.split.seed,
        "stratify_by_source": cfg.split.stratify_by_source,
        "scene_split": scene_split,
        "sampling": plan,
        "staged_images": len(stems),
        "written": written,
        "use_links": use_links,
    }
    finalize_dataset(layout, cfg.class_names, scene_records, split_plan)
    return SplitResult(scene_split, plan, written, scene_records)
