"""Dataset layout, sample writing and verification.

Layout on disk (YOLO-compatible)::

    <root>/
      images/train/<stem>.<ext>      labels/train/<stem>.txt
      images/val/<stem>.<ext>        labels/val/<stem>.txt
      meta/train/<stem>.json         meta/val/<stem>.json      (optional sidecar)
      classes.txt   data.yaml   scenes.json   split.json

``<stem>`` is ``<scene_id>__<camera>__p<pose>__f<frame>``, so a frame can always
be traced back to the scene, camera, pose and augmentation draw that produced
it -- and the scene can always be traced back to its seed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .labels import (
    LabelError,
    YoloBox,
    boxes_to_sidecar,
    read_labels,
    validate_box,
    write_classes,
    write_data_yaml,
    write_labels,
)

__all__ = [
    "SPLITS",
    "DatasetLayout",
    "VerificationReport",
    "image_stem",
    "save_image",
    "load_image",
    "write_sample",
    "finalize_dataset",
    "dataset_stats",
    "verify_dataset",
    "scene_of_stem",
]

SPLITS: Tuple[str, str] = ("train", "val")


@dataclass(frozen=True)
class DatasetLayout:
    root: str
    image_format: str = "jpg"
    image_quality: int = 95
    write_sidecar: bool = True
    classes_file: str = "classes.txt"
    data_yaml: str = "data.yaml"

    # -- directories ------------------------------------------------------- #
    @property
    def images_dir(self) -> str:
        return os.path.join(self.root, "images")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.root, "labels")

    @property
    def meta_dir(self) -> str:
        return os.path.join(self.root, "meta")

    @property
    def classes_path(self) -> str:
        return os.path.join(self.root, self.classes_file)

    @property
    def data_yaml_path(self) -> str:
        return os.path.join(self.root, self.data_yaml)

    @property
    def scenes_path(self) -> str:
        return os.path.join(self.root, "scenes.json")

    @property
    def split_path(self) -> str:
        return os.path.join(self.root, "split.json")

    # -- file names -------------------------------------------------------- #
    def image_path(self, split: str, stem: str) -> str:
        return os.path.join(self.images_dir, split, f"{stem}.{self.image_format}")

    def label_path(self, split: str, stem: str) -> str:
        return os.path.join(self.labels_dir, split, f"{stem}.txt")

    def meta_path(self, split: str, stem: str) -> str:
        return os.path.join(self.meta_dir, split, f"{stem}.json")

    def ensure_dirs(self) -> None:
        for split in SPLITS:
            os.makedirs(os.path.join(self.images_dir, split), exist_ok=True)
            os.makedirs(os.path.join(self.labels_dir, split), exist_ok=True)
            if self.write_sidecar:
                os.makedirs(os.path.join(self.meta_dir, split), exist_ok=True)
        os.makedirs(self.root, exist_ok=True)


@dataclass(frozen=True)
class RawLayout:
    """Flat staging area used *before* the scene split is decided.

    Collection and real-photo import write here; :mod:`rover_dataset.split`
    materialises the final ``images/{train,val}`` layout from it.  The split
    cannot be decided during collection, because importing real photos later
    changes the scene set (and therefore the split).

    The method signatures match :class:`DatasetLayout` (``split`` is ignored) so
    :func:`write_sample` works with either sink.
    """

    root: str
    image_format: str = "jpg"
    image_quality: int = 95
    write_sidecar: bool = True

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, "raw")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.raw_dir, "images")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.raw_dir, "labels")

    @property
    def meta_dir(self) -> str:
        return os.path.join(self.raw_dir, "meta")

    def image_path(self, split: str, stem: str) -> str:  # noqa: ARG002 (split ignored)
        return os.path.join(self.images_dir, f"{stem}.{self.image_format}")

    def label_path(self, split: str, stem: str) -> str:  # noqa: ARG002
        return os.path.join(self.labels_dir, f"{stem}.txt")

    def meta_path(self, split: str, stem: str) -> str:  # noqa: ARG002
        return os.path.join(self.meta_dir, f"{stem}.json")

    def ensure_dirs(self) -> None:
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        if self.write_sidecar:
            os.makedirs(self.meta_dir, exist_ok=True)


def raw_stems(layout: RawLayout) -> List[str]:
    """Every staged stem (sorted), one per image."""
    if not os.path.isdir(layout.images_dir):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(layout.images_dir)
        if not name.startswith(".") and os.path.isfile(os.path.join(layout.images_dir, name))
    )


def image_stem(scene_id: str, camera: str, pose_index: int, frame_index: int) -> str:
    return f"{scene_id}__{camera}__p{pose_index:04d}__f{frame_index:02d}"


def scene_of_stem(stem: str) -> str:
    return stem.split("__", 1)[0]


# --------------------------------------------------------------------------- #
# image I/O
# --------------------------------------------------------------------------- #
def save_image(path: str, image: np.ndarray, image_format: str = "jpg", quality: int = 95) -> None:
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    im = Image.fromarray(np.asarray(image, dtype=np.uint8), "RGB")
    if image_format in ("jpg", "jpeg"):
        im.save(path, format="JPEG", quality=int(quality))
    elif image_format == "png":
        im.save(path, format="PNG")
    else:
        raise ValueError(f"unsupported image format '{image_format}'")


def load_image(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# writing samples
# --------------------------------------------------------------------------- #
def write_sample(
    layout: DatasetLayout,
    *,
    split: str,
    scene_id: str,
    camera: str,
    pose_index: int,
    frame_index: int,
    image: np.ndarray,
    boxes: Sequence[YoloBox],
    dropped: Sequence[Dict[str, Any]] = (),
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write one image + label (+ sidecar) and return the sample record."""
    if split not in SPLITS:
        raise ValueError(f"unknown split '{split}' (expected one of {SPLITS})")
    layout.ensure_dirs()
    height, width = int(image.shape[0]), int(image.shape[1])
    stem = image_stem(scene_id, camera, pose_index, frame_index)
    image_path = layout.image_path(split, stem)
    label_path = layout.label_path(split, stem)

    for box in boxes:
        validate_box(box, width, height)

    save_image(image_path, image, layout.image_format, layout.image_quality)
    write_labels(label_path, boxes)
    if layout.write_sidecar:
        payload = {
            "scene_id": scene_id,
            "camera": camera,
            "pose_index": pose_index,
            "frame_index": frame_index,
            "image_width": width,
            "image_height": height,
        }
        payload.update(meta or {})
        boxes_to_sidecar(layout.meta_path(split, stem), boxes, dropped, payload)

    return {
        "stem": stem,
        "split": split,
        "scene_id": scene_id,
        "camera": camera,
        "pose_index": pose_index,
        "frame_index": frame_index,
        "width": width,
        "height": height,
        "boxes": len(boxes),
    }


def finalize_dataset(
    layout: DatasetLayout,
    class_names: Sequence[str],
    scenes: Sequence[Dict[str, Any]],
    split_plan: Dict[str, Any],
) -> None:
    """Write ``classes.txt``, ``data.yaml``, ``scenes.json`` and ``split.json``."""
    layout.ensure_dirs()
    write_classes(layout.classes_path, class_names)
    write_data_yaml(layout.data_yaml_path, layout.root, class_names)
    with open(layout.scenes_path, "w", encoding="utf-8") as handle:
        json.dump({"schema_version": 1, "scenes": list(scenes)}, handle, indent=1, sort_keys=True)
        handle.write("\n")
    with open(layout.split_path, "w", encoding="utf-8") as handle:
        json.dump(split_plan, handle, indent=1, sort_keys=True)
        handle.write("\n")


# --------------------------------------------------------------------------- #
# statistics / verification
# --------------------------------------------------------------------------- #
def _list_split(root: str, images_dir: str, split: str) -> List[str]:
    directory = os.path.join(images_dir, split)
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(directory)
        if not name.startswith(".") and os.path.isfile(os.path.join(directory, name))
    )


def dataset_stats(layout: DatasetLayout, class_names: Sequence[str]) -> Dict[str, Any]:
    """Per-split image/box/class counts (reads only the label files)."""
    stats: Dict[str, Any] = {"class_names": list(class_names), "splits": {}}
    for split in SPLITS:
        stems = _list_split(layout.root, layout.images_dir, split)
        per_class = {name: 0 for name in class_names}
        boxes = 0
        empty = 0
        cameras: Dict[str, int] = {}
        for stem in stems:
            path = layout.label_path(split, stem)
            if not os.path.isfile(path):
                continue
            try:
                label_boxes = read_labels(path)
            except LabelError:
                continue
            boxes += len(label_boxes)
            if not label_boxes:
                empty += 1
            for box in label_boxes:
                if 0 <= box.class_id < len(class_names):
                    per_class[class_names[box.class_id]] += 1
            camera = stem.split("__")[1] if "__" in stem else "?"
            cameras[camera] = cameras.get(camera, 0) + 1
        stats["splits"][split] = {
            "images": len(stems),
            "boxes": boxes,
            "empty_images": empty,
            "boxes_per_image": (boxes / len(stems)) if stems else 0.0,
            "per_class": per_class,
            "per_camera": cameras,
        }
    return stats


@dataclass
class VerificationReport:
    ok: bool = True
    problems: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    scene_split: Dict[str, str] = field(default_factory=dict)

    def problem(self, message: str) -> None:
        self.ok = False
        self.problems.append(message)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "problems": self.problems, "stats": self.stats,
                "scene_split": self.scene_split}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(
    layout: DatasetLayout,
    class_names: Sequence[str],
    *,
    check_duplicates: bool = False,
    expected_scene_split: Optional[Dict[str, str]] = None,
) -> VerificationReport:
    """Check dataset integrity.  ``ok`` is False if anything is wrong."""
    report = VerificationReport()
    for required in (layout.classes_path, layout.data_yaml_path):
        if not os.path.isfile(required):
            report.problem(f"missing {os.path.basename(required)}")
    if os.path.isfile(layout.classes_path):
        with open(layout.classes_path, "r", encoding="utf-8") as handle:
            on_disk = [line.strip() for line in handle if line.strip()]
        if on_disk != list(class_names):
            report.problem(f"classes.txt {on_disk} != configured {list(class_names)}")

    scene_split: Dict[str, str] = {}
    hashes: Dict[str, Dict[str, str]] = {split: {} for split in SPLITS}
    sizes: Dict[str, set] = {}

    for split in SPLITS:
        stems = _list_split(layout.root, layout.images_dir, split)
        for stem in stems:
            image_path = layout.image_path(split, stem)
            label_path = layout.label_path(split, stem)
            if not os.path.isfile(label_path):
                report.problem(f"{split}/{stem}: image without a label file")
                continue
            scene_id = scene_of_stem(stem)
            previous = scene_split.get(scene_id)
            if previous is not None and previous != split:
                report.problem(f"scene {scene_id} appears in both {previous} and {split}")
            scene_split[scene_id] = split

            try:
                from PIL import Image

                with Image.open(image_path) as im:
                    width, height = im.size
                sizes.setdefault(f"{width}x{height}", set()).add(scene_id)
            except Exception as exc:  # pragma: no cover - unreadable file
                report.problem(f"{split}/{stem}: cannot read image ({exc})")
                continue

            try:
                boxes = read_labels(label_path)
            except LabelError as exc:
                report.problem(f"{split}/{stem}: {exc}")
                continue
            for index, box in enumerate(boxes):
                if not 0 <= box.class_id < len(class_names):
                    report.problem(f"{split}/{stem}[{index}]: class_id {box.class_id} out of range")
                    continue
                try:
                    validate_box(box, width, height)
                except LabelError as exc:
                    report.problem(f"{split}/{stem}[{index}]: {exc}")

            if check_duplicates:
                hashes[split][stem] = _file_sha256(image_path)

    if check_duplicates:
        common = set(hashes["train"].values()) & set(hashes["val"].values())
        if common:
            report.problem(f"{len(common)} identical image(s) present in both train and val")

    if expected_scene_split is not None:
        for scene_id, split in expected_scene_split.items():
            actual = scene_split.get(scene_id)
            if actual != split:
                report.problem(f"scene {scene_id}: expected {split}, found {actual}")

    report.scene_split = scene_split
    report.stats = dataset_stats(layout, class_names)
    report.stats["image_sizes"] = sorted(sizes.keys())
    return report
