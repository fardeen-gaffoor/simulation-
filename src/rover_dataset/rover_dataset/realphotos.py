"""Merge real CSI-camera photographs into the dataset.

Real photos are the part of the dataset that actually decides whether the
detector works on the rover: the simulator cannot reproduce the real lens, the
real sand, or the real lighting.  This module imports a *labelled* batch of real
photos and gives it its own scene id, so:

* it participates in the scene-level split like any other scene,
* it can be oversampled (see :mod:`rover_dataset.split`) so real photos carry
  most of the weight,
* it is stamped with the camera it was taken with and the height it was taken
  from, so a mismatch with the module-2 calibration is caught immediately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .calibration_ref import CalibrationDoc, CameraModel, load_calibration
from .config import DatasetConfig
from .dataset import RawLayout, image_stem, load_image, save_image
from .labels import LabelError, read_labels, validate_box, write_labels

__all__ = ["ImportResult", "import_real_photos", "REAL_PHOTO_SUFFIXES"]

REAL_PHOTO_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

#: Warn when a stated mounting height differs from the calibration by this much.
_HEIGHT_TOLERANCE_M = 0.03


@dataclass
class ImportResult:
    scene_id: str
    camera: str
    imported: int = 0
    skipped: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    mount_height_m: Optional[float] = None
    calibration_height_m: Optional[float] = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_scene_record(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "source": "real",
            "camera": self.camera,
            "images": self.imported,
            "mount_height_m": self.mount_height_m,
            "calibration_height_m": self.calibration_height_m,
        }


def _find_pairs(photos_dir: str, labels_dir: Optional[str]) -> List[Tuple[str, str, str]]:
    """Return ``(stem, image_path, label_path)`` for every labelled photo."""
    pairs: List[Tuple[str, str, str]] = []
    for name in sorted(os.listdir(photos_dir)):
        root, ext = os.path.splitext(name)
        if ext not in REAL_PHOTO_SUFFIXES:
            continue
        image_path = os.path.join(photos_dir, name)
        label_dir = labels_dir or photos_dir
        label_path = os.path.join(label_dir, f"{root}.txt")
        pairs.append((root, image_path, label_path))
    return pairs


def import_real_photos(
    cfg: DatasetConfig,
    dataset_root: str,
    photos_dir: str,
    labels_dir: Optional[str],
    batch: str,
    camera: str,
    *,
    mount_height_m: Optional[float] = None,
    calibration: Optional[CalibrationDoc] = None,
    calibration_path: Optional[str] = None,
    unlabeled_as_negative: bool = False,
) -> ImportResult:
    """Import and validate one labelled batch of real photographs.

    A batch with *any* problem is not safe to train on: the caller
    (``import_real_photos.py``) refuses to register it until it is fixed.
    """
    scene_id = f"real_{batch}"
    result = ImportResult(scene_id=scene_id, camera=camera)

    if not os.path.isdir(photos_dir):
        result.problems.append(f"photos directory not found: {photos_dir}")
        return result
    calib = calibration or load_calibration(calibration_path)
    if camera not in calib.cameras:
        result.problems.append(
            f"camera '{camera}' is not in the calibration ({sorted(calib.cameras)})"
        )
        return result
    cam: CameraModel = calib.camera(camera)
    result.calibration_height_m = float(cam.t_base_cam[2])
    result.mount_height_m = mount_height_m
    if mount_height_m is not None:
        delta = abs(float(mount_height_m) - result.calibration_height_m)
        if delta > _HEIGHT_TOLERANCE_M:
            result.problems.append(
                f"stated mount height {mount_height_m:.3f} m differs from the calibration "
                f"({result.calibration_height_m:.3f} m) by {delta:.3f} m; update rover.yaml "
                "and regenerate calibration.yaml, or the ground projection will be wrong"
            )

    raw = RawLayout(
        root=dataset_root,
        image_format=cfg.paths.image_format,
        image_quality=cfg.paths.image_quality,
        write_sidecar=cfg.paths.write_sidecar,
    )
    raw.ensure_dirs()

    pairs = _find_pairs(photos_dir, labels_dir)
    if not pairs:
        result.problems.append(f"no photographs found in {photos_dir}")
        return result

    for index, (stem, image_path, label_path) in enumerate(pairs):
        if not os.path.isfile(label_path):
            if unlabeled_as_negative:
                boxes = []
            else:
                result.skipped.append(f"{stem}: no label file")
                continue
        else:
            try:
                boxes = read_labels(label_path)
            except LabelError as exc:
                result.problems.append(str(exc))
                continue
        try:
            image = load_image(image_path)
        except Exception as exc:  # pragma: no cover - unreadable image
            result.problems.append(f"{stem}: cannot read image ({exc})")
            continue
        height, width = int(image.shape[0]), int(image.shape[1])
        for box_index, box in enumerate(boxes):
            if not 0 <= box.class_id < len(cfg.classes):
                result.problems.append(
                    f"{stem}[{box_index}]: class_id {box.class_id} outside 0..{len(cfg.classes) - 1}"
                )
                continue
            try:
                validate_box(box, width, height)
            except LabelError as exc:
                result.problems.append(f"{stem}[{box_index}]: {exc}")

        out_stem = image_stem(scene_id, camera, 0, index)
        save_image(raw.image_path("train", out_stem), image, cfg.paths.image_format,
                   cfg.paths.image_quality)
        write_labels(raw.label_path("train", out_stem), boxes)
        if raw.write_sidecar:
            from .labels import boxes_to_sidecar

            boxes_to_sidecar(
                raw.meta_path("train", out_stem),
                boxes,
                [],
                {
                    "scene_id": scene_id,
                    "source": "real",
                    "camera": camera,
                    "original": os.path.basename(image_path),
                    "pose_index": 0,
                    "frame_index": index,
                    "image_width": width,
                    "image_height": height,
                    "mount_height_m": mount_height_m,
                },
            )
        result.imported += 1

    if result.imported == 0:
        result.problems.append("nothing was imported (see skipped)")
    return result
