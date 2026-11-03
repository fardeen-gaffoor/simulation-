"""Contact sheets: dataset frames with their labels drawn on.

The projection is the correctness heart of this package, and the only way to be
sure a label sits on the object is to look at it.  This module builds a single
grid image out of evenly spaced samples of a dataset so ``preview_labels.py`` can
answer "are the boxes on the rocks?" in one glance instead of twenty `cat`
commands.

Deterministic by construction: the sample is taken by even stride over the
sorted stems, never with a random number generator, so the same command always
shows the same sheet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .labels import YoloBox, read_labels

__all__ = [
    "IMAGE_SUFFIXES",
    "BOX_COLOURS",
    "PreviewResult",
    "pick_stems",
    "draw_boxes",
    "build_contact_sheet",
]

#: Image extensions a built dataset can contain.
IMAGE_SUFFIXES: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

#: One drawing colour per class index (cycling).  Index 0 is the boulder.
BOX_COLOURS: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),
    (255, 0, 255),
    (0, 176, 255),
    (255, 160, 0),
    (255, 255, 0),
    (0, 255, 255),
)

_BACKGROUND = (28, 28, 32)
_CAPTION = (235, 235, 235)


@dataclass
class PreviewResult:
    """What :func:`build_contact_sheet` produced."""

    output: str = ""
    cells: int = 0
    images: int = 0
    boxes: int = 0
    empty_images: int = 0
    per_class: Dict[str, int] = field(default_factory=dict)
    per_split: Dict[str, int] = field(default_factory=dict)
    missing_labels: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "output": self.output,
            "cells": self.cells,
            "images": self.images,
            "boxes": self.boxes,
            "empty_images": self.empty_images,
            "per_class": self.per_class,
            "per_split": self.per_split,
            "missing_labels": self.missing_labels,
        }


def _image_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        name
        for name in os.listdir(directory)
        if os.path.splitext(name)[1] in IMAGE_SUFFIXES
        and os.path.isfile(os.path.join(directory, name))
    )


def available_splits(root: str) -> List[str]:
    """The splits that actually contain images, in a stable order."""
    out = []
    for split in ("train", "val"):
        if _image_files(os.path.join(root, "images", split)):
            out.append(split)
    return out


def pick_stems(entries: Sequence[Any], count: int) -> List[Any]:
    """Evenly spaced sample of ``entries`` (deterministic, no RNG).

    Fewer than ``count`` entries returns all of them; ``count <= 0`` returns
    none.  Spreading the sample by stride is deliberate: adjacent frames from one
    pose look almost identical, so a contiguous slice would show one scene and
    hide every problem in the others.
    """
    total = len(entries)
    if count <= 0 or total == 0:
        return []
    if count >= total:
        return list(entries)
    step = total / float(count)
    picked: List[Any] = []
    seen = set()
    for index in range(count):
        position = min(total - 1, int(index * step))
        if position not in seen:
            seen.add(position)
            picked.append(entries[position])
    return picked


def draw_boxes(image: Image.Image, boxes: Sequence[YoloBox], width: int = 2) -> Image.Image:
    """A copy of ``image`` with every box outlined in its class colour."""
    out = image.copy()
    draw = ImageDraw.Draw(out)
    image_width, image_height = out.size
    for box in boxes:
        x0, y0, x1, y1 = box.pixels(image_width, image_height)
        colour = BOX_COLOURS[box.class_id % len(BOX_COLOURS)]
        # clamp so a box touching the edge still draws on-canvas
        x0 = max(0.0, min(float(image_width - 1), x0))
        x1 = max(0.0, min(float(image_width - 1), x1))
        y0 = max(0.0, min(float(image_height - 1), y0))
        y1 = max(0.0, min(float(image_height - 1), y1))
        draw.rectangle([x0, y0, x1, y1], outline=colour, width=max(1, width))
    return out


def build_contact_sheet(
    root: str,
    *,
    count: int = 9,
    columns: int = 3,
    cell_px: int = 320,
    output: Optional[str] = None,
    classes: Optional[Sequence[str]] = None,
    splits: Optional[Sequence[str]] = None,
) -> PreviewResult:
    """Draw ``count`` labelled frames from ``root`` into one grid image."""
    from .dataset import load_image

    result = PreviewResult()
    class_names = list(classes or [])
    use_splits = list(splits) if splits else available_splits(root)

    entries: List[Tuple[str, str]] = []
    for split in use_splits:
        for name in _image_files(os.path.join(root, "images", split)):
            entries.append((split, name))
    if not entries:
        raise FileNotFoundError(
            f"no images under {os.path.join(root, 'images')}; run collect_dataset.py or "
            "split_dataset.py first"
        )

    chosen = pick_stems(entries, int(count))
    columns = max(1, int(columns))
    rows = max(1, (len(chosen) + columns - 1) // columns)
    cell = max(64, int(cell_px))

    sheet = Image.new("RGB", (columns * cell, rows * cell), _BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    for index, (split, name) in enumerate(chosen):
        stem = os.path.splitext(name)[0]
        column, row = index % columns, index // columns
        origin_x, origin_y = column * cell, row * cell
        image_path = os.path.join(root, "images", split, name)
        image = Image.fromarray(load_image(image_path))
        if image.mode != "RGB":
            image = image.convert("RGB")

        label_path = os.path.join(root, "labels", split, f"{stem}.txt")
        boxes: List[YoloBox] = []
        if os.path.isfile(label_path):
            boxes = read_labels(label_path)
        else:
            result.missing_labels.append(f"{split}/{stem}")

        box_width = max(1, int(round(cell / 160.0)))
        cell_image = draw_boxes(image, boxes, width=box_width)
        cell_image.thumbnail((cell, cell), Image.BILINEAR)
        paste_x = origin_x + (cell - cell_image.width) // 2
        paste_y = origin_y + (cell - cell_image.height) // 2
        sheet.paste(cell_image, (paste_x, paste_y))

        counts: Dict[int, int] = {}
        for box in boxes:
            counts[box.class_id] = counts.get(box.class_id, 0) + 1
        parts = [
            f"{class_names[class_id] if class_id < len(class_names) else class_id}"
            f"x{number}"
            for class_id, number in sorted(counts.items())
        ]
        caption = f"{split}/{stem}  {' '.join(parts) if parts else 'background'}"
        draw.rectangle(
            [origin_x, origin_y, origin_x + cell - 1, origin_y + 13], fill=_BACKGROUND
        )
        draw.text((origin_x + 3, origin_y + 2), caption[:48], fill=_CAPTION)

        result.cells += 1
        result.images += 1
        result.boxes += len(boxes)
        result.empty_images += 0 if boxes else 1
        result.per_split[split] = result.per_split.get(split, 0) + 1
        for class_id, number in counts.items():
            name = class_names[class_id] if class_id < len(class_names) else str(class_id)
            result.per_class[name] = result.per_class.get(name, 0) + number

    destination = output or os.path.join(root, "preview", "grid.png")
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    sheet.save(destination)
    result.output = os.path.abspath(destination)
    return result
