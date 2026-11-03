"""Draw the AprilTag bitmaps that module 1's marker boards hold.

Module 1 owns the *boards* (three placements, all visual-only) and hands module 4
their exact poses in ``markers.yaml``.  Module 4 owns the *picture* on them.
That split only works if the picture is a real, decodable tag: the simulation
cannot demonstrate tag localisation with a flat black square, and a bitmap that
is mirrored or rotated reads as a different id - or as nothing.

So the bitmap is built from first principles out of :mod:`tagcodes`:

* module grid = ``quiet_zone_modules`` of white, then the black square, then the
  same margin again.  The black square is ``width_at_border`` modules on a side
  (8 for tag36h11) and contains a 1-module black border around a 6x6 payload.
* a payload bit is white where the code has a 1 and stays black where it has a 0,
  at exactly the ``(x, y)`` :func:`tagcodes.bits_for` reports.
* the drawing is scaled by an **integer** number of pixels per module, so no
  resampling ever blurs a bit boundary.

The physical size follows the same proportions: the *black square* edge is the
``tag_size`` module 1 reports (0.18 m), so the whole bitmap is
``tag_size * total_modules / 8`` metres across.

``test/test_tagimages.py`` renders ids 0-4 and requires the real
``dt_apriltags`` detector to read the same id back, which is what makes the
table, the layout and the detector agree (see docs/tags.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tagcodes import Family, bits_for, family as family_of

__all__ = [
    "TagImageError",
    "RenderedTag",
    "module_grid",
    "render_tag",
    "render",
    "drawing_size_m",
    "total_modules",
    "png_name",
    "write_png",
    "contact_sheet",
    "BLACK",
    "WHITE",
]

BLACK = 0
WHITE = 255


class TagImageError(ValueError):
    """Raised when a tag id cannot be drawn at the requested size."""


@dataclass(frozen=True)
class RenderedTag:
    """One rendered tag bitmap plus everything needed to place it in the world."""

    tag_id: int
    family: str
    #: (rows, cols) uint8 image, 0 = black, 255 = white
    array: np.ndarray
    pixels_per_module: int
    quiet_zone_modules: int
    #: edge length of the whole bitmap in metres (not just the black square)
    drawing_size_m: float

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.array.shape[0]), int(self.array.shape[1]))

    @property
    def modules(self) -> int:
        return int(self.array.shape[0]) // self.pixels_per_module

    def summary(self) -> Dict[str, object]:
        rows, cols = self.shape
        return {
            "id": self.tag_id,
            "family": self.family,
            "pixels": [cols, rows],
            "pixels_per_module": self.pixels_per_module,
            "modules": self.modules,
            "quiet_zone_modules": self.quiet_zone_modules,
            "drawing_size_m": round(self.drawing_size_m, 6),
        }


def total_modules(member: Family, quiet_zone_modules: int = 1) -> int:
    """Side length of the bitmap in modules, quiet zone included."""
    if quiet_zone_modules < 1:
        raise TagImageError("quiet_zone_modules must be >= 1")
    return member.width_at_border + 2 * int(quiet_zone_modules)


def module_grid(
    tag_id: int,
    family_name: str = "tag36h11",
    quiet_zone_modules: int = 1,
) -> np.ndarray:
    """The un-scaled module grid: ``total_modules`` square, uint8 0/255.

    Row 0 is the top of the tag as printed; the payload origin is the top-left
    corner of the 6x6 data grid, so no flipping happens here at all.
    """
    member = family_of(family_name)
    quiet = int(quiet_zone_modules)
    size = total_modules(member, quiet)
    grid = np.full((size, size), BLACK, dtype=np.uint8)
    grid[:quiet, :] = WHITE
    grid[-quiet:, :] = WHITE
    grid[:, :quiet] = WHITE
    grid[:, -quiet:] = WHITE
    for x, y, value in bits_for(tag_id, family_name):
        # payload coordinates are inside the black square (its own 1-module
        # border included), so they are already offset from module 0
        row = quiet + y
        col = quiet + x
        if not (0 <= row < size and 0 <= col < size):
            raise TagImageError(
                f"{family_name} id {tag_id}: payload bit ({x}, {y}) falls outside "
                f"the {size}-module drawing"
            )
        grid[row, col] = WHITE if value else BLACK
    return grid


def render_tag(
    tag_id: int,
    family_name: str = "tag36h11",
    pixels_per_module: int = 20,
    quiet_zone_modules: int = 1,
) -> np.ndarray:
    """Render a tag as a square uint8 image with crisp module boundaries."""
    if pixels_per_module < 1:
        raise TagImageError("pixels_per_module must be >= 1")
    grid = module_grid(tag_id, family_name, quiet_zone_modules)
    return np.kron(grid, np.ones((pixels_per_module, pixels_per_module), dtype=np.uint8))


def drawing_size_m(
    tag_size_m: float,
    family_name: str = "tag36h11",
    quiet_zone_modules: int = 1,
) -> float:
    """Physical width of the whole bitmap for a given black-square size."""
    member = family_of(family_name)
    if tag_size_m <= 0.0:
        raise TagImageError(f"tag_size_m must be > 0, got {tag_size_m}")
    return float(tag_size_m) * total_modules(member, quiet_zone_modules) / member.width_at_border


def render(
    tag_id: int,
    family_name: str = "tag36h11",
    tag_size_m: float = 0.18,
    texture_size_px: int = 512,
    quiet_zone_modules: int = 1,
) -> RenderedTag:
    """Render a tag sized for a physical board of ``tag_size_m`` black square.

    ``texture_size_px`` is an upper bound: the drawing is the largest integer
    number of pixels per module that fits (510 px for 512 requested and a
    10-module tag), so the bitmap is never resampled.
    """
    if texture_size_px < 16:
        raise TagImageError(f"texture_size_px {texture_size_px} is too small to hold a tag")
    member = family_of(family_name)
    modules = total_modules(member, quiet_zone_modules)
    pixels_per_module = max(1, int(texture_size_px) // modules)
    array = render_tag(tag_id, family_name, pixels_per_module, quiet_zone_modules)
    return RenderedTag(
        tag_id=int(tag_id),
        family=family_name,
        array=array,
        pixels_per_module=pixels_per_module,
        quiet_zone_modules=int(quiet_zone_modules),
        drawing_size_m=drawing_size_m(tag_size_m, family_name, quiet_zone_modules),
    )


def png_name(tag_id: int, family_name: str = "tag36h11") -> str:
    """File name convention module 4 writes and its SDFs reference."""
    return f"tag_{int(tag_id)}_{family_name}.png"


def write_png(path: str, array: np.ndarray) -> str:
    """Write a greyscale PNG (Pillow; the same image library module 3 uses)."""
    import os

    from PIL import Image

    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L")
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    image.save(path, format="PNG", optimize=True)
    return path


def contact_sheet(
    rendered: Sequence[RenderedTag],
    columns: int = 3,
    pad_px: int = 16,
    background: int = 128,
) -> np.ndarray:
    """Lay tags out in a grid with a mid-grey separator (preview tool)."""
    if not rendered:
        raise TagImageError("contact_sheet needs at least one rendered tag")
    columns = max(1, int(columns))
    cell = max(tag.array.shape[0] for tag in rendered)
    cell = max(cell, 1)
    rows = (len(rendered) + columns - 1) // columns
    width = columns * cell + (columns + 1) * pad_px
    height = rows * cell + (rows + 1) * pad_px
    canvas = np.full((height, width), int(background), dtype=np.uint8)
    for index, tag in enumerate(rendered):
        row = index // columns
        column = index % columns
        top = pad_px + row * (cell + pad_px)
        left = pad_px + column * (cell + pad_px)
        rows_px, cols_px = tag.array.shape
        canvas[top : top + rows_px, left : left + cols_px] = tag.array
    return canvas


def sheet_from_ids(
    tag_ids: Sequence[int],
    family_name: str = "tag36h11",
    tag_size_m: float = 0.18,
    texture_size_px: int = 512,
    quiet_zone_modules: int = 1,
    columns: int = 3,
) -> Tuple[np.ndarray, List[RenderedTag]]:
    """Render several ids and lay them out; returns ``(image, rendered)``."""
    rendered = [
        render(tag_id, family_name, tag_size_m, texture_size_px, quiet_zone_modules)
        for tag_id in tag_ids
    ]
    return contact_sheet(rendered, columns=columns), rendered
