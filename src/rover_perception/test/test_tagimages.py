"""Drawing AprilTag bitmaps from first principles - no library, no resampling.

The bitmaps module 4 spawns into Gazebo are built here, so the tests check the
geometry the whole localisation chain depends on: the black-square edge is
exactly ``tag_size``, the quiet zone is a full module of white, every module is
an integer number of pixels (so the camera sees crisp edges), and the payload is
placed at the module coordinates ``tagcodes`` declares.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from rover_perception import tagcodes
from rover_perception.tagimages import (
    BLACK,
    WHITE,
    TagImageError,
    contact_sheet,
    drawing_size_m,
    module_grid,
    png_name,
    render,
    render_tag,
    sheet_from_ids,
    total_modules,
    write_png,
)


def test_module_grid_shape_and_quiet_zone():
    member = tagcodes.family("tag36h11")
    grid = module_grid(0)
    assert grid.shape == (total_modules(member, 1), total_modules(member, 1)) == (10, 10)
    # the quiet zone is white all the way around
    assert np.all(grid[0, :] == WHITE)
    assert np.all(grid[-1, :] == WHITE)
    assert np.all(grid[:, 0] == WHITE)
    assert np.all(grid[:, -1] == WHITE)


def test_module_grid_values_are_binary():
    grid = module_grid(7)
    assert set(np.unique(grid)).issubset({BLACK, WHITE})


def test_payload_modules_match_the_code():
    """Every payload bit of the grid equals the corresponding code bit."""
    grid = module_grid(3)
    for x, y, value in tagcodes.bits_for(3):
        assert int(grid[1 + y, 1 + x]) == (WHITE if value else BLACK), (
            f"payload module ({x}, {y}) does not match the code"
        )


def test_border_of_the_black_square_is_black():
    member = tagcodes.family("tag36h11")
    grid = module_grid(5)
    quiet = 1
    border = quiet
    last = quiet + member.width_at_border - 1
    assert np.all(grid[border, border : last + 1] == BLACK)
    assert np.all(grid[last, border : last + 1] == BLACK)
    assert np.all(grid[border : last + 1, border] == BLACK)
    assert np.all(grid[border : last + 1, last] == BLACK)


def test_render_tag_scales_by_whole_pixels():
    pixels = 20
    image = render_tag(0, pixels_per_module=pixels)
    assert image.shape == (200, 200)
    assert image.dtype == np.uint8
    # every module is exactly `pixels` x `pixels`, so block edges are pixel-locked
    assert image[0, 0] == WHITE
    assert image[pixels, pixels] in (BLACK, WHITE)
    assert set(np.unique(image[pixels:, :])).issubset({BLACK, WHITE})


def test_render_uses_the_largest_integer_scale_that_fits():
    rendered = render(0, texture_size_px=512)
    assert rendered.shape == (510, 510)
    assert rendered.pixels_per_module == 51
    assert rendered.modules == 10
    assert rendered.drawing_size_m == pytest.approx(0.225)


def test_render_is_not_resampled():
    """Two scales of the same tag are proportional, module for module."""
    small = module_grid(11)
    large = render_tag(11, pixels_per_module=7)
    assert large.shape == (70, 70)
    for row in range(10):
        for column in range(10):
            assert large[row * 7, column * 7] == small[row, column]


def test_drawing_size_keeps_the_black_square_at_tag_size():
    """The black square is 8 modules; the drawing is 10, so it is 1.25x wider."""
    assert drawing_size_m(0.18) == pytest.approx(0.18 * 10 / 8)
    assert drawing_size_m(0.18) == pytest.approx(0.225)
    assert drawing_size_m(0.25) == pytest.approx(0.3125)
    with pytest.raises(TagImageError):
        drawing_size_m(0.0)


def test_quiet_zone_modules_is_configurable():
    assert total_modules(tagcodes.family("tag36h11"), 2) == 12
    assert module_grid(0, quiet_zone_modules=2).shape == (12, 12)
    with pytest.raises(TagImageError):
        total_modules(tagcodes.family("tag36h11"), 0)


def test_invalid_inputs_are_rejected():
    with pytest.raises(TagImageError):
        render(0, texture_size_px=8)
    with pytest.raises(TagImageError):
        render_tag(0, pixels_per_module=0)
    with pytest.raises(ValueError):
        render_tag(587)


def test_render_reports_its_summary():
    summary = render(2).summary()
    assert summary["id"] == 2
    assert summary["pixels"] == [510, 510]
    assert summary["modules"] == 10
    assert summary["family"] == "tag36h11"
    assert summary["drawing_size_m"] == pytest.approx(0.225)


def test_png_name_convention():
    assert png_name(4) == "tag_4_tag36h11.png"
    assert png_name(4, "tag36h11") == "tag_4_tag36h11.png"


def test_write_png_round_trips(tmp_path):
    path = str(tmp_path / "tag.png")
    written = write_png(path, render_tag(1, pixels_per_module=3))
    assert written == path
    assert os.path.isfile(path)
    from PIL import Image

    with Image.open(path) as image:
        assert image.mode == "L"
        assert image.size == (30, 30)


def test_contact_sheet_places_every_tag():
    rendered = [render(tag_id, texture_size_px=64) for tag_id in (0, 1, 2)]
    sheet = contact_sheet(rendered, columns=2, pad_px=4)
    cell = max(tag.array.shape[0] for tag in rendered)
    assert sheet.shape[1] == 2 * cell + 3 * 4
    assert sheet.shape[0] == 2 * cell + 3 * 4
    with pytest.raises(TagImageError):
        contact_sheet([])


def test_sheet_from_ids_returns_image_and_tags():
    image, rendered = sheet_from_ids([0, 1, 2], texture_size_px=64)
    assert [tag.tag_id for tag in rendered] == [0, 1, 2]
    assert image.ndim == 2
    assert image.shape[0] > 0


def test_every_reference_id_renders():
    for tag_id in (0, 1, 2, 586):
        rendered = render(tag_id, texture_size_px=128)
        assert rendered.shape[0] == rendered.shape[1]
