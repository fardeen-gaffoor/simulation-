"""Terrain mesh tests.

The terrain is the one part of module 1 that has already bitten us: a heightmap
produced a perfectly flat collidable surface, and an OBJ without vertex normals
made dartsim silently drop the sub-mesh (and then segfault).  These tests pin
both facts down:

* craters really are *negative* depressions in the mesh,
* every OBJ vertex has a normal and every face uses a full ``v/vt/vn`` index
  triplet with an identical count of ``v``/``vn`` entries.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cat_arena.terrain import (
    Crater,
    TerrainField,
    build_mesh,
    write_heightmap_png,
    write_obj_mesh,
    write_sand_albedo,
)
from cat_arena.png import read_png


def _crater(x=4.5, y=2.5, radius=0.25, depth=0.15) -> Crater:
    return Crater(crater_id="c0", x=x, y=y, radius=radius, depth=depth, zone="obstacle")


# --------------------------------------------------------------------------- #
# height field
# --------------------------------------------------------------------------- #
def test_flat_field_without_craters(cfg):
    field = TerrainField(cfg, [])
    for x, y in ((0.0, 0.0), (4.5, 2.5), (8.99, 4.99)):
        assert field.height(x, y) == pytest.approx(0.0)
    assert field.max_depth() == pytest.approx(0.0)
    assert field.max_height() == pytest.approx(0.0)


def test_crater_is_a_negative_depression(cfg):
    crater = _crater()
    field = TerrainField(cfg, [crater])
    # centre: exactly one depth below the nominal surface
    assert field.height(crater.x, crater.y) == pytest.approx(-crater.depth)
    # rim and beyond: back to the nominal surface, and never below it again
    assert field.height(crater.x + crater.radius, crater.y) == pytest.approx(0.0, abs=1e-12)
    assert field.height(crater.x + crater.radius + 0.5, crater.y) == pytest.approx(0.0)
    assert field.height(crater.x, crater.y + 10.0) == pytest.approx(0.0)
    assert field.max_depth() == pytest.approx(crater.depth)
    # monotonic bowl: every radius step out of the crater comes back up
    heights = [field.height(crater.x + r, crater.y) for r in np.linspace(0.0, crater.radius, 25)]
    assert all(b >= a - 1e-12 for a, b in zip(heights, heights[1:]))
    assert heights[0] < heights[-1]


def test_crater_depth_at_and_inside(cfg):
    crater = _crater()
    field = TerrainField(cfg, [crater])
    assert field.crater_depth_at(crater.x, crater.y) == pytest.approx(crater.depth)
    assert field.crater_depth_at(crater.x + 5.0, crater.y) == pytest.approx(0.0)


def test_height_grid_matches_pointwise_height(cfg):
    field = TerrainField(cfg, [_crater(x=3.0, y=2.0), _crater(x=6.0, y=3.5, radius=0.2, depth=0.08)])
    xs = np.linspace(0.0, cfg.length_x, 37)
    ys = np.linspace(0.0, cfg.width_y, 21)
    grid = field.height_grid(xs, ys)
    assert grid.shape == (ys.size, xs.size)
    for j in (0, 5, 12, 20):
        for i in (0, 7, 25, 36):
            assert grid[j, i] == pytest.approx(field.height(xs[i], ys[j]), abs=1e-9)
    assert grid.min() < -0.05


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
def test_build_mesh_geometry(cfg):
    field = TerrainField(cfg, [_crater(x=4.5, y=2.5, radius=0.25, depth=0.18)])
    mesh = build_mesh(cfg, field)
    nx = int(round(cfg.length_x / cfg.terrain.cell_size)) + 1
    ny = int(round(cfg.width_y / cfg.terrain.cell_size)) + 1
    assert mesh.xs.size == nx and mesh.ys.size == ny
    assert mesh.cell_x == pytest.approx(cfg.terrain.cell_size)
    assert mesh.z.shape == (ny, nx)
    expected_top = 2 * (nx - 1) * (ny - 1)
    if cfg.terrain.skirt:
        expected_top += 2 * (2 * (nx - 1) + 2 * (ny - 1))
    assert mesh.triangle_count == expected_top
    # the crater must show up as a real dent in the sampled grid
    assert mesh.z_min <= -0.18 + 0.02
    assert mesh.z_max == pytest.approx(0.0)


def test_mesh_resolution_is_playable(cfg):
    mesh = build_mesh(cfg, TerrainField(cfg, []))
    assert 10000 < mesh.triangle_count < 120000
    assert mesh.xs[0] == pytest.approx(0.0)
    assert mesh.xs[-1] == pytest.approx(cfg.length_x)
    assert mesh.ys[-1] == pytest.approx(cfg.width_y)


# --------------------------------------------------------------------------- #
# OBJ writer: the normals requirement is the important one
# --------------------------------------------------------------------------- #
def _count_lines(path, prefix):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.startswith(prefix))


def test_obj_has_matching_vertex_and_normal_counts(tmp_path, cfg):
    field = TerrainField(cfg, [_crater()])
    mesh = build_mesh(cfg, field)
    obj = tmp_path / "terrain.obj"
    stats = write_obj_mesh(str(obj), cfg, mesh)

    verts = _count_lines(str(obj), "v ")
    normals = _count_lines(str(obj), "vn ")
    uvs = _count_lines(str(obj), "vt ")
    assert verts == normals, "dartsim ignores (and then crashes on) OBJ without matching normals"
    assert verts == uvs
    assert verts == stats["vertices"]
    assert stats["triangles"] == mesh.triangle_count
    assert os.path.isfile(tmp_path / "terrain.mtl")


def test_obj_faces_use_full_index_triplets_and_stay_in_range(tmp_path, cfg):
    field = TerrainField(cfg, [])
    mesh = build_mesh(cfg, field)
    obj = tmp_path / "terrain.obj"
    stats = write_obj_mesh(str(obj), cfg, mesh)
    vertex_count = stats["vertices"]

    with open(str(obj), "r", encoding="utf-8") as handle:
        faces = [line.split()[1:] for line in handle if line.startswith("f ")]
    assert len(faces) == stats["triangles"]
    for face in faces[:2000] + faces[-50:]:
        assert len(face) == 3
        for token in face:
            parts = token.split("/")
            assert len(parts) == 3 and parts[0] == parts[1] == parts[2]
            index = int(parts[0])
            assert 1 <= index <= vertex_count


def test_obj_declares_its_material_library(tmp_path, cfg):
    obj = tmp_path / "terrain.obj"
    stats = write_obj_mesh(str(obj), cfg, build_mesh(cfg, TerrainField(cfg, [])))
    text = obj.read_text(encoding="utf-8")
    mtl = (tmp_path / "terrain.mtl").read_text(encoding="utf-8")
    assert "mtllib " + stats["mtl"] in text
    assert "usemtl sand" in text
    assert "newmtl sand" in mtl
    assert "newmtl sand_side" in mtl
    assert text.endswith("\n")


def test_skirt_is_missing_when_disabled(tmp_path, config_path):
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["terrain"]["skirt"] = False
    alt = tmp_path / "arena.yaml"
    alt.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from cat_arena.config import load_config

    cfg = load_config(str(alt))
    mesh = build_mesh(cfg, TerrainField(cfg, []))
    assert mesh.skirt_depth == 0.0
    stats = write_obj_mesh(str(tmp_path / "t.obj"), cfg, mesh)
    assert stats["triangles"] == 2 * (mesh.xs.size - 1) * (mesh.ys.size - 1)


def test_obj_generation_is_deterministic(tmp_path, cfg):
    mesh = build_mesh(cfg, TerrainField(cfg, [_crater()]))
    a, b = tmp_path / "a.obj", tmp_path / "b.obj"
    write_obj_mesh(str(a), cfg, mesh)
    write_obj_mesh(str(b), cfg, mesh)
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# side artifacts
# --------------------------------------------------------------------------- #
def test_heightmap_png_roundtrip(tmp_path, cfg):
    mesh = build_mesh(cfg, TerrainField(cfg, [_crater()]))
    path = tmp_path / "terrain_height.png"
    stats = write_heightmap_png(str(path), mesh, mesh.z_min - 0.01, mesh.z_max + 0.01)
    assert os.path.isfile(str(path))
    image = read_png(str(path))
    # mesh.z is (rows=y, cols=x); the PNG is flipped vertically for viewing
    assert image.width == mesh.xs.size
    assert image.height == mesh.ys.size
    assert stats["file"] == "terrain_height.png"
    assert stats["rows"] == mesh.ys.size and stats["cols"] == mesh.xs.size
    assert image.bit_depth == 8
    # the crater (and hence a real height difference) survives the encoding
    assert int(image.pixels.max()) - int(image.pixels.min()) > 20
    assert stats["quantization_m"] > 0.0


def test_sand_albedo_is_a_colour_texture(tmp_path, cfg):
    import numpy as np

    rng = np.random.default_rng(7)
    stats = write_sand_albedo(str(tmp_path / "sand_albedo.png"), cfg, rng)
    image = read_png(str(tmp_path / "sand_albedo.png"))
    assert image.width == image.height == cfg.terrain.texture_size
    assert image.pixels.shape[2] == 3
    assert int(image.pixels.max()) > int(image.pixels.min()), "albedo must not be a flat colour"
    assert stats["file"] == "sand_albedo.png"
