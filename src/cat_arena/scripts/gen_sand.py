#!/usr/bin/env python3
"""``ros2 run cat_arena gen_sand.py`` - procedural sand appearance generator.

Stage A of ``PROMPT_SAND.md``: for one seed and one quality preset, write the
procedural sand surface artefacts of a generated arena:

* ``albedo.png``    layered-noise sand albedo (base colour, damp/dry patches,
                    mid-frequency clumps, fine grain, painted track bands with
                    cleat ridges, footprints, chalk lines with broken edges);
* ``normal.png``    tangent-space normal map from the same height field that
                    drives the visual mesh;
* ``roughness.png`` sandy roughness from value noise;
* ``terrain_visual.obj`` visual mesh with micro-relief (ripples, tracks,
                    footprints, crater pits and rims);
* ``height.png``    simple collision heightmap (flat floor plus crater
                    depressions).  The visual mesh is never used for collision.

Usage::

    ros2 run cat_arena gen_sand.py --seed 7 --quality light
    python3 scripts/gen_sand.py --seed 7 --quality full --output-dir /tmp/sand

The same seed must give byte-identical artefacts (determinism requirement of
``PROMPT_SAND.md``); only the quality preset changes texture resolution and
relief scale.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from cat_arena.config import ArenaConfig, load_config
from cat_arena.terrain import Crater, Mesh, TerrainField, build_mesh, write_obj_mesh, write_heightmap_png
from cat_arena.png import write_gray8, write_rgb8

__all__ = ["main"]

_QUALITY_TEX_SIZE = {"light": 256, "full": 1024}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _blur(image: np.ndarray, passes: int = 1) -> np.ndarray:
    """Cheap wrap-around box blur used for the procedural textures."""
    out = image.astype(float)
    for _ in range(passes):
        out = 0.25 * (
            np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0) + np.roll(out, 1, axis=1) + np.roll(out, -1, axis=1)
        )
    return out


def _noise(size: int, rng: np.random.Generator, octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """Multi-scale value-ish noise in ``[0, 1]``."""
    total = np.zeros((size, size), dtype=float)
    amplitude = 1.0
    frequency = 2.0
    for _ in range(octaves):
        grid = rng.random((max(2, int(size / frequency) + 2), max(2, int(size / frequency) + 2)))
        # bilinear-ish upscale by cumulative sums (cheap smoothing substitute)
        small = grid[: size // int(frequency) + 2, : size // int(frequency) + 2]
        big = np.kron(small, np.ones((int(frequency), int(frequency))))
        total += amplitude * (big / big.max())
        amplitude *= persistence
        frequency *= 2.0
    return np.clip((total - total.min()) / (total.max() - total.min() + 1e-9), 0.0, 1.0)


def _load_appearance(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


# --------------------------------------------------------------------------- #
# sand height field with micro-relief
# --------------------------------------------------------------------------- #
def _craters(appearance: Dict[str, Any], count: int, width_m: float, depth_m: float, rng: np.random.Generator) -> List[Crater]:
    craters: List[Crater] = []
    for index in range(max(1, count)):
        x = float(rng.uniform(2.3, 7.2))
        y = float(rng.uniform(0.4, 4.4))
        craters.append(
            Crater(
                crater_id=f"crater_{index}",
                x=x,
                y=y,
                radius=0.5 * float(width_m),
                depth=float(depth_m),
                zone="obstacle",
            )
        )
    return craters


def _sand_field(
    cfg: ArenaConfig,
    craters: Sequence[Crater],
    ripple_amplitude: float,
    ripple_wavelength: float,
    track_depth: float,
    track_width: float,
    track_pitch: float,
    footprint_depth: float,
    size_x: float,
    size_y: float,
    cell: float,
) -> np.ndarray:
    """Height field ``z[j, i] = h(x = xs[i], y = ys[j])`` with micro-relief."""
    nx = int(round(size_x / cell)) + 1
    ny = int(round(size_y / cell)) + 1
    xs = np.linspace(0.0, size_x, nx)
    ys = np.linspace(0.0, size_y, ny)
    z = np.zeros((ny, nx), dtype=float)
    if ripple_amplitude > 0.0:
        lam = ripple_wavelength
        z += ripple_amplitude * (
            0.6 * np.sin(2.0 * np.pi * xs[None, :] / lam) * np.cos(2.0 * np.pi * ys[:, None] / lam)
            + 0.4 * np.sin(2.0 * np.pi * (xs[None, :] + ys[:, None]) / (1.7 * lam))
        )
    for crater in craters:
        radius = float(crater.radius)
        depth = float(crater.depth)
        bowl = 0.5 * (1.0 + np.cos(np.pi * np.hypot(xs[None, :] - crater.x, ys[:, None] - crater.y) / radius))
        z -= depth * bowl * (np.hypot(xs[None, :] - crater.x, ys[:, None] - crater.y) < radius)
        rim = float(0.5 * (crater.depth * 0.25))
        ring = (np.hypot(xs[None, :] - crater.x, ys[:, None] - crater.y) > radius) & (
            np.hypot(xs[None, :] - crater.x, ys[:, None] - crater.y) < radius * 1.6
        )
        z += rim * np.clip(1.0 - np.hypot(xs[None, :] - crater.x, ys[:, None] - crater.y) / (radius * 1.6), 0.0, 1.0) * ring
    if track_depth > 0.0:
        for band in range(int(np.ceil(size_y / max(track_width, 1e-6)))):
            y0 = float(band) * float(track_width) + float(track_width) * 0.5
            z -= track_depth * np.clip(1.0 - np.abs(ys[:, None] - y0) / (0.5 * float(track_width)), 0.0, 1.0)
            for cleat in range(int(max(1.0, float(track_pitch) / float(track_width)))):
                xc = float(cleat) * float(track_width) + float(track_width) * 0.5
                z -= track_depth * 0.35 * np.exp(-((xs[None, :] - xc) ** 2) / (2.0 * 0.02 ** 2))
    rng_local = np.random.default_rng(1234)
    for _ in range(int(60 * footprint_depth)):
        cx = float(rng_local.uniform(0.0, size_x))
        cy = float(rng_local.uniform(0.0, size_y))
        blob = np.exp(-((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) / (2.0 * 0.25 ** 2))
        z -= footprint_depth * blob
    return z


# --------------------------------------------------------------------------- #
# textures
# --------------------------------------------------------------------------- #
def _write_albedo(path: str, cfg: ArenaConfig, appearance: Dict[str, Any], rng: np.random.Generator, size: int, craters: Sequence[Crater] = ()) -> Dict[str, Any]:
    base = np.asarray(
        appearance.get("sand_base_color") or [float(c) * 255.0 for c in cfg.sand_color[:3]],
        dtype=float,
    )
    contrast_lo, contrast_hi = [float(v) for v in (appearance.get("patch_contrast") or [0.3, 0.7])]
    damp_fraction = float(appearance.get("damp_fraction") or 0.4)
    grain_scale = float(appearance.get("grain_scale") or 0.4)
    img = np.empty((size, size, 3), dtype=np.uint8)
    coarse = _blur(rng.normal(0.0, 1.0, (size, size)), passes=6)
    coarse /= max(1e-6, float(np.abs(coarse).max()))
    grain = _blur(rng.normal(0.0, 1.0, (size, size)), passes=2)
    grain /= max(1e-6, float(np.abs(grain).max()))
    modulation = 1.0 + contrast_lo * (1.0 - damp_fraction) * coarse + contrast_hi * damp_fraction * coarse + 0.15 * grain * grain_scale
    for channel in range(3):
        img[:, :, channel] = np.clip(base[channel] * modulation, 0, 255).astype(np.uint8)
    albedo = img.astype(float)
    # darker, damper sand inside craters (the visual floor of a pit)
    for crater in craters:
        radius_px = int(round(float(crater.radius) / float(cfg.length_x) * size))
        depth_px = max(4, int(round(float(crater.depth) / max(float(cfg.terrain.max_relief), 1e-6) * size * 2.0)))
        yy, xx = np.mgrid[0:size, 0:size]
        dist = np.sqrt((xx - float(crater.x) / float(cfg.length_x) * size) ** 2 + (yy - (1.0 - float(crater.y) / float(cfg.width_y)) * size) ** 2)
        mask = dist <= radius_px
        albedo[mask] *= float(crater.depth) / max(float(cfg.terrain.max_relief), 1e-6)
    # ------------------------------------------------------------------ #
    # painted surface features: track bands + cleat ridges, footprints, #
    # and jittered/broken chalk zone lines                              #
    # ------------------------------------------------------------------ #
    track_color = np.asarray(base, dtype=float) * 0.90
    footprint_color = np.asarray([118.0, 103.0, 84.0], dtype=float)
    band_width = float(appearance.get("track_band_width_cm") or 20.0) / 100.0
    band_count = int(appearance.get("track_band_count") or 2)
    cleat_pitch = max(0.02, float(appearance.get("track_cleat_pitch_cm") or 6.0) / 100.0)
    n_bands = max(1, int(np.ceil(float(cfg.width_y) / band_width - 1e-9)))
    band_count = max(1, min(band_count, n_bands))
    for band in range(band_count):
        yc = (float(band) + 0.5) * band_width
        y_lo = max(0.0, yc - 0.5 * band_width)
        y_hi = min(float(cfg.width_y), yc + 0.5 * band_width)
        row_lo = int(round((1.0 - y_hi / float(cfg.width_y)) * size))
        row_hi = int(round((1.0 - y_lo / float(cfg.width_y)) * size))
        if row_hi < row_lo or row_hi > size - 1:
            row_hi = min(row_hi, size - 1)
        if row_lo < 0:
            row_lo = 0
        if row_hi < row_lo:
            continue
        for row in range(row_lo, row_hi + 1):
            albedo[row, :] = track_color
        # cleat ridges across the band
        step = cleat_pitch if band_width > cleat_pitch else band_width
        k = 0
        pos = float(cfg.length_x) * 0.0
        while pos <= float(cfg.length_x) + 1e-9 and k * step <= float(cfg.width_y):
            xc = pos + 0.5 * step
            col_c = int(round((xc / float(cfg.length_x)) * size))
            col_d = int(round(((xc + step) / float(cfg.length_x)) * size))
            if 0 <= col_c < size and 0 <= col_d < size and col_d > col_c:
                for row in range(row_lo, row_hi + 1):
                    for col in range(col_c, col_d):
                        albedo[row, col] = track_color * 0.75
            pos += step
            k += 1
    # footprints: random dark blobs
    n_footprints = int(appearance.get("footprint_density") or 6)
    for _ in range(n_footprints):
        fx = float(rng.uniform(0.0, float(cfg.length_x)))
        fy = float(rng.uniform(0.0, float(cfg.width_y)))
        rx = 0.05
        ry = 0.04
        col_c = int(round((fx / float(cfg.length_x)) * size))
        col_d = int(round(((fx + 2 * rx) / float(cfg.length_x)) * size))
        row_c = int(round((1.0 - (fy + ry) / float(cfg.width_y)) * size))
        row_d = int(round((1.0 - (fy - ry) / float(cfg.width_y)) * size))
        for row in range(max(0, row_d), min(size, row_c + 1)):
            for col in range(max(0, col_c), min(size, col_d)):
                if ((row - row_c) ** 2 / (ry * ry) + (col - col_c) ** 2 / (rx * rx)) <= 1.0:
                    albedo[row, col] = footprint_color
    # chalk zone lines: thin, slightly irregular and broken
    chalk_color = np.asarray([238.0, 238.0, 230.0], dtype=float)
    chalk_break = float(appearance.get("chalk_line_breakup") or 0.5)
    chalk_half = max(0.25, 0.0175 * (1.0 + chalk_break))  # half-width with breakup
    for zone in cfg.zones.values():
        rect = zone.rect
        for y in (float(rect.y0), float(rect.y1)):
            row = int(round((1.0 - y / float(cfg.width_y)) * size))
            if not (0 <= row < size):
                continue
            count = max(2, int(abs(float(rect.x1) - float(rect.x0)) / 0.05) + 1)
            for x in np.linspace(float(rect.x0), float(rect.x1), count):
                col = int(round((x / float(cfg.length_x)) * size))
                if not (0 <= col < size):
                    continue
                col_j = int(round(col + float(rng.normal(0.0, chalk_half * 0.5))))
                row_j = int(round(row + float(rng.normal(0.0, chalk_half * 0.5))))
                if 0 <= col_j < size and 0 <= row_j < size:
                    albedo[row, col_j] = chalk_color
        for x in (float(rect.x0), float(rect.x1)):
            col = int(round((x / float(cfg.length_x)) * size))
            if not (0 <= col < size):
                continue
            count = max(2, int(abs(float(rect.y1) - float(rect.y0)) / 0.05) + 1)
            for y in np.linspace(float(rect.y0), float(rect.y1), count):
                row = int(round((1.0 - y / float(cfg.width_y)) * size))
                if not (0 <= row < size):
                    continue
                row_j = int(round(row + float(rng.normal(0.0, chalk_half * 0.5))))
                if 0 <= row_j < size:
                    albedo[row_j, col] = chalk_color
    img = np.clip(albedo, 0, 255).astype(np.uint8)
    write_rgb8(path, img)
    return {"file": os.path.basename(path), "size": [size, size], "type": "procedural_sand_albedo"}


def _write_normal(path: str, height: np.ndarray, size: int, size_x: float, size_y: float) -> Dict[str, Any]:
    """Tangent-space normal map from the same height field that drives the mesh."""
    ny, nx = height.shape
    j = np.linspace(0.0, float(ny - 1), size)
    i = np.linspace(0.0, float(nx - 1), size)
    Y, X = np.meshgrid(j, i, indexing="ij")
    j0 = np.clip(np.floor(Y).astype(int), 0, ny - 2)
    j1 = np.minimum(j0 + 1, ny - 1)
    i0 = np.clip(np.floor(X).astype(int), 0, nx - 2)
    i1 = np.minimum(i0 + 1, nx - 1)
    ty = (Y - j0).astype(np.float64)
    tx = (X - i0).astype(np.float64)
    top = height[j0, i0] * (1.0 - tx) + height[j0, i1] * tx
    bot = height[j1, i0] * (1.0 - tx) + height[j1, i1] * tx
    H = top * (1.0 - ty) + bot * ty
    dzdy, dzdx = np.gradient(H, axis=0), np.gradient(H, axis=1)
    nvec_x = -dzdx
    nvec_y = -dzdy
    norm = np.sqrt(nvec_x * nvec_x + nvec_y * nvec_y + 1.0)
    nxu = nvec_x / norm
    nyu = nvec_y / norm
    nzu = 1.0 / norm
    normal_map = np.empty((size, size, 3), dtype=np.uint8)
    normal_map[:, :, 0] = ((nxu + 1.0) * 0.5 * 255.0).astype(np.uint8)
    normal_map[:, :, 1] = ((nyu + 1.0) * 0.5 * 255.0).astype(np.uint8)
    normal_map[:, :, 2] = ((nzu + 1.0) * 0.5 * 255.0).astype(np.uint8)
    write_rgb8(path, normal_map)
    return {"file": os.path.basename(path), "size": [size, size], "type": "procedural_sand_normal"}


def _write_roughness(path: str, rng: np.random.Generator, size: int) -> Dict[str, Any]:
    rough = _blur(rng.normal(0.0, 1.0, (size, size)), passes=4)
    rough = np.clip(rough / max(1e-6, float(np.abs(rough).max())), 0.0, 1.0)
    write_gray8(path, (rough * 255.0).astype(np.uint8))
    return {"file": os.path.basename(path), "size": [size, size], "type": "procedural_sand_roughness"}


def _rock_vertices_faces(rng: np.random.Generator, center: Tuple[float, float, float], radius: float, segs_u: int = 16, segs_v: int = 12) -> Tuple[List[str], List[str]]:
    verts: List[str] = []
    faces: List[str] = []
    index: Dict[Tuple[int, int], int] = {}
    grid: List[Tuple[float, float, float]] = []
    for j in range(segs_v + 1):
        phi = math.pi * j / segs_v
        for i in range(segs_u):
            theta = 2.0 * math.pi * i / segs_u
            radial = radius * (1.0 + 0.22 * float(rng.normal(0.0, 1.0)))
            grid.append((center[0] + radial * math.cos(theta) * math.sin(phi), center[1] + radial * math.sin(theta) * math.sin(phi), center[2] + radial * math.cos(phi)))
    def vid(i: int, j: int) -> int:
        key = (i, j)
        if key not in index:
            index[key] = len(verts) + 1
            x, y, z = grid[j * segs_u + i]
            verts.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        return index[key]
    for j in range(segs_v):
        for i in range(segs_u - 1):
            faces.append(f"f {vid(i, j)} {vid(i + 1, j)} {vid(i + 1, j + 1)}")
            faces.append(f"f {vid(i, j)} {vid(i + 1, j + 1)} {vid(i, j + 1)}")
    return verts, faces


def _write_rocks(out_dir: str, rocks: List[Tuple[Tuple[float, float, float], float]], palette: Sequence[str]) -> List[str]:
    paths: List[str] = []
    for index, ((cx, cy, cz), radius) in enumerate(rocks):
        verts, faces = _rock_vertices_faces(np.random.default_rng(1000 + index), (cx, cy, cz), radius)
        path = os.path.join(out_dir, f"rock_{index}.obj")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# cat_arena procedural rock - generated file, do not edit\n")
            handle.write(f"mtllib rock.mtl\n")
            handle.write(f"o rock_{index}\n")
            handle.write("\n".join(verts) + "\n")
            handle.write("\n".join(faces) + "\n")
        paths.append(path)
    with open(os.path.join(out_dir, "rock.mtl"), "w", encoding="utf-8") as handle:
        for i, hexcolor in enumerate(palette):
            rgb = [int(hexcolor.lstrip("#")[k : k + 2], 16) for k in (0, 2, 4)]
            handle.write(f"newmtl rock_{i}\n")
            handle.write(f"Kd {rgb[0] / 255.0:.3f} {rgb[1] / 255.0:.3f} {rgb[2] / 255.0:.3f} 1.0\n")
            handle.write("newmtl default\n")
    return paths


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gen_sand",
        description="Write the procedural sand appearance artefacts for one seed and quality preset.",
    )
    parser.add_argument("--seed", type=int, default=1, help="arena seed (same seed => same artefacts)")
    parser.add_argument("--quality", choices=("light", "full"), default="light", help="texture/relief quality preset")
    parser.add_argument("--config", type=str, default=None, help="arena.yaml override")
    parser.add_argument("--appearance", type=str, default=None, help="appearance.yaml override")
    parser.add_argument("--output-dir", type=str, default=None, help="where to write the artefacts")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    appearance = _load_appearance(args.appearance)
    rng = np.random.default_rng(args.seed)

    quality = args.quality
    tex_size = int(_QUALITY_TEX_SIZE.get(quality, 256))
    cell = float(0.1) if quality == "light" else float(cfg.terrain.heightmap_resolution)
    ripple_amplitude = float(appearance.get("ripple_amplitude_mm") or 0.0) / 1000.0 * (0.25 if quality == "light" else 1.0)
    ripple_wavelength = float(appearance.get("ripple_wavelength_cm") or 30.0) / 100.0
    track_depth = 0.004 * (0.25 if quality == "light" else 1.0)
    footprint_depth = 0.002 * (0.25 if quality == "light" else 1.0)

    craters = _craters(appearance, int(appearance.get("crater_count") or 5), float(appearance.get("crater_width_cm") or 0.45) * 0.5, float(appearance.get("crater_depth_cm") or 0.3), rng)
    height = _sand_field(
        cfg,
        craters,
        ripple_amplitude=ripple_amplitude,
        ripple_wavelength=ripple_wavelength,
        track_depth=track_depth,
        track_width=float(appearance.get("track_band_width_cm") or 0.2),
        track_pitch=float(appearance.get("track_cleat_pitch_cm") or 6.0) / 100.0,
        footprint_depth=footprint_depth,
        size_x=float(cfg.length_x),
        size_y=float(cfg.width_y),
        cell=cell,
    )

    nx = int(round(float(cfg.length_x) / cell)) + 1
    ny = int(round(float(cfg.width_y) / cell)) + 1
    field = TerrainField(cfg, craters)
    mesh = Mesh(
        xs=np.linspace(0.0, float(cfg.length_x), nx),
        ys=np.linspace(0.0, float(cfg.width_y), ny),
        z=np.asarray(height, dtype=float),
        cell_x=cell,
        cell_y=cell,
        skirt_depth=float(cfg.terrain.sand_depth if bool(cfg.terrain.skirt) else 0.0),
        triangle_count=int(2 * (nx - 1) * (ny - 1)) + (int(2 * (2 * (nx - 1) + 2 * (ny - 1))) if bool(cfg.terrain.skirt) else 0),
    )

    out_dir = os.path.abspath(args.output_dir) if args.output_dir else tempfile.mkdtemp(prefix="cat_sand_")
    os.makedirs(out_dir, exist_ok=True)
    visual_path = os.path.join(out_dir, "terrain_visual.obj")
    stats = write_obj_mesh(visual_path, cfg, mesh)
    albedo_stats = _write_albedo(os.path.join(out_dir, "albedo.png"), cfg, appearance, rng, tex_size, craters)
    normal_stats = _write_normal(os.path.join(out_dir, "normal.png"), height, tex_size, float(cfg.length_x), float(cfg.width_y))
    rough_stats = _write_roughness(os.path.join(out_dir, "roughness.png"), rng, tex_size)
    collision = float(cfg.terrain.max_relief) + np.asarray(height, dtype=float)
    collision = np.clip(collision, 0.0, float(cfg.terrain.max_relief))
    write_heightmap_png(os.path.join(out_dir, "height.png"), collision, 0.0, float(cfg.terrain.max_relief))

    rock_count = int(appearance.get("rock_count") or 4)
    rock_diameter = float(appearance.get("rock_diameter_cm") or 35.0) / 100.0
    rocks: List[Tuple[Tuple[float, float, float], float]] = []
    rock_rng = np.random.default_rng(args.seed)
    for index in range(max(1, rock_count)):
        cx = float(rock_rng.uniform(2.2, 7.2))
        cy = float(rock_rng.uniform(0.3, 4.5))
        radius = 0.5 * rock_diameter * float(rock_rng.uniform(0.85, 1.15))
        rocks.append(((cx, cy, float(cfg.terrain.max_relief) - radius * 0.55), radius))
    rock_paths = _write_rocks(out_dir, rocks, [str(c) for c in (appearance.get("rock_color_palette") or ["#B4733F", "#C9A27A", "#8A5A33"])])

    print(f"[gen_sand] seed {args.seed} quality {quality} -> {out_dir}")
    print(f"[gen_sand] visual mesh : {stats.get('triangles')} triangles, z_min {stats.get('z_min')}")
    print(f"[gen_sand] albedo       : {albedo_stats}")
    print(f"[gen_sand] normal       : {normal_stats}")
    print(f"[gen_sand] roughness    : {rough_stats}")
    print(f"[gen_sand] collision    : flat floor + {len(craters)} craters")
    print(f"[gen_sand] rocks        : {len(rock_paths)} procedural meshes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
