"""Terrain height field, triangle-mesh export and texture generation.

The arena sand is a *real* triangle mesh:

* craters are baked into the geometry as smooth NEGATIVE depressions,
* the mesh is exported as OBJ **with vertex normals** - this is mandatory:
  Gazebo's dartsim ``CustomMeshShape`` reports
  "does not have a normal count [0] that matches its vertex count" and then
  segfaults when a sub-mesh has no normals (verified on gz-sim 8.15.0),
* the same mesh is used for the visual *and* for the collision geometry, so what
  the rover drives on is exactly what the cameras see,
* an optional skirt turns the open sheet into a sand layer with vertical sides.

Collision uses the DART bullet collision detector (verified).  The heightmap
collision path was tried first and rejected: on this build it silently produced
a flat surface, see README "verified facts".
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ArenaConfig
from .png import write_gray8, write_rgb8

__all__ = [
    "Crater",
    "TerrainField",
    "TerrainMesh",
    "build_mesh",
    "write_obj_mesh",
    "write_heightmap_png",
    "write_sand_albedo",
]


@dataclass(frozen=True)
class Crater:
    """A seeded crater: a negative depression baked into the terrain mesh."""

    crater_id: str
    x: float
    y: float
    radius: float
    depth: float
    zone: str

    @property
    def width(self) -> float:
        return 2.0 * self.radius

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.crater_id,
            "center": [round(self.x, 6), round(self.y, 6)],
            "radius": round(self.radius, 6),
            "width": round(self.width, 6),
            "depth": round(self.depth, 6),
            "zone": self.zone,
        }


def _bowl(t: float) -> float:
    """Smooth crater profile: 1 at the centre, 0 at the rim, zero slope at both."""
    return 0.5 * (1.0 + math.cos(math.pi * t))


class TerrainField:
    """Analytic sand height field ``z = h(x, y)`` in the arena frame.

    ``z = 0`` is the nominal (mean) sand surface; craters are the only negative
    features.  The optional long-wavelength ripple keeps the surface visually
    plausible while staying small enough not to perturb wheeled locomotion
    (default amplitude is 0.0 m, i.e. flat, for clean navigation tests).
    """

    def __init__(
        self,
        cfg: ArenaConfig,
        craters: Sequence[Crater],
        phases: Optional[Sequence[float]] = None,
    ):
        self._cfg = cfg
        self._craters = list(craters)
        self._amp = float(cfg.terrain.noise_amplitude)
        self._wavelength = float(cfg.terrain.noise_wavelength)
        self._phases = tuple(float(p) for p in (phases if phases is not None else (0.0, 0.0, 0.0)))

    @property
    def craters(self) -> List[Crater]:
        return list(self._craters)

    def _ripple(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self._amp <= 0.0:
            return np.zeros_like(x, dtype=float)
        lam = self._wavelength
        p0, p1, p2 = self._phases
        return self._amp * (
            0.6 * np.sin(2.0 * math.pi * x / lam + p0) * np.cos(2.0 * math.pi * y / lam + p1)
            + 0.4 * np.sin(2.0 * math.pi * (x + y) / (1.7 * lam) + p2)
        )

    def height(self, x: float, y: float) -> float:
        """Sand surface height (m) at arena coordinates ``(x, y)``."""
        z = float(self._ripple(np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
        for crater in self._craters:
            r = math.hypot(x - crater.x, y - crater.y)
            if r < crater.radius:
                z -= crater.depth * _bowl(r / crater.radius)
        return z

    def height_grid(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Vectorised height field of shape ``(ys.size, xs.size)``."""
        xx, yy = np.meshgrid(xs, ys)
        grid = np.asarray(self._ripple(xx, yy), dtype=float)
        for crater in self._craters:
            rr = np.hypot(xx - crater.x, yy - crater.y)
            inside = rr < crater.radius
            if bool(np.any(inside)):
                t = np.where(inside, rr / crater.radius, 1.0)
                profile = np.where(inside, np.cos(np.pi * t), -1.0)
                grid = grid - np.where(inside, crater.depth * 0.5 * (1.0 + profile), 0.0)
        return grid

    def crater_depth_at(self, x: float, y: float) -> float:
        """How deep the surface is below the nominal plane at ``(x, y)`` (>= 0)."""
        return -self.height(x, y) if self.height(x, y) < 0.0 else 0.0

    def max_depth(self) -> float:
        """Deepest point of the surface below the nominal plane (m)."""
        depth = -self._amp if self._amp > 0.0 else 0.0
        for crater in self._craters:
            depth = max(depth, crater.depth)
        return depth

    def max_height(self) -> float:
        return self._amp if self._amp > 0.0 else 0.0


class TerrainMesh:
    """Generated terrain grid plus the metadata needed to describe it."""

    __slots__ = ("xs", "ys", "z", "skirt_depth", "cell_x", "cell_y", "triangle_count", "vertex_count")

    def __init__(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        z: np.ndarray,
        skirt_depth: float,
        triangle_count: int,
        vertex_count: int,
    ):
        self.xs = xs
        self.ys = ys
        self.z = z
        self.skirt_depth = skirt_depth
        self.cell_x = float(xs[1] - xs[0])
        self.cell_y = float(ys[1] - ys[0])
        self.triangle_count = triangle_count
        self.vertex_count = vertex_count

    @property
    def z_min(self) -> float:
        return float(self.z.min())

    @property
    def z_max(self) -> float:
        return float(self.z.max())


def build_mesh(cfg: ArenaConfig, field: TerrainField) -> TerrainMesh:
    """Sample the height field on the arena grid (the grid used everywhere else)."""
    cell = float(cfg.terrain.cell_size)
    nx = int(round(cfg.length_x / cell)) + 1
    ny = int(round(cfg.width_y / cell)) + 1
    xs = np.linspace(0.0, cfg.length_x, nx)
    ys = np.linspace(0.0, cfg.width_y, ny)
    z = field.height_grid(xs, ys)
    tris = 2 * (nx - 1) * (ny - 1)
    verts = nx * ny
    if cfg.terrain.skirt:
        tris += 2 * (2 * (nx - 1) + 2 * (ny - 1))
        verts += 2 * (2 * nx + 2 * ny)
    return TerrainMesh(
        xs=xs,
        ys=ys,
        z=z,
        skirt_depth=float(cfg.sand_depth) if cfg.terrain.skirt else 0.0,
        triangle_count=tris,
        vertex_count=verts,
    )


def _grid_normals(xs: np.ndarray, ys: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-vertex normals of the top surface (central differences)."""
    dz_dy, dz_dx = np.gradient(z, ys, xs)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(z)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1)


class _ObjBuilder:
    """Tiny OBJ accumulator that keeps the v/vt/vn index spaces aligned."""

    def __init__(self) -> None:
        self.verts: List[Tuple[float, float, float]] = []
        self.uvs: List[Tuple[float, float]] = []
        self.norms: List[Tuple[float, float, float]] = []
        self.faces: List[str] = []

    def add_vertex(self, pos: Sequence[float], normal: Sequence[float], uv: Optional[Sequence[float]]) -> int:
        self.verts.append((float(pos[0]), float(pos[1]), float(pos[2])))
        self.norms.append((float(normal[0]), float(normal[1]), float(normal[2])))
        self.uvs.append((float(uv[0]), float(uv[1])) if uv is not None else (0.0, 0.0))
        return len(self.verts)  # 1-based OBJ index

    def _normal_of(self, ia: int, ib: int, ic: int) -> np.ndarray:
        pa = np.array(self.verts[ia - 1])
        pb = np.array(self.verts[ib - 1])
        pc = np.array(self.verts[ic - 1])
        return np.cross(pb - pa, pc - pa)

    def add_triangle(self, ia: int, ib: int, ic: int, outward: Sequence[float]) -> None:
        """Add a triangle, auto-correcting the winding to face ``outward``."""
        out = np.asarray(outward, dtype=float)
        if float(np.dot(self._normal_of(ia, ib, ic), out)) < 0.0:
            ib, ic = ic, ib
        self.faces.append(f"f {ia}/{ia}/{ia} {ib}/{ib}/{ib} {ic}/{ic}/{ic}")

    def add_quad(self, i0: int, i1: int, i2: int, i3: int, outward: Sequence[float]) -> None:
        self.add_triangle(i0, i1, i2, outward)
        self.add_triangle(i0, i2, i3, outward)

    def render(self) -> str:
        lines = [f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.6f}" for v in self.verts]
        lines += [f"vt {uv[0]:.6f} {uv[1]:.6f}" for uv in self.uvs]
        lines += [f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}" for n in self.norms]
        lines += self.faces
        return "\n".join(lines) + "\n"


def write_obj_mesh(
    obj_path: str,
    cfg: ArenaConfig,
    mesh: TerrainMesh,
    mtl_filename: str = "terrain.mtl",
    material_top: str = "sand",
    material_side: str = "sand_side",
) -> Dict[str, object]:
    """Write the terrain as an OBJ (with ``vn``/``vt``) plus its MTL file."""
    xs, ys, z = mesh.xs, mesh.ys, mesh.z
    nx, ny = int(xs.size), int(ys.size)
    normals = _grid_normals(xs, ys, z)
    builder = _ObjBuilder()

    header = [
        "# cat_arena terrain mesh - generated file, do not edit",
        f"# grid {nx} x {ny}, cell {mesh.cell_x:.4f} x {mesh.cell_y:.4f} m, skirt {mesh.skirt_depth:.3f} m",
        f"mtllib {mtl_filename}",
        "o terrain_top",
        f"usemtl {material_top}",
    ]

    # ---- top surface -------------------------------------------------------
    top_index = np.zeros((ny, nx), dtype=int)
    for j in range(ny):
        for i in range(nx):
            top_index[j, i] = builder.add_vertex(
                (xs[i], ys[j], z[j, i]),
                normals[j, i],
                (xs[i] / cfg.length_x, ys[j] / cfg.width_y),
            )
    up = (0.0, 0.0, 1.0)
    for j in range(ny - 1):
        for i in range(nx - 1):
            builder.add_quad(
                int(top_index[j, i]),
                int(top_index[j, i + 1]),
                int(top_index[j + 1, i + 1]),
                int(top_index[j + 1, i]),
                up,
            )

    # ---- skirt (vertical sand sides down to -sand_depth) -------------------
    if mesh.skirt_depth > 0.0:
        depth = mesh.skirt_depth
        bottom = z - depth
        header.append("o terrain_skirt")
        header.append(f"usemtl {material_side}")
        borders = (
            ("south", (0.0, -1.0, 0.0), [(xs[i], ys[0], z[0, i], bottom[0, i]) for i in range(nx)]),
            ("north", (0.0, 1.0, 0.0), [(xs[i], ys[-1], z[-1, i], bottom[-1, i]) for i in range(nx)]),
            ("west", (-1.0, 0.0, 0.0), [(xs[0], ys[j], z[j, 0], bottom[j, 0]) for j in range(ny)]),
            ("east", (1.0, 0.0, 0.0), [(xs[-1], ys[j], z[j, -1], bottom[j, -1]) for j in range(ny)]),
        )
        for _name, outward, samples in borders:
            bottom_idx: List[int] = []
            top_idx: List[int] = []
            for (x, y, z_top, z_bot) in samples:
                bottom_idx.append(builder.add_vertex((x, y, z_bot), outward, None))
                top_idx.append(builder.add_vertex((x, y, z_top), outward, None))
            for k in range(len(samples) - 1):
                builder.add_quad(
                    bottom_idx[k], bottom_idx[k + 1], top_idx[k + 1], top_idx[k], outward
                )

    with open(obj_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(header) + "\n")
        handle.write(builder.render())

    # ---- material library --------------------------------------------------
    r, g, b = cfg.sand_color[:3]
    side = tuple(max(0.0, c * 0.72) for c in (r, g, b))
    mtl_path = os.path.join(os.path.dirname(obj_path), mtl_filename)
    with open(mtl_path, "w", encoding="utf-8") as handle:
        handle.write(
            "# cat_arena terrain materials - generated file, do not edit\n"
            f"newmtl {material_top}\n"
            f"Ka {r * 0.25:.4f} {g * 0.25:.4f} {b * 0.25:.4f}\n"
            f"Kd {r:.4f} {g:.4f} {b:.4f}\n"
            "Ks 0.05 0.05 0.05\nNs 8.0\n\n"
            f"newmtl {material_side}\n"
            f"Ka {side[0] * 0.25:.4f} {side[1] * 0.25:.4f} {side[2] * 0.25:.4f}\n"
            f"Kd {side[0]:.4f} {side[1]:.4f} {side[2]:.4f}\n"
            "Ks 0.02 0.02 0.02\nNs 4.0\n"
        )

    return {
        "obj": os.path.basename(obj_path),
        "mtl": mtl_filename,
        "grid": [nx, ny],
        "cell_size": [round(mesh.cell_x, 6), round(mesh.cell_y, 6)],
        "triangles": int(mesh.triangle_count),
        "vertices": int(mesh.vertex_count),
        "z_min": round(mesh.z_min, 6),
        "z_max": round(mesh.z_max, 6),
    }


def write_heightmap_png(path: str, mesh: TerrainMesh, z_min: float, z_max: float) -> Dict[str, object]:
    """Write an 8-bit heightfield PNG artifact plus its encoding metadata.

    This PNG is *not* loaded by the simulation (the mesh drives collision).  It
    exists for analysis, quick inspection and later-module ground truth.
    """
    if z_max <= z_min:
        z_max = z_min + 1e-6
    scaled = (mesh.z - z_min) / (z_max - z_min)
    encoded = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
    write_gray8(path, np.flipud(encoded))
    return {
        "file": os.path.basename(path),
        "z_min": round(float(z_min), 6),
        "z_max": round(float(z_max), 6),
        "quantization_m": round(float((z_max - z_min) / 255.0), 8),
        "rows": int(mesh.ys.size),
        "cols": int(mesh.xs.size),
        "note": "row 0 = north edge (y = width_y); analysis artifact only, never loaded by gz",
    }


def _blur(image: np.ndarray, passes: int = 1) -> np.ndarray:
    """Cheap wrap-around box blur used for procedural textures."""
    out = image.astype(float)
    for _ in range(passes):
        out = 0.25 * (
            np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        )
    return out


def write_sand_albedo(path: str, cfg: ArenaConfig, rng: np.random.Generator) -> Dict[str, object]:
    """Generate a procedural dry-sand albedo texture (RGB8 PNG, visual only)."""
    size = int(cfg.terrain.texture_size)
    base = np.array(cfg.sand_color[:3], dtype=float) * 255.0
    coarse = _blur(rng.normal(0.0, 1.0, (size, size)), passes=6)
    coarse /= max(1e-6, float(np.abs(coarse).max()))
    grain = _blur(rng.normal(0.0, 1.0, (size, size)), passes=1)
    grain /= max(1e-6, float(np.abs(grain).max()))
    modulation = 1.0 + 0.10 * coarse + 0.06 * grain
    img = np.empty((size, size, 3), dtype=np.uint8)
    for channel in range(3):
        img[:, :, channel] = np.clip(base[channel] * modulation, 0, 255).astype(np.uint8)
    write_rgb8(path, img)
    return {"file": os.path.basename(path), "size": [size, size], "type": "procedural_sand_albedo"}
