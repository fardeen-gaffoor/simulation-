"""Terrain height field, heightmap collision, and visual mesh generation.

The arena sand uses a Gazebo ``<heightmap>`` for collision and a generated
triangle mesh for the visual.  The visual cannot use Ogre2's Terra heightmap
renderer because it unconditionally runs the ``TerraShadowGenerator`` compute
shader, which aborts the GUI on GL 4.1 devices without compute-shader support
(for example Intel iGPUs under Mesa D3D12).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import ArenaConfig
from .png import write_gray8, write_rgb8

__all__ = [
    "Crater",
    "Mesh",
    "TerrainField",
    "build_heightmap",
    "build_mesh",
    "write_heightmap_png",
    "write_heightmap_obj",
    "write_obj_mesh",
    "write_sand_albedo",
    "write_sand_normal",
    "write_sand_roughness",
    "write_sand_textures",
]


@dataclass(frozen=True)
class Crater:
    """A seeded crater: a negative depression in the heightmap."""

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


def _bowl_vec(t: np.ndarray) -> np.ndarray:
    """Vectorized bowl profile."""
    return 0.5 * (1.0 + np.cos(np.pi * t))


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
        self._amp = float(cfg.terrain.noise_amplitude) if hasattr(cfg.terrain, 'noise_amplitude') else 0.0
        self._wavelength = float(cfg.terrain.noise_wavelength) if hasattr(cfg.terrain, 'noise_wavelength') else 3.0
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


def _value_noise_2d(shape: Tuple[int, int], rng: np.random.Generator, frequency: float = 1.0) -> np.ndarray:
    """Generate simple value noise at given frequency."""
    ny, nx = shape
    grid_x = max(2, int(nx / frequency))
    grid_y = max(2, int(ny / frequency))
    grid = rng.normal(0, 1, (grid_y, grid_x)).astype(np.float32)

    x_indices = np.linspace(0, grid_x - 1, nx)
    y_indices = np.linspace(0, grid_y - 1, ny)

    x0 = np.floor(x_indices).astype(int)
    x1 = np.minimum(x0 + 1, grid_x - 1)
    y0 = np.floor(y_indices).astype(int)
    y1 = np.minimum(y0 + 1, grid_y - 1)

    wx = x_indices - x0
    wy = y_indices - y0

    c00 = grid[y0[:, None], x0[None, :]]
    c10 = grid[y0[:, None], x1[None, :]]
    c01 = grid[y1[:, None], x0[None, :]]
    c11 = grid[y1[:, None], x1[None, :]]

    wx = wx[None, :]
    wy = wy[:, None]

    return (1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c10 + (1 - wx) * wy * c01 + wx * wy * c11


def _multi_scale_noise(nx: int, ny: int, rng: np.random.Generator, octaves: int = 5, persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
    """Generate multi-scale noise by summing value noise at different frequencies."""
    result = np.zeros((ny, nx), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        noise = _value_noise_2d((ny, nx), rng, frequency)
        result += amplitude * noise
        amplitude *= persistence
        frequency *= lacunarity

    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def _make_flat_mask(nx: int, ny: int, arena_size_x: float, arena_size_y: float, flat_zones: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """Create boolean mask for flat zones (True = flat)."""
    flat_mask = np.zeros((ny, nx), dtype=bool)
    x_coords = np.linspace(0, arena_size_x, nx)
    y_coords = np.linspace(0, arena_size_y, ny)

    for (x0, y0, x1, y1) in flat_zones:
        x_mask = (x_coords >= x0) & (x_coords <= x1)
        y_mask = (y_coords >= y0) & (y_coords <= y1)
        flat_mask = flat_mask | (x_mask[np.newaxis, :] & y_mask[:, np.newaxis])

    return flat_mask


@dataclass
class Mesh:
    """Triangle mesh over the terrain height field.

    ``z[j, i]`` is the sand height at ``(x = xs[i], y = ys[j])``; ``z_min`` is
    negative at crater centres and ``z_max`` is the nominal (flat) level.
    """

    xs: np.ndarray
    ys: np.ndarray
    z: np.ndarray
    cell_x: float
    cell_y: float
    skirt_depth: float
    triangle_count: int

    @property
    def z_min(self) -> float:
        return float(self.z.min())

    @property
    def z_max(self) -> float:
        return float(self.z.max())


def build_mesh(cfg: ArenaConfig, field: TerrainField) -> Mesh:
    """Sample the analytic sand field on a square grid and build a triangle mesh."""
    cell = float(cfg.terrain.heightmap_resolution)
    nx = int(round(cfg.length_x / cell)) + 1
    ny = int(round(cfg.width_y / cell)) + 1
    xs = np.linspace(0.0, float(cfg.length_x), nx)
    ys = np.linspace(0.0, float(cfg.width_y), ny)
    z = np.asarray(field.height_grid(xs, ys), dtype=np.float64)
    top = int(2 * (nx - 1) * (ny - 1))
    skirt_tris = int(2 * (2 * (nx - 1) + 2 * (ny - 1))) if bool(cfg.terrain.skirt) else 0
    return Mesh(
        xs=xs,
        ys=ys,
        z=z,
        cell_x=cell,
        cell_y=cell,
        skirt_depth=float(cfg.terrain.sand_depth if bool(cfg.terrain.skirt) else 0.0),
        triangle_count=int(top + skirt_tris),
    )


def _mesh_normals(z: np.ndarray) -> np.ndarray:
    ny, nx = z.shape
    nv = np.zeros((ny, nx, 3), dtype=float)
    nv[:, :, 2] = 1.0
    if nx > 2 and ny > 2:
        dzdx = np.gradient(z, axis=1)
        dzdy = np.gradient(z, axis=0)
        nxv = -dzdx
        nyv = -dzdy
        nrm = np.sqrt(nxv * nxv + nyv * nyv + 1.0)
        nv[:, :, 0] = nxv / np.maximum(nrm, 1e-12)
        nv[:, :, 1] = nyv / np.maximum(nrm, 1e-12)
        nv[:, :, 2] = 1.0 / np.maximum(nrm, 1e-12)
    return nv


def write_obj_mesh(path: str, cfg: ArenaConfig, mesh: Mesh) -> Dict[str, object]:
    """Write the terrain height field as an OBJ visual mesh with MTL."""
    ny, nx = int(mesh.z.shape[0]), int(mesh.z.shape[1])
    xs, ys, z = mesh.xs, mesh.ys, np.asarray(mesh.z, dtype=float)
    # Center coordinates on the model origin so the visual mesh lands on the
    # same world surface as the collision heightmap.  The terrain model pose
    # is (length_x/2, width_y/2, max_relief/2), so the mesh spans world
    # [0, length_x] x [0, width_y] x [0, max_relief] exactly like the
    # heightmap.  Row 0 is flipped to be the north edge (y = width_y), matching
    # write_heightmap_png() and Gazebo's heightmap convention.
    cx = 0.5 * float(cfg.length_x)
    cy = 0.5 * float(cfg.width_y)
    z_centered = np.asarray(z, dtype=np.float64) - 0.5 * float(cfg.terrain.max_relief)
    rows = z_centered[::-1]
    normals = _mesh_normals(rows)
    mtl_path = os.path.join(os.path.dirname(os.path.abspath(path)), "terrain.mtl")
    mtl = (
        "newmtl sand\n"
        f"Kd {float(cfg.sand_color[0]):.3f} {float(cfg.sand_color[1]):.3f} {float(cfg.sand_color[2]):.3f} 1.0\n"
        "newmtl sand_side\n"
        f"Kd {float(cfg.sand_color[0]) * 0.7:.3f} {float(cfg.sand_color[1]) * 0.7:.3f} {float(cfg.sand_color[2]) * 0.7:.3f} 1.0\n"
    )
    with open(mtl_path, "w", encoding="utf-8") as handle:
        handle.write(mtl)

    header = [
        "# cat_arena terrain visual mesh - generated file, do not edit",
        f"# grid {nx} x {ny}, {float(cfg.length_x):.3f} x {float(cfg.width_y):.3f} m",
        f"mtllib {os.path.basename(mtl_path)}",
        "o terrain_top",
    ]
    v_lines: List[str] = []
    for j in range(ny):
        y = cy - float(ys[j])
        for i in range(nx):
            v_lines.append(f"v {float(xs[i]) - cx:.6f} {y:.6f} {float(rows[j, i]):.6f}")
    vt_lines: List[str] = []
    for j in range(ny):
        v = j / float(max(1, ny - 1))
        for i in range(nx):
            u = i / float(max(1, nx - 1))
            vt_lines.append(f"vt {u:.6f} {v:.6f}")
    vn_lines: List[str] = []
    for j in range(ny):
        for i in range(nx):
            vn_lines.append(f"vn {float(normals[j, i, 0]):.6f} {float(normals[j, i, 1]):.6f} {float(normals[j, i, 2]):.6f}")

    def _face(a: int, b: int, c: int) -> str:
        return f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}"

    grid_faces: List[str] = []
    for j in range(ny - 1):
        row0 = j * nx
        row1 = (j + 1) * nx
        for i in range(nx - 1):
            a = row0 + i + 1
            b = row0 + i + 2
            c = row1 + i + 2
            d = row1 + i + 1
            grid_faces.append(_face(a, c, b))
            grid_faces.append(_face(a, d, c))

    skirt_v: List[str] = []
    skirt_vt: List[str] = []
    skirt_vn: List[str] = []
    skirt_faces: List[str] = []
    skirt_vert_count = 0
    z_arr = np.asarray(mesh.z, dtype=np.float64)
    if bool(cfg.terrain.skirt) and float(mesh.skirt_depth) > 0.0:
        skirt_depth = float(mesh.skirt_depth)
        # Closed CCW boundary loop of grid slots (x right, y up, viewed from +z).
        loop = (
            [i for i in range(nx)]
            + [j * nx + (nx - 1) for j in range(1, ny)]
            + [(ny - 1) * nx + i for i in range(nx - 2, -1, -1)]
            + [j * nx for j in range(ny - 2, 0, -1)]
        )
        m = len(loop)
        for k in range(m):
            u = int(loop[k])
            v = int(loop[(k + 1) % m])
            ui = int(nx * ny) + skirt_vert_count + 1
            skirt_vert_count += 1
            vi = int(nx * ny) + skirt_vert_count + 1
            skirt_vert_count += 1
            ju, iu = divmod(u, nx)
            jv, iv = divmod(v, nx)
            skirt_v.append(f"v {float(xs[iu]) - cx:.6f} {cy - float(ys[ju]):.6f} {float(z_arr[ju, iu] - 0.5 * cfg.terrain.max_relief - skirt_depth):.6f}")
            skirt_v.append(f"v {float(xs[iv]) - cx:.6f} {cy - float(ys[jv]):.6f} {float(z_arr[jv, iv] - 0.5 * cfg.terrain.max_relief - skirt_depth):.6f}")
            skirt_vt.append(f"vt {float(iu) / max(1, nx - 1):.6f} {float(ju) / max(1, ny - 1):.6f}")
            skirt_vn.append(f"vn {float(normals[ju, iu, 0]):.6f} {float(normals[ju, iu, 1]):.6f} {float(normals[ju, iu, 2]):.6f}")
            skirt_vt.append(f"vt {float(iv) / max(1, nx - 1):.6f} {float(jv) / max(1, ny - 1):.6f}")
            skirt_vn.append(f"vn {float(normals[jv, iv, 0]):.6f} {float(normals[jv, iv, 1]):.6f} {float(normals[jv, iv, 2]):.6f}")
            skirt_faces.append(_face(u + 1, ui, vi))
            skirt_faces.append(_face(u + 1, vi, v + 1))

    if skirt_faces:
        v_lines.extend(skirt_v)
        vt_lines.extend(skirt_vt)
        vn_lines.extend(skirt_vn)
    lines = header + v_lines + vt_lines + vn_lines
    lines.append("usemtl sand")
    lines.extend(grid_faces)
    lines.extend(skirt_faces)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    return {
        "file": os.path.basename(path),
        "vertices": int(nx * ny) + len(skirt_v),
        "triangles": int(mesh.triangle_count),
        "mtl": os.path.basename(mtl_path),
        "z_min": float(np.min(z_arr)),
        "z_max": float(np.max(z_arr)),
        "rows": int(ny),
        "cols": int(nx),
    }


def write_sand_albedo(path: str, cfg: ArenaConfig, rng: np.random.Generator) -> Dict[str, object]:
    """Procedural sand albedo: damp/dry patches, grain and ripples.  RGB PNG."""
    size = int(cfg.terrain.texture_size)
    base = np.asarray(cfg.sand_color[:3], dtype=float) * 255.0
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


def write_sand_normal(path: str, cfg: ArenaConfig, rng: np.random.Generator) -> Dict[str, object]:
    """Procedural sand normal map from multi-scale noise.  Tangent-space RGB PNG."""
    size = int(cfg.terrain.texture_size)
    height_detail = _multi_scale_noise(size, size, rng, octaves=4, persistence=0.5)
    dx = np.zeros((size, size), dtype=np.float32)
    dx[:, 1:-1] = (height_detail[:, 2:] - height_detail[:, :-2]) * 0.5
    dx[:, 0] = height_detail[:, 1] - height_detail[:, 0]
    dx[:, -1] = height_detail[:, -1] - height_detail[:, -2]
    dy = np.zeros((size, size), dtype=np.float32)
    dy[1:-1, :] = (height_detail[2:, :] - height_detail[:-2, :]) * 0.5
    dy[0, :] = height_detail[1, :] - height_detail[0, :]
    dy[-1, :] = height_detail[-1, :] - height_detail[-2, :]
    norm = np.sqrt(dx * dx + dy * dy + 1.0)
    nx = -dx / norm
    ny = -dy / norm
    nz = 1.0 / norm
    normal_map = np.empty((size, size, 3), dtype=np.uint8)
    normal_map[:, :, 0] = ((nx + 1.0) * 0.5 * 255).astype(np.uint8)
    normal_map[:, :, 1] = ((ny + 1.0) * 0.5 * 255).astype(np.uint8)
    normal_map[:, :, 2] = ((nz + 1.0) * 0.5 * 255).astype(np.uint8)
    write_rgb8(path, normal_map)
    return {"file": os.path.basename(path), "size": [size, size], "type": "procedural_sand_normal"}


def write_sand_roughness(path: str, cfg: ArenaConfig, rng: np.random.Generator) -> Dict[str, object]:
    """Procedural sand roughness map from value noise.  Grayscale PNG."""
    size = int(cfg.terrain.texture_size)
    rough = _blur(rng.normal(0.0, 1.0, (size, size)), passes=4)
    rough = np.clip(rough / max(1e-6, float(np.abs(rough).max())), 0.0, 1.0)
    write_gray8(path, (rough * 255.0).astype(np.uint8))
    return {"file": os.path.basename(path), "size": [size, size], "type": "procedural_sand_roughness"}


def build_heightmap(
    cfg: ArenaConfig,
    craters: Sequence[Crater],
    seed: int = 42,
    flat_zones: Optional[List[Tuple[float, float, float, float]]] = None,
) -> np.ndarray:
    """Build heightmap array in meters (0 = base, max_relief = max height).

    Gazebo heightmaps must be square (2^n+1 x 2^n+1). We use the larger dimension.
    """
    rng = np.random.default_rng(seed)

    nx = int(round(cfg.length_x / cfg.terrain.heightmap_resolution)) + 1
    ny = int(round(cfg.width_y / cfg.terrain.heightmap_resolution)) + 1

    # Ensure 2^n + 1 for heightmap compatibility (Gazebo requirement - MUST BE SQUARE)
    def next_pow2_plus1(n):
        p = 1
        while p + 1 < n:
            p <<= 1
        return p + 1

    # Use the same size for both dimensions (square heightmap required by Gazebo)
    n = max(nx, ny)
    n = next_pow2_plus1(n)
    nx = ny = n

    max_relief = cfg.terrain.max_relief

    # Generate base dune noise
    noise = _multi_scale_noise(nx, ny, rng, octaves=5)
    height = noise * max_relief

    # Add craters (negative depressions)
    for crater in craters:
        radius_m = crater.radius
        depth_m = crater.depth

        cx_m = crater.x
        cy_m = crater.y

        # Check if center falls in flat zone
        in_flat_zone = False
        if flat_zones:
            for (x0, y0, x1, y1) in flat_zones:
                if x0 - radius_m - 0.1 <= cx_m <= x1 + radius_m + 0.1 and \
                   y0 - radius_m - 0.1 <= cy_m <= y1 + radius_m + 0.1:
                    in_flat_zone = True
                    break

        if in_flat_zone:
            continue

        # Convert to pixel coordinates
        cx_px = cx_m / cfg.length_x * (nx - 1)
        cy_px = cy_m / cfg.width_y * (ny - 1)
        radius_px = radius_m / cfg.length_x * (nx - 1)

        # Create crater mask
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dist = np.sqrt((x_grid - cx_px) ** 2 + (y_grid - cy_px) ** 2)

        mask = dist < radius_px
        if not np.any(mask):
            continue

        t = np.where(mask, dist / radius_px, 1.0)
        profile = np.where(mask, _bowl_vec(t), 0.0)

        # Subtract crater (negative height)
        height = np.where(mask, height - depth_m * profile, height)

    # Apply flat zones mask
    if flat_zones:
        flat_mask = _make_flat_mask(nx, ny, cfg.length_x, cfg.width_y, flat_zones)
        height = np.where(flat_mask, 0.0, height)

    # Ensure non-negative (Gazebo heightmaps expect 0 to max)
    height = np.maximum(height, 0.0)

    return height


def write_heightmap_png(path: str, height_m: Any, z_min: float = 0.0, z_max: float = 0.4) -> Dict[str, object]:
    """Write an 8-bit heightfield PNG for Gazebo heightmap.

    ``height_m`` may be a plain ``(rows, cols)`` array of heights in metres or
    the :class:`Mesh`` built by :func:`build_mesh` (in which case ``mesh.z`` is
    used and the nominal surface is re-derived from the mesh bounds).
    """
    if isinstance(height_m, Mesh):
        height_m = np.asarray(height_m.z, dtype=np.float64)
    height_m = np.asarray(height_m, dtype=np.float64)
    if height_m.ndim != 2:
        raise ValueError(f"write_heightmap_png: expected a 2-D array, got {height_m.shape}")
    if z_max <= z_min:
        z_max = z_min + 1e-6

    # Normalize to 0-255
    height_norm = (height_m - z_min) / (z_max - z_min)
    height_norm = np.clip(height_norm, 0.0, 1.0)
    height_uint8 = (height_norm * 255).astype(np.uint8)

    # Flip vertically because image origin is top-left, arena origin is bottom-left
    height_uint8 = np.flipud(height_uint8)

    write_gray8(path, height_uint8)

    return {
        "file": os.path.basename(path),
        "z_min": round(float(z_min), 6),
        "z_max": round(float(z_max), 6),
        "quantization_m": round(float((z_max - z_min) / 255.0), 8),
        "rows": int(height_m.shape[0]),
        "cols": int(height_m.shape[1]),
        "note": "row 0 = north edge (y = width_y); used by Gazebo heightmap",
    }


def write_heightmap_obj(path: str, height_m: np.ndarray, cfg: ArenaConfig) -> Dict[str, object]:
    """Write the height field as an OBJ visual mesh.

    The *collision* geometry remains a Gazebo ``<heightmap>``.  The visual is a
    plain triangle mesh because Ogre2/Terra's heightmap renderer runs a compute
    shader (``TerraShadowGenerator``) unconditionally; on GL 4.1 devices such as
    Intel iGPUs under Mesa D3D12 compute shaders are unavailable and the GUI
    aborts while compiling it.  A mesh visual has no Terra dependency and is
    safe on all of those devices.

    Vertices are expressed in the local frame of the terrain model, whose pose
    is ``(length_x/2, width_y/2, max_relief/2)``, so the mesh lands on exactly
    the same world surface as the collision heightmap.
    """
    z = np.asarray(height_m, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"write_heightmap_obj: expected a 2-D array, got {z.shape}")

    ny, nx = z.shape
    size_x = float(cfg.length_x)
    size_y = float(cfg.width_y)
    z_max = max(float(cfg.terrain.max_relief), 1e-6)

    # Match the 8-bit quantization used by the collision heightmap PNG so the
    # visible surface and the collision surface agree to within ~1.6 mm.
    z = np.round(np.clip(z, 0.0, z_max) / z_max * 255.0) / 255.0 * z_max

    xs = np.linspace(0.0, size_x, nx)
    ys = np.linspace(0.0, size_y, ny)
    lx = xs - 0.5 * size_x

    # Gazebo/DART reads heightmap row 0 as the *north* edge and rows advance
    # towards -y, which is why write_heightmap_png() stores the image flipped.
    # Do the same here so the visual surface and the collision heightfield
    # agree, including under the rover.
    rows = z[::-1]
    ly = 0.5 * size_y - ys
    lz = rows - 0.5 * z_max

    dz_dy, dz_dx = np.gradient(rows, ly, xs)
    nxv = -dz_dx
    nyv = -dz_dy
    nzv = np.ones_like(rows)
    norm = np.sqrt(nxv * nxv + nyv * nyv + nzv * nzv)
    nxv /= norm
    nyv /= norm
    nzv /= norm

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# cat_arena terrain visual mesh - generated file, do not edit\n")
        handle.write(f"# grid {nx} x {ny}, {size_x:.3f} x {size_y:.3f} m\n")
        handle.write("o terrain_top\n")

        # OBJ indices are 1-based and row-major: index = j * nx + i + 1.
        for j in range(ny):
            y = ly[j]
            for i in range(nx):
                handle.write(f"v {lx[i]:.6f} {y:.6f} {lz[j, i]:.6f}\n")

        for j in range(ny):
            v = j / float(max(1, ny - 1))
            for i in range(nx):
                u = i / float(max(1, nx - 1))
                handle.write(f"vt {u:.6f} {v:.6f}\n")

        for j in range(ny):
            for i in range(nx):
                handle.write(f"vn {nxv[j, i]:.6f} {nyv[j, i]:.6f} {nzv[j, i]:.6f}\n")

        for j in range(ny - 1):
            row0 = j * nx
            row1 = (j + 1) * nx
            for i in range(nx - 1):
                a = row0 + i + 1
                b = row0 + i + 2
                c = row1 + i + 2
                d = row1 + i + 1
                # Rows advance south (-y), so keep the winding CCW seen from
                # above (+z) or the whole surface is backface-culled.
                handle.write(f"f {a}/{a}/{a} {c}/{c}/{c} {b}/{b}/{b}\n")
                handle.write(f"f {a}/{a}/{a} {d}/{d}/{d} {c}/{c}/{c}\n")

    return {
        "file": os.path.basename(path),
        "rows": int(ny),
        "cols": int(nx),
        "triangles": int(2 * (nx - 1) * (ny - 1)),
        "vertices": int(nx * ny),
        "z_min": round(float(z.min()), 6),
        "z_max": round(float(z.max()), 6),
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


def _tex_coord(value_m: float, span_m: float, size_px: int) -> int:
    """Map an arena-frame metre value to a texture pixel index."""
    return int(round((value_m / span_m) * size_px))


def _paint_surface_features(
    albedo: np.ndarray, cfg: ArenaConfig, rng: np.random.Generator, size: int
) -> np.ndarray:
    """Paint Stage B surface disturbance onto the sand albedo texture.

    Adds, deterministically from ``rng``:

    * parallel track bands with cleat ridges (tracked-rover marks),
    * random dark damp footprints,
    * light chalk zone lines at zone boundaries (slightly irregular/broken).

    ``albedo`` is a float ``(h, w, 3)`` array in 0-255 and is updated in place
    (also returned for convenience).
    """
    length_x = float(cfg.length_x)
    width_y = float(cfg.width_y)
    base_color = np.asarray(cfg.sand_color[:3], dtype=float)

    # ------------------------------------------------------------------ #
    # track bands with cleat ridges                                    #
    # ------------------------------------------------------------------ #
    band_width_m = 0.20
    band_count = 2
    n_bands = max(1, int(np.ceil(width_y / band_width_m - 1e-9)))
    band_count = max(1, min(band_count, n_bands))
    track_color = base_color * 0.85
    cleat_color = base_color * 0.70
    cleat_pitch_m = 0.06
    for band in range(band_count):
        yc = (float(band) + 0.5) * band_width_m
        y_lo = max(0.0, yc - 0.5 * band_width_m)
        y_hi = min(width_y, yc + 0.5 * band_width_m)
        r_lo = max(0, _tex_coord(y_lo, width_y, size))
        r_hi = min(size - 1, _tex_coord(y_hi, width_y, size))
        if r_hi < r_lo:
            continue
        for r in range(r_lo, r_hi + 1):
            albedo[r, :] = track_color
        # cleat ridges across the band
        x = 0.0
        while x <= length_x + 1e-9:
            c = _tex_coord(x, length_x, size)
            c2 = _tex_coord(x + 0.5 * cleat_pitch_m, length_x, size)
            if 0 <= c < size and 0 <= c2 < size and c2 > c:
                for r in range(r_lo, r_hi + 1):
                    albedo[r, c:c2] = cleat_color
            x += cleat_pitch_m

    # ------------------------------------------------------------------ #
    # footprints: random dark damp blobs                               #
    # ------------------------------------------------------------------ #
    footprint_color = np.array([118.0, 103.0, 84.0], dtype=float)
    for _ in range(8):
        fx = float(rng.uniform(0.0, length_x))
        fy = float(rng.uniform(0.0, width_y))
        rx_m = 0.05
        ry_m = 0.04
        cc = _tex_coord(fx, length_x, size)
        rc = _tex_coord(fy, width_y, size)
        cr = max(1, int(rx_m * size / length_x))
        rr = max(1, int(ry_m * size / width_y))
        for r in range(max(0, rc - rr), min(size, rc + rr + 1)):
            for c in range(max(0, cc - cr), min(size, cc + cr + 1)):
                if ((r - rc) ** 2 / max(1.0, rr * rr) + (c - cc) ** 2 / max(1.0, cr * cr)) <= 1.0:
                    albedo[r, c] = footprint_color

    # ------------------------------------------------------------------ #
    # chalk zone lines: light, jittered and broken                     #
    # ------------------------------------------------------------------ #
    chalk_color = np.array([238.0, 238.0, 230.0], dtype=float)
    chalk_half = 1.0  # pixels
    for zone in cfg.zones.values():
        rect = zone.rect
        for yy in (float(rect.y0), float(rect.y1)):
            r = _tex_coord(yy, width_y, size)
            if not (0 <= r < size):
                continue
            for xx in np.linspace(float(rect.x0), float(rect.x1), max(2, int(abs(float(rect.x1) - float(rect.x0)) / 0.05) + 1)):
                c = _tex_coord(float(xx), length_x, size)
                if not (0 <= c < size):
                    continue
                cj = int(round(c + float(rng.normal(0.0, chalk_half))))
                rj = int(round(r + float(rng.normal(0.0, chalk_half))))
                if 0 <= cj < size and 0 <= rj < size:
                    albedo[rj, cj] = chalk_color
        for xx in (float(rect.x0), float(rect.x1)):
            c = _tex_coord(float(xx), length_x, size)
            if not (0 <= c < size):
                continue
            for yy in np.linspace(float(rect.y0), float(rect.y1), max(2, int(abs(float(rect.y1) - float(rect.y0)) / 0.05) + 1)):
                r = _tex_coord(float(yy), width_y, size)
                if not (0 <= r < size):
                    continue
                rj = int(round(r + float(rng.normal(0.0, chalk_half))))
                if 0 <= rj < size:
                    albedo[rj, c] = chalk_color

    return albedo


def write_sand_textures(
    diffuse_path: str,
    normal_path: Optional[str],
    cfg: ArenaConfig,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Generate procedural sand diffuse and normal map textures."""
    size = int(cfg.terrain.texture_size)
    base = np.array(cfg.sand_color[:3], dtype=float) * 255.0

    # Diffuse texture
    coarse = _blur(rng.normal(0.0, 1.0, (size, size)), passes=6)
    coarse /= max(1e-6, float(np.abs(coarse).max()))
    grain = _blur(rng.normal(0.0, 1.0, (size, size)), passes=1)
    grain /= max(1e-6, float(np.abs(grain).max()))
    modulation = 1.0 + 0.10 * coarse + 0.06 * grain

    img = np.empty((size, size, 3), dtype=np.uint8)
    for channel in range(3):
        img[:, :, channel] = np.clip(base[channel] * modulation, 0, 255).astype(np.uint8)

    # Stage B: painted surface disturbance (tracks, footprints, chalk lines).
    albedo = _paint_surface_features(img.astype(float), cfg, rng, size)
    img = np.clip(albedo, 0, 255).astype(np.uint8)

    write_rgb8(diffuse_path, img)

    result = {"diffuse": {"file": os.path.basename(diffuse_path), "size": [size, size], "type": "procedural_sand_diffuse"}}

    # Normal map
    if normal_path and cfg.terrain.normal_map:
        height_detail = _multi_scale_noise(size, size, rng, octaves=4, persistence=0.5)

        # Compute gradients
        dx = np.zeros((size, size), dtype=np.float32)
        dx[:, 1:-1] = (height_detail[:, 2:] - height_detail[:, :-2]) * 0.5
        dx[:, 0] = height_detail[:, 1] - height_detail[:, 0]
        dx[:, -1] = height_detail[:, -1] - height_detail[:, -2]

        dy = np.zeros((size, size), dtype=np.float32)
        dy[1:-1, :] = (height_detail[2:, :] - height_detail[:-2, :]) * 0.5
        dy[0, :] = height_detail[1, :] - height_detail[0, :]
        dy[-1, :] = height_detail[-1, :] - height_detail[-2, :]

        # Normal vector: (-dx, -dy, 1) normalized
        norm = np.sqrt(dx * dx + dy * dy + 1.0)
        nx = -dx / norm
        ny = -dy / norm
        nz = 1.0 / norm

        # Convert to [0, 255] range for normal map (tangent space)
        normal_map = np.empty((size, size, 3), dtype=np.uint8)
        normal_map[:, :, 0] = ((nx + 1.0) * 0.5 * 255).astype(np.uint8)
        normal_map[:, :, 1] = ((ny + 1.0) * 0.5 * 255).astype(np.uint8)
        normal_map[:, :, 2] = ((nz + 1.0) * 0.5 * 255).astype(np.uint8)

        write_rgb8(normal_path, normal_map)
        result["normal"] = {"file": os.path.basename(normal_path), "size": [size, size], "type": "procedural_sand_normal"}

    return result


# --------------------------------------------------------------------------- #
# procedural rock meshes                                                      #
# --------------------------------------------------------------------------- #
def write_background_png(path: str, size: int = 2048, seed: int = 0) -> Dict[str, object]:
    """Stage C: blurred backdrop image (tent / crowd / orange ground).

    Builds an orange-ground base with blurred noise and a few darker patches to
    suggest a tent/crowd beyond the arena walls, then heavily blurs it so it
    reads as a soft out-of-focus backdrop rather than sharp detail.
    """
    rng = np.random.default_rng(seed)
    base = np.array([205, 133, 63], dtype=float)  # orange ground
    img = np.empty((size, size, 3), dtype=float)
    for channel in range(3):
        img[:, :, channel] = base[channel]
    noise = _blur(rng.normal(0.0, 1.0, (size, size)), passes=10)
    noise /= max(1e-6, float(np.abs(noise).max()))
    img += 28.0 * noise[:, :, None]
    for _ in range(6):
        cx = int(rng.integers(0, size))
        cy = int(rng.integers(0, size))
        rr = int(rng.uniform(0.08, 0.22) * size)
        yy, xx = np.mgrid[0:size, 0:size]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < rr * rr
        img[mask] *= 0.6
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = _blur(img.astype(float), passes=12).astype(np.uint8)
    write_rgb8(path, img)
    return {"file": os.path.basename(path), "size": [size, size], "type": "procedural_background"}



def _rock_vertices_faces(
    rng: np.random.Generator,
    center: Tuple[float, float, float],
    radius: float,
    segs_u: int = 20,
    segs_v: int = 14,
) -> Tuple[List[str], List[str], List[str]]:
    """Build an irregular convex-ish rock blob as a perturbed sphere.

    Returns (verts, faces, normals) with 1-based OBJ index references.
    """
    cx, cy, cz = center
    verts: List[str] = []
    norms: List[str] = []
    index: Dict[Tuple[int, int], int] = {}
    grid: List[Tuple[float, float, float]] = []
    for j in range(segs_v + 1):
        phi = math.pi * j / segs_v
        for i in range(segs_u):
            theta = 2.0 * math.pi * i / segs_u
            radial = radius * (1.0 + 0.25 * float(rng.normal(0.0, 1.0)))
            nx_ = math.sin(phi) * math.cos(theta)
            ny_ = math.sin(phi) * math.sin(theta)
            nz_ = math.cos(phi)
            grid.append(
                (
                    cx + radial * nx_,
                    cy + radial * ny_,
                    cz + radial * nz_,
                )
            )
            norms.append(f"vn {nx_:.6f} {ny_:.6f} {nz_:.6f}")

    def vid(i: int, j: int) -> int:
        key = (i, j)
        if key not in index:
            index[key] = len(verts) + 1
            x, y, z = grid[j * segs_u + i]
            verts.append(f"v {x:.6f} {y:.6f} {z:.6f}")
        return index[key]

    faces: List[str] = []
    for j in range(segs_v):
        for i in range(segs_u - 1):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i + 1, j + 1)
            d = vid(i, j + 1)
            # CCW from the outside (+normal)
            faces.append(f"f {a}/{a}/{a} {c}/{c}/{c} {b}/{b}/{b}")
            faces.append(f"f {a}/{a}/{a} {d}/{d}/{d} {c}/{c}/{c}")
    return verts, norms, faces


def write_rock_meshes(
    out_dir: str,
    boulders: Sequence[Boulder],
    seed: int,
    palette: Sequence[str],
) -> Dict[str, str]:
    """Write one procedural rock OBJ (+MTL) per boulder.

    Returns a mapping ``boulder_id -> obj filename``.  Rock meshes are
    deterministic for a given seed and boulder index.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}
    mtl_path = os.path.join(out_dir, "rock.mtl")
    with open(mtl_path, "w", encoding="utf-8") as handle:
        handle.write("# cat_arena procedural rock material - generated file, do not edit\n")
        for i, hexcolor in enumerate(palette):
            rgb = [int(hexcolor.lstrip("#")[k : k + 2], 16) for k in (0, 2, 4)]
            handle.write(f"newmtl rock_{i}\n")
            handle.write(f"Kd {rgb[0]/255.0:.3f} {rgb[1]/255.0:.3f} {rgb[2]/255.0:.3f} 1.0\n")
        handle.write("newmtl default\n")
        handle.write("Kd 0.70 0.45 0.25 1.0\n")

    for i, boulder in enumerate(boulders):
        rng = np.random.default_rng(seed * 1000003 + i)
        radius = 0.5 * boulder.diameter
        center = (boulder.x, boulder.y, boulder.center_z)
        verts, norms, faces = _rock_vertices_faces(rng, center, radius)
        obj_path = os.path.join(out_dir, f"rock_{boulder.boulder_id}.obj")
        mtl_name = f"rock_{i % len(palette)}.mtl" if palette else "rock.mtl"
        with open(obj_path, "w", encoding="utf-8") as handle:
            handle.write("# cat_arena procedural rock - generated file, do not edit\n")
            handle.write("mtllib rock.mtl\n")
            handle.write(f"o {boulder.boulder_id}\n")
            handle.write("usemtl default\n")
            handle.write("\n".join(verts) + "\n")
            handle.write("\n".join(norms) + "\n")
            handle.write("\n".join(faces) + "\n")
        paths[boulder.boulder_id] = os.path.basename(obj_path)
    return paths


__all__ = [
    "Mesh",
    "TerrainField",
    "Crater",
    "build_mesh",
    "build_heightmap",
    "write_obj_mesh",
    "write_heightmap_png",
    "write_heightmap_obj",
    "write_sand_albedo",
    "write_sand_normal",
    "write_sand_roughness",
    "write_sand_textures",
    "_paint_surface_features",
    "write_rock_meshes",
    "_rock_vertices_faces",
]
