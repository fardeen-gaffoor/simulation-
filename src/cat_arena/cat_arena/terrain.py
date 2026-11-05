"""Terrain height field and heightmap generation for Gazebo.

The arena sand uses a Gazebo <heightmap> for both collision and visual.
This replaces the old triangle mesh approach.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .config import ArenaConfig
from .png import write_gray8, write_rgb8

__all__ = [
    "Crater",
    "TerrainField",
    "build_heightmap",
    "write_heightmap_png",
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


def write_heightmap_png(path: str, height_m: np.ndarray, z_min: float = 0.0, z_max: float = 0.4) -> Dict[str, object]:
    """Write an 8-bit heightfield PNG for Gazebo heightmap."""
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