#!/usr/bin/env python3
"""
Generate a 257x257 grayscale height.png with multi-scale dune noise and craters.
Keeps start, excavation, and berm zones perfectly flat by masking.

Usage:
    python3 tools/make_height.py --seed 7 --output height.png
    python3 tools/make_height.py --seed 7 --output height.png --mask-zones config/arena.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from PIL import Image

# Import cat_arena config to get zone coordinates
sys.path.insert(0, "/home/fewest/cat/src")
from cat_arena.config import load_config, ArenaConfig, Rect


@dataclass
class CratersConfig:
    count: int = 12
    min_radius: float = 0.125
    max_radius: float = 0.225
    min_depth: float = 0.06
    max_depth: float = 0.15
    margin: float = 0.4


def bowl_profile(t: np.ndarray) -> np.ndarray:
    """Smooth crater profile: 1 at center, 0 at rim, zero slope at both."""
    return 0.5 * (1.0 + np.cos(np.pi * t))


def value_noise_2d(shape: Tuple[int, int], rng: np.random.Generator, frequency: float = 1.0) -> np.ndarray:
    """Generate simple value noise at given frequency."""
    ny, nx = shape
    # Grid of random values at frequency
    grid_x = max(2, int(nx / frequency))
    grid_y = max(2, int(ny / frequency))
    grid = rng.normal(0, 1, (grid_y, grid_x)).astype(np.float32)

    # Bilinear interpolation to full resolution
    x_indices = np.linspace(0, grid_x - 1, nx)
    y_indices = np.linspace(0, grid_y - 1, ny)

    x0 = np.floor(x_indices).astype(int)
    x1 = np.minimum(x0 + 1, grid_x - 1)
    y0 = np.floor(y_indices).astype(int)
    y1 = np.minimum(y0 + 1, grid_y - 1)

    wx = x_indices - x0
    wy = y_indices - y0

    # Vectorized bilinear interpolation
    c00 = grid[y0[:, None], x0[None, :]]
    c10 = grid[y0[:, None], x1[None, :]]
    c01 = grid[y1[:, None], x0[None, :]]
    c11 = grid[y1[:, None], x1[None, :]]

    wx = wx[None, :]
    wy = wy[:, None]

    return (1 - wx) * (1 - wy) * c00 + wx * (1 - wy) * c10 + (1 - wx) * wy * c01 + wx * wy * c11


def multi_scale_noise(
    nx: int,
    ny: int,
    rng: np.random.Generator,
    octaves: int = 5,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Generate multi-scale noise by summing value noise at different frequencies."""
    result = np.zeros((ny, nx), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        noise = value_noise_2d((ny, nx), rng, frequency)
        result += amplitude * noise
        amplitude *= persistence
        frequency *= lacunarity

    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def generate_heightmap(
    seed: int = 42,
    nx: int = 257,
    ny: int = 257,
    arena_size_x: float = 9.0,
    arena_size_y: float = 5.0,
    max_relief: float = 0.4,
    craters_cfg: Optional[CratersConfig] = None,
    flat_zones: Optional[List[Rect]] = None,
    noise_amplitude: float = 1.0,
) -> np.ndarray:
    """Generate heightmap as float array in meters (0 = base, max_relief = max height)."""
    rng = np.random.default_rng(seed)

    if craters_cfg is None:
        craters_cfg = CratersConfig()

    # Generate base dune noise
    noise = multi_scale_noise(nx, ny, rng, octaves=5)

    # Scale to max relief
    height = noise * max_relief * noise_amplitude

    # Add craters (negative depressions)
    for i in range(craters_cfg.count):
        # Random crater parameters
        radius_m = rng.uniform(craters_cfg.min_radius, craters_cfg.max_radius)
        depth_m = rng.uniform(craters_cfg.min_depth, craters_cfg.max_depth)

        # Random position (in arena coordinates)
        cx_m = rng.uniform(craters_cfg.margin, arena_size_x - craters_cfg.margin)
        cy_m = rng.uniform(craters_cfg.margin, arena_size_y - craters_cfg.margin)

        # Check if center falls in flat zone
        in_flat_zone = False
        if flat_zones:
            for zone in flat_zones:
                if zone.contains(cx_m, cy_m, margin=radius_m + 0.1):
                    in_flat_zone = True
                    break

        if in_flat_zone:
            continue  # Skip this crater

        # Convert to pixel coordinates
        cx_px = cx_m / arena_size_x * (nx - 1)
        cy_px = cy_m / arena_size_y * (ny - 1)
        radius_px = radius_m / arena_size_x * (nx - 1)

        # Create crater mask
        y_grid, x_grid = np.ogrid[:ny, :nx]
        dist = np.sqrt((x_grid - cx_px) ** 2 + (y_grid - cy_px) ** 2)

        mask = dist < radius_px
        if not np.any(mask):
            continue

        t = np.where(mask, dist / radius_px, 1.0)
        profile = np.where(mask, bowl_profile(t), 0.0)

        # Subtract crater (negative height)
        height = np.where(mask, height - depth_m * profile, height)

    # Apply flat zones mask
    if flat_zones:
        # Create mask for flat zones
        flat_mask = np.zeros((ny, nx), dtype=bool)
        x_coords = np.linspace(0, arena_size_x, nx)
        y_coords = np.linspace(0, arena_size_y, ny)

        for zone in flat_zones:
            x_mask = (x_coords >= zone.x0) & (x_coords <= zone.x1)
            y_mask = (y_coords >= zone.y0) & (y_coords <= zone.y1)
            flat_mask = flat_mask | (x_mask[np.newaxis, :] & y_mask[:, np.newaxis])

        # Set flat zones to zero height (base level)
        height = np.where(flat_mask, 0.0, height)

    # Ensure non-negative (Gazebo heightmaps expect 0 to max)
    height = np.maximum(height, 0.0)

    return height


def height_to_png(height_m: np.ndarray, output_path: str, z_min: float = 0.0, z_max: float = 0.4):
    """Convert height array in meters to 8-bit PNG."""
    # Normalize to 0-255
    height_norm = (height_m - z_min) / (z_max - z_min)
    height_norm = np.clip(height_norm, 0.0, 1.0)
    height_uint8 = (height_norm * 255).astype(np.uint8)

    # Flip vertically because image origin is top-left, arena origin is bottom-left
    height_uint8 = np.flipud(height_uint8)

    img = Image.fromarray(height_uint8, mode='L')
    img.save(output_path)
    print(f"Saved heightmap to {output_path} ({height_uint8.shape[1]}x{height_uint8.shape[0]})")


def main():
    parser = argparse.ArgumentParser(description="Generate heightmap for arena terrain")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="height.png", help="Output PNG path")
    parser.add_argument("--nx", type=int, default=257, help="Width in pixels (must be 2^n+1 for heightmap)")
    parser.add_argument("--ny", type=int, default=257, help="Height in pixels")
    parser.add_argument("--max-relief", type=float, default=0.4, help="Max height variation in meters")
    parser.add_argument("--arena-x", type=float, default=9.0, help="Arena size X in meters")
    parser.add_argument("--arena-y", type=float, default=5.0, help="Arena size Y in meters")
    parser.add_argument("--craters", type=int, default=12, help="Number of craters")
    parser.add_argument("--mask-zones", type=str, help="Path to arena.yaml to get flat zone coordinates")
    parser.add_argument("--noise-amplitude", type=float, default=1.0, help="Multiplier for base noise")

    args = parser.parse_args()

    # Load flat zones from arena.yaml if provided
    flat_zones = None
    if args.mask_zones:
        cfg = load_config(args.mask_zones)
        # Flat zones: start, construction, berm_target, left_corridor
        zone_names = ['start', 'construction', 'berm_target', 'left_corridor']
        flat_zones = [cfg.zone(name).rect for name in zone_names]
        print(f"Masking {len(flat_zones)} flat zones: {zone_names}")

    # Crater config
    craters_cfg = CratersConfig(count=args.craters)

    # Generate heightmap
    height = generate_heightmap(
        seed=args.seed,
        nx=args.nx,
        ny=args.ny,
        arena_size_x=args.arena_x,
        arena_size_y=args.arena_y,
        max_relief=args.max_relief,
        craters_cfg=craters_cfg,
        flat_zones=flat_zones,
        noise_amplitude=args.noise_amplitude,
    )

    # Save as PNG
    height_to_png(height, args.output, z_min=0.0, z_max=args.max_relief)

    # Print stats
    print(f"Height stats: min={height.min():.4f}m, max={height.max():.4f}m, mean={height.mean():.4f}m")


if __name__ == "__main__":
    main()