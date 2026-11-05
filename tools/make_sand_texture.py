#!/usr/bin/env python3
"""
Generate procedural sand textures (diffuse + normal map) for heightmap.

Usage:
    python3 tools/make_sand_texture.py --size 1024 --output-dir /tmp/sand_textures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/fewest/cat/src")


def value_noise_2d(shape, rng, frequency=1.0):
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


def multi_scale_noise(shape, rng, octaves=6, persistence=0.5, lacunarity=2.0):
    ny, nx = shape
    result = np.zeros((ny, nx), dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0

    for _ in range(octaves):
        noise = value_noise_2d((ny, nx), rng, frequency)
        result += amplitude * noise
        amplitude *= persistence
        frequency *= lacunarity

    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def generate_sand_diffuse(size: int, seed: int = 42) -> np.ndarray:
    """Generate sand-colored diffuse texture."""
    rng = np.random.default_rng(seed)

    # Base sand color (RGB)
    base_color = np.array([0.84, 0.78, 0.58], dtype=np.float32)  # Sand color from arena.yaml

    # Multi-scale noise for variation
    coarse = multi_scale_noise((size, size), rng, octaves=4, persistence=0.6)
    medium = multi_scale_noise((size, size), rng, octaves=3, persistence=0.5)
    fine = multi_scale_noise((size, size), rng, octaves=2, persistence=0.4)

    # Combine noise layers
    modulation = 1.0 + 0.08 * coarse + 0.05 * medium + 0.03 * fine

    # Apply to each channel with slight variation
    img = np.empty((size, size, 3), dtype=np.float32)
    for c in range(3):
        channel_mod = modulation * (1.0 + 0.02 * rng.normal(0, 0.1, (size, size)))
        img[:, :, c] = base_color[c] * channel_mod

    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def generate_sand_normal(size: int, seed: int = 42, strength: float = 0.5) -> np.ndarray:
    """Generate normal map from height noise."""
    rng = np.random.default_rng(seed + 1000)

    # Generate height detail for normal map
    height_detail = multi_scale_noise((size, size), rng, octaves=4, persistence=0.5)

    # Compute gradients using Sobel-like filters
    # dx
    dx = np.zeros((size, size), dtype=np.float32)
    dx[:, 1:-1] = (height_detail[:, 2:] - height_detail[:, :-2]) * 0.5
    dx[:, 0] = height_detail[:, 1] - height_detail[:, 0]
    dx[:, -1] = height_detail[:, -1] - height_detail[:, -2]

    # dy
    dy = np.zeros((size, size), dtype=np.float32)
    dy[1:-1, :] = (height_detail[2:, :] - height_detail[:-2, :]) * 0.5
    dy[0, :] = height_detail[1, :] - height_detail[0, :]
    dy[-1, :] = height_detail[-1, :] - height_detail[-2, :]

    # Scale by strength
    dx *= strength
    dy *= strength

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

    return normal_map


def main():
    parser = argparse.ArgumentParser(description="Generate procedural sand textures")
    parser.add_argument("--size", type=int, default=1024, help="Texture size (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="/tmp/sand_textures", help="Output directory")
    parser.add_argument("--normal-strength", type=float, default=0.5, help="Normal map strength")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.size}x{args.size} sand textures (seed={args.seed})...")

    # Diffuse
    diffuse = generate_sand_diffuse(args.size, args.seed)
    diffuse_path = output_dir / "sand_diffuse.png"
    Image.fromarray(diffuse, 'RGB').save(diffuse_path)
    print(f"Saved diffuse to {diffuse_path}")

    # Normal map
    normal = generate_sand_normal(args.size, args.seed, args.normal_strength)
    normal_path = output_dir / "sand_normal.png"
    Image.fromarray(normal, 'RGB').save(normal_path)
    print(f"Saved normal map to {normal_path}")

    print("Done!")


if __name__ == "__main__":
    main()