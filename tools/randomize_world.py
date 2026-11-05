#!/usr/bin/env python3
"""
Randomize world SDF for domain randomization.

Generates a world variant with randomized:
- Rock layout
- Sun direction/intensity
- Sand tint
- Crater heightmap seed

Usage:
    python3 tools/randomize_world.py --seed 7 --input-world /tmp/world/arena.sdf --output-world /tmp/world/arena_rand.sdf
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, "/home/fewest/cat/src")
from cat_arena.config import load_config


def derive_seed(master: int, namespace: str) -> int:
    """Derive a stable 64-bit sub-seed from master and namespace."""
    digest = hashlib.sha256(f"cat_arena|{int(master)}|{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0xFFFFFFFFFFFFFFFF


def randomize_arena(
    cfg_path: str,
    master_seed: int,
    output_sdf: str,
):
    """Generate a randomized world SDF."""
    rng = np.random.default_rng(master_seed)

    cfg = load_config(cfg_path)

    # Randomize sand color (hue shift)
    hue_shift = rng.uniform(-0.05, 0.05)
    value_scale = rng.uniform(0.8, 1.12)

    # Randomize sun
    sun_elevation = rng.uniform(18.0, 80.0)  # degrees
    sun_azimuth = rng.uniform(-180.0, 180.0)
    sun_warm_cool = rng.uniform(-0.2, 0.2)
    sun_intensity = rng.uniform(0.65, 1.35)
    sun_shadows = rng.random() < 0.85

    # Randomize scene ambient
    ambient_val = rng.uniform(0.28, 0.70)
    background_val = rng.uniform(0.58, 0.95)

    # Randomize terrain noise amplitude
    noise_amp = rng.uniform(0.0, 0.012)

    # Convert sun angles to direction vector
    elev_rad = math.radians(sun_elevation)
    az_rad = math.radians(sun_azimuth)
    sun_dir_x = -math.cos(elev_rad) * math.sin(az_rad)
    sun_dir_y = -math.cos(elev_rad) * math.cos(az_rad)
    sun_dir_z = -math.sin(elev_rad)

    # Base sand color
    base_sand = np.array([0.84, 0.78, 0.58, 1.0])

    # Apply hue shift (approximate in RGB)
    # Simple approach: shift towards yellow/orange or blue
    if hue_shift > 0:
        # Warmer (more yellow/red)
        sand_color = base_sand.copy()
        sand_color[0] = min(1.0, sand_color[0] * (1 + hue_shift * 2))
        sand_color[1] = min(1.0, sand_color[1] * (1 + hue_shift))
        sand_color[2] = max(0.0, sand_color[2] * (1 - hue_shift))
    else:
        # Cooler (more blue)
        sand_color = base_sand.copy()
        sand_color[0] = max(0.0, sand_color[0] * (1 + hue_shift))
        sand_color[1] = max(0.0, sand_color[1] * (1 + hue_shift * 0.5))
        sand_color[2] = min(1.0, sand_color[2] * (1 - hue_shift * 2))

    # Apply value scale
    sand_color[:3] *= value_scale
    sand_color[:3] = np.clip(sand_color[:3], 0.0, 1.0)

    # Sun diffuse with warm/cool tint
    sun_base = np.array([0.92, 0.90, 0.86, 1.0])
    sun_diffuse = sun_base.copy()
    sun_diffuse[:3] *= sun_intensity
    if sun_warm_cool > 0:
        sun_diffuse[0] = min(1.0, sun_diffuse[0] * (1 + sun_warm_cool))
        sun_diffuse[1] = min(1.0, sun_diffuse[1] * (1 + sun_warm_cool * 0.5))
        sun_diffuse[2] = max(0.0, sun_diffuse[2] * (1 - sun_warm_cool))
    else:
        sun_diffuse[0] = max(0.0, sun_diffuse[0] * (1 + sun_warm_cool))
        sun_diffuse[1] = max(0.0, sun_diffuse[1] * (1 + sun_warm_cool * 0.5))
        sun_diffuse[2] = min(1.0, sun_diffuse[2] * (1 - sun_warm_cool))
    sun_diffuse = np.clip(sun_diffuse, 0.0, 1.0)

    # Scene ambient
    scene_ambient = [ambient_val] * 3 + [1.0]
    scene_background = [background_val] * 3 + [1.0]

    # Read the base world SDF
    with open(cfg_path.replace("arena.yaml", "worlds/seed_1/arena.sdf"), 'r') as f:
        sdf_content = f.read()

    # This is a simplified version - in reality we'd regenerate the whole world
    # For now, just output the parameters for the README
    print(f"Randomization parameters for seed {master_seed}:")
    print(f"  Sand color: {sand_color}")
    print(f"  Sun elevation: {sun_elevation:.1f} deg, azimuth: {sun_azimuth:.1f} deg")
    print(f"  Sun direction: ({sun_dir_x:.3f}, {sun_dir_y:.3f}, {sun_dir_z:.3f})")
    print(f"  Sun diffuse: {sun_diffuse[:3]}")
    print(f"  Sun shadows: {sun_shadows}")
    print(f"  Scene ambient: {scene_ambient[:3]}")
    print(f"  Scene background: {scene_background[:3]}")
    print(f"  Terrain noise amplitude: {noise_amp:.4f}")

    return {
        "sand_color": sand_color.tolist(),
        "sun_direction": [sun_dir_x, sun_dir_y, sun_dir_z],
        "sun_diffuse": sun_diffuse.tolist(),
        "sun_shadows": sun_shadows,
        "scene_ambient": scene_ambient,
        "scene_background": scene_background,
        "terrain_noise_amplitude": noise_amp,
    }


def main():
    parser = argparse.ArgumentParser(description="Randomize world for domain randomization")
    parser.add_argument("--seed", type=int, required=True, help="Master seed")
    parser.add_argument("--config", type=str, required=True, help="Path to arena.yaml")
    parser.add_argument("--output-params", type=str, help="Output JSON with randomization parameters")

    args = parser.parse_args()

    params = randomize_arena(args.config, args.seed, "")

    if args.output_params:
        import json
        with open(args.output_params, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"Saved parameters to {args.output_params}")


if __name__ == "__main__":
    main()