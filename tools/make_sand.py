#!/usr/bin/env python3
"""
Generate SDF with small dynamic sand spheres for excavation zone.

Usage:
    python3 tools/make_sand.py --count 250 --seed 7 --output sand_pile.sdf --arena-yaml src/cat_arena/config/arena.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

sys.path.insert(0, "/home/fewest/cat/src")
from cat_arena.config import load_config, Rect


@dataclass
class SandConfig:
    count: int = 250
    max_count: int = 500
    radius: float = 0.02
    mass: float = 0.02
    friction: float = 0.9
    color: tuple = (0.84, 0.78, 0.58, 1.0)  # Sand color
    z_min: float = 0.02  # Start slightly above ground
    z_max: float = 0.15  # Max height of pile


def generate_sand_pile_sdf(
    count: int,
    seed: int,
    cfg_path: str,
    output_path: str,
    sand_cfg: Optional[SandConfig] = None,
):
    if sand_cfg is None:
        sand_cfg = SandConfig()

    if count > sand_cfg.max_count:
        print(f"WARNING: Requested {count} particles, capping at {sand_cfg.max_count}")
        count = sand_cfg.max_count

    rng = np.random.default_rng(seed)
    cfg = load_config(cfg_path)

    # Excavation zone bounds
    excavation_zone = cfg.zone("excavation").rect

    # Generate positions in a box within the excavation zone
    margin = 0.3
    x_min = excavation_zone.x0 + margin
    x_max = excavation_zone.x1 - margin
    y_min = excavation_zone.y0 + margin
    y_max = excavation_zone.y1 - margin

    print(f"Excavation zone: x=[{excavation_zone.x0:.2f}, {excavation_zone.x1:.2f}], y=[{excavation_zone.y0:.2f}, {excavation_zone.y1:.2f}]")
    print(f"Sand spawn box: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")

    # Generate random positions with some layering
    positions = []
    for i in range(count):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        z = rng.uniform(sand_cfg.z_min, sand_cfg.z_max)
        positions.append((x, y, z))

    # Write SDF
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write(f'<!-- Generated sand pile SDF - include this in your world -->\n')
        f.write(f'<!-- seed={seed}, count={count} -->\n')
        f.write('<sdf version="1.9">\n')
        f.write('  <world name="sand_pile">\n')

        for i, (x, y, z) in enumerate(positions):
            f.write(f'    <model name="sand_{i}">\n')
            f.write('      <static>false</static>\n')
            f.write(f'      <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>\n')
            f.write('      <link name="link">\n')
            f.write(f'        <mass>{sand_cfg.mass:.6f}</mass>\n')
            f.write('        <inertial>\n')
            f.write('          <pose>0 0 0 0 0 0</pose>\n')
            # Sphere inertia
            inertia = (2.0/5.0) * sand_cfg.mass * sand_cfg.radius * sand_cfg.radius
            f.write(f'          <inertia>\n')
            f.write(f'            <ixx>{inertia:.8f}</ixx>\n')
            f.write(f'            <ixy>0</ixy>\n')
            f.write(f'            <ixz>0</ixz>\n')
            f.write(f'            <iyy>{inertia:.8f}</iyy>\n')
            f.write(f'            <iyz>0</iyz>\n')
            f.write(f'            <izz>{inertia:.8f}</izz>\n')
            f.write(f'          </inertia>\n')
            f.write('        </inertial>\n')
            f.write('        <collision name="collision">\n')
            f.write('          <geometry>\n')
            f.write(f'            <sphere>\n')
            f.write(f'              <radius>{sand_cfg.radius:.6f}</radius>\n')
            f.write(f'            </sphere>\n')
            f.write('          </geometry>\n')
            f.write('          <surface>\n')
            f.write('            <friction>\n')
            f.write('              <ode>\n')
            f.write(f'                <mu>{sand_cfg.friction:.4f}</mu>\n')
            f.write(f'                <mu2>{sand_cfg.friction:.4f}</mu2>\n')
            f.write('              </ode>\n')
            f.write('            </friction>\n')
            f.write('            <contact>\n')
            f.write('              <ode>\n')
            f.write('                <kp>100000.0</kp>\n')
            f.write('                <kd>10.0</kd>\n')
            f.write('                <max_vel>0.1</max_vel>\n')
            f.write('                <min_depth>0.0001</min_depth>\n')
            f.write('              </ode>\n')
            f.write('            </contact>\n')
            f.write('          </surface>\n')
            f.write('        </collision>\n')
            f.write('        <visual name="visual">\n')
            f.write('          <geometry>\n')
            f.write(f'            <sphere>\n')
            f.write(f'              <radius>{sand_cfg.radius:.6f}</radius>\n')
            f.write(f'            </sphere>\n')
            f.write('          </geometry>\n')
            r, g, b, a = sand_cfg.color
            f.write(f'          <material>\n')
            f.write(f'            <ambient>{r*0.6:.4f} {g*0.6:.4f} {b*0.6:.4f} {a:.4f}</ambient>\n')
            f.write(f'            <diffuse>{r:.4f} {g:.4f} {b:.4f} {a:.4f}</diffuse>\n')
            f.write(f'            <specular>0.05 0.05 0.05 {a:.4f}</specular>\n')
            f.write(f'          </material>\n')
            f.write('        </visual>\n')
            f.write('      </link>\n')
            f.write('    </model>\n')

        f.write('  </world>\n')
        f.write('</sdf>\n')

    print(f"Saved {count} sand particles to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate sand pile SDF for excavation zone")
    parser.add_argument("--count", type=int, default=250, help="Number of sand particles (max 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="sand_pile.sdf", help="Output SDF file")
    parser.add_argument("--arena-yaml", type=str, required=True, help="Path to arena.yaml")
    parser.add_argument("--radius", type=float, default=0.02, help="Particle radius (m)")
    parser.add_argument("--mass", type=float, default=0.02, help="Particle mass (kg)")
    parser.add_argument("--friction", type=float, default=0.9, help="Friction coefficient")

    args = parser.parse_args()

    sand_cfg = SandConfig(
        count=args.count,
        radius=args.radius,
        mass=args.mass,
        friction=args.friction,
    )

    generate_sand_pile_sdf(
        count=args.count,
        seed=args.seed,
        cfg_path=args.arena_yaml,
        output_path=args.output,
        sand_cfg=sand_cfg,
    )


if __name__ == "__main__":
    main()