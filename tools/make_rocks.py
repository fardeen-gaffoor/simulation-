#!/usr/bin/env python3
"""
Generate SDF with randomized rocks (spheres/ellipsoids) outside flat zones.

Usage:
    python3 tools/make_rocks.py --count 50 --seed 7 --output rocks.sdf --arena-yaml src/cat_arena/config/arena.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, "/home/fewest/cat/src")
from cat_arena.config import load_config, Rect


@dataclass
class RockConfig:
    min_radius: float = 0.05
    max_radius: float = 0.15
    min_mass: float = 0.5
    max_mass: float = 5.0
    friction: float = 0.8
    margin: float = 0.3  # margin from zone borders


def _sample_in_zone(rng: np.random.Generator, rect: Rect, margin: float) -> Tuple[float, float]:
    """Sample a random point inside rect with margin."""
    x = rng.uniform(rect.x0 + margin, rect.x1 - margin)
    y = rng.uniform(rect.y0 + margin, rect.y1 - margin)
    return x, y


def _point_in_zones(x: float, y: float, zones: List[Rect], margin: float) -> bool:
    """Check if point is inside any of the zones (with margin)."""
    for zone in zones:
        if zone.contains(x, y, margin=margin):
            return True
    return False


def generate_rocks_sdf(
    count: int,
    seed: int,
    cfg_path: str,
    output_path: str,
    rock_cfg: Optional[RockConfig] = None,
):
    rng = np.random.default_rng(seed)

    if rock_cfg is None:
        rock_cfg = RockConfig()

    cfg = load_config(cfg_path)

    # Flat zones where rocks should NOT be placed
    flat_zone_names = ['start', 'construction', 'berm_target', 'left_corridor']
    flat_zones = [cfg.zone(name).rect for name in flat_zone_names]

    # Obstacle zones where rocks CAN be placed
    obstacle_zone_names = ['obstacle', 'excavation']
    obstacle_zones = [cfg.zone(name).rect for name in obstacle_zone_names]

    # Pre-compute zone areas for weighted sampling
    zone_areas = [z.width * z.height for z in obstacle_zones]
    total_area = sum(zone_areas)
    zone_weights = [a / total_area for a in zone_areas]

    rocks = []
    attempts = 0
    max_attempts = count * 100

    while len(rocks) < count and attempts < max_attempts:
        attempts += 1

        # Pick a zone based on area
        zone_idx = rng.choice(len(obstacle_zones), p=zone_weights)
        zone = obstacle_zones[zone_idx]

        # Sample position
        x, y = _sample_in_zone(rng, zone, rock_cfg.margin)

        # Check not in flat zone
        if _point_in_zones(x, y, flat_zones, rock_cfg.margin):
            continue

        # Check not too close to existing rocks
        too_close = False
        for (rx, ry, rz, r_radius, r_mass) in rocks:
            dist = math.hypot(x - rx, y - ry)
            min_dist = r_radius + rock_cfg.max_radius + 0.1
            if dist < min_dist:
                too_close = True
                break

        if too_close:
            continue

        # Random rock properties
        radius = rng.uniform(rock_cfg.min_radius, rock_cfg.max_radius)
        mass = rng.uniform(rock_cfg.min_mass, rock_cfg.max_mass)

        # Slightly bury the rock (5-15% of radius)
        bury_fraction = rng.uniform(0.05, 0.15)
        z = radius * (1.0 - bury_fraction)

        rocks.append((x, y, z, radius, mass))

    print(f"Generated {len(rocks)} rocks after {attempts} attempts")

    # Write SDF as a world with multiple models (for include)
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" ?>\n')
        f.write('<!-- Generated rocks SDF - include this in your world -->\n')
        f.write(f'<!-- seed={seed}, count={len(rocks)} -->\n')
        f.write('<sdf version="1.9">\n')
        f.write('  <world name="rocks">\n')

        for i, (x, y, z, radius, mass) in enumerate(rocks):
            # Random color variation (rocky gray/brown)
            shade = rng.uniform(0.3, 0.5)
            color_r = shade * rng.uniform(0.9, 1.1)
            color_g = shade * rng.uniform(0.85, 1.0)
            color_b = shade * rng.uniform(0.8, 0.95)

            f.write(f'    <model name="rock_{i}">\n')
            f.write('      <static>false</static>\n')
            f.write(f'      <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>\n')
            f.write('      <link name="link">\n')
            f.write('        <mass>{:.4f}</mass>\n'.format(mass))
            f.write('        <inertial>\n')
            f.write('          <pose>0 0 0 0 0 0</pose>\n')
            # Sphere inertia: I = 2/5 * m * r^2
            inertia = (2.0/5.0) * mass * radius * radius
            f.write(f'          <inertia>\n')
            f.write(f'            <ixx>{inertia:.6f}</ixx>\n')
            f.write(f'            <ixy>0</ixy>\n')
            f.write(f'            <ixz>0</ixz>\n')
            f.write(f'            <iyy>{inertia:.6f}</iyy>\n')
            f.write(f'            <iyz>0</iyz>\n')
            f.write(f'            <izz>{inertia:.6f}</izz>\n')
            f.write(f'          </inertia>\n')
            f.write('        </inertial>\n')
            f.write('        <collision name="collision">\n')
            f.write('          <geometry>\n')
            f.write(f'            <sphere>\n')
            f.write(f'              <radius>{radius:.6f}</radius>\n')
            f.write(f'            </sphere>\n')
            f.write('          </geometry>\n')
            f.write('          <surface>\n')
            f.write('            <friction>\n')
            f.write('              <ode>\n')
            f.write(f'                <mu>{rock_cfg.friction:.4f}</mu>\n')
            f.write(f'                <mu2>{rock_cfg.friction:.4f}</mu2>\n')
            f.write('              </ode>\n')
            f.write('            </friction>\n')
            f.write('            <contact>\n')
            f.write('              <ode>\n')
            f.write('                <kp>10000000.0</kp>\n')
            f.write('                <kd>1000.0</kd>\n')
            f.write('                <max_vel>0.01</max_vel>\n')
            f.write('                <min_depth>0.0005</min_depth>\n')
            f.write('              </ode>\n')
            f.write('            </contact>\n')
            f.write('          </surface>\n')
            f.write('        </collision>\n')
            f.write('        <visual name="visual">\n')
            f.write('          <geometry>\n')
            f.write(f'            <sphere>\n')
            f.write(f'              <radius>{radius:.6f}</radius>\n')
            f.write(f'            </sphere>\n')
            f.write('          </geometry>\n')
            f.write('          <material>\n')
            f.write(f'            <ambient>{color_r:.4f} {color_g:.4f} {color_b:.4f} 1</ambient>\n')
            f.write(f'            <diffuse>{min(1.0, color_r*1.2):.4f} {min(1.0, color_g*1.2):.4f} {min(1.0, color_b*1.2):.4f} 1</diffuse>\n')
            f.write('            <specular>0.05 0.05 0.05 1</specular>\n')
            f.write('          </material>\n')
            f.write('        </visual>\n')
            f.write('      </link>\n')
            f.write('    </model>\n')

        f.write('  </world>\n')
        f.write('</sdf>\n')

    print(f"Saved rocks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate rocks SDF for arena")
    parser.add_argument("--count", type=int, default=50, help="Number of rocks to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="rocks.sdf", help="Output SDF file")
    parser.add_argument("--arena-yaml", type=str, required=True, help="Path to arena.yaml")
    parser.add_argument("--min-radius", type=float, default=0.05, help="Minimum rock radius (m)")
    parser.add_argument("--max-radius", type=float, default=0.15, help="Maximum rock radius (m)")
    parser.add_argument("--min-mass", type=float, default=0.5, help="Minimum rock mass (kg)")
    parser.add_argument("--max-mass", type=float, default=5.0, help="Maximum rock mass (kg)")
    parser.add_argument("--friction", type=float, default=0.8, help="Rock friction coefficient")
    parser.add_argument("--margin", type=float, default=0.3, help="Margin from zone borders (m)")

    args = parser.parse_args()

    rock_cfg = RockConfig(
        min_radius=args.min_radius,
        max_radius=args.max_radius,
        min_mass=args.min_mass,
        max_mass=args.max_mass,
        friction=args.friction,
        margin=args.margin,
    )

    generate_rocks_sdf(
        count=args.count,
        seed=args.seed,
        cfg_path=args.arena_yaml,
        output_path=args.output,
        rock_cfg=rock_cfg,
    )


if __name__ == "__main__":
    main()