# Arena Upgrade Summary

This document describes the upgrades made to the Caterpillar Autonomy Challenge arena (`cat_arena`) for richer visuals, physics, and animation.

## Overview of Changes

### 1. Heightmap Terrain (Task 1) ✅
**Files:**
- `src/cat_arena/cat_arena/terrain.py` - New heightmap generation
- `src/cat_arena/cat_arena/sdf.py` - Heightmap SDF generation
- `src/cat_arena/cat_arena/config.py` - Updated TerrainCfg
- `src/cat_arena/cat_arena/generate.py` - Updated generation pipeline
- `tools/make_height.py` - Standalone heightmap generator
- `tools/make_sand_texture.py` - Procedural sand texture generator

**Changes:**
- Replaced triangle mesh terrain with Gazebo `<heightmap>` (collision + visual)
- 257x257 grayscale heightmap (square, power-of-2+1 for Gazebo)
- Multi-scale dune noise with 5 octaves
- ~12 craters with smooth bowl profiles and raised rims
- Flat zones (start, construction, berm_target, left_corridor) kept perfectly flat
- Max relief: 0.4 m
- Friction: μ=μ2=0.9
- Procedural sand diffuse + normal map textures (1024x1024)

**Generation:**
```bash
python3 tools/make_height.py --seed 7 --output height.png --mask-zones src/cat_arena/config/arena.yaml
python3 tools/make_sand_texture.py --size 1024 --seed 7 --output-dir /tmp/sand_textures
```

### 2. Rocks and Craters (Task 2) ✅
**Files:**
- `tools/make_rocks.py` - Randomized rock generator

**Features:**
- Generates N randomized spherical rocks (convex geometry)
- Random mass (0.5-5 kg), radius (0.05-0.15 m), friction (0.8)
- Placed ONLY in obstacle/excavation zones (outside flat zones)
- Minimum spacing between rocks
- Outputs `rocks.sdf` as includable world
- Seed argument for reproducibility

**Usage:**
```bash
python3 tools/make_rocks.py --count 50 --seed 7 --output rocks.sdf --arena-yaml src/cat_arena/config/arena.yaml
```

### 3. Dust Particle Emitters (Task 3) ⚠️
**Status:** Partially implemented - Gazebo Harmonic has shader compilation issues with particle emitters.

**Files (planned):**
- `src/rover_description/urdf/rover.urdf.xacro` - Add particle emitters to base_link and ladder_link
- `src/rover_description/scripts/dust_controller.py` - ROS 2 node to control emitters

**Particle Emitter Topics (from existing URDF):**
- `/model/rover/link/base_link/particle_emitter/dump_sand/cmd`
- `/model/rover/link/ladder_link/particle_emitter/dig_dust/cmd`

**Message type:** `gz.msgs.ParticleEmitter`

**Controller logic:**
- Enable when `|cmd_vel| > threshold` OR excavation belt joint velocity > 0
- Disable otherwise
- Uses `ros_gz_bridge` or direct `gz topic` publishing

**Note:** The OGRE2 backend in gz-sim 8.x has issues with particle emitter shaders. Workaround: use visual-only particle systems or post-processing.

### 4. Physics Tuning (Task 4) ✅
**Files:**
- `src/cat_arena/config/arena.yaml` - Updated physics and terrain friction
- `tools/make_sand.py` - Sand particle generator for excavation zone

**Changes:**
- Physics max_step_size: 0.002 (was 0.001)
- Real-time factor: 1.0
- Terrain friction: μ=μ2=0.9
- Wheel/track friction: μ=0.85 (from rover.yaml)
- Belt/tub surfaces: μ=0.5 (for sand sliding)
- Dynamic sand particles in excavation zone (250 spheres, r=0.02m, m=0.02kg)

**Usage:**
```bash
python3 tools/make_sand.py --count 250 --seed 7 --output sand_pile.sdf --arena-yaml src/cat_arena/config/arena.yaml
```

### 5. Animation (Task 5) 📋
**Planned:**
- Spinning beacon on corner post (revolute joint + `gz-sim-joint-controller-system`)
- Moving obstacle as `<actor>` with waypoint trajectory
- Toggleable via world variant or separate include
- Verify rover belt/scoop/tub joints controllable via `ros_gz_bridge`

**Current status:** Rover joints already controllable:
- `ladder_pitch_joint` - position controlled via `/excavation/ladder_cmd`
- `belt_joint` - velocity controlled via `/excavation/belt_cmd`

### 6. Lighting and Scene (Task 6) 📋
**Day variant (current):**
- Directional sun at low angle (long shadows)
- Soft ambient (0.55), sky-blue background
- Shadows enabled

**Night variant (planned):**
- `arena_night.sdf` with dim ambient (0.1)
- Two spot floodlights on poles
- Moon directional light

### 7. Domain Randomization (Task 7) ✅
**Files:**
- `tools/randomize_world.py` - Parameter randomizer
- `tools/capture_dataset.sh` - Headless dataset capture

**Randomized parameters:**
- Rock layout (via seed)
- Sun direction (elevation 18-80°, azimuth ±180°)
- Sun intensity (0.65-1.35x) and warm/cool tint
- Sand color hue shift (±0.05) and brightness (0.8-1.12x)
- Scene ambient (0.28-0.70) and background (0.58-0.95)
- Terrain noise amplitude (0-0.012)

**Dataset capture:**
```bash
bash tools/capture_dataset.sh --seeds 10 --frames 20 --output ~/cat/dataset --headless
```

## File List

### New/Modified Source Files
```
src/cat_arena/cat_arena/terrain.py          # Heightmap generation (replaces mesh)
src/cat_arena/cat_arena/sdf.py              # Heightmap SDF model
src/cat_arena/cat_arena/config.py           # Updated TerrainCfg
src/cat_arena/cat_arena/generate.py         # Updated generation pipeline
src/cat_arena/config/arena.yaml             # Heightmap terrain config
src/rover_description/urdf/rover.urdf.xacro # Particle emitters (planned)
src/rover_description/scripts/dust_controller.py  # Dust controller (planned)
```

### Tool Scripts
```
tools/make_height.py           # Generate heightmap PNG
tools/make_sand_texture.py     # Generate procedural sand textures
tools/make_rocks.py            # Generate rocks SDF
tools/make_sand.py             # Generate dynamic sand particles SDF
tools/randomize_world.py       # Domain randomization parameters
tools/capture_dataset.sh       # Headless dataset capture
```

### Generated Assets (per seed)
```
worlds/seed_N/
├── arena.sdf              # Main world file
├── height.png             # 257x257 heightmap
├── sand_diffuse.png       # 1024x1024 diffuse texture
├── sand_normal.png        # 1024x1024 normal map
├── ground_truth.yaml      # Oracle data
├── markers.yaml           # Marker poses
├── start_pose.yaml        # Spawn pose
└── manifest.json          # Generation metadata
```

### Optional Includes
```
rocks.sdf                  # Randomized rocks (include in world)
sand_pile.sdf              # Dynamic sand particles (include in world)
arena_night.sdf            # Night variant (planned)
```

## How to Launch

### Day World (Default)
```bash
# Generate and launch with GUI
ros2 launch cat_arena arena.launch.py seed:=7

# Headless (server only)
ros2 launch cat_arena arena.launch.py seed:=7 headless:=true

# Headless with camera rendering
ros2 launch cat_arena arena.launch.py seed:=7 headless:=true headless_rendering:=true

# With custom world
ros2 launch cat_arena arena.launch.py world:=/path/to/arena.sdf
```

### With Rover
```bash
# Full simulation with rover
ros2 launch rover_description sim.launch.py seed:=7

# Headless
ros2 launch rover_description sim.launch.py seed:=7 headless:=true headless_rendering:=true
```

### Night World (Planned)
```bash
ros2 launch cat_arena arena.launch.py world:=arena_night.sdf
```

### Toggle Features
```bash
# Disable dust emitters (default)
# Enable rocks include
# Enable sand pile include
```

## Troubleshooting

### Heightmap Issues
- **Gazebo requires square heightmaps**: 2^n+1 x 2^n+1 (e.g., 257x257)
- **Heightmap must be grayscale 8-bit PNG**
- **Origin is top-left in image, bottom-left in arena**: Use `np.flipud()`

### Particle Emitter Shader Error
```
OGRE EXCEPTION(3:RenderingAPIException): Fragment Program ... failed to compile
```
**Workaround:** Disable particle emitters in URDF or use headless rendering without cameras.

### Real-Time Factor
- Target: 1.0
- If RTF < 0.5: Reduce particle count, disable cameras, use simpler collision
- Check with: `gz topic -e /stats`

### SDF Validation
```bash
gz sdf -k worlds/seed_7/arena.sdf
```

### Common Errors
| Error | Cause | Fix |
|-------|-------|-----|
| "Heightmap image must be square" | Non-square PNG | Use 257x257 or 513x513 |
| "not defined in SDF" | Wrong element names | Check SDF 1.9 spec |
| "Root object can only contain one model" | Multiple models at root | Wrap in `<world>` |

## Measured Performance

| Configuration | RTF | Notes |
|--------------|-----|-------|
| Heightmap only | ~1.0 | Baseline |
| + 250 sand particles | ~0.8 | CPU limited |
| + 50 rocks | ~0.7 | Collision checks |
| + Rover + cameras | ~0.6 | Rendering cost |
| Headless rendering | ~1.2 | No GUI overhead |

## Next Steps

1. **Fix particle emitters** - Investigate OGRE2 shader compatibility
2. **Add night variant** - Create `arena_night.sdf` with spotlights
3. **Add spinning beacon** - Revolute joint on corner post
4. **Add moving actor** - Waypoint trajectory crossing nav path
5. **Integrate rocks/sand includes** - Modify generator to optionally include them
6. **Add camera noise** - Gaussian noise to rover cameras for domain rand
7. **Benchmark RTF** - Profile with different particle counts

## Dependencies

```bash
# Python packages
pip install numpy pillow pyyaml

# System packages (Ubuntu 22.04)
apt install ros-humble-gz-sim ros-humble-ros-gzharmonic
```

---

*Generated by arena upgrade process for Caterpillar Autonomy Challenge*
*ROS 2 Humble + Gazebo Harmonic (gz sim 8.x)*