"""Builds ``arena.sdf``: terrain, walls, zone markings, fiducial boards, boulders.

The world is a pure function of ``arena.yaml`` + the run seed: regeneration with
the same seed produces a byte-identical file.

Geometry choices that were *verified* on this machine before being committed
(see README "verified facts"):

* the terrain is a triangle **mesh** for visual *and* collision - the mesh-based
  heightmap collision path silently produced a flat surface on gz-sim 8.15.0,
* DART needs ``<collision_detector>bullet</collision_detector>`` for a
  triangle-mesh terrain,
* every marking/board is visual-only (``<visual>`` without ``<collision>``) so
  the robot can drive over the painted zones,
* zone outline strips are drawn *inside* their own zone rectangle, which avoids
  coplanar z-fighting between neighbouring zones.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import ArenaConfig, Rect
from .obstacles import Boulder
from .terrain import Crater

__all__ = [
    "WorldAssets",
    "build_world_sdf",
    "rotation_matrix_to_rpy",
    "rpy_matrix",
    "look_at_rpy",
    "board_rpy",
]


@dataclass(frozen=True)
class WorldAssets:
    """File names of the generated assets referenced from the world SDF."""

    obj_file: str
    mtl_file: str
    albedo_file: Optional[str] = None
    heightmap_file: Optional[str] = None


# --------------------------------------------------------------------------- #
# small rotation helpers (shared with the tests)
# --------------------------------------------------------------------------- #
def rpy_matrix(roll: float, pitch: float, yaw: float) -> List[List[float]]:
    """ZYX Euler (Gazebo/SDF convention) to a 3x3 rotation matrix."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def rotation_matrix_to_rpy(matrix: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    """Inverse of :func:`rpy_matrix`, handling the gimbal-lock case."""
    m = [[float(v) for v in row] for row in matrix]
    sp = -m[2][0]
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(m[2][0]) < 1.0 - 1e-9:
        roll = math.atan2(m[2][1], m[2][2])
        yaw = math.atan2(m[1][0], m[0][0])
    else:  # gimbal lock: pitch = +-pi/2
        roll = 0.0
        yaw = math.atan2(-m[0][1], m[1][1])
    return roll, pitch, yaw


def look_at_rpy(
    position: Sequence[float], target: Sequence[float], roll: float = 0.0
) -> Tuple[float, float, float]:
    """RPY that aims a Gazebo camera (+x forward, +z up) from position to target."""
    fx = target[0] - position[0]
    fy = target[1] - position[1]
    fz = target[2] - position[2]
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    if norm <= 0.0:
        raise ValueError("look_at_rpy: position and target coincide")
    f = (fx / norm, fy / norm, fz / norm)
    # camera "left" axis = world up x forward, i.e. the horizontal axis
    # perpendicular to the view direction.  This is what keeps the camera
    # unrotated (roll == 0) instead of wallowing around the view axis.
    lx = -f[1]
    ly = f[0]
    lz = 0.0
    lnorm = math.hypot(lx, ly)
    if lnorm < 1e-9:  # looking straight up/down
        lx, ly, lz = (0.0, 1.0, 0.0)
    else:
        lx, ly, lz = (lx / lnorm, ly / lnorm, 0.0)
    # camera "up" = forward x left
    ux = f[1] * lz - f[2] * ly
    uy = f[2] * lx - f[0] * lz
    uz = f[0] * ly - f[1] * lx
    matrix = [
        [f[0], lx, ux],
        [f[1], ly, uy],
        [f[2], lz, uz],
    ]
    return rotation_matrix_to_rpy(matrix)


def board_rpy(normal: Tuple[float, float]) -> Tuple[float, float, float]:
    """RPY for a flat board whose *local +z* axis points along a horizontal normal.

    ``normal`` is the outward direction of the wall face the board is mounted on
    (i.e. the direction the board looks at, into the arena).
    """
    yaw = math.atan2(normal[1], normal[0])
    # R = Rz(yaw) * Ry(pi/2) -> R*ez = (cos(yaw), sin(yaw), 0)
    matrix = rpy_matrix(0.0, math.pi / 2.0, yaw)
    return rotation_matrix_to_rpy(matrix)


def _pose(x: float, y: float, z: float, rpy: Sequence[float] = (0.0, 0.0, 0.0)) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f} {rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"


def _v3(v: Sequence[float]) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def _v4(v: Sequence[float]) -> str:
    return " ".join(f"{x:.4f}" for x in v)


def _v6(v: Sequence[float]) -> str:
    return " ".join(f"{x:.6f}" for x in v)


def _color(rgba: Sequence[float], specular: float = 0.15) -> str:
    r, g, b, a = rgba
    return (
        f"<material>\n"
        f"          <ambient>{r * 0.6:.4f} {g * 0.6:.4f} {b * 0.6:.4f} {a:.4f}</ambient>\n"
        f"          <diffuse>{r:.4f} {g:.4f} {b:.4f} {a:.4f}</diffuse>\n"
        f"          <specular>{specular:.4f} {specular:.4f} {specular:.4f} {a:.4f}</specular>\n"
        f"        </material>"
    )


def stable_shade(name: str, spread: float = 0.28) -> float:
    """Deterministic per-object shade factor (``hash()`` is salted, so not usable)."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    unit = int.from_bytes(digest[:4], "big") / float(0xFFFFFFFF)
    return (1.0 - 0.5 * spread) + spread * unit


def _surface(mu: float, mu2: float) -> str:
    return (
        "<surface>\n"
        "          <friction>\n"
        "            <ode>\n"
        f"              <mu>{mu:.4f}</mu>\n"
        f"              <mu2>{mu2:.4f}</mu2>\n"
        "            </ode>\n"
        "          </friction>\n"
        "        </surface>"
    )


def _file_uri(path: str) -> str:
    return "file://" + os.path.abspath(path)


def _comment_safe(text: object) -> str:
    """Make dynamic text safe inside an XML comment.

    ``--`` may not appear inside an XML comment (``<!-- ... -- ... -->`` is a
    syntax error), and both marker notes and obstacle ids are user/config
    supplied, so anything that goes into a comment passes through here.
    """
    return str(text).replace("--", "- -").replace("<!--", "< !--").replace("-->", "-- >")


# --------------------------------------------------------------------------- #
# section builders
# --------------------------------------------------------------------------- #
def _terrain_model(cfg: ArenaConfig, assets: WorldAssets, output_dir: str) -> str:
    obj_uri = _file_uri(os.path.join(output_dir, assets.obj_file))
    r, g, b = cfg.sand_color[:3]
    if assets.albedo_file:
        material = (
            "<material>\n"
            f"          <ambient>{r * 0.55:.4f} {g * 0.55:.4f} {b * 0.55:.4f} 1</ambient>\n"
            f"          <diffuse>{r:.4f} {g:.4f} {b:.4f} 1</diffuse>\n"
            "          <specular>0.04 0.04 0.04 1</specular>\n"
            "          <pbr>\n"
            "            <metal>\n"
            f"              <albedo_map>{_file_uri(os.path.join(output_dir, assets.albedo_file))}</albedo_map>\n"
            "              <roughness>0.92</roughness>\n"
            "              <metalness>0.0</metalness>\n"
            "            </metal>\n"
            "          </pbr>\n"
            "        </material>"
        )
    else:
        material = _color(cfg.sand_color, specular=0.05)
    mu, mu2 = cfg.friction.mu, cfg.friction.mu2
    return f"""    <!-- ==================== tilled dry sand terrain ==================== -->
    <!-- Collision and visual both use the generated triangle mesh, so what the
         cameras see and what the wheels feel are the same surface.
         gz mesh loading REQUIRES vertex normals in the OBJ (verified). -->
    <model name="terrain">
      <static>true</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <collision name="terrain_collision">
          <geometry>
            <mesh>
              <uri>{obj_uri}</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
          {_surface(mu, mu2)}
        </collision>
        <visual name="terrain_visual">
          <geometry>
            <mesh>
              <uri>{obj_uri}</uri>
              <scale>1 1 1</scale>
            </mesh>
          </geometry>
          {material}
        </visual>
      </link>
    </model>"""


def _walls_model(cfg: ArenaConfig) -> str:
    if not cfg.walls.present:
        return "    <!-- arena walls disabled by configuration -->"
    L, W = cfg.length_x, cfg.width_y
    t, h = cfg.walls.thickness, cfg.walls.height
    mu, mu2 = 0.9, 0.9
    color = _color(cfg.walls.color, specular=0.05)
    links: List[str] = []
    specs = [
        ("south_wall", (0.5 * L, -0.5 * t, 0.5 * h), (L + 2 * t, t, h)),
        ("north_wall", (0.5 * L, W + 0.5 * t, 0.5 * h), (L + 2 * t, t, h)),
        ("west_wall", (-0.5 * t, 0.5 * W, 0.5 * h), (t, W, h)),
        ("east_wall", (L + 0.5 * t, 0.5 * W, 0.5 * h), (t, W, h)),
    ]
    for name, pos, size in specs:
        links.append(
            f"""      <link name="{name}">
        <pose>{_pose(pos[0], pos[1], pos[2])}</pose>
        <collision name="{name}_collision">
          <geometry>
            <box>
              <size>{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}</size>
            </box>
          </geometry>
          {_surface(mu, mu2)}
        </collision>
        <visual name="{name}_visual">
          <geometry>
            <box>
              <size>{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}</size>
            </box>
          </geometry>
          {color}
        </visual>
      </link>"""
        )
    return (
        "    <!-- ==================== arena walls (collidable) ==================== -->\n"
        f"    <!-- height {h} m is a flagged assumption; a wall ram costs -50 points. -->\n"
        '    <model name="arena_walls">\n'
        "      <static>true</static>\n"
        + "\n".join(links)
        + "\n    </model>"
    )


def _zone_markings_model(cfg: ArenaConfig) -> str:
    visuals: List[str] = []
    width = cfg.marking.width
    thickness = cfg.marking.thickness
    z_out = cfg.marking.z + 0.5 * thickness
    half = 0.5 * width

    for zone in cfg.zones.values():
        rect = zone.rect
        inner = Rect(rect.x0 + half, rect.y0 + half, rect.x1 - half, rect.y1 - half)
        if inner.width <= 0 or inner.height <= 0:
            continue
        strips = [
            (f"{zone.name}_south", (0.5 * (inner.x0 + inner.x1), inner.y0), (inner.width + width, width)),
            (f"{zone.name}_north", (0.5 * (inner.x0 + inner.x1), inner.y1), (inner.width + width, width)),
            (f"{zone.name}_west", (inner.x0, 0.5 * (inner.y0 + inner.y1)), (width, inner.height + width)),
            (f"{zone.name}_east", (inner.x1, 0.5 * (inner.y0 + inner.y1)), (width, inner.height + width)),
        ]
        if zone.fill:
            # filled target patch, slightly below its own outline
            visuals.append(
                f"""        <visual name="{zone.name}_fill">
          <pose>{_pose(rect.center[0], rect.center[1], cfg.marking.z)}</pose>
          <geometry>
            <box>
              <size>{rect.width:.6f} {rect.height:.6f} {thickness:.6f}</size>
            </box>
          </geometry>
          {_color(zone.color, specular=0.05)}
        </visual>"""
            )
        for name, center, size in strips:
            visuals.append(
                f"""        <visual name="{name}">
          <pose>{_pose(center[0], center[1], z_out)}</pose>
          <geometry>
            <box>
              <size>{size[0]:.6f} {size[1]:.6f} {thickness:.6f}</size>
            </box>
          </geometry>
          {_color(zone.color, specular=0.05)}
        </visual>"""
            )
    return (
        "    <!-- ============ zone markings (VISUAL ONLY, no collision) ============ -->\n"
        "    <!-- Outlines are drawn inside each zone rectangle so that adjacent\n"
        "         zones cannot produce coplanar z-fighting. -->\n"
        '    <model name="zone_markings">\n'
        "      <static>true</static>\n"
        '      <link name="link">\n'
        + "\n".join(visuals)
        + "\n      </link>\n    </model>"
    )


def _markers_model(cfg: ArenaConfig) -> str:
    if not cfg.markers.enabled or not cfg.markers.placements:
        return "    <!-- arena fiducial markers disabled -->"
    from .markers import marker_poses  # local import: avoids a circular import at module load

    m = cfg.markers
    visuals: List[str] = []
    for pose in marker_poses(cfg):
        bx, by, bz = pose.position
        rpy = pose.rpy
        visuals.append(
            f"""        <!-- marker {pose.marker_id}: {_comment_safe(pose.mount)}, {_comment_safe(pose.note)} -->
        <visual name="marker_{pose.marker_id}_board">
          <pose>{_pose(bx, by, bz, rpy)}</pose>
          <geometry>
            <box>
              <size>{m.board_size:.6f} {m.board_size:.6f} {m.board_thickness:.6f}</size>
            </box>
          </geometry>
          {_color(m.board_color, specular=0.10)}
        </visual>
        <visual name="marker_{pose.marker_id}_tag">
          <pose>{_pose(bx, by, bz, rpy)}</pose>
          <geometry>
            <box>
              <size>{m.tag_size:.6f} {m.tag_size:.6f} {m.board_thickness * 1.4:.6f}</size>
            </box>
          </geometry>
          {_color(m.tag_color, specular=0.02)}
        </visual>"""
        )
    return (
        "    <!-- ================= arena fiducial marker boards ================= -->\n"
        f"    <!-- family {m.family}, {len(m.placements)} of max {m.max_count} allowed,\n"
        f"         board {m.board_size} m, top edge {m.center_height + 0.5 * m.board_size:.3f} m\n"
        "         (competition limits: <=25x25 cm, <=60 cm high, <=5 total).\n"
        "         Module 4 draws the AprilTag bitmaps into the dark inner square. -->\n"
        '    <model name="arena_markers">\n'
        "      <static>true</static>\n"
        '      <link name="link">\n'
        + "\n".join(visuals)
        + "\n      </link>\n    </model>"
    )


def _boulders_models(cfg: ArenaConfig, boulders: Sequence[Boulder]) -> str:
    if not boulders:
        return "    <!-- no boulders this run -->"
    mu, mu2 = cfg.friction.mu, cfg.friction.mu2
    blocks: List[str] = []
    for boulder in boulders:
        r, g, b, a = cfg.boulders.color
        # deterministically vary the rock shade per boulder (no RNG stream use)
        shade = stable_shade(boulder.boulder_id)
        color = (
            min(1.0, r * shade),
            min(1.0, g * shade),
            min(1.0, b * shade),
            a,
        )
        blocks.append(
            f"""    <model name="{boulder.boulder_id}">
      <static>{str(cfg.boulders.static).lower()}</static>
      <pose>{_pose(boulder.x, boulder.y, boulder.center_z)}</pose>
      <link name="link">
        <collision name="collision">
          <geometry>
            <sphere>
              <radius>{boulder.radius:.6f}</radius>
            </sphere>
          </geometry>
          {_surface(mu, mu2)}
        </collision>
        <visual name="visual">
          <geometry>
            <sphere>
              <radius>{boulder.radius:.6f}</radius>
            </sphere>
          </geometry>
          {_color(color, specular=0.08)}
        </visual>
      </link>
    </model>"""
        )
    return (
        "    <!-- ============ seeded boulders (static: must not be moved) ============ -->\n"
        + "\n".join(blocks)
    )


def _overview_camera_model(cfg: ArenaConfig, enabled: bool) -> str:
    if not enabled:
        return "    <!-- overview camera disabled (turn it on with the generator's overview camera option) -->"
    ov = cfg.overview_camera
    rpy = look_at_rpy(ov.position, ov.look_at)
    return f"""    <!-- ============ verification camera (not part of the robot) ============ -->
    <model name="arena_overview_camera">
      <static>true</static>
      <pose>{_pose(ov.position[0], ov.position[1], ov.position[2], rpy)}</pose>
      <link name="link">
        <sensor name="overview" type="camera">
          <always_on>1</always_on>
          <update_rate>{ov.update_rate:.3f}</update_rate>
          <visualize>false</visualize>
          <topic>{ov.topic.lstrip('/')}</topic>
          <camera>
            <horizontal_fov>{math.radians(ov.hfov_deg):.9f}</horizontal_fov>
            <image>
              <width>{ov.width}</width>
              <height>{ov.height}</height>
              <format>{ov.fmt}</format>
            </image>
            <clip>
              <near>0.1</near>
              <far>200</far>
            </clip>
          </camera>
        </sensor>
      </link>
    </model>"""


# --------------------------------------------------------------------------- #
# world
# --------------------------------------------------------------------------- #
def build_world_sdf(
    cfg: ArenaConfig,
    assets: WorldAssets,
    output_dir: str,
    boulders: Sequence[Boulder] = (),
    craters: Sequence[Crater] = (),
    seed: int = 0,
    include_overview_camera: Optional[bool] = None,
    include_robot_plugins: Optional[bool] = None,
) -> str:
    """Render the complete world description for one seeded run."""
    ov_enabled = cfg.overview_camera.enabled if include_overview_camera is None else bool(include_overview_camera)
    # A robot that carries cameras / an IMU needs those systems present in the
    # *world*: gz-sim-sensors-system renders cameras and gz-sim-imu-system
    # publishes IMU sensors (the sensors system alone does not).  The overview
    # camera implies the sensors system; module 2 turns both on explicitly with
    # include_robot_plugins=True.  Default (None/False) preserves module 1.
    robot_plugins = bool(include_robot_plugins)
    sensors_system = ov_enabled or robot_plugins
    plugins = [
        '    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>',
        '    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>',
        '    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>',
        '    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>',
    ]
    if robot_plugins:
        plugins.append(
            '    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>'
        )
    if sensors_system:
        plugins.append(
            '    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">\n'
            "      <render_engine>ogre2</render_engine>\n"
            "    </plugin>"
        )
    gx, gy, gz = cfg.physics.gravity
    crater_lines = "\n".join(
        f"         crater {_comment_safe(c.crater_id)}: centre ({c.x:.3f}, {c.y:.3f}) r={c.radius:.3f} "
        f"depth={c.depth:.3f} (zone {c.zone})"
        for c in craters
    ) or "         none"
    header = f"""<?xml version="1.0" ?>
<!--
  cat_arena world - GENERATED FILE, do not edit by hand.
  Edit src/cat_arena/config/arena.yaml and re-run the generator instead.

  seed                 : {seed}
  arena                : {cfg.length_x} x {cfg.width_y} m, sand depth {cfg.sand_depth} m
  sand friction        : mu={cfg.friction.mu} mu2={cfg.friction.mu2}
  wall height          : {cfg.walls.height} m (flagged assumption)
  terrain resolution   : {cfg.terrain.cell_size} m
  terrain mesh         : {assets.obj_file} (visual + collision, vertex normals mandatory)
  boulders             : {len(boulders)}
  craters              : {len(craters)}
{crater_lines}
  ground truth         : ground_truth.yaml (TEST / LABEL USE ONLY - never loaded by autonomy)
-->
"""
    body = f"""<sdf version="1.9">
  <world name="{cfg.world_name}">
    <gravity>{gx} {gy} {gz}</gravity>
    <physics name="arena_physics" type="{cfg.physics.engine}">
      <max_step_size>{cfg.physics.max_step_size}</max_step_size>
      <real_time_factor>{cfg.physics.real_time_factor}</real_time_factor>
      <dart>
        <!-- REQUIRED for triangle-mesh terrain on this Gazebo Harmonic build. -->
        <collision_detector>{cfg.physics.collision_detector}</collision_detector>
      </dart>
    </physics>

""" + "\n".join(plugins) + f"""

    <scene>
      <ambient>{_v4(cfg.scene.ambient)}</ambient>
      <background>{_v4(cfg.scene.background)}</background>
      <shadows>{'true' if cfg.scene.shadows else 'false'}</shadows>
      <grid>{'true' if cfg.scene.grid else 'false'}</grid>
      <origin_visual>{'true' if cfg.scene.origin_visual else 'false'}</origin_visual>
    </scene>

    <light type="directional" name="{cfg.light.name}">
      <cast_shadows>{'true' if cfg.light.cast_shadows else 'false'}</cast_shadows>
      <pose>{_v6(cfg.light.pose)}</pose>
      <diffuse>{_v4(cfg.light.diffuse)}</diffuse>
      <specular>{_v4(cfg.light.specular)}</specular>
      <attenuation>
        <range>{cfg.light.attenuation[0]:g}</range>
        <constant>{cfg.light.attenuation[1]:g}</constant>
        <linear>{cfg.light.attenuation[2]:g}</linear>
        <quadratic>{cfg.light.attenuation[3]:g}</quadratic>
      </attenuation>
      <direction>{_v3(cfg.light.direction)}</direction>
    </light>

    <!-- No <gui> block on purpose. A world-level <gui> element REPLACES
         Gazebo's default GUI plugin list (gz-sim8/gui/gui.config) instead of
         adding to it. Emitting only MinimalScene therefore dropped
         GzSceneManager - the plugin that populates the render scene from the
         entity-component-manager - and the 3D view came up black with working
         Qt panels around it. Leaving <gui> out keeps the stock GUI (scene
         manager, entity tree, view controls) which also frames the arena
         automatically. Verified on gz-sim 8.15.0: with <gui> the viewport is
         black, without it the arena renders. -->

{_terrain_model(cfg, assets, output_dir)}

{_walls_model(cfg)}

{_zone_markings_model(cfg)}

{_markers_model(cfg)}

{_boulders_models(cfg, boulders)}

{_overview_camera_model(cfg, ov_enabled)}
  </world>
</sdf>
"""
    return header + body
