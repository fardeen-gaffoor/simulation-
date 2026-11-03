"""World SDF tests.

The generated world is the interface between module 1 and Gazebo, so these tests
check the structure that Gazebo needs rather than string equality:

* the physics block selects dartsim + the bullet collision detector (the only
  combination on this build where a triangle-mesh terrain actually collides),
* the terrain model references the generated OBJ for **both** visual and
  collision, and every geometry carries the configured friction,
* the walls, zone markings, markers and boulders are present exactly as
  configured,
* the world always carries physics / user-commands / scene-broadcaster /
  contact plugins (a world without systems is silently inert),
* the sensors system is only emitted when the verification camera is on,
* no world-level `<gui>` block: it would replace Gazebo's default GUI plugin
  list (dropping GzSceneManager) and blank the 3D view.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from cat_arena.sdf import (
    WorldAssets,
    board_rpy,
    build_world_sdf,
    look_at_rpy,
    rpy_matrix,
    rotation_matrix_to_rpy,
    stable_shade,
)
from cat_arena.terrain import Crater
from cat_arena.obstacles import Boulder


def _rotate(rpy, vector):
    m = rpy_matrix(*rpy)
    return tuple(sum(m[i][j] * vector[j] for j in range(3)) for i in range(3))


# --------------------------------------------------------------------------- #
# rotation helpers
# --------------------------------------------------------------------------- #
def test_rpy_round_trip():
    for rpy in ((0, 0, 0), (0.3, -0.2, 1.1), (-0.4, 0.5, -2.0), (0.0, 0.0, math.pi)):
        matrix = rpy_matrix(*rpy)
        back = rotation_matrix_to_rpy(matrix)
        for value, expected in zip(back, rpy):
            assert value == pytest.approx(expected, abs=1e-9), f"{rpy} -> {back}"


def test_look_at_aims_the_forward_axis_at_the_target():
    position = (5.05, -2.40, 7.20)
    target = (4.50, 2.50, 0.00)
    rpy = look_at_rpy(position, target)
    forward = _rotate(rpy, (1.0, 0.0, 0.0))
    expected = tuple((t - p) for t, p in zip(target, position))
    norm = math.sqrt(sum(v * v for v in expected))
    expected = tuple(v / norm for v in expected)
    for value, want in zip(forward, expected):
        assert value == pytest.approx(want, abs=1e-9)
    # the camera never ends up rolled onto its side for a normal down-tilt
    assert abs(rpy[0]) < 1e-9


def test_look_at_rejects_a_degenerate_pair():
    with pytest.raises(ValueError):
        look_at_rpy((1, 1, 1), (1, 1, 1))


def test_board_rpy_points_the_board_normal_into_the_arena():
    for normal in ((1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0)):
        rpy = board_rpy(normal)
        pointed = _rotate(rpy, (0.0, 0.0, 1.0))
        assert pointed[0] == pytest.approx(normal[0], abs=1e-9)
        assert pointed[1] == pytest.approx(normal[1], abs=1e-9)
        assert pointed[2] == pytest.approx(0.0, abs=1e-9)


def test_look_at_keeps_the_camera_level():
    """A camera looking down at the arena must not be rolled on its side."""
    rpy = look_at_rpy((5.05, -2.40, 7.20), (4.50, 2.50, 0.00))
    assert abs(rpy[0]) < 1e-9, "roll must stay 0 for a look-at down-tilt"
    up = _rotate(rpy, (0.0, 0.0, 1.0))
    assert up[2] > 0.4, "the camera's up axis must still point mostly upwards"


def test_stable_shade_is_stable_and_bounded():
    a = stable_shade("boulder_0")
    b = stable_shade("boulder_0")
    assert a == b
    assert a != stable_shade("boulder_1")
    for name in ("a", "b", "c", "d", "e"):
        shade = stable_shade(name, spread=0.28)
        assert 1.0 - 0.14 - 1e-9 <= shade <= 1.0 + 0.14 + 1e-9
    assert stable_shade("anything", spread=0.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# world structure
# --------------------------------------------------------------------------- #
def _assets(obj="terrain.obj"):
    return WorldAssets(obj_file=obj, mtl_file="terrain.mtl", albedo_file="sand_albedo.png")


def _world(cfg, boulders=(), craters=(), camera=None):
    return build_world_sdf(
        cfg=cfg,
        assets=_assets(),
        output_dir="/tmp/cat_arena_test",
        boulders=boulders,
        craters=craters,
        seed=42,
        include_overview_camera=camera,
    )


def test_world_is_well_formed_xml(cfg):
    root = ET.fromstring(_world(cfg))
    assert root.tag == "sdf"
    world = root.find("world")
    assert world is not None
    assert world.get("name") == cfg.world_name


def test_physics_block_and_systems(cfg):
    text = _world(cfg)
    root = ET.fromstring(text)
    world = root.find("world")
    physics = world.find("physics")
    assert physics.get("type") == "dartsim"
    assert physics.find("max_step_size").text == str(cfg.physics.max_step_size)
    detector = physics.find("dart/collision_detector")
    assert detector is not None and detector.text == "bullet"
    gravity = world.find("gravity").text.split()
    assert [float(v) for v in gravity] == pytest.approx([0.0, 0.0, -9.8])

    plugins = {p.get("filename") for p in world.findall("plugin")}
    for required in (
        "gz-sim-physics-system",
        "gz-sim-user-commands-system",
        "gz-sim-scene-broadcaster-system",
        "gz-sim-contact-system",
    ):
        assert required in plugins, f"world is missing the {required} system plugin"


def test_sensors_system_only_with_the_camera(cfg):
    assert "gz-sim-sensors-system" not in _world(cfg, camera=False)
    assert "gz-sim-sensors-system" in _world(cfg, camera=True)


def test_world_does_not_override_the_gui_config(cfg):
    # A world-level <gui> element REPLACES Gazebo's default GUI plugin list
    # rather than extending it. Emitting just MinimalScene dropped
    # GzSceneManager, which is what adds entities to the render scene, and the
    # 3D view rendered black on gz-sim 8.15.0 while the Qt panels drew fine.
    # Leave the GUI configuration to Gazebo.
    root = ET.fromstring(_world(cfg))
    assert root.find("world/gui") is None


def test_terrain_model_uses_the_mesh_for_visual_and_collision(cfg):
    root = ET.fromstring(_world(cfg))
    world = root.find("world")
    terrain = world.find("./model[@name='terrain']")
    assert terrain is not None
    link = terrain.find("link")
    uris = [mesh.find("uri").text for mesh in link.findall(".//mesh")]
    assert len(uris) == 2, "terrain needs a visual mesh and a collision mesh"
    for uri in uris:
        assert uri.endswith("terrain.obj")
        assert uri.startswith("file://")
    for surface in link.findall(".//surface"):
        mu = surface.find("friction/ode/mu")
        mu2 = surface.find("friction/ode/mu2")
        assert float(mu.text) == pytest.approx(cfg.friction.mu)
        assert float(mu2.text) == pytest.approx(cfg.friction.mu2)
    assert link.find("collision").get("name") == "terrain_collision"
    assert link.find("visual").get("name") == "terrain_visual"


def test_walls_markings_and_markers_are_present(cfg):
    root = ET.fromstring(_world(cfg))
    world = root.find("world")
    names = {model.get("name") for model in world.findall("model")}
    assert "arena_walls" in names
    assert "zone_markings" in names
    assert "arena_markers" in names
    # walls are static and high enough for the configured height
    walls = world.find("./model[@name='arena_walls']")
    assert walls.find("static").text in ("true", "1")
    heights = [float(box.find("size").text.split()[2]) for box in walls.findall(".//collision/geometry/box")]
    assert heights and max(heights) == pytest.approx(cfg.walls.height)


def test_zone_markings_are_visual_only(cfg):
    root = ET.fromstring(_world(cfg))
    markings = root.find("./world/model[@name='zone_markings']")
    assert markings.findall(".//collision") == [], "zone markings must never collide"


def test_markers_can_be_disabled(cfg, tmp_path):
    import yaml

    raw = yaml.safe_load(open(cfg.path, "r", encoding="utf-8"))
    raw["markers"]["enabled"] = False
    alt = tmp_path / "arena.yaml"
    alt.write_text(yaml.safe_dump(raw), encoding="utf-8")
    from cat_arena.config import load_config

    cfg_off = load_config(str(alt))
    root = ET.fromstring(_world(cfg_off))
    names = {model.get("name") for model in root.find("world").findall("model")}
    assert "arena_markers" not in names


def test_boulders_and_craters_are_rendered(cfg):
    boulders = [
        Boulder("boulder_0", 3.0, 2.0, 0.35, 0.25, 0.0, "obstacle"),
        Boulder("boulder_1", 8.0, 3.0, 0.30, 0.30, 0.0, "excavation"),
    ]
    craters = [Crater("crater_0", 5.0, 2.0, 0.22, 0.12, "obstacle")]
    root = ET.fromstring(_world(cfg, boulders=boulders, craters=craters))
    world = root.find("world")
    names = {model.get("name") for model in world.findall("model")}
    assert {"boulder_0", "boulder_1"} <= names
    assert "crater_0" not in names, "craters are carved into the mesh, not separate models"
    for boulder in boulders:
        model = world.find(f"./model[@name='{boulder.boulder_id}']")
        pose = [float(v) for v in model.find("pose").text.split()]
        assert pose[0] == pytest.approx(boulder.x)
        assert pose[1] == pytest.approx(boulder.y)
        assert pose[2] == pytest.approx(boulder.center_z)
        radius = float(model.find(".//collision/geometry/sphere/radius").text)
        assert radius == pytest.approx(boulder.radius)
        assert model.find("static").text in ("true", "1")


def test_header_documents_the_seed_and_assumptions(cfg):
    text = _world(cfg)
    assert "seed                 : 42" in text
    assert "flagged assumption" in text
    assert "TEST / LABEL USE ONLY" in text


def test_crater_geometry_is_documented_in_the_header(cfg):
    craters = [Crater("crater_0", 5.0, 2.0, 0.22, 0.12, "obstacle")]
    text = _world(cfg, craters=craters)
    assert "crater_0" in text
    assert "depth=0.120" in text
