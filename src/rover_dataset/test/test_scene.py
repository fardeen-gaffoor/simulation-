"""Per-scene domain randomization tests.

The scene config must be (a) deterministic, (b) actually random across scenes,
and (c) still a *valid* config for modules 1 and 2 -- a scene that module 1
refuses to load is worse than no randomization at all.
"""

from __future__ import annotations

import json
import math
import os

import pytest
import yaml

from rover_dataset.scene import (
    prepare_scenes,
    resolve_arena_config_path,
    resolve_rover_config_path,
    scene_specs,
    write_scene_files,
)

pytest.importorskip("cat_arena", reason="module 1 must be built for scene validation")


def test_base_configs_are_locatable():
    assert os.path.isfile(resolve_arena_config_path())
    assert os.path.isfile(resolve_rover_config_path())


def test_scene_specs_are_deterministic(cfg):
    first = scene_specs(cfg, master_seed=1234, count=3)
    second = scene_specs(cfg, master_seed=1234, count=3)
    assert [s.as_dict() for s in first] == [s.as_dict() for s in second]
    assert [s.arena_yaml for s in first] == [s.arena_yaml for s in second]
    assert [s.rover_yaml for s in first] == [s.rover_yaml for s in second]


def test_scene_specs_differ_between_scenes(cfg):
    specs = scene_specs(cfg, master_seed=1234, count=4)
    directions = [tuple(s.arena_yaml["light"]["direction"]) for s in specs]
    assert len(set(directions)) == len(directions)
    seeds = [s.seed for s in specs]
    assert len(set(seeds)) == len(seeds)


def test_scene_specs_differ_between_seeds(cfg):
    a = scene_specs(cfg, master_seed=1, count=2)
    b = scene_specs(cfg, master_seed=2, count=2)
    assert a[0].arena_yaml["light"]["diffuse"] != b[0].arena_yaml["light"]["diffuse"]


def test_lighting_stays_physical(cfg):
    for spec in scene_specs(cfg, master_seed=99, count=8):
        direction = spec.arena_yaml["light"]["direction"]
        norm = math.sqrt(sum(v * v for v in direction))
        assert norm == pytest.approx(1.0, abs=1e-4)
        assert direction[2] < 0.0  # light travels downward
        diffuse = spec.arena_yaml["light"]["diffuse"]
        assert all(0.0 <= c <= 1.0 for c in diffuse)
        ambient = spec.arena_yaml["scene"]["ambient"]
        assert all(0.0 <= c <= 1.0 for c in ambient)
        background = spec.arena_yaml["scene"]["background"]
        assert all(0.0 <= c <= 1.0 for c in background)


def test_shadows_toggle_matches_the_light(cfg):
    for spec in scene_specs(cfg, master_seed=7, count=6):
        assert spec.arena_yaml["light"]["cast_shadows"] == spec.arena_yaml["scene"]["shadows"]


def test_camera_mount_jitter_stays_within_its_bound(cfg):
    for spec in scene_specs(cfg, master_seed=42, count=4):
        record = spec.params["camera_mount_jitter"]
        assert record["cameras"]
        for entry in record["cameras"].values():
            delta = entry["origin_delta"]
            assert math.sqrt(sum(v * v for v in delta)) <= record["translation_m"] + 1e-9
            for axis in entry["rpy_delta_deg"]:
                assert abs(axis) <= record["rotation_deg"] + 1e-9


def test_jittered_rover_keeps_the_cameras_where_they_were(cfg):
    base = yaml.safe_load(open(resolve_rover_config_path(), encoding="utf-8"))
    base_origins = {c["name"]: list(c["origin"]) for c in base["sensors"]["cameras"]}
    spec = scene_specs(cfg, master_seed=5, count=1)[0]
    for camera in spec.rover_yaml["sensors"]["cameras"]:
        origin = camera["origin"]
        delta = max(abs(origin[i] - base_origins[camera["name"]][i]) for i in range(3))
        assert delta <= cfg.scene.mount_translation_m.hi + 1e-9


def test_generated_arena_config_is_valid_for_module_1(cfg, tmp_path):
    from cat_arena.config import load_config as load_arena

    spec = scene_specs(cfg, master_seed=2024, count=1)[0]
    files = write_scene_files(spec, str(tmp_path))
    arena = load_arena(files.arena_config)
    # module 1's rules still hold after the override
    assert arena.length_x == pytest.approx(9.0)
    assert arena.width_y == pytest.approx(5.0)
    assert arena.sand_depth == pytest.approx(0.30)
    # and the randomized fields really moved, inside their configured ranges
    assert cfg.scene.ambient.contains(arena.scene.ambient[0])
    assert cfg.scene.sun_elevation_deg.lo <= spec.params["sun_elevation_deg"]
    assert cfg.scene.sun_elevation_deg.hi >= spec.params["sun_elevation_deg"]


def test_generated_rover_config_is_valid_for_module_2(cfg, tmp_path):
    from rover_description.config import load_config as load_rover

    spec = scene_specs(cfg, master_seed=2024, count=1)[0]
    files = write_scene_files(spec, str(tmp_path))
    rover = load_rover(files.rover_config)
    assert len(rover.sensors.cameras) == 4


def test_write_scene_files_lays_out_three_files(cfg, tmp_path):
    spec = scene_specs(cfg, master_seed=11, count=1)[0]
    files = write_scene_files(spec, str(tmp_path))
    assert os.path.isfile(files.arena_config)
    assert os.path.isfile(files.rover_config)
    assert os.path.isfile(files.record)
    record = json.loads(open(files.record, encoding="utf-8").read())
    assert record["scene_id"] == spec.scene_id
    assert record["seed"] == spec.seed
    assert len(record["files"]["arena.yaml"]) == 64


def test_prepare_scenes_writes_an_index(cfg, tmp_path):
    from dataclasses import replace

    local = replace(cfg, paths=replace(cfg.paths, scenes_dir=str(tmp_path)))
    specs, files = prepare_scenes(local, master_seed=3, count=3)
    assert len(specs) == len(files) == 3
    index = json.loads(open(os.path.join(str(tmp_path), "scenes.json"), encoding="utf-8").read())
    assert index["count"] == 3
    assert index["master_seed"] == 3
    assert [entry["scene_id"] for entry in index["scenes"]] == [s.scene_id for s in specs]


def test_scene_ids_are_zero_padded_and_sortable(cfg):
    specs = scene_specs(cfg, master_seed=1, count=3)
    assert [s.scene_id for s in specs] == ["sim_0000", "sim_0001", "sim_0002"]
