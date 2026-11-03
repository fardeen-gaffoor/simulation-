"""The scene registries that feed the split."""

from __future__ import annotations

from rover_dataset.registry import (
    COLLECTION_FILE,
    IMPORTS_FILE,
    all_scenes,
    load_collection,
    load_imports,
    upsert_scene,
)


def test_empty_registry_is_empty(tmp_path):
    assert load_collection(str(tmp_path)) == {}
    assert load_imports(str(tmp_path)) == {}
    assert all_scenes(str(tmp_path)) == []


def test_upsert_adds_and_replaces(tmp_path):
    root = str(tmp_path)
    upsert_scene(root, {"scene_id": "real_a", "source": "real", "images": 10})
    upsert_scene(root, {"scene_id": "real_a", "source": "real", "images": 25})
    scenes = load_imports(root)["scenes"]
    assert len(scenes) == 1
    assert scenes[0]["images"] == 25


def test_registries_are_separate_files(tmp_path):
    root = str(tmp_path)
    upsert_scene(root, {"scene_id": "real_a", "source": "real", "images": 3})
    assert (tmp_path / IMPORTS_FILE).is_file()
    assert not (tmp_path / COLLECTION_FILE).exists()


def test_all_scenes_merges_and_sorts(tmp_path):
    root = str(tmp_path)
    upsert_scene(root, {"scene_id": "sim_0001", "source": "sim", "images": 5, "ok": True})
    upsert_scene(root, {"scene_id": "real_b", "source": "real", "images": 2})
    upsert_scene(root, {"scene_id": "sim_0000", "source": "sim", "images": 7, "ok": True})
    scenes = all_scenes(root)
    assert [scene["scene_id"] for scene in scenes] == ["real_b", "sim_0000", "sim_0001"]


def test_all_scenes_skips_failed_collections(tmp_path):
    root = str(tmp_path)
    upsert_scene(root, {"scene_id": "sim_0000", "source": "sim", "images": 5, "ok": True})
    upsert_scene(root, {"scene_id": "sim_0001", "source": "sim", "images": 0, "ok": False})
    assert [s["scene_id"] for s in all_scenes(root)] == ["sim_0000"]
    assert len(all_scenes(root, only_ok=False)) == 2
