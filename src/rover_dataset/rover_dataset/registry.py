"""The little scene registries that keep the dataset reproducible.

Two JSON files live at the dataset root:

* ``collection.json`` -- written by ``collect_dataset.py`` (sim scenes),
* ``imports.json``    -- written by ``import_real_photos.py`` (real photo batches).

:func:`all_scenes` merges them into the single scene list the split consumes.
Keeping them separate means importing more real photos never rewrites the record
of what the simulator produced.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

__all__ = [
    "COLLECTION_FILE",
    "IMPORTS_FILE",
    "load_collection",
    "load_imports",
    "upsert_scene",
    "all_scenes",
]

COLLECTION_FILE = "collection.json"
IMPORTS_FILE = "imports.json"


def _read(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, sort_keys=True)
        handle.write("\n")


def load_collection(root: str) -> Dict[str, Any]:
    return _read(os.path.join(root, COLLECTION_FILE))


def load_imports(root: str) -> Dict[str, Any]:
    return _read(os.path.join(root, IMPORTS_FILE))


def upsert_scene(root: str, scene: Dict[str, Any], registry: str = IMPORTS_FILE) -> Dict[str, Any]:
    """Insert or replace one scene record in a registry file."""
    path = os.path.join(root, registry)
    payload = _read(path) or {"schema_version": 1, "scenes": []}
    scenes: List[Dict[str, Any]] = list(payload.get("scenes") or [])
    scene_id = str(scene["scene_id"])
    scenes = [existing for existing in scenes if str(existing.get("scene_id")) != scene_id]
    scenes.append(scene)
    scenes.sort(key=lambda entry: str(entry.get("scene_id", "")))
    payload["scenes"] = scenes
    payload["schema_version"] = 1
    _write(path, payload)
    return payload


def all_scenes(root: str, *, only_ok: bool = True) -> List[Dict[str, Any]]:
    """Every known scene, merged and de-duplicated by ``scene_id``."""
    merged: Dict[str, Dict[str, Any]] = {}
    for payload in (load_collection(root), load_imports(root)):
        for scene in payload.get("scenes") or []:
            scene_id = str(scene.get("scene_id", ""))
            if not scene_id:
                continue
            if only_ok and "ok" in scene and not scene["ok"]:
                continue
            merged[scene_id] = dict(scene)
    return [merged[key] for key in sorted(merged)]
