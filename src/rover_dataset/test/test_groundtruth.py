"""Ground-truth parsing tests (module 1's contract)."""

from __future__ import annotations

import pytest
import yaml

from rover_dataset.groundtruth import GroundTruthError, load_ground_truth

SAMPLE = {
    "schema_version": 1,
    "seed": 7,
    "world_name": "cat_arena",
    "arena": {"length_x": 9.0, "width_y": 5.0, "spawn_z_offset": 0.08},
    "boulders": [
        {
            "id": "boulder_0",
            "center": [3.0, 2.5, 0.084785],
            "diameter": 0.36988,
            "radius": 0.18494,
            "exposed_height": 0.269725,
            "top_z": 0.269725,
            "zone": "obstacle",
        }
    ],
    "craters": [
        {"id": "crater_0", "center": [5.0, 1.5], "radius": 0.22, "width": 0.44,
         "depth": 0.11, "zone": "obstacle"}
    ],
    "start_pose": {"position": [1.294, 3.753, 0.08], "yaw_rad": 1.386},
    "zones": {"obstacle": {"rect": [2.0, 0.0, 7.5, 5.0]}},
}


def _write(tmp_path, payload):
    path = tmp_path / "ground_truth.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return str(path)


def test_loads_the_module_one_schema(tmp_path):
    ground_truth = load_ground_truth(_write(tmp_path, SAMPLE))
    assert ground_truth.seed == 7
    assert ground_truth.length_x == pytest.approx(9.0)
    assert len(ground_truth.boulders) == 1
    assert len(ground_truth.craters) == 1
    boulder = ground_truth.boulders[0]
    assert boulder.radius == pytest.approx(0.18494)
    assert boulder.top_z == pytest.approx(boulder.center[2] + boulder.radius)
    assert boulder.xy == pytest.approx((3.0, 2.5))
    assert ground_truth.craters[0].xy == (5.0, 1.5)
    assert ground_truth.start_pose[3] == pytest.approx(1.386)


def test_objects_are_ordered_boulders_then_craters(tmp_path):
    ground_truth = load_ground_truth(_write(tmp_path, SAMPLE))
    kinds = [kind for kind, _ in ground_truth.objects()]
    assert kinds == ["boulder", "crater"]


def test_derives_diameter_and_width_when_absent(tmp_path):
    payload = dict(SAMPLE)
    payload["boulders"] = [{"id": "b", "center": [1.0, 1.0, 0.2], "radius": 0.15}]
    payload["craters"] = [{"id": "c", "center": [2.0, 2.0], "radius": 0.2}]
    ground_truth = load_ground_truth(_write(tmp_path, payload))
    assert ground_truth.boulders[0].diameter == pytest.approx(0.30)
    assert ground_truth.craters[0].width == pytest.approx(0.40)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(GroundTruthError, match="not found"):
        load_ground_truth(str(tmp_path / "nope.yaml"))


def test_bad_schema_version_is_rejected(tmp_path):
    payload = dict(SAMPLE, schema_version=9)
    with pytest.raises(GroundTruthError, match="schema_version"):
        load_ground_truth(_write(tmp_path, payload))


def test_empty_arena_is_rejected(tmp_path):
    payload = dict(SAMPLE, boulders=[], craters=[])
    with pytest.raises(GroundTruthError, match="no boulders or craters"):
        load_ground_truth(_write(tmp_path, payload))


def test_malformed_centre_is_rejected(tmp_path):
    payload = dict(SAMPLE)
    payload["boulders"] = [{"id": "b", "center": [1.0, 2.0], "radius": 0.15}]
    with pytest.raises(GroundTruthError, match="center"):
        load_ground_truth(_write(tmp_path, payload))
