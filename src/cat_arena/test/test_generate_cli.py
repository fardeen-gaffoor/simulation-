"""End-to-end generator tests (CLI + determinism + manifest).

The generator is the module's deliverable: everything else reads what it writes.
These tests run the real CLI, check that every promised artifact exists, that a
seed is reproducible byte for byte, and that a broken configuration fails with a
non-zero exit code instead of writing a half-valid world.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil

import pytest
import yaml

from cat_arena.generate import generate, main

EXPECTED_FILES = (
    "arena.sdf",
    "terrain.obj",
    "terrain.mtl",
    "sand_albedo.png",
    "terrain_height.png",
    "ground_truth.yaml",
    "markers.yaml",
    "start_pose.yaml",
    "manifest.json",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# artifacts
# --------------------------------------------------------------------------- #
def test_generate_writes_every_documented_artifact(generated):
    for name in EXPECTED_FILES:
        path = os.path.join(generated.directory, name)
        assert os.path.isfile(path), f"missing artifact {name}"
        assert os.path.getsize(path) > 0
    assert os.path.basename(generated.directory) == f"seed_{generated.seed}"


def test_generated_world_validates(generated):
    assert generated.validation_output == "gz sdf -k: OK"


def test_world_file_uses_the_generated_mesh(generated):
    text = open(generated.world_path, "r", encoding="utf-8").read()
    assert "terrain.obj" in text
    assert os.path.basename(generated.terrain_mesh) == "terrain.obj"
    assert os.path.isfile(generated.terrain_mesh)


def test_manifest_records_everything_needed_to_reproduce(generated):
    with open(generated.manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["seed"] == generated.seed
    assert manifest["world_name"] == generated.config.world_name
    assert manifest["assumptions"]
    assert manifest["outputs"]["arena.sdf"]["sha256"] == _sha256(generated.world_path)
    assert manifest["rule_compliance"]["all"] is True
    assert manifest["validation"] == "gz sdf -k: OK"
    assert str(generated.seed) in manifest["reproduce"]
    assert manifest["counts"]["boulders"] >= 3
    assert manifest["counts"]["craters"] >= 3
    assert manifest["config"]["sha256"] == generated.config.config_sha256()


def test_start_pose_yaml_matches_the_result(generated):
    with open(generated.start_pose_path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    assert document["seed"] == generated.seed
    assert document["position"][0] == pytest.approx(generated.start_pose.x)
    assert document["position"][1] == pytest.approx(generated.start_pose.y)
    assert document["yaw_rad"] == pytest.approx(generated.start_pose.yaw)
    assert document["declared_forward_deg"] == pytest.approx(
        generated.config.start_pose.declared_forward_deg
    )


def test_markers_yaml_carries_the_apriltag_handoff(generated):
    with open(generated.markers_path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    assert document["family"] == "tag36h11"
    assert document["count"] == len(generated.config.markers.placements)
    assert len(document["apriltag_ros"]["standalone_tags"]) == document["count"]
    tag = document["apriltag_ros"]["standalone_tags"][0]
    assert set(tag) == {"id", "size", "name", "pose"}
    assert set(tag["pose"]["orientation"]) == {"x", "y", "z", "w"}
    # quaternions must be unit length
    q = tag["pose"]["orientation"]
    norm = sum(q[axis] ** 2 for axis in "xyzw") ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_spawn_arguments_match_the_start_pose(generated):
    x, y, z, yaw = generated.spawn_arguments()
    pose = generated.start_pose
    assert (x, y, z, yaw) == (pose.x, pose.y, pose.z, pose.yaw)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_same_seed_reproduces_byte_identical_artifacts(tmp_path, config_path):
    """Regenerating a seed into the same directory must be byte-identical.

    (The world embeds absolute mesh paths, so two generations into *different*
    directories legitimately differ in those URI lines - see the path-independent
    comparison below.)
    """
    out = tmp_path / "same"
    first = generate(seed=11, config_path=config_path, output_dir=str(out),
                     validate=False, verbose=False, include_timestamp=False)
    hashes = {name: _sha256(os.path.join(first.directory, name)) for name in (
        "arena.sdf", "terrain.obj", "terrain.mtl", "sand_albedo.png",
        "terrain_height.png", "ground_truth.yaml", "markers.yaml",
        "start_pose.yaml", "manifest.json",
    )}
    second = generate(seed=11, config_path=config_path, output_dir=str(out),
                      validate=False, verbose=False, include_timestamp=False)
    assert second.directory == first.directory
    for name, digest in hashes.items():
        assert _sha256(os.path.join(second.directory, name)) == digest, f"{name} is not reproducible"


def test_seed_reproducibility_is_independent_of_the_output_directory(tmp_path, config_path, generated):
    """The path-independent artifacts must match across output directories."""
    again = generate(seed=generated.seed, config_path=config_path, output_dir=str(tmp_path),
                     validate=False, verbose=False, include_timestamp=False)
    for name in ("terrain.obj", "terrain.mtl", "start_pose.yaml", "markers.yaml"):
        first = os.path.join(generated.directory, name)
        second = os.path.join(again.directory, name)
        assert _sha256(first) == _sha256(second), f"{name} is not reproducible"


def test_world_differs_between_directories_only_by_absolute_paths(tmp_path, config_path, generated):
    again = generate(seed=generated.seed, config_path=config_path, output_dir=str(tmp_path),
                     validate=False, verbose=False, include_timestamp=False)
    first = open(generated.world_path, "r", encoding="utf-8").read().splitlines()
    second = open(again.world_path, "r", encoding="utf-8").read().splitlines()
    assert len(first) == len(second)
    differences = [(a, b) for a, b in zip(first, second) if a != b]
    assert differences, "expected the embedded mesh paths to differ"
    for a, b in differences:
        assert "file://" in a and "file://" in b


def test_different_seeds_produce_different_worlds(tmp_path, config_path):
    a = generate(seed=1, config_path=config_path, output_dir=str(tmp_path / "a"),
                 validate=False, verbose=False, include_timestamp=False)
    b = generate(seed=2, config_path=config_path, output_dir=str(tmp_path / "b"),
                 validate=False, verbose=False, include_timestamp=False)
    assert _sha256(a.world_path) != _sha256(b.world_path)
    assert (a.start_pose.x, a.start_pose.y) != (b.start_pose.x, b.start_pose.y)


def test_random_seed_is_drawn_and_reported(tmp_path, config_path):
    a = generate(seed=None, config_path=config_path, output_dir=str(tmp_path),
                 validate=False, verbose=False, include_timestamp=False)
    assert a.seed >= 0
    assert os.path.basename(a.directory) == f"seed_{a.seed}"
    assert a.master_seed == a.seed


def test_output_directory_is_created_recursively(tmp_path, config_path):
    target = tmp_path / "deeply" / "nested"
    result = generate(seed=5, config_path=config_path, output_dir=str(target),
                      validate=False, verbose=False, include_timestamp=False)
    assert os.path.isdir(result.directory)
    assert str(target) in result.directory


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_quiet_mode_prints_the_output_directory(tmp_path, config_path, capsys):
    rc = main(["--seed", "3", "--config", config_path, "--output-dir", str(tmp_path),
               "--no-validate", "--quiet"])
    out = capsys.readouterr().out.strip().splitlines()
    assert rc == 0
    assert out[-1].endswith("seed_3")
    assert os.path.isfile(os.path.join(out[-1], "arena.sdf"))


def test_cli_json_mode(tmp_path, config_path, capsys):
    rc = main(["--seed", "4", "--config", config_path, "--output-dir", str(tmp_path),
               "--no-validate", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["seed"] == 4
    assert payload["boulders"] >= 3
    assert payload["craters"] >= 3
    assert payload["start_pose"]["yaw_deg"] == pytest.approx(
        math.degrees(payload["start_pose"]["yaw_rad"]), abs=1e-3
    )
    assert os.path.isfile(payload["world"])


def test_cli_print_paths_emits_shell_assignments(tmp_path, config_path, capsys):
    rc = main(["--seed", "6", "--config", config_path, "--output-dir", str(tmp_path),
               "--no-validate", "--print-paths"])
    out = capsys.readouterr().out
    assert rc == 0
    for key in ("CAT_ARENA_SEED=6", "CAT_ARENA_DIR=", "CAT_ARENA_WORLD=", "CAT_ARENA_START_YAW="):
        assert key in out


def test_cli_rejects_a_broken_config(tmp_path, config_path, capsys):
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["boulders"]["count"] = {"min": 1, "max": 2}
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(raw), encoding="utf-8")
    rc = main(["--seed", "1", "--config", str(broken), "--output-dir", str(tmp_path), "--quiet"])
    err = capsys.readouterr().err
    assert rc != 0
    assert "count" in err


def test_cli_reports_a_missing_config(tmp_path, capsys):
    rc = main(["--seed", "1", "--config", str(tmp_path / "nope.yaml"),
               "--output-dir", str(tmp_path), "--quiet"])
    assert rc != 0
    assert "not found" in capsys.readouterr().err


@pytest.mark.skipif(shutil.which("gz") is None, reason="gz CLI not available")
def test_invalid_world_is_rejected(tmp_path, capsys):
    """A world that gz cannot parse must raise instead of being written quietly."""
    from cat_arena.generate import GenerationError, _validate_sdf

    bad = tmp_path / "bad.sdf"
    bad.write_text("<sdf version='1.9'><world name='x'>", encoding="utf-8")
    with pytest.raises(GenerationError):
        _validate_sdf(str(bad))


@pytest.mark.skipif(shutil.which("gz") is None, reason="gz CLI not available")
def test_unknown_world_element_is_rejected(tmp_path):
    """gz sdf -k warns about 'not defined in SDF' - that is a defect for us.

    A world that carries an element Gazebo does not know about is a typo or a
    version mismatch, not a warning to live with, so ``_validate_sdf`` treats it
    as a failure and the generator never writes such a world.
    """
    from cat_arena.generate import GenerationError, _validate_sdf

    bad = tmp_path / "bad2.sdf"
    bad.write_text(
        "<?xml version='1.0'?>\n<sdf version='1.9'><world name='x'><not_a_sdf_thing/>"
        "<physics name='p' type='dartsim'><max_step_size>0.001</max_step_size>"
        "<real_time_factor>1</real_time_factor></physics></world></sdf>",
        encoding="utf-8",
    )
    with pytest.raises(GenerationError, match="not defined in SDF"):
        _validate_sdf(str(bad))


@pytest.mark.skipif(shutil.which("gz") is None, reason="gz CLI not available")
def test_the_generated_world_validates_without_warnings(generated):
    """'Valid.' on its own - no 'not defined in SDF' warnings either."""
    from cat_arena.generate import _validate_sdf

    assert _validate_sdf(generated.world_path) == "gz sdf -k: OK"
