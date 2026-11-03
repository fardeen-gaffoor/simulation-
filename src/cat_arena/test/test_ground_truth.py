"""Ground-truth document tests.

``ground_truth.yaml`` is the oracle file.  Two things must hold:

1. it is complete and correct (it is what the tests, the dataset labeller in
   module 3 and the batch scorer in module 5 rely on), and
2. it can never leak into the autonomy packages - the challenge forbids giving
   the autonomy stack the obstacle locations.
"""

from __future__ import annotations

import os

import pytest
import yaml

from cat_arena.ground_truth import (
    GROUND_TRUTH_HEADER,
    assert_autonomy_clean,
    write_ground_truth,
)

#: Packages that must never see hidden obstacle state.  rover_dataset is
#: deliberately NOT in this list: module 3 uses ground truth for automatic
#: labelling, which the challenge allows (labels are not fed to autonomy).
AUTONOMY_PACKAGES = ("rover_perception", "rover_navigation", "rover_excavation")


@pytest.fixture(scope="module")
def truth(generated):
    with open(generated.ground_truth_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_header_warns_loudly(generated):
    text = open(generated.ground_truth_path, "r", encoding="utf-8").read()
    assert text.startswith(GROUND_TRUTH_HEADER)
    assert "TEST / LABEL USE ONLY" in text
    assert "FORBIDDEN" in text
    assert "rover_perception" in text


def test_rule_compliance_passes_for_all_sub_rules(truth):
    rules = truth["rule_compliance"]
    assert rules["all"] is True, f"violated rules: {[k for k, v in rules.items() if v is False]}"
    for key, value in rules.items():
        if key != "all":
            assert value is True, f"{key} is False"


def test_document_matches_the_generated_layout(truth, generated):
    assert truth["seed"] == generated.seed
    assert len(truth["boulders"]) == len(generated.layout.boulders)
    assert len(truth["craters"]) == len(generated.layout.craters)
    assert truth["world_name"] == generated.config.world_name


def test_boulders_are_described_completely(truth, cfg):
    assert len(truth["boulders"]) >= 3
    for boulder in truth["boulders"]:
        assert 0.30 - 1e-9 <= boulder["diameter"] <= 0.40 + 1e-9
        assert boulder["exposed_height"] > 0.0
        assert boulder["buried_depth"] >= 0.0
        assert boulder["top_z"] == pytest.approx(boulder["center"][2] + boulder["radius"])
        assert boulder["zone"] in cfg.boulders.zones
        assert boulder["static"] is True


def test_craters_are_described_as_negative_features(truth):
    assert len(truth["craters"]) >= 3
    for crater in truth["craters"]:
        assert crater["depth"] > 0.0, "depth is positive downwards"
        assert 0.25 - 1e-9 <= crater["width"] <= 0.50 + 1e-9
        assert crater["zone"] == "obstacle"


def test_zones_are_exported_matching_the_config(truth, cfg):
    for name, zone in cfg.zones.items():
        entry = truth["zones"][name]
        assert entry["rect"] == pytest.approx(zone.rect.as_list())
        assert entry["center"] == pytest.approx(list(zone.rect.center))
        assert entry["tiling"] == zone.tiling
        assert entry["obstacles_allowed"] == zone.obstacles_allowed


def test_start_pose_in_the_document_is_the_generated_pose(truth, generated):
    pose = truth["start_pose"]
    assert pose["position"][0] == pytest.approx(generated.start_pose.x)
    assert pose["position"][1] == pytest.approx(generated.start_pose.y)
    assert pose["yaw_rad"] == pytest.approx(generated.start_pose.yaw)
    assert len(pose["corners_xy"]) == 4


def test_obstacles_never_reported_in_the_start_zone(truth):
    start = truth["zones"]["start"]["rect"]
    for item in truth["boulders"] + truth["craters"]:
        cx, cy = item["center"][0], item["center"][1]
        radius = item["radius"]
        assert (
            cx + radius <= start[0]
            or cx - radius >= start[2]
            or cy + radius <= start[1]
            or cy - radius >= start[3]
        )


def test_terrain_section_describes_the_mesh(truth):
    terrain = truth["terrain"]
    assert terrain["files"]["mesh"] == "terrain.obj"
    assert terrain["triangles"] > 10000
    assert terrain["z_min"] < 0.0, "craters must show up as a negative z in the mesh"
    assert terrain["skirt_depth"] == pytest.approx(0.30)


def test_files_and_versions_are_recorded(truth):
    assert "arena.sdf" in truth["files"]
    for name, entry in truth["files"].items():
        assert entry["bytes"] > 0
        assert len(entry["sha256"]) == 64
    assert "gz_sim" in truth["versions"]
    assert "numpy" in truth["versions"]


def test_assumptions_are_carried_into_the_document(truth):
    joined = " | ".join(truth["assumptions"]).lower()
    assert "wall height" in joined
    assert "friction" in joined
    assert "organizer confirmation" in joined


# --------------------------------------------------------------------------- #
# the leak check
# --------------------------------------------------------------------------- #
def test_leak_check_flags_a_package_that_reads_ground_truth(tmp_path):
    pkg = tmp_path / "rover_navigation"
    pkg.mkdir()
    (pkg / "planner.py").write_text(
        "with open('ground_truth.yaml') as fh:\n    pass\n", encoding="utf-8"
    )
    violations = assert_autonomy_clean([str(pkg)])
    assert violations and "ground_truth.yaml" in violations[0]


def test_leak_check_passes_for_clean_packages(tmp_path):
    pkg = tmp_path / "rover_perception"
    pkg.mkdir()
    (pkg / "detector.py").write_text("import rclpy\nprint('clean')\n", encoding="utf-8")
    (pkg / "config.yaml").write_text("camera: /cam_nav\n", encoding="utf-8")
    assert assert_autonomy_clean([str(pkg)]) == []


def test_leak_check_ignores_missing_directories(tmp_path):
    assert assert_autonomy_clean([str(tmp_path / "does_not_exist")]) == []


def test_autonomy_packages_are_clean_if_they_exist():
    """Enforces the rule for every autonomy package that exists in this repo."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    roots = [os.path.join(src, name) for name in AUTONOMY_PACKAGES]
    violations = assert_autonomy_clean(roots)
    assert violations == [], "autonomy packages must never reference ground truth:\n" + "\n".join(violations)


def test_writer_round_trip(tmp_path):
    document = {"schema_version": 1, "seed": 1, "boulders": [], "craters": []}
    path = tmp_path / "ground_truth.yaml"
    write_ground_truth(str(path), document)
    text = path.read_text(encoding="utf-8")
    assert text.startswith(GROUND_TRUTH_HEADER)
    assert yaml.safe_load(text)["seed"] == 1
