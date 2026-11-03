"""Tests for the generator: artifacts, determinism and URDF validation."""

from __future__ import annotations

import json
import os

import pytest

from conftest import gz_available, xacro_available
from rover_description.config import CAMERA_TOPICS
from rover_description.generate import (
    generate,
    inertias_document,
    validate_urdf,
)


def test_files_created(generated):
    assert os.path.isfile(generated.urdf_path)
    assert os.path.isfile(generated.inertias_path)
    assert os.path.isfile(generated.calibration_path)
    assert os.path.isfile(generated.mass_path)
    assert os.path.isfile(generated.envelope_path)
    assert os.path.isfile(generated.manifest_path)
    assert generated.urdf_path.endswith("rover.urdf")


def test_result_ok(generated):
    assert generated.ok
    assert generated.warnings == []
    assert generated.envelope.ok and generated.mass.ok


def test_validate_urdf_ok(generated, cfg):
    with open(generated.urdf_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    assert validate_urdf(text, cfg) == []


def test_validate_urdf_flags_magnetometer(generated, cfg):
    with open(generated.urdf_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    bad = text.replace("</robot>", '  <sensor name="mag" type="magnetometer"/>\n</robot>')
    problems = validate_urdf(bad, cfg)
    assert any("magnetometer" in p or "forbidden" in p for p in problems)


def test_validate_urdf_flags_missing_camera(generated, cfg):
    with open(generated.urdf_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    # drop the cam_dump sensor block entirely
    start = text.find('name="cam_dump"')
    assert start > 0
    bad = text[: start - 40] + text[start + 400:]
    problems = validate_urdf(bad, cfg)
    assert problems  # at least one structural/camera problem


def test_inertias_doc_matches_geometry(generated, cfg):
    import yaml

    with open(generated.inertias_path, "r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    expected = inertias_document(cfg)
    for link in ("base_link", "ladder_link", "belt_link", "tub_link", "funnel_link"):
        assert doc[link]["mass"] == pytest.approx(expected[link]["mass"], rel=1e-6)
        assert doc[link]["ixx"] == pytest.approx(expected[link]["ixx"], rel=1e-6)


def test_manifest_contents(generated):
    with open(generated.manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["module"] == "rover_description"
    assert manifest["robot_name"] == "rover"
    assert len(manifest["config_sha256"]) == 64
    assert manifest["assumptions"]
    assert manifest["envelope"]["ok"] is True
    assert manifest["mass"]["ok"] is True
    assert manifest["files"]["rover.urdf"]


def test_determinism(tmp_path, config_path):
    if not xacro_available():
        pytest.skip("xacro not available")
    a = generate(config_path=config_path, output_dir=str(tmp_path / "a"), write_sdf=False)
    b = generate(config_path=config_path, output_dir=str(tmp_path / "b"), write_sdf=False)
    for name in ("inertias.yaml", "rover.urdf", "calibration.yaml", "mass_budget.yaml", "envelope.yaml"):
        with open(os.path.join(a.output_dir, name), "rb") as ha, open(
            os.path.join(b.output_dir, name), "rb"
        ) as hb:
            assert ha.read() == hb.read(), f"{name} is not deterministic"


def test_sdf_conversion(tmp_path, config_path):
    if not (gz_available() and xacro_available()):
        pytest.skip("gz or xacro not available")
    result = generate(config_path=config_path, output_dir=str(tmp_path / "sdf"), write_sdf=True)
    assert result.sdf_path and os.path.isfile(result.sdf_path)
    with open(result.sdf_path, "r", encoding="utf-8") as handle:
        sdf = handle.read()
    # every camera topic and the imu survive the URDF -> SDF conversion
    for topic in CAMERA_TOPICS:
        assert f"<topic>{topic}</topic>" in sdf
    assert "<topic>/imu</topic>" in sdf
    assert "gz-sim-diff-drive-system" in sdf
