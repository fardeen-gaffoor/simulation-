"""Tests for the Stage 1 geometry consistency check (geometry_check.py).

Flags the wheelbase conflict: 0.65 m wheelbase + 0.20 m wheel diameter
= 0.85 m > 0.81 m overall length.
"""

from __future__ import annotations

import pytest

from rover_description.geometry_check import GeometryReport, check_geometry, load_and_check


@pytest.fixture
def robot_cfg(tmp_path):
    import os
    import shutil

    src = os.path.join(os.path.dirname(__file__), "..", "config", "robot_params.yaml")
    dst = tmp_path / "robot_params.yaml"
    shutil.copy(src, dst)
    return load_and_check(str(dst))[0]


def test_wheelbase_conflict_is_flagged(robot_cfg):
    report = check_geometry(robot_cfg)
    wb_check = next(c for c in report.checks if c.name == "wheelbase_vs_length")
    assert wb_check.status == "conflict"
    assert "0.85" in wb_check.detail or "protrude" in wb_check.detail
    assert len(report.conflicts) > 0


def test_track_is_consistent(robot_cfg):
    report = check_geometry(robot_cfg)
    track_check = next(c for c in report.checks if c.name == "track_vs_width")
    assert track_check.status == "pass"


def test_clearance_is_consistent(robot_cfg):
    report = check_geometry(robot_cfg)
    clearance_check = next(c for c in report.checks if c.name == "clearance_vs_wheel_radius")
    assert clearance_check.status == "pass"


def test_envelope_checks(robot_cfg):
    report = check_geometry(robot_cfg)
    for c in report.checks:
        if c.name.startswith("envelope_"):
            assert c.status in ("pass", "conflict")


def test_mass_target_vs_ceiling(robot_cfg):
    report = check_geometry(robot_cfg)
    mc_check = next(c for c in report.checks if c.name == "mass_target_vs_ceiling")
    assert mc_check.status == "pass"


def test_report_ok_false_on_conflict(robot_cfg):
    report = check_geometry(robot_cfg)
    assert report.ok is False  # wheelbase conflict


def test_table_prints(robot_cfg):
    report = check_geometry(robot_cfg)
    text = report.table()
    assert "wheelbase_vs_length" in text
    assert "conflict" in text.lower()


def test_to_dict(robot_cfg):
    report = check_geometry(robot_cfg)
    d = report.to_dict()
    assert "ok" in d
    assert "checks" in d
    assert "conflicts" in d
    assert "flags" in d


def test_load_and_check(tmp_path):
    import os
    import shutil

    src = os.path.join(os.path.dirname(__file__), "..", "config", "robot_params.yaml")
    dst = tmp_path / "robot_params.yaml"
    shutil.copy(src, dst)
    robot, report = load_and_check(str(dst))
    assert isinstance(robot, object)
    assert isinstance(report, GeometryReport)
