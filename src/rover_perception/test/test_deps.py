"""The dependency probe: what module 4 needs, and what it refuses.

``perception_info.py --deps`` is the first thing to run on a new machine, so the
probe must never raise, must distinguish required from optional, and must state
*why* cv2 / cv_bridge are refused rather than simply listing them as missing.
"""

from __future__ import annotations

import pytest

from rover_perception.deps import (
    PYTHON,
    ROS,
    Dependency,
    check_dependencies,
    missing_required,
    report,
    summary_dict,
)


def test_check_dependencies_never_raises_and_probes_real_modules():
    rows = check_dependencies()
    assert rows, "the probe knows about something"
    names = {row.name for row in rows}
    for expected in ("numpy", "yaml", "Pillow", "dt-apriltags", "rover_perception_msgs"):
        assert expected in names
    # the three the classical stack genuinely needs are importable here
    available = {row.name: row.available for row in rows}
    assert available["numpy"] and available["yaml"] and available["Pillow"] and available["dt-apriltags"]


def test_every_row_is_well_formed():
    for row in check_dependencies():
        assert row.name
        assert row.kind in (PYTHON, ROS)
        assert isinstance(row.required, bool)
        assert isinstance(row.available, bool)
        assert row.detail, "a row always says something"
        if not row.available and row.install:
            assert "install" in row.install or "colcon" in row.install or "pip" in row.install


def test_the_state_string_reflects_availability_and_requirement():
    assert Dependency("numpy", PYTHON, True, True, "ok").state == "ok"
    assert Dependency("numpy", PYTHON, True, False, "gone").state == "MISSING"
    assert Dependency("onnxruntime", PYTHON, False, False, "gone").state == "optional"


def test_to_dict_round_trips_the_fields():
    row = Dependency("robot_localization", ROS, False, False, "not found", "apt install it")
    entry = row.to_dict()
    assert entry["name"] == "robot_localization"
    assert entry["kind"] == ROS
    assert entry["required"] is False
    assert entry["available"] is False
    assert entry["state"] == "optional"
    assert entry["install"] == "apt install it"


def test_cv2_and_cv_bridge_are_refused_by_design():
    rows = {row.name: row for row in check_dependencies()}
    for name in ("cv2", "cv_bridge"):
        assert name in rows
        row = rows[name]
        assert not row.available
        assert not row.required, "the refusal is planned, not a blocker"
        assert "refused" in row.detail
        assert row.install == "", "there is no install hint for something we refuse"


def test_missing_required_defaults_to_a_fresh_probe():
    rows = check_dependencies()
    assert missing_required(rows) == tuple(row for row in rows if row.required and not row.available)


def test_missing_required_can_be_given_a_synthetic_list():
    gone = Dependency("thing", PYTHON, True, False, "gone", "apt install thing")
    optional_gone = Dependency("extra", PYTHON, False, False, "gone", "pip install extra")
    present = Dependency("numpy", PYTHON, True, True, "here")
    assert missing_required([gone, optional_gone, present]) == (gone,)


def test_summary_dict_names_only_what_blocks_a_run():
    gone = Dependency("thing", PYTHON, True, False, "gone", "apt install thing")
    optional_gone = Dependency("extra", PYTHON, False, False, "gone", "pip install extra")
    present = Dependency("numpy", PYTHON, True, True, "here")
    summary = summary_dict([present, gone, optional_gone])
    assert summary["checked"] == 3
    assert summary["ok"] is False
    assert summary["missing_required"] == ["thing"]
    assert summary["missing_optional"] == ["extra"], "only optional rows with an install hint"
    assert len(summary["dependencies"]) == 3


def test_summary_is_ok_when_nothing_required_is_missing():
    present = Dependency("numpy", PYTHON, True, True, "here")
    summary = summary_dict([present])
    assert summary["ok"] is True
    assert summary["missing_required"] == []


def test_report_prints_a_row_and_its_install_hint():
    gone = Dependency("thing", PYTHON, True, False, "gone", "apt install thing")
    text = report([gone])
    assert "thing" in text
    assert "MISSING" in text
    assert "install: apt install thing" in text
    assert "1 REQUIRED dependency/ies missing" in text


def test_report_of_a_healthy_set_has_no_missing_line():
    present = Dependency("numpy", PYTHON, True, True, "here")
    text = report([present])
    assert "REQUIRED" not in text
    assert "ok" in text


def test_the_live_report_is_printable():
    text = report()
    assert "numpy" in text
    assert isinstance(text, str) and text.strip()
