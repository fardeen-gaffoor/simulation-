"""Tests for the Stage 1 motor and torque analysis (motor_analysis.py)."""

from __future__ import annotations

import pytest

from rover_description.motor_analysis import (
    G,
    KG_CM_TO_NM,
    MotorAnalysis,
    analyze_motor_torque,
    load_and_analyze,
)


@pytest.fixture
def analysis(tmp_path):
    import os
    import shutil

    src_root = os.path.join(os.path.dirname(__file__), "..", "config")
    dst = tmp_path / "cfg"
    dst.mkdir()
    for name in ("robot_params.yaml", "motors.yaml"):
        shutil.copy(os.path.join(src_root, name), dst / name)
    _, motors, analysis = load_and_analyze(
        robot_path=str(dst / "robot_params.yaml"),
        motors_path=str(dst / "motors.yaml"),
        mass_kg=50.0,
    )
    return motors, analysis


def test_torque_against_limits(analysis):
    motors, ana = analysis
    for c in ana.cases:
        # Each case must have a comparison result
        assert c.vs_rated in ("OK", "EXCEEDS")
        assert c.vs_advised in ("OK", "EXCEEDS")
        assert c.vs_gearbox in ("OK", "EXCEEDS")


def test_stall_requirement_not_met(analysis):
    motors, ana = analysis
    assert ana.stall_requirement_met is False
    assert ana.motor_stall_req_kgcm == 125.0
    assert ana.motor_gearbox_break_kgcm == 100.0


def test_speed_is_positive(analysis):
    motors, ana = analysis
    for c in ana.cases:
        assert c.speed_mps > 0.0


def test_current_is_positive(analysis):
    motors, ana = analysis
    for c in ana.cases:
        assert c.current_a > 0.0


def test_force_increases_with_slope(analysis):
    motors, ana = analysis
    # Force at slope=10 should be > force at slope=0 for same rr
    f0 = [c for c in ana.cases if c.slope_deg == 0.0 and c.rr == 0.25][0]
    f10 = [c for c in ana.cases if c.slope_deg == 10.0 and c.rr == 0.25][0]
    assert f10.total_force_n > f0.total_force_n


def test_torque_increases_with_rr(analysis):
    motors, ana = analysis
    # Torque at rr=0.4 should be > torque at rr=0.15 for same slope
    t_low = [c for c in ana.cases if c.slope_deg == 5.0 and c.rr == 0.15][0]
    t_high = [c for c in ana.cases if c.slope_deg == 5.0 and c.rr == 0.40][0]
    assert t_high.torque_motor_kgcm > t_low.torque_motor_kgcm


def test_table_prints(analysis):
    motors, ana = analysis
    text = ana.table()
    assert "Motor & torque analysis" in text
    assert "Rated torque" in text
    assert "Gearbox break" in text
    assert "Stall requirement" in text


def test_kgcm_conversion():
    assert KG_CM_TO_NM == 0.0980665
    # 40 kg.cm = 3.92266 N.m
    assert 40.0 * KG_CM_TO_NM == pytest.approx(3.92266, abs=1e-4)


def test_gravity_constant():
    assert G == 9.81
