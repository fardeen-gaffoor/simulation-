"""Tests for the Stage 1 configuration loader (params.py).

Every value in robot_params.yaml, motors.yaml, battery.yaml and cameras.yaml
is tagged given / assumed / derived.  These tests assert that the tags parse
and that the computed values (frame mass, capacity Wh) are correct.
"""

from __future__ import annotations

import pytest

from rover_description.params import (
    load_battery,
    load_cameras,
    load_motors,
    load_robot_params,
)


def test_robot_params_loads(cfg_path):
    robot = load_robot_params(cfg_path)
    assert robot.name
    assert robot.declared_forward == "+x"
    assert robot.limits.envelope_length_x.value == 1.5
    assert robot.limits.mass_ceiling_kg.value == 80.0
    assert robot.mass.total_max_kg.value == 50.0


def test_frame_mass_is_computed(cfg_path):
    robot = load_robot_params(cfg_path)
    # frame = 50 - (4*3.0 + 4*0.30 + 3.5 + 5.0 + 4.0 + 2.0) = 50 - 27.7 = 22.3
    expected = 50.0 - (4.0 * 3.0 + 4.0 * 0.30 + 3.5 + 5.0 + 4.0 + 2.0)
    assert robot.mass.frame_mass_computed() == pytest.approx(expected, abs=1e-9)


def test_tags_are_valid(cfg_path):
    robot = load_robot_params(cfg_path)
    for entry in (
        robot.mass.total_max_kg,
        robot.mass.wheel_mass_kg,
        robot.mass.motor_mass_kg,
        robot.mass.battery_mass_kg,
        robot.mass.ladder_mass_kg,
        robot.mass.tub_funnel_mass_kg,
        robot.mass.electronics_wiring_mass_kg,
        robot.mass.frame_mass_kg,
    ):
        assert entry.tag in ("given", "assumed", "derived"), f"bad tag {entry.tag!r}"


def test_motors_loads(motors_path):
    motors = load_motors(motors_path)
    assert motors.model == "Robokites Rhino"
    assert motors.torque.rated_torque_kgcm.value == 40.0
    assert motors.torque.advised_max_torque_kgcm.value == 75.0
    assert motors.torque.gearbox_breaking_torque_kgcm.value == 100.0
    assert motors.torque.stall_torque_requirement_kgcm.value == 125.0
    assert motors.conversion.kgcm_to_nm.value == 0.0980665


def test_stall_requirement_not_met(motors_path):
    motors = load_motors(motors_path)
    # 125 kg.cm requirement > 100 kg.cm gearbox break -> NOT MET
    assert motors.torque.stall_torque_requirement_kgcm.value > motors.torque.gearbox_breaking_torque_kgcm.value


def test_battery_loads(battery_path):
    battery = load_battery(battery_path)
    assert battery.electrical.voltage_nominal_v.value == 12.0
    assert battery.electrical.capacity_mah.value == 30000
    # 12 V x 30 Ah = 360 Wh
    assert battery.electrical.capacity_wh.value == pytest.approx(360.0)


def test_cameras_loads(cameras_path):
    cameras = load_cameras(cameras_path)
    assert cameras.preset.default == "A"
    assert cameras.preset.image_format == "RGB"
    assert cameras.preset.resolution == (1280, 720)
    assert set(cameras.cameras.keys()) == {"cam_front", "cam_rear", "cam_belt", "cam_dump"}
    for name, spec in cameras.cameras.items():
        assert spec.placeholder is True
        assert spec.tag in ("given", "assumed")


def test_all_assumptions_listed(cfg_path, motors_path, battery_path, cameras_path):
    robot = load_robot_params(cfg_path)
    motors = load_motors(motors_path)
    battery = load_battery(battery_path)
    cameras = load_cameras(cameras_path)
    all_assumptions = (
        robot.assumptions + motors.assumptions + battery.assumptions + cameras.assumptions
    )
    # At least the wheelbase conflict must be flagged
    assert any("wheelbase" in a.lower() or "CONFLICT" in a.upper() for a in all_assumptions)
    assert len(all_assumptions) > 0


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    import os
    import shutil

    src = os.path.join(os.path.dirname(__file__), "..", "config", "robot_params.yaml")
    dst = tmp_path / "robot_params.yaml"
    shutil.copy(src, dst)
    return str(dst)


@pytest.fixture
def motors_path(tmp_path):
    import os
    import shutil

    src = os.path.join(os.path.dirname(__file__), "..", "config", "motors.yaml")
    dst = tmp_path / "motors.yaml"
    shutil.copy(src, dst)
    return str(dst)


@pytest.fixture
def battery_path(tmp_path):
    import os
    import shutil

    src = os.path.join(os.path.dirname(__file__), "..", "config", "battery.yaml")
    dst = tmp_path / "battery.yaml"
    shutil.copy(src, dst)
    return str(dst)


@pytest.fixture
def cameras_path(tmp_path):
    import os
    import shutil

    src = os.path.join(os.path.dirname(__file__), "..", "config", "cameras.yaml")
    dst = tmp_path / "cameras.yaml"
    shutil.copy(src, dst)
    return str(dst)
