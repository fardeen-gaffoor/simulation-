"""Tests for the DC motor model (shared with sandsim)."""

from __future__ import annotations

import pytest

from rover_sim_tools.motor_model import DcMotorModel, KG_CM_TO_NM


@pytest.fixture
def motor() -> DcMotorModel:
    return DcMotorModel()


def test_ke_equals_kt(motor: DcMotorModel) -> None:
    """SI units: Kt [N.m/A] == Ke [V/(rad/s)]."""
    assert motor.kt == motor.ke


def test_back_emf_linear(motor: DcMotorModel) -> None:
    e1 = motor.back_emf_v(100.0)
    e2 = motor.back_emf_v(200.0)
    assert e2 == pytest.approx(2.0 * e1)


def test_current_positive_when_voltage_above_back_emf(motor: DcMotorModel) -> None:
    i = motor.motor_current_a(12.0, 0.0)  # stalled motor, 12 V applied
    assert i > 0.0
    assert i == pytest.approx(12.0 / motor.r_winding_ohm)


def test_current_zero_when_back_emf_equals_applied(motor: DcMotorModel) -> None:
    """At no-load speed, back-EMF ~ nominal voltage, current ~ no-load current."""
    omega_nl = 1885.0  # motor rad/s at 60 rpm output
    i = motor.motor_current_a(12.0, omega_nl)
    assert i == pytest.approx(0.8, abs=0.2)


def test_torque_from_current(motor: DcMotorModel) -> None:
    i = 7.5  # A
    tau_m = motor.motor_torque_nm(i)
    tau_out = motor.output_torque_nm(i)
    assert tau_m == pytest.approx(motor.kt * i)
    assert tau_out == pytest.approx(tau_m * motor.gear_ratio * motor.eta_gb)


def test_torque_status_ok(motor: DcMotorModel) -> None:
    status, kgcm = motor.torque_status(3.923)  # 40 kg.cm rated
    assert status == "OK"

    status, kgcm = motor.torque_status(7.355)  # 75 kg.cm advised max
    assert status == "WARNING"

    status, kgcm = motor.torque_status(9.807)  # 100 kg.cm gearbox break
    assert status == "FAULT"


def test_stall_current(motor: DcMotorModel) -> None:
    """At stall (omega=0), current = V / R."""
    i_stall = motor.motor_current_a(12.0, 0.0)
    assert i_stall == pytest.approx(20.0, abs=0.5)


def test_kgcm_conversion() -> None:
    assert KG_CM_TO_NM == pytest.approx(0.0980665)
    assert 40.0 * KG_CM_TO_NM == pytest.approx(3.92266, abs=1e-4)


def test_max_torques(motor: DcMotorModel) -> None:
    assert motor.max_advised_output_torque_nm() == pytest.approx(7.355, abs=1e-3)
    assert motor.max_safe_output_torque_nm() == pytest.approx(9.807, abs=1e-3)
