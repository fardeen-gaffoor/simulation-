"""DC motor model shared by ``rover_sim_tools`` and ``sandsim``.

Implements a brushed DC motor with gearbox:

    V_applied = duty * V_terminal                      (PWM, after battery sag)
    E_back    = Ke * omega_motor                       (back-EMF)
    I         = (V_applied - E_back) / R_winding       (winding current)
    tau_motor = Kt * I                                 (motor torque)
    tau_out   = tau_motor * gear_ratio * eta_gb        (output shaft torque)
    omega_out = omega_motor / gear_ratio               (output shaft speed)

SI units: Kt [N.m/A] == Ke [V/(rad/s)].

Parameters are fitted to the Robokites Rhino 12 V 60 RPM listing:

* R_winding = 0.60 ohm  (from assumed 20 A stall current at 12 V)
* Ke = Kt = 0.0061  (from no-load 60 rpm output = 1885 rad/s motor at 12 V)
* gear_ratio = 300  (18000 rpm base motor / 60 rpm output)
* eta_gb = 0.85     (assumed gearbox efficiency)
"""

from __future__ import annotations

import math

KG_CM_TO_NM = 0.0980665  # 1 kg.cm = 0.0980665 N.m (given)


class DcMotorModel:
    """Brushed DC motor with gearbox, suitable for rover wheel drive.

    The model computes steady-state electrical and mechanical quantities for a
    given output-shaft speed and load torque.  It is intentionally simple (no
    rotor inertia dynamics) so it can run in real time inside a ROS 2 node and
    be reused by sandsim.
    """

    def __init__(
        self,
        voltage_nominal_v: float = 12.0,
        r_winding_ohm: float = 0.60,
        ke_v_per_rads: float = 0.00611,
        gear_ratio: float = 300.0,
        eta_gb: float = 0.85,
        rated_torque_kgcm: float = 40.0,
        advised_max_torque_kgcm: float = 75.0,
        gearbox_break_torque_kgcm: float = 100.0,
        no_load_current_a: float = 0.8,
        stall_current_a: float = 20.0,
    ) -> None:
        self.voltage_nominal_v = voltage_nominal_v
        self.r_winding_ohm = r_winding_ohm
        self.ke = ke_v_per_rads
        self.kt = ke_v_per_rads  # SI: Kt [N.m/A] == Ke [V/(rad/s)]
        self.gear_ratio = gear_ratio
        self.eta_gb = eta_gb
        self.rated_torque_nm = rated_torque_kgcm * KG_CM_TO_NM
        self.advised_max_torque_nm = advised_max_torque_kgcm * KG_CM_TO_NM
        self.gearbox_break_torque_nm = gearbox_break_torque_kgcm * KG_CM_TO_NM
        self.no_load_current_a = no_load_current_a
        self.stall_current_a = stall_current_a

    # ------------------------------------------------------------------ #
    # electrical                                                         #
    # ------------------------------------------------------------------ #
    def back_emf_v(self, omega_motor_rads: float) -> float:
        """Back-EMF voltage at motor shaft speed."""
        return self.ke * omega_motor_rads

    def motor_current_a(
        self, v_applied: float, omega_motor_rads: float
    ) -> float:
        """Winding current for applied voltage and motor speed."""
        e_back = self.back_emf_v(omega_motor_rads)
        v_over_r = v_applied - e_back
        if v_over_r <= 0.0:
            return 0.0  # motor is generating, not motoring
        return v_over_r / self.r_winding_ohm

    def required_voltage_v(
        self, current_a: float, omega_motor_rads: float
    ) -> float:
        """Terminal voltage needed to drive ``current_a`` at ``omega_motor``."""
        return self.back_emf_v(omega_motor_rads) + current_a * self.r_winding_ohm

    # ------------------------------------------------------------------ #
    # mechanical                                                         #
    # ------------------------------------------------------------------ #
    def motor_torque_nm(self, current_a: float) -> float:
        """Motor shaft torque from current."""
        return self.kt * current_a

    def output_torque_nm(self, current_a: float) -> float:
        """Output shaft torque after gearbox."""
        return self.motor_torque_nm(current_a) * self.gear_ratio * self.eta_gb

    def current_for_output_torque_nm(self, torque_out_nm: float) -> float:
        """Current required to produce ``torque_out_nm`` at the output shaft."""
        tau_motor = torque_out_nm / (self.gear_ratio * self.eta_gb)
        return tau_motor / self.kt

    def output_speed_rads(self, omega_motor_rads: float) -> float:
        """Output shaft speed from motor shaft speed."""
        return omega_motor_rads / self.gear_ratio

    def motor_speed_rads(self, omega_out_rads: float) -> float:
        """Motor shaft speed from output shaft speed."""
        return omega_out_rads * self.gear_ratio

    # ------------------------------------------------------------------ #
    # limits / faults                                                    #
    # ------------------------------------------------------------------ #
    def torque_status(
        self, torque_out_nm: float
    ) -> tuple[str, float]:
        """Return (status, torque_kgcm) for output torque.

        status is one of: ``OK``, ``WARNING`` (above advised max), ``FAULT``
        (above gearbox break torque).
        """
        torque_kgcm = torque_out_nm / KG_CM_TO_NM
        if torque_out_nm > self.gearbox_break_torque_nm:
            return "FAULT", torque_kgcm
        if torque_out_nm > self.advised_max_torque_nm:
            return "WARNING", torque_kgcm
        return "OK", torque_kgcm

    def max_safe_output_torque_nm(self) -> float:
        """Maximum output torque before gearbox break."""
        return self.gearbox_break_torque_nm

    def max_advised_output_torque_nm(self) -> float:
        """Maximum advised output torque."""
        return self.advised_max_torque_nm
