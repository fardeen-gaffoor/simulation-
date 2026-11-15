#!/usr/bin/env python3
"""``ros2 run rover_sim_tools motor_battery_model.py``

Stage 3 of PROMPT_ROVER_SIM.md: DC motor + battery electrical model.

Subscribes to /joint_states (wheel velocities) and computes, per wheel:

* motor current (A) and output torque (N.m) from the DC motor model
* battery energy (Wh) from cumulative energy integration
* hardware faults (gearbox break torque exceeded, advised max exceeded)

Publishes:

* /battery/energy_wh (std_msgs/Float64) - remaining battery energy
* /wheel/{name}/current (std_msgs/Float64) - per-motor current
* /wheel/{name}/torque (std_msgs/Float64) - per-motor output torque
* /hw/fault (std_msgs/String) - fault/warning messages

The motor model (``rover_sim_tools.motor_model.DcMotorModel``) and battery
model (``rover_sim_tools.battery_model.BatteryModel``) are shared with
sandsim and parameterised entirely from YAML config files.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String

from rover_sim_tools.motor_model import DcMotorModel, KG_CM_TO_NM
from rover_sim_tools.battery_model import BatteryModel
from rover_description.config import load_config as load_rover_config


WHEEL_NAMES = (
    "wheel_front_left",
    "wheel_front_right",
    "wheel_rear_left",
    "wheel_rear_right",
)

QOS_SENSOR = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)


class MotorBatteryModel(Node):
    """ROS 2 node implementing the motor + battery electrical model."""

    def __init__(self) -> None:
        super().__init__("motor_battery_model")

        # ------------------------------------------------------------------ #
        # parameters                                                       #
        # ------------------------------------------------------------------ #
        self.declare_parameter("rover_config", "")
        self.declare_parameter("motors_config", "")
        self.declare_parameter("battery_config", "")
        self.declare_parameter("soft_start_tau_s", 0.5)
        self.declare_parameter("pwm_duty_cap", 1.0)
        self.declare_parameter("rolling_resistance_rr", 0.25)
        self.declare_parameter("publish_rate_hz", 50.0)

        rover_config = self.get_parameter("rover_config").value
        motors_config = self.get_parameter("motors_config").value
        battery_config = self.get_parameter("battery_config").value
        self.soft_start_tau = float(self.get_parameter("soft_start_tau_s").value)
        self.pwm_duty_cap = float(self.get_parameter("pwm_duty_cap").value)
        self.rr = float(self.get_parameter("rolling_resistance_rr").value)

        # ------------------------------------------------------------------ #
        # config loading                                                   #
        # ------------------------------------------------------------------ #
        from rover_description.params import (
            load_robot_params,
            load_motors,
            load_battery,
        )

        self.robot_cfg = load_rover_config(rover_config or None)
        self.motors_cfg = load_motors(motors_config or None)
        self.battery_cfg = load_battery(battery_config or None)

        # Motor model
        d = self.robot_cfg.drive
        m = self.motors_cfg
        self.motor = DcMotorModel(
            voltage_nominal_v=float(self.battery_cfg.electrical.voltage_nominal_v.value),
            r_winding_ohm=0.60,  # from stall current assumption
            ke_v_per_rads=0.00611,  # from no-load speed
            gear_ratio=300.0,
            eta_gb=0.85,
            rated_torque_kgcm=float(m.torque.rated_torque_kgcm.value),
            advised_max_torque_kgcm=float(m.torque.advised_max_torque_kgcm.value),
            gearbox_break_torque_kgcm=float(
                m.torque.gearbox_breaking_torque_kgcm.value
            ),
            no_load_current_a=float(m.electrical.no_load_current_a.value),
            stall_current_a=float(m.electrical.stall_current_a.value),
        )

        # Battery model
        self.battery = BatteryModel(
            voltage_nominal_v=float(
                self.battery_cfg.electrical.voltage_nominal_v.value
            ),
            capacity_wh=float(self.battery_cfg.electrical.capacity_wh.value),
            voltage_cutoff_v=float(
                self.battery_cfg.electrical.voltage_cutoff_v.value
            ),
            r_internal_ohm=float(
                self.battery_cfg.electrical.internal_resistance_ohm.value
            ),
        )

        # ------------------------------------------------------------------ #
        # state                                                            #
        # ------------------------------------------------------------------ #
        self.wheel_velocities: dict[str, float] = {name: 0.0 for name in WHEEL_NAMES}
        self.wheel_accelerations: dict[str, float] = {
            name: 0.0 for name in WHEEL_NAMES
        }
        self.last_wheel_velocities: dict[str, float] = {
            name: 0.0 for name in WHEEL_NAMES
        }
        self.last_time: Optional[float] = None

        # Soft-start: target duty cycle per wheel, ramped with time constant
        self.target_duty: dict[str, float] = {name: 0.0 for name in WHEEL_NAMES}
        self.active_duty: dict[str, float] = {name: 0.0 for name in WHEEL_NAMES}

        # Fault state
        self.fault_active = False
        self.fault_message = ""

        # Mass for rolling resistance computation
        self.mass_kg = 50.0  # design mass from Stage 1
        try:
            self.mass_kg = float(self.robot_cfg.mass.total_max_kg.value or 50.0)
        except Exception:
            pass

        self.get_logger().info(
            f"motor_battery_model: mass={self.mass_kg} kg, "
            f"rr={self.rr}, duty_cap={self.pwm_duty_cap}, "
            f"soft_start_tau={self.soft_start_tau}s"
        )

        # ------------------------------------------------------------------ #
        # publishers / subscribers                                           #
        # ------------------------------------------------------------------ #
        self.js_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_states_cb, QOS_SENSOR
        )

        self.energy_pub = self.create_publisher(
            Float64, "/battery/energy_wh", 10
        )
        self.fault_pub = self.create_publisher(String, "/hw/fault", 10)

        self.current_pubs: dict[str, rclpy.publisher.Publisher] = {}
        self.torque_pubs: dict[str, rclpy.publisher.Publisher] = {}
        for name in WHEEL_NAMES:
            self.current_pubs[name] = self.create_publisher(
                Float64, f"/wheel/{name}/current", 10
            )
            self.torque_pubs[name] = self.create_publisher(
                Float64, f"/wheel/{name}/torque", 10
            )

        # ------------------------------------------------------------------ #
        # timer                                                            #
        # ------------------------------------------------------------------ #
        rate = float(self.get_parameter("publish_rate_hz").value)
        period = 1.0 / rate if rate > 0 else 0.02
        self.timer = self.create_timer(period, self._timer_cb)

    # ------------------------------------------------------------------ #
    # callbacks                                                          #
    # ------------------------------------------------------------------ #
    def _joint_states_cb(self, msg: JointState) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_time if self.last_time is not None else 0.0
        self.last_time = now

        name_to_vel = dict(zip(msg.name, msg.velocity))
        for name in WHEEL_NAMES:
            if name in name_to_vel:
                new_vel = float(name_to_vel[name])
                if dt > 0.0:
                    self.wheel_accelerations[name] = (
                        new_vel - self.last_wheel_velocities[name]
                    ) / dt
                else:
                    self.wheel_accelerations[name] = 0.0
                self.last_wheel_velocities[name] = new_vel
                self.wheel_velocities[name] = new_vel

    # ------------------------------------------------------------------ #
    # computation                                                        #
    # ------------------------------------------------------------------ #
    def _compute_wheel_load_torque_nm(self, name: str) -> float:
        """Required torque at the wheel for rolling resistance + acceleration."""
        d = self.robot_cfg.drive
        wr = float(d.wheel_radius)
        mass = self.mass_kg

        # Rolling resistance torque per wheel (flat ground)
        tau_rr = mass * 9.81 * self.rr * wr / 4.0

        # Acceleration torque per wheel (force = m*a / 4, torque = F * wr)
        alpha = self.wheel_accelerations[name]
        lin_acc = alpha * wr
        tau_acc = mass * lin_acc * wr / 4.0

        return tau_rr + max(0.0, tau_acc)

    def _compute_wheel_electrical(self, name: str) -> tuple[float, float, str]:
        """Compute (current_a, torque_out_nm, status) for one wheel."""
        omega_out = self.wheel_velocities[name]
        tau_load = self._compute_wheel_load_torque_nm(name)

        # Motor speed from output speed
        omega_motor = self.motor.motor_speed_rads(omega_out)

        # Back-EMF at motor speed
        e_back = self.motor.back_emf_v(omega_motor)

        # Battery terminal voltage under total current (iterative: sag depends on current)
        # First pass: estimate current ignoring sag
        tau_motor_required = tau_load / (self.motor.gear_ratio * self.motor.eta_gb)
        i_required = tau_motor_required / self.motor.kt

        # Total current estimate for voltage sag
        total_i = sum(
            abs(
                self.motor.current_for_output_torque_nm(
                    self._compute_wheel_load_torque_nm(n)
                )
            )
            for n in WHEEL_NAMES
        )
        v_terminal = self.battery.terminal_voltage_v(total_i)

        # Available voltage after PWM duty cap
        v_applied = self.pwm_duty_cap * v_terminal

        # Required voltage to produce needed current
        v_required = e_back + i_required * self.motor.r_winding_ohm

        if v_required <= v_applied:
            # Can produce required torque
            i_actual = i_required
        else:
            # Voltage limited - compute actual current
            v_over_e = v_applied - e_back
            if v_over_e <= 0.0:
                i_actual = 0.0
            else:
                i_actual = v_over_e / self.motor.r_winding_ohm

        # Actual output torque
        tau_out = self.motor.output_torque_nm(i_actual)

        # Fault/warning check
        status, torque_kgcm = self.motor.torque_status(tau_out)

        return i_actual, tau_out, status

    def _ramp_duty(self, name: str, target: float, dt: float) -> float:
        """Soft-start: ramp active duty toward target with time constant."""
        current = self.active_duty[name]
        if abs(target - current) < 1e-6:
            return target
        step = (target - current) * (dt / self.soft_start_tau) if self.soft_start_tau > 0 else (target - current)
        return current + step

    # ------------------------------------------------------------------ #
    # timer callback                                                     #
    # ------------------------------------------------------------------ #
    def _timer_cb(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if self.last_time is None:
            self.last_time = now
            return
        dt = now - self.last_time
        self.last_time = now

        # Compute per-wheel electrical quantities
        total_current = 0.0
        fault_parts: list[str] = []
        warning_parts: list[str] = []

        for name in WHEEL_NAMES:
            i_a, tau_nm, status = self._compute_wheel_electrical(name)
            total_current += abs(i_a)

            self.current_pubs[name].publish(Float64(data=i_a))
            self.torque_pubs[name].publish(Float64(data=tau_nm))

            if status == "FAULT":
                fault_parts.append(f"{name}: {torque_kgcm:.1f} kg.cm > gearbox break")
            elif status == "WARNING":
                warning_parts.append(f"{name}: {torque_kgcm:.1f} kg.cm > advised max")

        # Battery energy integration
        self.battery.step(total_current, dt)

        # Fault logic
        new_fault = False
        if fault_parts:
            new_fault = True
            self.fault_message = "HARD FAULT: " + "; ".join(fault_parts)
        elif warning_parts:
            self.fault_message = "WARNING: " + "; ".join(warning_parts)
        else:
            self.fault_message = ""
        self.fault_active = new_fault

        # Battery depleted
        if self.battery.is_depleted(total_current):
            new_fault = True
            self.fault_message = (
                self.fault_message + "; " if self.fault_message else ""
            ) + "BATTERY DEPLETED"

        # Publish
        self.energy_pub.publish(Float64(data=self.battery.energy_wh))
        fault_msg = String()
        fault_msg.data = self.fault_message
        self.fault_pub.publish(fault_msg)


def main() -> None:
    rclpy.init()
    node = MotorBatteryModel()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
