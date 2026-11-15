"""Integration test for the motor_battery_model node."""

from __future__ import annotations

import pytest

import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from rover_sim_tools.motor_battery_model import MotorBatteryModel


@pytest.fixture
def node() -> MotorBatteryModel:
    rclpy.init()
    node = MotorBatteryModel()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def test_node_starts(node: MotorBatteryModel) -> None:
    assert node is not None
    assert node.motor is not None
    assert node.battery is not None


def test_joint_states_callback(node: MotorBatteryModel) -> None:
    msg = JointState()
    msg.name = [
        "wheel_front_left",
        "wheel_front_right",
        "wheel_rear_left",
        "wheel_rear_right",
    ]
    msg.velocity = [2.0, 2.0, 2.0, 2.0]  # rad/s at each wheel
    node._joint_states_cb(msg)
    for name in ["wheel_front_left", "wheel_front_right",
                 "wheel_rear_left", "wheel_rear_right"]:
        assert node.wheel_velocities[name] == pytest.approx(2.0)


def test_publishers_exist(node: MotorBatteryModel) -> None:
    assert node.energy_pub is not None
    assert node.fault_pub is not None
    for name in ["wheel_front_left", "wheel_front_right",
                 "wheel_rear_left", "wheel_rear_right"]:
        assert name in node.current_pubs
        assert name in node.torque_pubs
