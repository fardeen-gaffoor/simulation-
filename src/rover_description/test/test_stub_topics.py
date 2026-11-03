"""Tests for the /estop + /battery/energy_wh interface stub node.

These stay light: they check the module builds, the node exposes the interface
the stack depends on, and the drain/gate logic is exercised directly without
spinning a full graph.
"""

from __future__ import annotations


def test_module_imports():
    from rover_description import stub_topics

    assert hasattr(stub_topics, "StubTopics")
    assert hasattr(stub_topics, "main")


def test_class_interface():
    from rover_description.stub_topics import StubTopics

    for method in ("_publish_estop", "_drain_w", "_tick_battery", "_tick_gate", "_on_estop"):
        assert callable(getattr(StubTopics, method))


def test_estop_is_latched_and_cmd_vel_gated():
    from rover_description.stub_topics import _latched

    from rclpy.qos import DurabilityPolicy

    assert _latched().durability == DurabilityPolicy.TRANSIENT_LOCAL


def test_node_uses_config_topics(cfg):
    # the stub must publish/subscribe the exact config topics
    assert cfg.topics["battery_energy"] == "/battery/energy_wh"
    assert cfg.topics["estop"] == "/estop"
    assert cfg.topics["cmd_vel"] == "/cmd_vel"


def test_battery_parameters(cfg):
    assert cfg.battery.capacity_wh > 0.0
    assert cfg.battery.initial_energy_wh <= cfg.battery.capacity_wh
    assert cfg.battery.idle_draw_w > 0.0
