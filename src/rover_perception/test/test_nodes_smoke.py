"""Every node: it imports, it parses its flags, and it can be constructed.

The nodes are thin ROS wrappers around the tested libraries, so this file does
not re-test the algorithms.  What it *does* pin is the glue that only shows up
when the whole thing is started: that each module exposes ``NODE_NAME`` /
``add_arguments`` / ``main``, that the constructor accepts ``--config`` and
therefore honours an override rather than silently using the default, and that a
node builds its publishers, subscriptions and timers without a live DDS peer.
"""

from __future__ import annotations

import argparse
import importlib
import sys

import pytest

from rover_perception.nodes import (
    camera_manager_node,
    health_monitor_node,
    obstacle_detector_node,
    tag_detector_node,
    tag_localizer_node,
)

_MODULES = {
    "camera_manager": camera_manager_node,
    "health_monitor": health_monitor_node,
    "obstacle_detector": obstacle_detector_node,
    "tag_detector": tag_detector_node,
    "tag_localizer": tag_localizer_node,
}


# --------------------------------------------------------------------------- #
# module shape
# --------------------------------------------------------------------------- #
def test_every_module_declares_its_node_name():
    for label, module in _MODULES.items():
        assert isinstance(module.NODE_NAME, str) and module.NODE_NAME
        assert module.NODE_NAME.startswith("perception_"), label


def test_every_module_exports_the_entry_points():
    for label, module in _MODULES.items():
        assert callable(module.main), label
        assert callable(module.add_arguments), label
        node_class = next(
            getattr(module, name)
            for name in module.__all__
            if name.endswith("Node")
        )
        assert isinstance(node_class, type), label


def test_the_node_classes_are_importable_by_name():
    for module_name, class_name in (
        ("rover_perception.nodes.camera_manager_node", "CameraManagerNode"),
        ("rover_perception.nodes.health_monitor_node", "HealthMonitorNode"),
        ("rover_perception.nodes.obstacle_detector_node", "ObstacleDetectorNode"),
        ("rover_perception.nodes.tag_detector_node", "TagDetectorNode"),
        ("rover_perception.nodes.tag_localizer_node", "TagLocalizerNode"),
    ):
        module = importlib.import_module(module_name)
        assert hasattr(module, class_name)


# --------------------------------------------------------------------------- #
# flags
# --------------------------------------------------------------------------- #
def _flags(module):
    parser = argparse.ArgumentParser(prog="node")
    parser.add_argument("--config", default="")
    module.add_arguments(parser)
    return parser


def test_the_shared_config_flag_exists_with_the_default():
    for label, module in _MODULES.items():
        args = _flags(module).parse_args([])
        assert args.config == "", label


def test_publish_hz_override_flags_parse():
    for label, module in (("camera_manager", camera_manager_node), ("health_monitor", health_monitor_node)):
        args = _flags(module).parse_args(["--publish-hz", "7.5"])
        assert args.publish_hz == pytest.approx(7.5), label


def test_tag_detector_flags():
    args = _flags(tag_detector_node).parse_args(["--fallback-tag-size", "0.2", "--tags-file", "/tmp/t.yaml"])
    assert args.fallback_tag_size == pytest.approx(0.2)
    assert args.tags_file == "/tmp/t.yaml"


def test_tag_localizer_flags():
    args = _flags(tag_localizer_node).parse_args(["--publish-tf"])
    assert args.publish_tf is True
    args = _flags(tag_localizer_node).parse_args([])
    assert args.publish_tf is False, "the sim default must not publish map -> odom twice"


def test_obstacle_detector_debug_flag():
    assert _flags(obstacle_detector_node).parse_args(["--debug-image"]).debug_image is True


# --------------------------------------------------------------------------- #
# the argument split (what a launch file actually exercises)
# --------------------------------------------------------------------------- #
def test_ros_args_are_split_from_own_args():
    from rover_perception import nodeutil

    ros, own = nodeutil.split_node_arguments(
        ["--config", "/x.yaml", "--ros-args", "-r", "__node:=n", "--params-file", "/p.yaml"]
    )
    assert own == ["--config", "/x.yaml"]
    assert ros == ["--ros-args", "-r", "__node:=n", "--params-file", "/p.yaml"]

    ros, own = nodeutil.split_node_arguments(["--publish-hz", "2.0"])
    assert ros is None
    assert own == ["--publish-hz", "2.0"]


def test_sys_argv_carrying_ros_args_is_split_too(monkeypatch):
    """A launch file passes ``--ros-args ...`` in ``sys.argv``; that path used to leak."""
    from rover_perception import nodeutil

    monkeypatch.setattr(
        sys,
        "argv",
        ["camera_manager.py", "--ros-args", "-r", "__node:=perception_camera_manager"],
    )
    ros, own = nodeutil.split_node_arguments(None)
    assert own == []
    assert ros == ["--ros-args", "-r", "__node:=perception_camera_manager"]


def test_no_ros_args_leaves_own_args_alone(monkeypatch):
    from rover_perception import nodeutil

    monkeypatch.setattr(sys, "argv", ["tag_detector.py", "--fallback-tag-size", "0.2"])
    ros, own = nodeutil.split_node_arguments(None)
    assert ros is None
    assert own == ["--fallback-tag-size", "0.2"]


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
@pytest.fixture()
def built(rclpy_session):
    """Construct a node, then always destroy it so the DDS graph stays clean."""
    made = []

    def _build(factory):
        node = factory()
        made.append(node)
        return node

    yield _build
    for node in made:
        node.destroy_node()


def test_camera_manager_node_builds(built, cfg):
    node = built(lambda: camera_manager_node.CameraManagerNode(config_override=cfg.path))
    assert node.config is not None
    assert node.config.path == cfg.path, "the --config override is honoured"
    assert node.publish_hz > 0.0
    snapshot = node.snapshot()
    assert snapshot["config"] == cfg.path
    assert len(snapshot["cameras"]) == len(cfg.cameras)
    assert set(node.controller.camera_names) == set(cfg.camera_names)


def test_health_monitor_node_builds(built, cfg):
    node = built(lambda: health_monitor_node.HealthMonitorNode(config_override=cfg.path))
    snapshot = node.snapshot()
    assert snapshot["config"] == cfg.path
    assert "published" in snapshot


def test_obstacle_detector_node_builds(built, cfg):
    node = built(lambda: obstacle_detector_node.ObstacleDetectorNode(config_override=cfg.path))
    snapshot = node.snapshot()
    assert snapshot["config"] == cfg.path
    assert snapshot["backend"] == cfg.detector.backend
    assert snapshot["camera"] == cfg.detector.camera


def test_tag_detector_node_builds(built, cfg, catalogue):
    node = built(lambda: tag_detector_node.TagDetectorNode(config_override=cfg.path))
    snapshot = node.snapshot()
    assert snapshot["config"] == cfg.path
    assert snapshot["camera"] == cfg.tags.camera
    assert sorted(snapshot["catalogue_ids"]) == sorted(catalogue.tags)


def test_tag_localizer_node_builds(built, cfg, catalogue):
    # catalogue guarantees the tags.yaml the localizer needs exists on disk
    node = built(lambda: tag_localizer_node.TagLocalizerNode(config_override=cfg.path))
    snapshot = node.snapshot()
    assert snapshot["config"] == cfg.path
    assert snapshot["catalogue"]["count"] == len(catalogue.tags)
    assert snapshot["localized"] is False, "no fix has been seen yet"


def test_tag_localizer_refuses_when_localization_is_disabled(built, cfg):
    import dataclasses
    import os

    from support import write_config

    disabled = write_config(
        os.path.join(os.path.dirname(cfg.path), "disabled.yaml"),
        cfg.path,
        {"localization": {"enabled": False}},
    )
    with pytest.raises(ValueError):
        built(lambda: tag_localizer_node.TagLocalizerNode(config_override=disabled))


def test_a_bad_config_path_fails_loudly(built):
    from rover_perception.config import PerceptionConfigError

    with pytest.raises((PerceptionConfigError, FileNotFoundError, ValueError)):
        built(lambda: camera_manager_node.CameraManagerNode(config_override="/nonexistent/perception.yaml"))
