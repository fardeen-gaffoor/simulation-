"""Tests for the rover.yaml loader and its competition-rule validation."""

from __future__ import annotations

import math

import pytest

from rover_description.config import (
    CAMERA_TOPICS,
    RoverConfigError,
    default_config_path,
    load_config,
)


def test_default_config_path_exists():
    assert default_config_path().endswith("rover.yaml")


def test_load_ok(cfg):
    assert cfg.schema_version == 1
    assert cfg.name
    assert cfg.declared_forward == "+x"


def test_competition_limits(cfg):
    assert cfg.envelope_limit.as_tuple() == pytest.approx((1.5, 0.75, 0.75))
    assert cfg.mass_limit == pytest.approx(80.0)


def test_mass_total_within_limit(cfg):
    assert cfg.total_mass <= cfg.mass_limit
    assert cfg.total_mass == pytest.approx(cfg.structural_mass + cfg.payload_mass)


def test_four_cameras_exact_topics(cfg):
    assert len(cfg.sensors.cameras) == 4
    assert sorted(c.topic for c in cfg.sensors.cameras) == sorted(CAMERA_TOPICS)
    assert [c.name for c in cfg.sensors.cameras] == [
        "cam_nav",
        "cam_tag",
        "cam_excavation",
        "cam_dump",
    ]


def test_camera_helpers(cfg):
    cam = cfg.camera("cam_nav")
    assert cam.camera_info_topic == "/cam_nav/camera_info"
    assert cam.optical_frame == "cam_nav_optical_frame"
    assert cam.rpy_rad[1] == pytest.approx(math.radians(18.0))
    # fx from a 100 deg HFOV at 640 px
    assert cam.fx == pytest.approx((320.0) / math.tan(math.radians(50.0)))


def test_cam_tag_height_within_rule(cfg):
    # competition: the tag camera sits at most 60 cm high
    assert cfg.camera("cam_tag").origin[2] <= 0.60


def test_imu_present_without_magnetometer_keys(cfg):
    assert cfg.sensors.imu.enabled
    assert cfg.sensors.imu.link == "imu_link"


def test_ladder_pitch_range(cfg):
    lad = cfg.excavation.ladder
    assert lad.min_pitch_deg < lad.stowed_pitch_deg < lad.max_pitch_deg


def test_assumptions_non_empty(cfg):
    assert cfg.assumptions_report()
    assert any("forward" in a.lower() for a in cfg.assumptions_report())


def test_config_sha256(cfg):
    digest = cfg.config_sha256()
    assert len(digest) == 64
    assert all(ch in "0123456789abcdef" for ch in digest)


def test_missing_section_raises(make_config):
    path = make_config(lambda d: d.pop("drive"))
    with pytest.raises(RoverConfigError):
        load_config(path)


def test_bad_camera_topic_raises(make_config):
    path = make_config(lambda d: d["sensors"]["cameras"][0].__setitem__("topic", "/cam_foo"))
    with pytest.raises(RoverConfigError):
        load_config(path)


def test_duplicate_camera_name_raises(make_config):
    path = make_config(lambda d: d["sensors"]["cameras"][1].__setitem__("name", "cam_nav"))
    with pytest.raises(RoverConfigError):
        load_config(path)


def test_overweight_raises(make_config):
    path = make_config(lambda d: d["payload"].append({"name": "lead", "mass": 40.0}))
    with pytest.raises(RoverConfigError):
        load_config(path)


def test_bad_ladder_pitch_range_raises(make_config):
    def mutate(d):
        d["excavation"]["ladder"]["min_pitch_deg"] = 30.0
        d["excavation"]["ladder"]["stowed_pitch_deg"] = -25.0
        d["excavation"]["ladder"]["max_pitch_deg"] = 55.0

    with pytest.raises(RoverConfigError):
        load_config(make_config(mutate))


def test_track_must_exceed_wheel_width(make_config):
    path = make_config(lambda d: d["drive"].__setitem__("track", d["drive"]["wheel_width"]))
    with pytest.raises(RoverConfigError):
        load_config(path)
