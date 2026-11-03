"""gz backend tests: pose parsing and teleport argument construction."""

from __future__ import annotations

import math

import pytest

from rover_dataset.capture import (
    FakeGzBackend,
    GzBackend,
    GzCliBackend,
    GzError,
    parse_model_pose,
)

# captured verbatim from `gz model -m rover -p` on gz-Harmonic
_REAL_OUTPUT = """Requesting state for world [cat_arena]...

Model: [78]
  - Name: rover
  - Pose [ XYZ (m) ] [ RPY (rad) ]:
    [3.000000 2.500000 0.100000]
    [0.000000 0.000000 1.570800]
"""


def test_parses_the_real_gz_output():
    pose = parse_model_pose(_REAL_OUTPUT)
    assert pose.x == pytest.approx(3.0)
    assert pose.y == pytest.approx(2.5)
    assert pose.z == pytest.approx(0.1)
    assert pose.yaw == pytest.approx(1.5708, abs=1e-4)
    assert pose.roll == pytest.approx(0.0)
    assert pose.pitch == pytest.approx(0.0)


def test_parses_negative_and_scientific_values():
    text = "blah\n[1e-1 -2.5 3]\n[0 0.5 -0.25]\n"
    pose = parse_model_pose(text)
    assert pose.x == pytest.approx(0.1)
    assert pose.yaw == pytest.approx(-0.25)


def test_rejects_unparseable_output():
    with pytest.raises(GzError, match="could not parse"):
        parse_model_pose("no pose here")


def test_entity_pose_converts_to_a_rover_pose():
    pose = parse_model_pose(_REAL_OUTPUT).as_rover_pose()
    assert pose.as_tuple() == pytest.approx((3.0, 2.5, 0.1, 0.0, 0.0, 1.5708), abs=1e-4)


def test_set_pose_request_encodes_the_yaw_as_a_quaternion(monkeypatch):
    backend = GzCliBackend(world="cat_arena")
    captured = {}

    def fake_run(args):
        captured["args"] = list(args)
        return 0, "data: true"

    monkeypatch.setattr(backend, "_run", fake_run)
    assert backend.set_pose("rover", 1.0, 2.0, 0.1, math.pi / 2)
    args = captured["args"]
    assert "/world/cat_arena/set_pose" in args
    assert "gz.msgs.Pose" in args
    request = args[args.index("--req") + 1]
    assert 'name: "rover"' in request
    # 90 deg -> z = sin(45 deg) = qw = cos(45 deg)
    assert "z: 0.707106781" in request
    assert "w: 0.707106781" in request


def test_set_pose_reports_failure(monkeypatch):
    backend = GzCliBackend()
    monkeypatch.setattr(backend, "_run", lambda args: (0, "data: false"))
    assert backend.set_pose("rover", 0.0, 0.0, 0.0, 0.0) is False


def test_fake_backend_records_commands_and_settles():
    backend = FakeGzBackend(settle_offset=0.1)
    assert backend.set_pose("rover", 1.0, 2.0, 0.12, 0.5)
    assert backend.commanded == [("rover", 1.0, 2.0, 0.12, 0.5)]
    pose = backend.get_pose("rover")
    assert pose.z == pytest.approx(0.02)


def test_fake_backend_can_fail_on_demand():
    backend = FakeGzBackend(error_every=2)
    assert backend.set_pose("rover", 0.0, 0.0, 0.0, 0.0) is True
    assert backend.set_pose("rover", 0.0, 0.0, 0.0, 0.0) is False


def test_fake_backend_raises_for_unknown_entity():
    with pytest.raises(GzError):
        FakeGzBackend().get_pose("nope")


def test_backends_share_the_interface():
    assert issubclass(GzCliBackend, GzBackend)
    assert issubclass(FakeGzBackend, GzBackend)
