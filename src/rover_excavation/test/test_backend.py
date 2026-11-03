"""The IO layer: the three mechanism backends, the serial protocol, and the berm box."""

from __future__ import annotations

import pytest

from rover_excavation.backend import (
    GzBermVisual,
    HardwareIO,
    MechanismIOError,
    NullIO,
    SimIO,
    berm_sdf,
    build_io,
    crc8,
    decode_status,
    encode_command,
)
from rover_excavation.mechanism import MechanismCommand


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
def test_null_io_records_what_it_was_told(config):
    io = NullIO()
    io.start()
    io.send(MechanismCommand(0.1, 2.0, 0.5))
    assert io.sent == 1
    assert io.last.as_tuple() == (0.1, 2.0, 0.5)
    io.close()
    assert "null" in io.describe()


def test_sim_io_publishes_both_joint_targets(config):
    published = []
    io = SimIO(config.backend, lambda topic, value: published.append((topic, value)))
    io.start()
    io.send(MechanismCommand(0.4, -6.0, 1.0))
    assert published == [
        (config.backend.ladder_cmd_topic, 0.4),
        (config.backend.belt_cmd_topic, -6.0),
    ]
    assert io.sent == 1
    assert config.backend.ladder_cmd_topic in io.describe()


def test_build_io_sim_needs_a_publisher(config):
    with pytest.raises(MechanismIOError, match="publish"):
        build_io(config.backend)


def test_build_io_selects_the_backend(config):
    import dataclasses

    sim = build_io(config.backend, publish=lambda topic, value: None)
    assert isinstance(sim, SimIO)
    null_config = dataclasses.replace(config.backend, kind="null")
    assert isinstance(build_io(null_config), NullIO)
    hardware_config = dataclasses.replace(config.backend, kind="hardware")
    assert isinstance(build_io(hardware_config), HardwareIO)


def test_hardware_io_refuses_clearly_without_pyserial(config):
    import dataclasses

    hardware_config = dataclasses.replace(config.backend, kind="hardware")
    io = HardwareIO(hardware_config)
    try:
        import serial  # noqa: F401

        has_serial = True
    except Exception:
        has_serial = False
    if has_serial:
        pytest.skip("pyserial is installed; the missing-dependency path cannot be tested here")
    with pytest.raises(MechanismIOError, match="pyserial"):
        io.start()
    assert config.backend.serial.port in io.describe()


def test_send_before_start_is_an_error(config):
    import dataclasses

    io = HardwareIO(dataclasses.replace(config.backend, kind="hardware"))
    with pytest.raises(MechanismIOError, match="before start"):
        io.send(MechanismCommand(0.0, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# the serial protocol
# --------------------------------------------------------------------------- #
def test_crc8_known_vector():
    # CRC-8/ATM (poly 0x07, init 0x00) of "123456789" is 0xF4.
    assert crc8(b"123456789") == 0xF4
    assert crc8(b"") == 0x00


def test_encode_command_shape():
    frame = encode_command(MechanismCommand(0.6109, -6.0, 1.0))
    assert frame.startswith(b"$C,") and frame.endswith(b"\n")
    body = frame[1 : frame.index(b"*")]
    assert body.decode() == "C,0.6109,-6.0000,1.0000"
    assert frame[frame.index(b"*") + 1 : -1] == f"{crc8(body):02X}".encode()


def test_decode_status_round_trip():
    body = "S,0.5000,2.0000,1.0000,0"
    line = f"${body}*{crc8(body.encode('ascii')):02X}\n".encode("ascii")
    assert decode_status(line) == (0.5, 2.0, 1.0, 0)


def test_decode_status_rejects_a_bad_checksum():
    body = "S,0.5000,2.0000,1.0000,0"
    line = f"${body}*00\n".encode("ascii")
    assert decode_status(line) is None


def test_decode_status_rejects_junk():
    assert decode_status(b"") is None
    assert decode_status(b"garbage\n") is None
    assert decode_status(b"$C,1.0,2.0,3.0*FF\n") is None  # a command frame, not status
    assert decode_status(b"$S,1.0,2.0*FF\n") is None
    assert decode_status(b"$S,a,b,c,d*FF\n") is None


def test_decode_status_accepts_a_torn_line_without_raising():
    assert decode_status(b"$S,0.5") is None


# --------------------------------------------------------------------------- #
# the berm in Gazebo
# --------------------------------------------------------------------------- #
class _Recorder:
    def __init__(self, code: int = 0):
        self.calls = []
        self.code = code

    def __call__(self, command):
        self.calls.append(list(command))
        return self.code

    def services(self):
        out = []
        for call in self.calls:
            index = call.index("-s")
            out.append(call[index + 1])
        return out


def _visual(config, recorder, target_height=0.15):
    target = config.berm_target
    return GzBermVisual(
        config.visual,
        (target.x0, target.y0, target.x1, target.y1),
        target_height,
        run=recorder,
    )


def test_berm_sdf_is_a_static_box_without_collision():
    sdf = berm_sdf("berm", 1.25, 0.9, 0.15, visual_only=True)
    assert 'name="berm"' in sdf
    assert "<static>true</static>" in sdf
    assert "1.2500 0.9000 0.1500" in sdf
    assert "<collision" not in sdf
    with_collision = berm_sdf("berm", 1.25, 0.9, 0.15, visual_only=False)
    assert "<collision" in with_collision


def test_visual_spawns_once_then_only_moves(config):
    recorder = _Recorder()
    visual = _visual(config, recorder)
    assert visual.update(0.0) is True
    assert visual.spawned
    # The create and the first placement happen together, so the box starts out
    # buried rather than popping to the surface.
    assert recorder.services() == [
        f"/world/{config.visual.gz_world}/create",
        f"/world/{config.visual.gz_world}/set_pose",
    ]
    # A tiny change is below the step threshold: nothing is sent.
    assert visual.update(0.001) is False
    assert visual.update(0.05) is True
    assert recorder.services()[-1] == f"/world/{config.visual.gz_world}/set_pose"
    assert recorder.services().count(f"/world/{config.visual.gz_world}/create") == 1


def test_visual_raises_the_box_by_the_mean_height(config):
    recorder = _Recorder()
    visual = _visual(config, recorder)
    visual.update(0.0)
    recorder.calls.clear()
    visual.update(0.15)
    request = recorder.calls[-1][recorder.calls[-1].index("--req") + 1]
    assert "name: \"berm\"" in request
    # The box is 0.15 m tall, so its centre sits at 0.075 for a 0.15 m top.
    assert "z: 0.0750" in request


def test_visual_is_clamped_to_the_target_height(config):
    recorder = _Recorder()
    visual = _visual(config, recorder)
    visual.update(0.0)
    visual.update(10.0)
    assert visual.height_m == pytest.approx(0.15)


def test_visual_disables_itself_if_gz_is_missing(config):
    recorder = _Recorder(code=1)
    visual = _visual(config, recorder)
    assert visual.update(0.05) is False
    assert not visual.enabled
    assert "gz service create failed" in visual.note
    assert visual.update(0.10) is False
    assert len(recorder.calls) == 1


def test_visual_does_nothing_when_disabled(config):
    import dataclasses

    recorder = _Recorder()
    visual_config = dataclasses.replace(config.visual, gz_berm=False)
    target = config.berm_target
    visual = GzBermVisual(
        visual_config, (target.x0, target.y0, target.x1, target.y1), 0.15, run=recorder
    )
    assert visual.update(0.1) is False
    assert recorder.calls == []


def test_visual_close_removes_the_box_once(config):
    recorder = _Recorder()
    visual = _visual(config, recorder)
    visual.update(0.05)
    visual.close()
    assert recorder.services()[-1] == f"/world/{config.visual.gz_world}/remove"
    recorder.calls.clear()
    visual.close()
    assert recorder.calls == []


def test_visual_requests_are_well_formed(config):
    recorder = _Recorder()
    visual = _visual(config, recorder)
    visual.update(0.05)
    for call in recorder.calls:
        assert call[0] == "gz"
        assert call[1] == "service"
        assert "--reqtype" in call and "--reptype" in call and "--timeout" in call
        assert call[call.index("--timeout") + 1] == "2000"
