"""The ROS-free image codec: module 4 never touches ``cv_bridge``/``cv2``.

``cv2`` cannot import on this machine (this file's numpy shadows the apt OpenCV),
so ``imagecodec`` is the only way pixels move between a ``sensor_msgs/Image`` and
numpy.  A bug here is a bug in *every* node, so the encodings, the padding case
(a real stereo/camera ``step``) and the drawing helpers are all pinned down.
"""

from __future__ import annotations

import numpy as np
import pytest

from rover_perception.imagecodec import (
    ENCODINGS,
    ImageCodecError,
    array_to_bytes,
    build_image,
    draw_cross,
    draw_filled_rect,
    draw_rect,
    encoding_info,
    payload_bytes,
    to_array,
    to_gray,
    to_rgb,
)


class FakeImage:
    """The attribute surface ``sensor_msgs/Image`` presents to the codec."""

    def __init__(self, height, width, encoding, data, step=0):
        self.height = height
        self.width = width
        self.encoding = encoding
        self.data = data
        self.step = step or (width * ENCODINGS[encoding][1] * np.dtype(ENCODINGS[encoding][0]).itemsize)


def _rgb_image(array):
    return FakeImage(array.shape[0], array.shape[1], "rgb8", array.tobytes())


def test_encodings_are_known():
    assert ENCODINGS["rgb8"] == ("uint8", 3)
    assert ENCODINGS["mono8"] == ("uint8", 1)
    assert encoding_info("mono8") == ("uint8", 1)
    with pytest.raises(ImageCodecError) as excinfo:
        encoding_info("yuv420")
    assert "yuv420" in str(excinfo.value)


def test_payload_bytes():
    assert payload_bytes(640, 480, "rgb8") == 640 * 480 * 3
    assert payload_bytes(640, 480, "mono8") == 640 * 480
    assert payload_bytes(1280, 720, "rgba8") == 1280 * 720 * 4
    assert payload_bytes(64, 64, "16UC1") == 64 * 64 * 2


def test_rgb_round_trip():
    array = (np.arange(4 * 3 * 3, dtype=np.uint8).reshape(4, 3, 3))
    decoded = to_array(_rgb_image(array))
    assert decoded.shape == (4, 3, 3)
    assert np.array_equal(decoded, array)
    assert np.array_equal(to_rgb(_rgb_image(array)), array)


def test_mono_round_trip_and_grey_passthrough():
    array = np.arange(6 * 5, dtype=np.uint8).reshape(6, 5)
    image = FakeImage(6, 5, "mono8", array.tobytes())
    assert to_array(image).shape == (6, 5)
    assert np.array_equal(to_gray(image), array)


def test_grey_uses_bt601_weights():
    pixel = np.array([[[200, 100, 50]]], dtype=np.uint8)
    expected = int(np.clip(0.299 * 200 + 0.587 * 100 + 0.114 * 50 + 0.5, 0, 255))
    assert int(to_gray(_rgb_image(pixel))[0, 0]) == expected


def test_bgr_is_reordered_not_reinterpreted():
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    bgr = FakeImage(1, 1, "bgr8", rgb[:, :, ::-1].tobytes())
    assert np.array_equal(to_rgb(bgr), rgb)


def test_rgba_drops_alpha():
    rgba = np.array([[[1, 2, 3, 255]]], dtype=np.uint8)
    assert np.array_equal(to_rgb(FakeImage(1, 1, "rgba8", rgba.tobytes())), rgba[:, :, :3])


def test_padded_rows_are_honoured():
    """A camera may pad each row; the codec must not shear the image."""
    width, height, step = 3, 2, 16
    raw = np.zeros((height, step), dtype=np.uint8)
    raw[0, :9] = np.arange(1, 10)
    raw[1, :9] = np.arange(11, 20)
    image = FakeImage(height, width, "rgb8", raw.tobytes(), step=step)
    decoded = to_array(image)
    assert decoded.shape == (height, width, 3)
    assert np.array_equal(decoded[0].reshape(-1), np.arange(1, 10))
    assert np.array_equal(decoded[1].reshape(-1), np.arange(11, 20))


def test_short_payload_is_rejected():
    image = FakeImage(4, 3, "rgb8", b"\x00" * 5, step=9)
    with pytest.raises(ImageCodecError) as excinfo:
        to_array(image)
    assert "smaller than" in str(excinfo.value)


def test_degenerate_size_is_rejected():
    image = FakeImage(0, 3, "rgb8", b"", step=9)
    with pytest.raises(ImageCodecError):
        to_array(image)


def test_step_smaller_than_row_is_rejected():
    image = FakeImage(2, 3, "rgb8", b"\x00" * 18, step=2)
    with pytest.raises(ImageCodecError) as excinfo:
        to_array(image)
    assert "step" in str(excinfo.value)


def test_array_to_bytes_shape_is_checked():
    matrix = np.zeros((2, 2, 3), dtype=np.uint8)
    assert len(array_to_bytes(matrix, "rgb8")) == 12
    with pytest.raises(ImageCodecError):
        array_to_bytes(np.zeros((2, 2), dtype=np.uint8), "rgb8")


def test_build_image_sets_every_field():
    class StubHeader:
        frame_id = ""
        stamp = None

    class StubImage:
        def __init__(self):
            self.header = StubHeader()
            self.height = 0
            self.width = 0
            self.encoding = ""
            self.is_bigendian = True
            self.step = 0
            self.data = b""

    array = np.arange(4 * 2 * 3, dtype=np.uint8).reshape(4, 2, 3)
    message = build_image(array, "rgb8", "cam_nav_optical_frame", "STAMP", StubImage)
    assert message.header.frame_id == "cam_nav_optical_frame"
    assert message.header.stamp == "STAMP"
    assert (message.height, message.width) == (4, 2)
    assert message.encoding == "rgb8"
    assert message.is_bigendian is False
    assert message.step == 2 * 3
    assert np.array_equal(to_array(message), array)


def test_uint16_scales_into_range():
    array = (np.array([[0, 65535]], dtype=np.uint16))
    grey = to_gray(FakeImage(1, 2, "mono16", array.tobytes()))
    assert grey[0, 0] == 0
    assert grey[0, 1] == 255


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def test_draw_rect_draws_a_box_and_clips():
    canvas = np.zeros((10, 10, 3), dtype=np.uint8)
    draw_rect(canvas, 2, 3, 7, 8, (255, 0, 0), thickness=1)
    assert np.all(canvas[3, 2:8] == (255, 0, 0))
    assert np.all(canvas[8, 2:8] == (255, 0, 0))
    assert np.all(canvas[3:9, 2] == (255, 0, 0))
    assert np.all(canvas[3:9, 7] == (255, 0, 0))
    assert np.all(canvas[5, 5] == 0), "the box must be hollow"


def test_draw_rect_outside_the_canvas_is_clipped():
    canvas = np.zeros((8, 8, 3), dtype=np.uint8)
    draw_rect(canvas, -5, -5, 100, 100, (0, 255, 0), thickness=2)
    assert np.all(canvas[0, :] == (0, 255, 0))
    assert np.all(canvas[:, 0] == (0, 255, 0))


def test_draw_filled_rect():
    canvas = np.zeros((6, 6, 3), dtype=np.uint8)
    draw_filled_rect(canvas, 1, 1, 3, 3, (0, 0, 255))
    assert np.all(canvas[1:4, 1:4] == (0, 0, 255))
    assert np.all(canvas[0, :] == 0)


def test_draw_cross_is_centred():
    canvas = np.zeros((9, 9, 3), dtype=np.uint8)
    draw_cross(canvas, 4, 4, (255, 255, 255), size=2)
    assert np.all(canvas[4, 2:7] == 255)
    assert np.all(canvas[2:7, 4] == 255)
    assert np.all(canvas[0, 0] == 0)


def test_draw_helpers_accept_a_grey_canvas():
    canvas = np.zeros((5, 5), dtype=np.uint8)
    draw_cross(canvas, 2, 2, 255, size=1)
    assert canvas[2, 2] == 255
