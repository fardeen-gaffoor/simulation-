"""Augmentation tests.

Two properties matter: photometric ops must never move a label, geometric ops
must move them exactly, and the whole thing must be reproducible from a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from rover_dataset.augment import OP_ORDER, augment_image, transform_boxes
from rover_dataset.config import AugmentCfg, RandomOp, Range
from rover_dataset.labels import YoloBox


def _op(prob=1.0, **params):
    return RandomOp("op", prob, {key: Range(*value) for key, value in params.items()})


def _cfg(ops, enabled=True) -> AugmentCfg:
    return AugmentCfg(enabled=enabled, seed_namespace="augment", ops=ops)


def _rng(seed=0):
    return np.random.default_rng(seed)


BOXES = [
    YoloBox.from_pixels(0, 4.0, 8.0, 20.0, 24.0, 64, 48, object_id="b0"),
    YoloBox.from_pixels(1, 30.0, 10.0, 60.0, 40.0, 64, 48, object_id="c0"),
]


# --------------------------------------------------------------------------- #
# shape / dtype / determinism
# --------------------------------------------------------------------------- #
def test_disabled_augmentation_is_a_no_op(image):
    outcome = augment_image(image, _cfg({}, enabled=False), _rng())
    assert np.array_equal(outcome.image, image)
    assert outcome.applied == []
    assert not outcome.geometry_changed


def test_output_is_uint8_rgb_of_the_same_shape(image):
    cfg = _cfg(
        {
            "gaussian_blur": _op(sigma=(1.0, 2.0)),
            "gaussian_noise": _op(sigma=(3.0, 9.0)),
            "brightness": _op(delta=(-0.2, 0.2)),
            "contrast": _op(factor=(0.7, 1.3)),
            "gamma": _op(value=(0.7, 1.4)),
            "saturation": _op(factor=(0.6, 1.4)),
            "hue_shift_deg": _op(delta=(-10.0, 10.0)),
            "vignette": _op(strength=(0.1, 0.4)),
            "jpeg": _op(quality=(50, 90)),
            "motion_blur": _op(kernel=(3.0, 7.0)),
        }
    )
    outcome = augment_image(image, cfg, _rng(7))
    assert outcome.image.shape == image.shape
    assert outcome.image.dtype == np.uint8
    assert outcome.image.min() >= 0 and outcome.image.max() <= 255
    assert set(outcome.applied) <= set(OP_ORDER)


def test_same_seed_gives_the_same_result(image):
    cfg = _cfg(
        {
            "gaussian_noise": _op(sigma=(4.0, 8.0)),
            "brightness": _op(delta=(-0.2, 0.2)),
            "gamma": _op(value=(0.8, 1.2)),
            "jpeg": _op(quality=(60, 90)),
        }
    )
    first = augment_image(image, cfg, _rng(11))
    second = augment_image(image, cfg, _rng(11))
    assert np.array_equal(first.image, second.image)
    assert first.applied == second.applied


def test_different_seeds_differ(image):
    cfg = _cfg({"brightness": _op(delta=(-0.3, 0.3)), "gaussian_noise": _op(sigma=(5.0, 5.0))})
    first = augment_image(image, cfg, _rng(1))
    second = augment_image(image, cfg, _rng(2))
    assert not np.array_equal(first.image, second.image)


# --------------------------------------------------------------------------- #
# photometric ops do not touch labels
# --------------------------------------------------------------------------- #
def test_photometric_ops_leave_boxes_untouched(image):
    cfg = _cfg(
        {
            "gaussian_blur": _op(sigma=(2.0, 2.0)),
            "gaussian_noise": _op(sigma=(6.0, 6.0)),
            "brightness": _op(delta=(0.2, 0.2)),
            "contrast": _op(factor=(1.2, 1.2)),
            "gamma": _op(value=(0.8, 0.8)),
            "saturation": _op(factor=(1.3, 1.3)),
            "hue_shift_deg": _op(delta=(8.0, 8.0)),
            "vignette": _op(strength=(0.3, 0.3)),
        }
    )
    outcome = augment_image(image, cfg, _rng(3))
    assert not outcome.geometry_changed
    assert transform_boxes(BOXES, outcome.flip_h, outcome.flip_v) == BOXES


def test_gaussian_blur_reduces_high_frequency_energy(image):
    cfg = _cfg({"gaussian_blur": _op(sigma=(2.0, 2.0))})
    outcome = augment_image(image, cfg, _rng(0))
    assert "gaussian_blur" in outcome.applied
    original = image.astype(np.float64)
    blurred = outcome.image.astype(np.float64)
    assert np.abs(np.diff(blurred, axis=1)).mean() < np.abs(np.diff(original, axis=1)).mean()


def test_brightness_increases_the_mean(image):
    cfg = _cfg({"brightness": _op(delta=(0.25, 0.25))})
    outcome = augment_image(image, cfg, _rng(0))
    assert outcome.image.mean() > image.mean()


def test_vignette_darkens_the_corners(image):
    cfg = _cfg({"vignette": _op(strength=(0.5, 0.5))})
    outcome = augment_image(image, cfg, _rng(0))
    corner = outcome.image[:4, :4].mean()
    centre = outcome.image[22:26, 30:34].mean()
    assert corner < centre


# --------------------------------------------------------------------------- #
# geometric ops move labels
# --------------------------------------------------------------------------- #
def test_flip_180_mirrors_boxes():
    cfg = _cfg({"flip_180": RandomOp("flip_180", 1.0, {})})
    outcome = augment_image(np.zeros((48, 64, 3), dtype=np.uint8), cfg, _rng(0))
    assert outcome.flip_h and outcome.flip_v
    flipped = transform_boxes(BOXES, outcome.flip_h, outcome.flip_v)
    assert flipped[0].cx == pytest.approx(1.0 - BOXES[0].cx)
    assert flipped[0].cy == pytest.approx(1.0 - BOXES[0].cy)
    assert flipped[0].w == pytest.approx(BOXES[0].w)
    assert flipped[1].cx == pytest.approx(1.0 - BOXES[1].cx)


def test_horizontal_flip_un_mirrors_a_preflipped_box():
    """flip_180 then horizontal_flip is a vertical mirror of the original."""
    cfg = _cfg(
        {
            "flip_180": RandomOp("flip_180", 1.0, {}),
            "horizontal_flip": RandomOp("horizontal_flip", 1.0, {}),
        }
    )
    outcome = augment_image(np.zeros((48, 64, 3), dtype=np.uint8), cfg, _rng(0))
    assert outcome.flip_h is False
    assert outcome.flip_v is True
    flipped = transform_boxes(BOXES, outcome.flip_h, outcome.flip_v)
    assert flipped[0].cx == pytest.approx(BOXES[0].cx)
    assert flipped[0].cy == pytest.approx(1.0 - BOXES[0].cy)


def test_image_is_actually_flipped():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[0, 0] = 255
    cfg = _cfg({"flip_180": RandomOp("flip_180", 1.0, {})})
    outcome = augment_image(image, cfg, _rng(0))
    assert outcome.image[3, 3, 0] == 255


def test_transform_boxes_is_identity_without_flips():
    assert transform_boxes(BOXES, False, False) == BOXES


def test_probability_zero_never_applies():
    cfg = _cfg({"salt_pepper": RandomOp("salt_pepper", 0.0, {"amount": Range(0.5, 0.5)})})
    for seed in range(20):
        outcome = augment_image(
            np.full((8, 8, 3), 128, dtype=np.uint8), cfg, _rng(seed)
        )
        assert "salt_pepper" not in outcome.applied


def test_salt_pepper_extremes_appear():
    cfg = _cfg({"salt_pepper": RandomOp("salt_pepper", 1.0, {"amount": Range(0.2, 0.2)})})
    outcome = augment_image(np.full((32, 32, 3), 128, dtype=np.uint8), cfg, _rng(0))
    values = set(np.unique(outcome.image).tolist())
    assert 0 in values and 255 in values
