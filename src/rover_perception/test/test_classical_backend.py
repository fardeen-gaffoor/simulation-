"""The classical bootstrap detector: what runs before a trained model exists.

It is honest about being a heuristic, which makes it easy to test: the sand is a
flat reference, a boulder is darker *and lit from above* (top brighter than
bottom), a crater is darker *and shaded from above* (top darker than bottom).
The tests drive it with synthetic images where the answer is known by
construction, and check the gates (area, fill, aspect, bbox fraction) that keep
noise out.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from rover_perception.backends.classical import (
    Blob,
    ClassicalDetector,
    Region,
    anomaly_mask,
    classify_blob,
    extract_blobs,
    label_components,
    region_for,
    row_runs,
)
from rover_perception.calibration_ref import load_calibration
from rover_perception.config import load_config


@pytest.fixture(scope="module")
def cfg():
    here = os.path.dirname(os.path.abspath(__file__))
    return load_config(os.path.join(here, "..", "config", "perception.yaml")).detector.classical


@pytest.fixture(scope="module")
def camera(calibration_path):
    return load_calibration(calibration_path).camera("cam_nav")


@pytest.fixture(scope="module")
def detector(cfg, camera):
    return ClassicalDetector(cfg, camera)


# --------------------------------------------------------------------------- #
# runs and connected components
# --------------------------------------------------------------------------- #
def test_row_runs():
    assert row_runs(np.zeros(10, dtype=bool)) == []
    assert row_runs(np.array([0, 1, 1, 1, 0, 1, 0], dtype=bool)) == [(1, 3), (5, 5)]
    assert row_runs(np.ones(4, dtype=bool)) == [(0, 3)]


def test_label_components_counts_separate_blobs():
    mask = np.zeros((6, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[4:6, 6:9] = True
    labels, count = label_components(mask)
    assert count == 2
    assert labels[2, 2] != labels[5, 7]
    assert labels[0, 0] == 0, "background is label 0"


def test_label_components_uses_eight_connectivity():
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 1] = True
    labels, count = label_components(mask)
    assert count == 1, "a diagonal touch joins the components"
    assert labels[0, 0] == labels[1, 1]


def test_label_components_rejects_a_non_mask():
    with pytest.raises(ValueError):
        label_components(np.zeros((2, 2, 2), dtype=bool))


# --------------------------------------------------------------------------- #
# the region
# --------------------------------------------------------------------------- #
def test_region_starts_below_the_horizon(cfg, camera):
    region = region_for(cfg, camera)
    assert region.row0 == pytest.approx(157, abs=1), "horizon 153 + 4 px margin"
    assert region.row1 == 480
    assert (region.col0, region.col1) == (0, 640)
    assert not region.empty
    assert region.as_tuple() == (region.row0, region.row1, region.col0, region.col1)


def test_a_degenerate_region_reports_empty(cfg, camera):
    import dataclasses

    empty = dataclasses.replace(cfg, bottom_fraction=0.0)
    region = region_for(empty, camera)
    assert region.empty
    assert ClassicalDetector(empty, camera).detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


# --------------------------------------------------------------------------- #
# the anomaly mask
# --------------------------------------------------------------------------- #
def test_flat_sand_has_no_anomalies(cfg, camera):
    grey = np.full((480, 640), 200.0, dtype=np.float32)
    chroma = np.zeros((480, 640, 2), dtype=np.float32)
    mask, stats = anomaly_mask(grey, chroma, region_for(cfg, camera), cfg)
    assert not mask.any()
    assert stats["reference_grey"] == pytest.approx(200.0)
    assert stats["anomaly_px"] == 0.0


def test_a_dark_patch_is_an_anomaly(cfg, camera):
    grey = np.full((480, 640), 200.0, dtype=np.float32)
    grey[300:360, 280:360] = 150.0
    chroma = np.zeros((480, 640, 2), dtype=np.float32)
    mask, stats = anomaly_mask(grey, chroma, region_for(cfg, camera), cfg)
    assert mask[300:360, 280:360].all()
    assert not mask[:296].any(), "above the horizon is out of the region"
    assert stats["anomaly_px"] == pytest.approx(60.0 * 80.0)
    assert stats["mad"] == pytest.approx(0.0), "a clear majority is uniform sand"


def test_an_off_colour_patch_is_an_anomaly_even_at_the_same_brightness(cfg, camera):
    grey = np.full((480, 640), 200.0, dtype=np.float32)
    chroma = np.zeros((480, 640, 2), dtype=np.float32)
    chroma[300:360, 280:360, 0] = 60.0  # strongly red where the sand is neutral
    mask, _stats = anomaly_mask(grey, chroma, region_for(cfg, camera), cfg)
    assert mask[330, 300]


def test_the_reference_is_robust_to_a_minority_of_dark_rock(cfg, camera):
    grey = np.full((480, 640), 200.0, dtype=np.float32)
    grey[200:400, 200:400] = 20.0  # 25% of the region is very dark
    chroma = np.zeros((480, 640, 2), dtype=np.float32)
    _mask, stats = anomaly_mask(grey, chroma, region_for(cfg, camera), cfg)
    assert stats["reference_grey"] == pytest.approx(200.0), "the median ignores the rock"


# --------------------------------------------------------------------------- #
# blobs and their gates
# --------------------------------------------------------------------------- #
def _labels_for(mask):
    return label_components(mask)


def test_extract_blobs_measures_a_square(cfg, camera):
    region = region_for(cfg, camera)
    mask = np.zeros((480, 640), dtype=bool)
    mask[300:360, 280:360] = True
    labels, count = _labels_for(mask)
    grey = np.full((480, 640), 200.0, dtype=np.float32)
    blobs = extract_blobs(labels, count, grey, region, cfg)
    assert len(blobs) == 1
    blob = blobs[0]
    assert blob.area == 60 * 80
    assert (blob.x0, blob.y0, blob.x1, blob.y1) == (280, 300, 359, 359)
    assert blob.width == 80 and blob.height == 60
    assert blob.fill == pytest.approx(1.0)
    assert blob.aspect == pytest.approx(80.0 / 60.0)
    assert not blob.touches_edge
    assert set(blob.as_features()) == {
        "area_px",
        "fill",
        "aspect",
        "bbox_fraction",
        "mean_grey",
        "grey_top",
        "grey_bottom",
        "shading_delta",
    }


def test_a_too_small_blob_is_dropped(cfg, camera):
    region = region_for(cfg, camera)
    mask = np.zeros((480, 640), dtype=bool)
    mask[300:305, 280:285] = True  # 25 px < min_area_px (60)
    labels, count = _labels_for(mask)
    grey = np.full((480, 640), 200.0, dtype=np.float32)
    assert extract_blobs(labels, count, grey, region, cfg) == []


def test_a_thin_blob_is_dropped_by_aspect(cfg, camera):
    region = region_for(cfg, camera)
    mask = np.zeros((480, 640), dtype=bool)
    mask[300:302, 100:400] = True  # 2 x 300: aspect 150 >> 3.0
    labels, count = _labels_for(mask)
    grey = np.full((480, 640), 200.0, dtype=np.float32)
    assert extract_blobs(labels, count, grey, region, cfg) == []


def test_a_blob_touching_the_region_edge_is_flagged(cfg, camera):
    region = region_for(cfg, camera)
    mask = np.zeros((480, 640), dtype=bool)
    mask[300:360, 0:80] = True
    labels, count = _labels_for(mask)
    grey = np.full((480, 640), 200.0, dtype=np.float32)
    blobs = extract_blobs(labels, count, grey, region, cfg)
    assert len(blobs) == 1
    assert blobs[0].touches_edge


def test_no_components_is_no_blobs(cfg, camera):
    region = region_for(cfg, camera)
    grey = np.zeros((480, 640), dtype=np.float32)
    assert extract_blobs(np.zeros((480, 640), dtype=np.int32), 0, grey, region, cfg) == []


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def _blob(grey_top: float, grey_bottom: float, **changes) -> Blob:
    defaults = dict(
        label=1,
        area=4800,
        x0=280,
        y0=300,
        x1=359,
        y1=359,
        fill=1.0,
        aspect=80.0 / 60.0,
        bbox_fraction=0.02,
        mean_grey=0.5 * (grey_top + grey_bottom),
        grey_top=grey_top,
        grey_bottom=grey_bottom,
        touches_edge=False,
    )
    defaults.update(changes)
    return Blob(**defaults)


def test_top_brighter_is_a_boulder(cfg):
    kind, score, conservative = classify_blob(_blob(200.0, 150.0), 210.0, cfg)
    assert kind == "boulder"
    assert not conservative
    assert 0.3 <= score <= 1.0


def test_top_darker_is_a_crater(cfg):
    kind, _score, conservative = classify_blob(_blob(150.0, 200.0), 210.0, cfg)
    assert kind == "crater"
    assert not conservative


def test_an_ambiguous_blob_follows_the_config(cfg):
    flat = _blob(180.0, 180.0)
    kind, _score, conservative = classify_blob(flat, 210.0, cfg)
    assert kind == cfg.ambiguous_kind
    assert conservative, "an ambiguous blob is published as a guess, and says so"

    import dataclasses

    dropping = dataclasses.replace(cfg, ambiguous_kind="drop")
    assert classify_blob(flat, 210.0, dropping) == ("", 0.0, True)


def test_a_blob_on_the_region_edge_loses_score(cfg):
    interior = classify_blob(_blob(200.0, 150.0), 210.0, cfg)
    edge = classify_blob(_blob(200.0, 150.0, touches_edge=True), 210.0, cfg)
    assert edge[2] is True, "an edge blob might be a cut-off obstacle"
    assert edge[1] < interior[1]


def test_a_darker_blob_scores_higher(cfg):
    dark = classify_blob(_blob(160.0, 120.0), 210.0, cfg)
    pale = classify_blob(_blob(200.0, 195.0), 210.0, cfg)
    assert dark[1] > pale[1]


# --------------------------------------------------------------------------- #
# the whole detector
# --------------------------------------------------------------------------- #
def _sand_with_patch(grey_top: float, grey_bottom: float) -> np.ndarray:
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    image[300:330, 280:360] = grey_top
    image[330:360, 280:360] = grey_bottom
    return image


def test_detects_a_boulder(detector):
    # both halves must be darker than the sand reference (200 - delta_luma 18),
    # or the brighter half reads as sand and the blob loses its shading
    detections = detector.detect(_sand_with_patch(180, 130))
    assert len(detections) == 1
    found = detections[0]
    assert found.kind == "boulder"
    assert found.class_id == 0
    assert found.x0 <= 280 and found.x1 >= 360
    assert found.y1 >= 360
    assert not found.conservative
    assert detector.last_stats["detections"] == 1.0
    assert detector.last_stats["blobs"] >= 1.0


def test_detects_a_crater(detector):
    detections = detector.detect(_sand_with_patch(130, 180))
    assert len(detections) == 1
    assert detections[0].kind == "crater"
    assert detections[0].class_id == 1


def test_clean_sand_yields_nothing(detector):
    assert detector.detect(np.full((480, 640, 3), 200, dtype=np.uint8)) == []


def test_above_the_horizon_is_ignored(detector):
    """A dark cloud in the sky is not an obstacle."""
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    image[20:100, 200:400] = 20
    assert detector.detect(image) == []


def test_the_box_never_leaves_the_image(detector):
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    image[300:480, 560:640] = 150  # touches the right edge and the bottom
    detections = detector.detect(image)
    for found in detections:
        assert 0.0 <= found.x0 <= found.x1 <= 640.0
        assert 0.0 <= found.y0 <= found.y1 <= 480.0


def test_max_detections_is_honoured(cfg, camera):
    import dataclasses

    capped = dataclasses.replace(cfg, max_detections=1, min_area_px=20)
    detector = ClassicalDetector(capped, camera)
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    for index in range(3):
        image[300:330, 100 + index * 120 : 160 + index * 120] = 180
        image[330:360, 100 + index * 120 : 160 + index * 120] = 130
    assert len(detector.detect(image)) == 1


def test_a_grey_image_is_accepted(detector):
    grey = np.full((480, 640), 200, dtype=np.uint8)
    grey[300:330, 280:360] = 180
    grey[330:360, 280:360] = 130
    detections = detector.detect(grey)
    assert len(detections) == 1
    assert detections[0].kind == "boulder"


def test_describe_reports_the_region(detector):
    description = detector.describe()
    assert description["backend"] == "classical"
    assert len(description["region"]) == 4
    assert description["ambiguous_kind"] == "crater"


def test_region_accessor_is_a_region(detector):
    assert isinstance(detector.region, Region)
    assert not detector.region.empty
