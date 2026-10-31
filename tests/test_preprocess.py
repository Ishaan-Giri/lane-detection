import cv2
import numpy as np
import pytest

from lanedet.config import PreprocessConfig, ROIConfig
from lanedet.preprocess import (
    ThresholdMaskExtractor,
    apply_roi,
    color_mask,
    gradient_mask,
    morphology,
    to_bgr,
)


def _paint_mask(transform, scene):
    """Where lane paint actually is in the camera image (ground truth for the mask)."""
    from lanedet.metrics import sample_lane_x

    h = transform.size[1]
    mask = np.zeros((h, transform.size[0]), np.uint8)
    rows = np.arange(int(transform.src[:, 1].min()) + 5, h)
    for coeffs in (scene.left, scene.right):
        xs = sample_lane_x(transform.unwarp_curve(coeffs, 300), rows)
        for x, y in zip(xs, rows, strict=True):
            if x >= 0:
                mask[y, int(round(x))] = 255
    return mask


def test_mask_finds_lane_paint_and_little_else(make_frame, transform):
    frame, scene = make_frame(curve=2e-4, right_dashed=False)  # solid lines: every row is paint
    mask = ThresholdMaskExtractor()(frame)
    assert mask.dtype == np.uint8 and mask.shape == frame.shape[:2]
    assert set(np.unique(mask)) <= {0, 255}

    centre = _paint_mask(transform, scene)
    near = cv2.dilate(centre, np.ones((41, 41), np.uint8))  # paint is ~31 px wide near the car
    hit = np.count_nonzero(mask[centre > 0]) / np.count_nonzero(centre)
    stray = np.count_nonzero(mask[near == 0]) / np.count_nonzero(mask)
    assert hit > 0.8, f"only {hit:.0%} of lane centre-line pixels in the mask"
    assert stray < 0.05, f"{stray:.0%} of mask pixels are away from the lanes"


def test_grayscale_and_bgra_input(straight_frame):
    frame, _ = straight_frame
    ex = ThresholdMaskExtractor()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert ex(gray).shape == gray.shape
    bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    np.testing.assert_array_equal(ex(bgra), ex(frame))


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((0, 0, 3), np.uint8),
        np.zeros((10, 10, 2), np.uint8),
        np.zeros((10, 10), np.float32),
    ],
)
def test_rejects_bad_images(bad):
    with pytest.raises(ValueError):
        to_bgr(bad)


def test_empty_frame_gives_empty_mask():
    blank = np.full((720, 1280, 3), 90, np.uint8)
    assert not ThresholdMaskExtractor()(blank).any()


def test_yellow_and_white_colour_mask():
    img = np.full((20, 30, 3), 90, np.uint8)
    img[:, :10] = (0, 200, 235)  # yellow (BGR)
    img[:, 20:] = (240, 240, 240)  # white
    m = color_mask(img, (100, 255), (200, 255), 70)
    assert m[:, :10].all() and m[:, 20:].all() and not m[:, 10:20].any()


def test_dark_saturated_pixels_ignored():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:] = (0, 0, 40)  # dark red: high S in HLS, low L
    assert not color_mask(img, (100, 255), (200, 255), 70).any()
    assert color_mask(img, (100, 255), (200, 255), 0).any()


def test_gradient_is_vertical_edge_detector():
    vertical = np.zeros((40, 40, 3), np.uint8)
    vertical[:, 20:] = 200
    horizontal = np.zeros((40, 40, 3), np.uint8)
    horizontal[20:, :] = 200
    assert gradient_mask(vertical, (40, 255)).any()
    assert not gradient_mask(horizontal, (40, 255)).any()


def test_gradient_does_not_amplify_noise():
    rng = np.random.default_rng(0)
    noise = np.clip(rng.normal(90, 3, (100, 100, 3)), 0, 255).astype(np.uint8)
    assert np.count_nonzero(gradient_mask(noise, (40, 255))) == 0


def test_morphology_close_fills_gap_and_open_removes_speck():
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5] = 255
    m[5:15, 7] = 255
    closed = morphology(m, close=3)
    assert closed[10, 6] == 255
    speck = np.zeros((20, 20), np.uint8)
    speck[10, 10] = 255
    assert not morphology(speck, open_=3).any()
    np.testing.assert_array_equal(morphology(m), m)


def test_roi_clears_outside():
    m = np.full((100, 100), 255, np.uint8)
    out = apply_roi(m, [(0.0, 1.0), (0.5, 0.5), (1.0, 1.0)])
    assert out[99, 50] == 255
    assert out[0, 0] == 0 and out[10, 50] == 0


def test_stages_keys(straight_frame):
    stages = ThresholdMaskExtractor(PreprocessConfig(), ROIConfig()).stages(straight_frame[0])
    assert set(stages) == {"color", "gradient", "combined", "mask"}
