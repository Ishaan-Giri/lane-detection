import math

import numpy as np
import pytest

from lanedet.config import Config, GeometryConfig
from lanedet.lanefit import (
    LaneFit,
    LaneGeometry,
    SlidingWindowFinder,
    fit_lane,
    fit_lanes,
    search_around_fits,
)
from lanedet.preprocess import ThresholdMaskExtractor

H = 720


def _binary_from_fits(coeffs_list, h=H, w=1280, width=20):
    img = np.zeros((h, w), np.uint8)
    ys = np.arange(h)
    for c in coeffs_list:
        xs = np.polyval(c, ys).round().astype(int)
        for y, x in zip(ys, xs, strict=True):
            img[y, max(x - width // 2, 0) : max(x + width // 2, 0)] = 255
    return img


def _geometry(transform):
    cfg = Config()
    return LaneGeometry.from_config(GeometryConfig(), transform, cfg.perspective.dst)


def test_fit_recovers_known_coefficients():
    true = np.array([2e-4, -0.1, 400.0])
    ys = np.arange(0, H, 2).astype(float)
    xs = np.polyval(true, ys) + np.random.default_rng(0).normal(0, 1.0, ys.size)
    fit = fit_lane(xs, ys)
    np.testing.assert_allclose(fit.coeffs, true, rtol=0.05, atol=0.5)


def test_fit_needs_enough_pixels():
    assert fit_lane(np.array([1, 2]), np.array([1, 2])) is None
    assert fit_lane(np.arange(10), np.full(10, 5)) is None  # one row only
    assert fit_lane(np.arange(10), np.arange(10), min_pixels=50) is None


def test_sliding_window_on_synthetic_binary():
    left, right = np.array([1.5e-4, -0.2, 380.0]), np.array([1.5e-4, -0.2, 1020.0])
    binary = _binary_from_fits([left, right])
    pixels = SlidingWindowFinder(9, 100, 50).find(binary)
    lf, rf = fit_lanes(pixels, 100)
    ys = np.linspace(0, H - 1, 20)
    assert np.max(np.abs(lf.x_at(ys) - np.polyval(left, ys))) < 3
    assert np.max(np.abs(rf.x_at(ys) - np.polyval(right, ys))) < 3
    assert len(pixels.windows) == 18


def test_bases_none_when_empty():
    assert SlidingWindowFinder.find_bases(np.zeros((100, 200), np.uint8)) == (None, None)
    pixels = SlidingWindowFinder().find(np.zeros((100, 200), np.uint8))
    assert len(pixels.left_x) == 0 and len(pixels.right_x) == 0


def test_single_lane_visible():
    binary = _binary_from_fits([np.array([0, 0, 380.0])])
    lf, rf = fit_lanes(SlidingWindowFinder().find(binary), 100)
    assert lf is not None and rf is None


def test_search_around_previous_fit():
    left, right = np.array([0, 0, 380.0]), np.array([0, 0, 1020.0])
    binary = _binary_from_fits([left, right])
    pixels = search_around_fits(binary, LaneFit(left + [0, 0, 30]), None, margin=80)
    assert len(pixels.left_x) > 0 and len(pixels.right_x) == 0
    assert pixels.method == "around_fit"


def test_rejects_colour_input():
    with pytest.raises(ValueError):
        SlidingWindowFinder().find(np.zeros((10, 10, 3), np.uint8))


def test_curvature_matches_analytic(transform):
    geo = _geometry(transform)
    a_px = 2.5e-4
    fit = LaneFit(np.array([a_px, 0.0, 320.0]))
    a_m = a_px * geo.xm / geo.ym**2
    y_m = (H - 1) * geo.ym
    expected = (1 + (2 * a_m * y_m) ** 2) ** 1.5 / abs(2 * a_m)
    assert geo.curvature_radius(fit) == pytest.approx(expected, rel=1e-9)


def test_straight_lane_infinite_radius(transform):
    geo = _geometry(transform)
    assert math.isinf(geo.curvature_radius(LaneFit(np.array([0.0, 0.0, 320.0]))))
    assert math.isinf(geo.combined_radius(LaneFit(np.zeros(3)), LaneFit(np.zeros(3))))
    assert geo.combined_radius(None, None) is None


def test_offset_sign_and_value(transform):
    geo = _geometry(transform)
    centre = geo.camera_x
    # Lane centre 64 px to the left of the camera -> vehicle is right of centre.
    left = LaneFit(np.array([0, 0, centre - 320 - 64]))
    right = LaneFit(np.array([0, 0, centre + 320 - 64]))
    assert geo.offset(left, right) == pytest.approx(64 * geo.xm)
    assert geo.offset(left, right) > 0
    np.testing.assert_allclose(geo.lane_widths(left, right), 640 * geo.xm)


def test_end_to_end_curve_recovery(curved_frame, transform):
    """Mask -> warp -> windows -> fit on a rendered frame recovers the drawn lanes."""
    frame, scene = curved_frame
    warped = transform.warp(ThresholdMaskExtractor()(frame), nearest=True)
    lf, rf = fit_lanes(SlidingWindowFinder().find(warped), 150)
    ys = np.linspace(100, H - 1, 20)
    assert np.max(np.abs(lf.x_at(ys) - np.polyval(scene.left, ys))) < 10
    assert np.max(np.abs(rf.x_at(ys) - np.polyval(scene.right, ys))) < 10
    geo = _geometry(transform)
    true_r = geo.curvature_radius(LaneFit(scene.left))
    assert geo.curvature_radius(lf) == pytest.approx(true_r, rel=0.25)
