import math

import numpy as np
import pytest

from lanedet import Config, LanePipeline
from lanedet.hough import HoughLaneDetector
from lanedet.metrics import sample_lane_x
from lanedet.overlay import debug_panel, draw_lanes, format_offset, format_radius
from lanedet.synthetic import drive, ground_truth, h_samples_for


def test_lanes_on_every_frame_of_synthetic_video(synthetic_video):
    pipe = LanePipeline()
    results = [pipe.process(f) for f in synthetic_video]
    assert all(r.detected for r in results)
    assert all(r.accepted for r in results)
    assert results[0].search == "windows"
    assert results[-1].search == "around_fit"


def test_matches_ground_truth(curved_frame, transform):
    frame, scene = curved_frame
    r = LanePipeline(use_tracker=False).process(frame)
    rows = h_samples_for(transform)
    for pred, gt in zip(r.lanes_image, ground_truth(scene, transform, rows), strict=True):
        xs = sample_lane_x(pred, rows, 1280)
        valid = np.array(gt) >= 0
        assert np.max(np.abs(xs[valid] - np.array(gt)[valid])) < 15


def test_offset_and_curvature_reported(make_frame):
    frame, _ = make_frame(curve=0.0, shift=-64.0)  # lane shifted left -> car right of centre
    r = LanePipeline().process(frame)
    assert r.offset_m == pytest.approx(64 * 3.7 / 640, abs=0.08)
    assert r.curvature_m > 2000


def test_recovers_after_bad_frame(synthetic_video):
    pipe = LanePipeline()
    for f in synthetic_video[:6]:
        good = pipe.process(f)
    blank = np.full_like(synthetic_video[0], 90)
    bad = pipe.process(blank)
    assert not bad.accepted and bad.detected  # falls back to recent fits
    np.testing.assert_allclose(bad.left.coeffs, good.left.coeffs, rtol=0.2, atol=5)
    after = pipe.process(synthetic_video[6])
    assert after.accepted


def test_empty_frame_without_history():
    r = LanePipeline().process(np.zeros((720, 1280, 3), np.uint8))
    assert not r.detected and r.lanes_image == [] and r.offset_m is None


def test_single_lane_visible():
    frames = list(drive(20, seed=2, scenario="dropout"))
    pipe = LanePipeline(use_tracker=False)
    r = pipe.process(frames[-1][0])  # right line hidden
    assert r.left_image is not None and r.right_image is None
    assert r.offset_m is None and r.curvature_m is not None


def test_wrong_image_shape():
    with pytest.raises(ValueError):
        LanePipeline().process(np.zeros((10, 10, 5), np.uint8))


def test_resolution_change_and_reset(synthetic_video):
    import cv2

    pipe = LanePipeline()
    pipe.process(synthetic_video[0])
    small = cv2.resize(synthetic_video[0], (640, 360))
    pipe.process(small)
    assert pipe.size == (640, 360)
    pipe.reset()
    assert not pipe.tracker.has_fit


def test_custom_mask_extractor(straight_frame):
    calls = []

    def extractor(frame):
        calls.append(frame.shape)
        return np.zeros(frame.shape[:2], np.uint8)

    r = LanePipeline(mask_extractor=extractor).process(straight_frame[0])
    assert calls and not r.detected

    with pytest.raises(ValueError):
        LanePipeline(mask_extractor=lambda f: np.zeros((5, 5), np.uint8)).process(straight_frame[0])


def test_debug_and_overlay(curved_frame):
    frame, _ = curved_frame
    r = LanePipeline(debug=True).process(frame)
    assert {"mask", "warped"} <= set(r.debug)
    out = draw_lanes(frame, r)
    assert out.shape == frame.shape and not np.array_equal(out, frame)
    panel = debug_panel(out, r)
    assert panel.shape[0] == frame.shape[0] and panel.shape[1] > frame.shape[1]
    d = r.to_dict()
    assert len(d["left_coeffs"]) == 3 and d["accepted"]


def test_hough_baseline(straight_frame):
    frame, _ = straight_frame
    r = HoughLaneDetector(debug=True).process(frame)
    assert r.detected and math.isinf(r.curvature_m)
    assert abs(r.offset_m) < 0.3
    assert "mask" in r.debug
    assert not HoughLaneDetector().process(np.zeros((720, 1280, 3), np.uint8)).lanes_image
    assert debug_panel(draw_lanes(frame, r), r).shape[1] > frame.shape[1]


def test_formatters():
    assert format_radius(None) == "Radius: n/a"
    assert format_radius(math.inf) == "Radius: straight"
    assert format_radius(512.4) == "Radius: 512 m"
    assert format_offset(None) == "Offset: n/a"
    assert format_offset(-0.2) == "Offset: 0.20 m left of centre"


def test_config_is_validated():
    cfg = Config()
    cfg.tracker.method = "median"
    with pytest.raises(ValueError):
        LanePipeline(cfg)


def test_extend_to_reaches_requested_row(straight_frame):
    frame, _ = straight_frame if isinstance(straight_frame, tuple) else (straight_frame, None)
    h = frame.shape[0]
    cfg = Config()
    cfg.perspective.extend_to = 0.45
    for det in (LanePipeline(cfg, use_tracker=False), HoughLaneDetector(cfg)):
        r = det.process(frame)
        assert r.lanes_image
        for line in r.lanes_image:
            assert line[:, 1].min() == pytest.approx(0.45 * (h - 1), abs=1.0)
    base = LanePipeline(use_tracker=False).process(frame)
    assert min(line[:, 1].min() for line in base.lanes_image) > 0.5 * (h - 1)


def test_extend_to_is_validated():
    cfg = Config()
    cfg.perspective.extend_to = 1.5
    with pytest.raises(ValueError):
        cfg.validate()
