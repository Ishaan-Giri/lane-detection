import numpy as np
import pytest

from lanedet.metrics import (
    FrameScore,
    aggregate,
    line_accuracy,
    sample_lane_x,
    score_frame,
    select_ego_lanes,
)

ROWS = list(range(400, 720, 40))  # 8 rows
L = [300.0 + i for i in range(8)]
R = [900.0 - i for i in range(8)]


def test_perfect_match():
    s = score_frame([L, R], [L, R])
    assert (s.correct_points, s.gt_points, s.n_matched) == (16, 16, 2)
    summary = aggregate([s])
    assert (summary.accuracy, summary.fp_rate, summary.fn_rate) == (1.0, 0.0, 0.0)


def test_one_missed_lane():
    summary = aggregate([score_frame([L], [L, R])])
    assert summary.accuracy == pytest.approx(0.5)
    assert summary.fp_rate == 0.0
    assert summary.fn_rate == pytest.approx(0.5)


def test_one_extra_lane():
    extra = [600.0] * 8
    summary = aggregate([score_frame([L, R, extra], [L, R])])
    assert summary.accuracy == 1.0
    assert summary.fp_rate == pytest.approx(1 / 3)
    assert summary.fn_rate == 0.0


def test_partial_hit_below_ratio_counts_points_but_not_match():
    pred = list(L)
    pred[0] += 50
    pred[1] += 50  # 6/8 = 0.75 < 0.85
    s = score_frame([pred], [L])
    assert s.correct_points == 6 and s.n_matched == 0
    summary = aggregate([s])
    assert summary.accuracy == pytest.approx(0.75)
    assert summary.fp_rate == 1.0 and summary.fn_rate == 1.0


def test_threshold_is_strict():
    assert line_accuracy([319.9], [300.0], 20) == 1.0
    assert line_accuracy([320.0], [300.0], 20) == 0.0


def test_missing_points_ignored_in_gt_and_wrong_in_pred():
    gt = [-2, -2, 300, 300]
    assert line_accuracy([300, 300, 300, 300], gt) == 1.0
    assert line_accuracy([300, 300, -2, 300], gt) == 0.5
    with pytest.raises(ValueError):
        line_accuracy([1, 2], [1, 2, 3])


def test_one_to_one_matching():
    # Two predictions on top of one GT lane: only one can match.
    s = score_frame([L, L], [L])
    assert s.n_matched == 1 and s.n_pred == 2


def test_empty_cases():
    assert aggregate([score_frame([], [])]).accuracy == 0.0
    s = score_frame([[-2] * 8], [[-2] * 8])
    assert s.n_pred == 0 and s.n_gt == 0
    assert aggregate([]).n_frames == 0
    assert FrameScore(0, 0, 0, 0, 0).accuracy == 0.0


def test_aggregate_pools_points():
    a = FrameScore(10, 10, 2, 2, 2)
    b = FrameScore(0, 30, 0, 2, 0)
    summary = aggregate([a, b])
    assert summary.accuracy == pytest.approx(0.25)
    assert summary.fn_rate == pytest.approx(0.5)


def test_sample_lane_x():
    poly = np.array([[100, 400], [200, 600]], float)
    xs = sample_lane_x(poly, [300, 400, 500, 600, 700])
    np.testing.assert_allclose(xs, [-2, 100, 150, 200, -2])
    xs = sample_lane_x(np.array([[-50, 400], [50, 600]]), [400, 600], width=100)
    np.testing.assert_allclose(xs, [-2, 50])
    assert (sample_lane_x(np.empty((0, 2)), [1, 2]) == -2).all()


def test_select_ego_lanes():
    rows = [400, 500, 600, 700]
    far_left = [100, 50, -2, -2]
    ego_left = [550, 450, 350, 250]
    ego_right = [700, 800, 900, 1000]
    far_right = [800, 1000, 1200, -2]
    ego = select_ego_lanes([far_left, ego_right, far_right, ego_left], rows, 1280)
    assert ego == [ego_left, ego_right]
    assert select_ego_lanes([[-2, -2, -2, 5]], rows, 1280) == []
