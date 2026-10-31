import numpy as np
import pytest

from lanedet.config import Config, TrackerConfig
from lanedet.lanefit import LaneFit, LaneGeometry
from lanedet.tracker import LaneTracker

H = 720
XM = 3.7 / 640


def _geo():
    return LaneGeometry(XM, 30 / H, H, 640.0)


def _pair(shift=0.0, width=640.0, curve=0.0):
    left = LaneFit(np.array([curve, 0.0, 320.0 + shift]))
    right = LaneFit(np.array([curve, 0.0, 320.0 + width + shift]))
    return left, right


def _tracker(**kw):
    return LaneTracker(_geo(), TrackerConfig(**kw))


def test_accepts_good_measurement():
    t = _tracker()
    s = t.update(*_pair())
    assert s.accepted and s.reason == "ok" and t.has_fit


def test_single_bad_frame_barely_changes_output():
    t = _tracker()
    for _ in range(5):
        t.update(*_pair())
    before = t.smoothed()[0].x_at(H - 1)
    s = t.update(*_pair(width=1500))  # impossible width
    assert not s.accepted and s.reason == "width"
    assert s.left.x_at(H - 1) == pytest.approx(before)


def test_impossible_width_rejected_without_history():
    s = _tracker().update(*_pair(width=100))
    assert not s.accepted and s.reason == "width"
    assert s.left is None and s.right is None


def test_crossed_and_non_parallel_rejected():
    t = _tracker()
    left = LaneFit(np.array([0, 0, 900.0]))
    right = LaneFit(np.array([0, 0, 300.0]))
    assert t.check(left, right) == (False, "crossed")
    left = LaneFit(np.array([0.0, 0.0, 320.0]))
    right = LaneFit(np.array([0.0, -0.6, 960.0 + 0.6 * (H - 1)]))  # converges toward the top
    assert t.check(left, right)[1] == "not_parallel"


def test_jump_rejected():
    t = _tracker()
    t.update(*_pair())
    ok, reason = t.check(*_pair(shift=300))
    assert not ok and reason == "jump"


def test_n_bad_frames_trigger_reset():
    t = _tracker(max_failures=3)
    t.update(*_pair())
    states = [t.update(None, None) for _ in range(3)]
    assert [s.reset for s in states] == [False, False, True]
    assert not t.has_fit and t.failures == 0
    assert states[-1].left is None


def test_reset_allows_lane_change():
    t = _tracker(max_failures=2)
    t.update(*_pair())
    moved = _pair(shift=300)
    assert not t.update(*moved).accepted
    assert t.update(*moved).reset
    assert t.update(*moved).accepted


def test_missing_lane_passthrough_without_history():
    left, _ = _pair()
    s = _tracker().update(left, None)
    assert s.reason == "missing_right" and s.left is left and s.right is None


def test_mean_smoothing():
    t = _tracker(n_history=2)
    t.update(*_pair(shift=0))
    s = t.update(*_pair(shift=20))
    assert s.left.x_at(0) == pytest.approx(330.0)


def test_ema_smoothing():
    t = _tracker(method="ema", ema_alpha=0.5)
    t.update(*_pair(shift=0))
    s = t.update(*_pair(shift=20))
    assert s.left.x_at(0) == pytest.approx(330.0)
    s = t.update(*_pair(shift=20))
    assert s.left.x_at(0) == pytest.approx(335.0)


def test_config_default_tracker():
    assert LaneTracker(_geo()).cfg == Config().tracker
