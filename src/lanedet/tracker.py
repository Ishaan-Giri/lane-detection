"""Temporal smoothing with sanity checks and fallback to recent good fits."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from lanedet.config import TrackerConfig
from lanedet.lanefit import LaneFit, LaneGeometry


@dataclass
class TrackState:
    """Output of :meth:`LaneTracker.update`."""

    left: LaneFit | None
    right: LaneFit | None
    accepted: bool  # the new measurement passed the sanity checks
    reason: str  # "ok" or why it was rejected
    reset: bool  # history was cleared on this frame
    failures: int  # consecutive rejections so far


class LaneTracker:
    """Keeps the last N good fits and outputs their average.

    A new pair of fits is accepted only if both lanes exist, the lane width is plausible, the lines are
    roughly parallel and neither line jumped too far from the smoothed estimate. Rejected frames reuse
    the smoothed estimate. After ``max_failures`` consecutive rejections the history is cleared so the
    pipeline goes back to a full sliding-window search.
    """

    def __init__(self, geometry: LaneGeometry, cfg: TrackerConfig | None = None) -> None:
        self.geometry = geometry
        self.cfg = cfg or TrackerConfig()
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.cfg.n_history)
        self._ema: tuple[np.ndarray, np.ndarray] | None = None
        self.failures = 0

    @property
    def has_fit(self) -> bool:
        return len(self._history) > 0

    def reset(self) -> None:
        self._history.clear()
        self._ema = None
        self.failures = 0

    def smoothed(self) -> tuple[LaneFit, LaneFit] | None:
        """Current smoothed (left, right) fits, or ``None`` without history."""
        if not self._history:
            return None
        if self.cfg.method == "ema" and self._ema is not None:
            left, right = self._ema
        else:
            left = np.mean([h[0] for h in self._history], axis=0)
            right = np.mean([h[1] for h in self._history], axis=0)
        return LaneFit(left), LaneFit(right)

    def check(self, left: LaneFit | None, right: LaneFit | None) -> tuple[bool, str]:
        """Sanity-check a new measurement. Returns (ok, reason)."""
        if left is None and right is None:
            return False, "no_lanes"
        if left is None:
            return False, "missing_left"
        if right is None:
            return False, "missing_right"
        c, g = self.cfg, self.geometry
        widths = g.lane_widths(left, right)
        if widths.min() <= 0:
            return False, "crossed"
        if not c.min_lane_width_m <= widths[-1] <= c.max_lane_width_m:
            return False, "width"
        if widths.std() > c.max_width_std_m:
            return False, "not_parallel"
        prev = self.smoothed()
        if prev is not None:
            ys = np.linspace(0, g.height - 1, 10)
            for new, old in ((left, prev[0]), (right, prev[1])):
                if np.max(np.abs(new.x_at(ys) - old.x_at(ys))) * g.xm > c.max_jump_m:
                    return False, "jump"
        return True, "ok"

    def update(self, left: LaneFit | None, right: LaneFit | None) -> TrackState:
        """Feed one frame's raw fits; get the fits to display."""
        ok, reason = self.check(left, right)
        if ok:
            assert left is not None and right is not None
            self._history.append((left.coeffs, right.coeffs))
            if self._ema is None:
                self._ema = (left.coeffs, right.coeffs)
            else:
                a = self.cfg.ema_alpha
                self._ema = (
                    a * left.coeffs + (1 - a) * self._ema[0],
                    a * right.coeffs + (1 - a) * self._ema[1],
                )
            self.failures = 0
            sm = self.smoothed()
            assert sm is not None
            return TrackState(sm[0], sm[1], True, reason, False, 0)

        self.failures += 1
        failures = self.failures
        reset = False
        if self.failures >= self.cfg.max_failures:
            self.reset()
            reset = True
        sm = self.smoothed()
        if sm is not None:
            return TrackState(sm[0], sm[1], False, reason, reset, failures)
        # No history to fall back on: pass through a lone lane (one side missing), drop bad pairs.
        if reason in ("missing_left", "missing_right"):
            return TrackState(left, right, False, reason, reset, failures)
        return TrackState(None, None, False, reason, reset, failures)
