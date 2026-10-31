"""Hough-line baseline: Canny edges + probabilistic Hough, one straight line per side.

Used as an ablation against the sliding-window polynomial pipeline. It cannot model curves, so its
curvature output is always ``inf`` (straight).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from lanedet.config import Config
from lanedet.pipeline import FrameResult
from lanedet.preprocess import apply_roi, to_bgr


class HoughLaneDetector:
    """Same ``process(frame) -> FrameResult`` interface as :class:`LanePipeline`."""

    def __init__(self, config: Config | None = None, debug: bool = False) -> None:
        self.config = (config or Config()).validate()
        self.debug = debug

    def reset(self) -> None:
        """No temporal state; present for interface parity."""

    def _segments(self, bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self.config.hough
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (c.blur_kernel, c.blur_kernel), 0)
        edges = apply_roi(cv2.Canny(blur, *c.canny), self.config.roi.vertices)
        lines = cv2.HoughLinesP(
            edges,
            c.rho,
            math.radians(c.theta_deg),
            c.threshold,
            minLineLength=c.min_line_length,
            maxLineGap=c.max_line_gap,
        )
        segs = np.empty((0, 4)) if lines is None else lines.reshape(-1, 4).astype(np.float64)
        return segs, edges

    @staticmethod
    def _fit_side(segs: np.ndarray) -> tuple[float, float] | None:
        """Length-weighted fit of ``x = m*y + c`` through segment endpoints."""
        if len(segs) == 0:
            return None
        xs = np.concatenate([segs[:, 0], segs[:, 2]])
        ys = np.concatenate([segs[:, 1], segs[:, 3]])
        lengths = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
        weights = np.concatenate([lengths, lengths])
        if np.ptp(ys) < 1:
            return None
        m, c = np.polyfit(ys, xs, 1, w=np.sqrt(weights))
        return float(m), float(c)

    def process(self, frame: np.ndarray) -> FrameResult:
        bgr = to_bgr(frame)
        h, w = bgr.shape[:2]
        segs, edges = self._segments(bgr)
        dx = segs[:, 2] - segs[:, 0]
        dy = segs[:, 3] - segs[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(dx == 0, np.inf, dy / dx)
        mid_x = (segs[:, 0] + segs[:, 2]) / 2.0
        steep = np.abs(slope) >= self.config.hough.min_abs_slope
        # Image y grows downwards: the left lane rises to the right (negative slope).
        left = self._fit_side(segs[steep & (slope < 0) & (mid_x < w / 2)])
        right = self._fit_side(segs[steep & (slope > 0) & (mid_x >= w / 2)])

        y_top = min(y for _, y in self.config.roi.vertices)
        if self.config.perspective.extend_to is not None:
            y_top = min(y_top, self.config.perspective.extend_to)
        y_top *= h - 1
        ys = np.linspace(y_top, h - 1, 20)

        def polyline(line: tuple[float, float] | None) -> np.ndarray | None:
            return None if line is None else np.column_stack([line[0] * ys + line[1], ys])

        left_img, right_img = polyline(left), polyline(right)
        offset = None
        if left_img is not None and right_img is not None:
            lx, rx = left_img[-1, 0], right_img[-1, 0]
            if rx > lx:
                xm = self.config.geometry.lane_width_m / (rx - lx)
                offset = float((w / 2.0 - (lx + rx) / 2.0) * xm)
        found = left_img is not None or right_img is not None
        result = FrameResult(
            left_image=left_img,
            right_image=right_img,
            curvature_m=math.inf if found else None,
            offset_m=offset,
            accepted=left_img is not None and right_img is not None,
            reason="raw",
            search="hough",
        )
        if self.debug:
            result.debug = {"mask": edges}
        return result
