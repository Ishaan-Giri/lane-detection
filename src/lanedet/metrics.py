"""TuSimple-style lane metrics.

A lane is a list of x positions at fixed image rows (``h_samples``); ``x < 0`` means "no lane at this
row". A predicted lane matches a ground-truth (GT) lane when at least ``ratio`` of the GT lane's valid
points are within ``pixel_thresh`` pixels. Over a set of frames:

* accuracy = correctly predicted GT points / total GT points
* FP rate  = predicted lanes with no match / predicted lanes
* FN rate  = GT lanes with no match / GT lanes

Each GT lane is paired with at most one prediction (greedy, best point accuracy first).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MISSING = -2.0


def sample_lane_x(polyline: np.ndarray, h_samples: Sequence[float], width: int | None = None):
    """Interpolate an image-space polyline at rows ``h_samples``; ``-2`` outside its y-range/image."""
    pts = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    ys_query = np.asarray(h_samples, dtype=np.float64)
    out = np.full(len(ys_query), MISSING)
    if len(pts) < 2:
        return out
    order = np.argsort(pts[:, 1])
    xs, ys = pts[order, 0], pts[order, 1]
    inside = (ys_query >= ys[0]) & (ys_query <= ys[-1])
    out[inside] = np.interp(ys_query[inside], ys, xs)
    if width is not None:
        out[(out < 0) | (out >= width)] = MISSING
    return out


def line_accuracy(pred: Sequence[float], gt: Sequence[float], pixel_thresh: float = 20.0) -> float:
    """Fraction of the GT lane's valid points that the prediction hits within ``pixel_thresh``."""
    p = np.asarray(pred, dtype=np.float64)
    g = np.asarray(gt, dtype=np.float64)
    if p.shape != g.shape:
        raise ValueError("pred and gt must be sampled at the same rows")
    valid = g >= 0
    if not valid.any():
        return 0.0
    hit = (p >= 0) & (np.abs(p - g) < pixel_thresh) & valid
    return float(hit.sum() / valid.sum())


@dataclass
class FrameScore:
    """Counts for one frame; add them up across frames with :func:`aggregate`."""

    correct_points: int
    gt_points: int
    n_pred: int
    n_gt: int
    n_matched: int

    @property
    def accuracy(self) -> float:
        return self.correct_points / self.gt_points if self.gt_points else 0.0


def score_frame(
    pred_lanes: Sequence[Sequence[float]],
    gt_lanes: Sequence[Sequence[float]],
    pixel_thresh: float = 20.0,
    ratio: float = 0.85,
) -> FrameScore:
    """Score one frame. Lanes are x positions at the same ``h_samples``."""
    gt = [np.asarray(g, dtype=np.float64) for g in gt_lanes]
    gt = [g for g in gt if (g >= 0).any()]
    pred = [np.asarray(p, dtype=np.float64) for p in pred_lanes]
    pred = [p for p in pred if (p >= 0).any()]
    gt_points = int(sum((g >= 0).sum() for g in gt))

    pairs = sorted(
        (
            (line_accuracy(p, g, pixel_thresh), gi, pi)
            for gi, g in enumerate(gt)
            for pi, p in enumerate(pred)
        ),
        reverse=True,
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    correct = 0
    matched = 0
    for acc, gi, pi in pairs:
        if gi in used_gt or pi in used_pred or acc <= 0:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        correct += int(round(acc * (gt[gi] >= 0).sum()))
        if acc >= ratio:
            matched += 1
    return FrameScore(correct, gt_points, len(pred), len(gt), matched)


@dataclass
class Summary:
    accuracy: float
    fp_rate: float
    fn_rate: float
    n_frames: int


def aggregate(scores: Sequence[FrameScore]) -> Summary:
    """Pool frame counts into dataset-level accuracy / FP / FN rates."""
    correct = sum(s.correct_points for s in scores)
    gt_pts = sum(s.gt_points for s in scores)
    n_pred = sum(s.n_pred for s in scores)
    n_gt = sum(s.n_gt for s in scores)
    matched = sum(s.n_matched for s in scores)
    return Summary(
        accuracy=correct / gt_pts if gt_pts else 0.0,
        fp_rate=(n_pred - matched) / n_pred if n_pred else 0.0,
        fn_rate=(n_gt - matched) / n_gt if n_gt else 0.0,
        n_frames=len(scores),
    )


def select_ego_lanes(
    gt_lanes: Sequence[Sequence[float]], h_samples: Sequence[float], image_width: int
) -> list[list[float]]:
    """Keep the GT lanes immediately left and right of the image centre (the ego lane's borders).

    Each lane's position at the bottom of the image is found by fitting a line to its valid points
    and extrapolating, so lanes that stop short of the bottom row are still placed correctly.
    """
    ys = np.asarray(h_samples, dtype=np.float64)
    bottom = ys.max()
    centre = image_width / 2.0
    left, right = None, None
    for lane in gt_lanes:
        xs = np.asarray(lane, dtype=np.float64)
        valid = xs >= 0
        if valid.sum() < 2:
            continue
        m, c = np.polyfit(ys[valid], xs[valid], 1)
        xb = m * bottom + c
        if xb < centre and (left is None or xb > left[0]):
            left = (xb, list(lane))
        elif xb >= centre and (right is None or xb < right[0]):
            right = (xb, list(lane))
    return [item[1] for item in (left, right) if item is not None]
