"""Bird's-eye-view perspective transform."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


def extend_up(polyline: np.ndarray, y_top: float, fit_fraction: float = 0.25) -> np.ndarray:
    """Extend an image-space polyline (ordered top to bottom) up to row ``y_top``.

    A straight line is fitted to the top ``fit_fraction`` of the points and continued upwards. In the
    camera image a straight lane is a straight line, so this is exact on straight roads and an
    approximation on curves. Polylines that already reach ``y_top`` are returned unchanged.
    """
    pts = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2 or pts[:, 1].min() <= y_top:
        return pts
    pts = pts[np.argsort(pts[:, 1])]
    k = max(3, int(round(len(pts) * fit_fraction)))
    top = pts[: min(k, len(pts))]
    if np.ptp(top[:, 1]) < 1e-6:
        return pts
    m, c = np.polyfit(top[:, 1], top[:, 0], 1)
    step = max(float(np.median(np.diff(pts[:, 1]))), 1.0)
    ys = np.arange(y_top, pts[0, 1], step)
    return np.vstack([np.column_stack([m * ys + c, ys]), pts])


class PerspectiveTransform:
    """Maps the road trapezoid (``src``) to a rectangle (``dst``) and back.

    Args:
        src: four pixel points in the camera image (TL, TR, BR, BL).
        dst: four pixel points in the warped image, same order.
        size: (width, height) of both images.
    """

    def __init__(self, src: np.ndarray, dst: np.ndarray, size: tuple[int, int]) -> None:
        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError("size must be positive")
        self.src = np.asarray(src, dtype=np.float32).reshape(4, 2)
        self.dst = np.asarray(dst, dtype=np.float32).reshape(4, 2)
        self.size = (int(w), int(h))
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        if not np.all(np.isfinite(self.M)) or abs(np.linalg.det(self.M)) < 1e-12:
            raise ValueError("perspective points are degenerate")
        self.Minv = np.linalg.inv(self.M)

    @classmethod
    def from_fractions(
        cls,
        src_frac: Sequence[tuple[float, float]],
        dst_frac: Sequence[tuple[float, float]],
        size: tuple[int, int],
    ) -> PerspectiveTransform:
        """Build from points given as fractions of (width, height)."""
        w, h = size
        scale = np.array([w, h], dtype=np.float64)
        return cls(np.array(src_frac) * scale, np.array(dst_frac) * scale, size)

    def warp(self, image: np.ndarray, nearest: bool = False) -> np.ndarray:
        """Camera view -> bird's-eye view. Use ``nearest=True`` for binary masks."""
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpPerspective(image, self.M, self.size, flags=flags)

    def unwarp(self, image: np.ndarray, nearest: bool = False) -> np.ndarray:
        """Bird's-eye view -> camera view."""
        flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.warpPerspective(image, self.Minv, self.size, flags=flags)

    @staticmethod
    def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if pts.size == 0:
            return np.empty((0, 2))
        return cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)

    def warp_points(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` camera-image points into the warped image."""
        return self._apply(self.M, points)

    def unwarp_points(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N, 2)`` warped-image points back to the camera image."""
        return self._apply(self.Minv, points)

    def unwarp_curve(self, coeffs: np.ndarray, n: int = 72) -> np.ndarray:
        """Sample ``x = poly(y)`` over the warped height and return it in camera coordinates.

        Returns an ``(n, 2)`` float array ordered from the far end (top) to the near end (bottom).
        """
        h = self.size[1]
        ys = np.linspace(0, h - 1, n)
        xs = np.polyval(coeffs, ys)
        return self.unwarp_points(np.column_stack([xs, ys]))
