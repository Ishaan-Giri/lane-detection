"""Lane pixel search (sliding windows / around a previous fit), polynomial fit, curvature, offset.

All fits live in the warped (bird's-eye) image: ``x = A*y**2 + B*y + C`` with ``y`` in pixels,
``y = 0`` at the far end of the view and ``y = height - 1`` next to the car.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from lanedet.config import GeometryConfig, WindowConfig
from lanedet.perspective import PerspectiveTransform


@dataclass(frozen=True)
class LaneFit:
    """A 2nd-order lane polynomial in warped-image pixels."""

    coeffs: np.ndarray  # (A, B, C)
    n_pixels: int = 0

    def x_at(self, y: np.ndarray | float) -> np.ndarray:
        """Lane x position(s) at warped row(s) ``y``."""
        return np.polyval(self.coeffs, y)


@dataclass
class LanePixels:
    """Pixels assigned to each lane, plus the search windows used (for drawing)."""

    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    windows: list[tuple[int, int, int, int]] = field(default_factory=list)  # x0, y0, x1, y1
    method: str = "windows"


def _empty() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


def _check_binary(binary: np.ndarray) -> None:
    if binary.ndim != 2:
        raise ValueError("expected a single-channel binary image")


class SlidingWindowFinder:
    """Locate lane bases with a column histogram, then follow each lane upwards with windows."""

    def __init__(self, n_windows: int = 9, margin: int = 100, min_pix: int = 50) -> None:
        if n_windows < 1 or margin < 1:
            raise ValueError("n_windows and margin must be >= 1")
        self.n_windows = n_windows
        self.margin = margin
        self.min_pix = min_pix

    @classmethod
    def from_config(cls, cfg: WindowConfig) -> SlidingWindowFinder:
        return cls(cfg.n_windows, cfg.margin, cfg.min_pix)

    @staticmethod
    def find_bases(binary: np.ndarray) -> tuple[int | None, int | None]:
        """Peak columns of the bottom-half histogram, one per image half (``None`` if empty)."""
        _check_binary(binary)
        h, w = binary.shape
        hist = np.count_nonzero(binary[h // 2 :, :], axis=0)
        mid = w // 2
        left = int(np.argmax(hist[:mid])) if hist[:mid].any() else None
        right = int(np.argmax(hist[mid:])) + mid if hist[mid:].any() else None
        return left, right

    def find(self, binary: np.ndarray) -> LanePixels:
        """Collect left/right lane pixels with a sliding-window search."""
        _check_binary(binary)
        h, w = binary.shape
        ys, xs = binary.nonzero()
        left_base, right_base = self.find_bases(binary)
        win_h = max(h // self.n_windows, 1)
        windows: list[tuple[int, int, int, int]] = []

        def follow(base: int | None) -> np.ndarray:
            if base is None:
                return _empty()
            x_cur = base
            picked = []
            for i in range(self.n_windows):
                y_hi = h - i * win_h
                y_lo = max(y_hi - win_h, 0)
                x_lo, x_hi = x_cur - self.margin, x_cur + self.margin
                windows.append((max(x_lo, 0), y_lo, min(x_hi, w - 1), y_hi))
                inside = ((ys >= y_lo) & (ys < y_hi) & (xs >= x_lo) & (xs < x_hi)).nonzero()[0]
                picked.append(inside)
                if len(inside) > self.min_pix:
                    x_cur = int(np.mean(xs[inside]))
            return np.concatenate(picked) if picked else _empty()

        li, ri = follow(left_base), follow(right_base)
        return LanePixels(xs[li], ys[li], xs[ri], ys[ri], windows, "windows")


def search_around_fits(
    binary: np.ndarray, left: LaneFit | None, right: LaneFit | None, margin: int
) -> LanePixels:
    """Collect pixels within ``margin`` of the previous fits (faster and more stable than windows)."""
    _check_binary(binary)
    ys, xs = binary.nonzero()

    def band(fit: LaneFit | None) -> np.ndarray:
        if fit is None:
            return np.zeros(len(xs), dtype=bool)
        return np.abs(xs - fit.x_at(ys)) < margin

    lm, rm = band(left), band(right)
    return LanePixels(xs[lm], ys[lm], xs[rm], ys[rm], [], "around_fit")


def fit_lane(xs: np.ndarray, ys: np.ndarray, min_pixels: int = 3) -> LaneFit | None:
    """Least-squares ``x = f(y)`` quadratic, or ``None`` if there is too little data."""
    if len(xs) < max(min_pixels, 3) or len(np.unique(ys)) < 3:
        return None
    coeffs = np.polyfit(ys.astype(np.float64), xs.astype(np.float64), 2)
    return LaneFit(coeffs=coeffs, n_pixels=len(xs))


def fit_lanes(pixels: LanePixels, min_pixels: int) -> tuple[LaneFit | None, LaneFit | None]:
    """Fit both lanes from a :class:`LanePixels` result."""
    return (
        fit_lane(pixels.left_x, pixels.left_y, min_pixels),
        fit_lane(pixels.right_x, pixels.right_y, min_pixels),
    )


class LaneGeometry:
    """Converts warped-pixel fits into metric curvature, lane width and vehicle offset.

    Args:
        xm_per_pix: metres per warped pixel, horizontally.
        ym_per_pix: metres per warped pixel, vertically.
        height: warped image height (the row next to the car is ``height - 1``).
        camera_x: x of the camera centre-line in the warped image.
    """

    def __init__(self, xm_per_pix: float, ym_per_pix: float, height: int, camera_x: float) -> None:
        self.xm = float(xm_per_pix)
        self.ym = float(ym_per_pix)
        self.height = int(height)
        self.camera_x = float(camera_x)

    @classmethod
    def from_config(
        cls,
        cfg: GeometryConfig,
        transform: PerspectiveTransform,
        dst_frac: list[tuple[float, float]],
    ) -> LaneGeometry:
        """Lane width in the warped image = dst bottom-right x minus bottom-left x."""
        w, h = transform.size
        lane_px = (dst_frac[2][0] - dst_frac[3][0]) * w
        # Camera assumed on the image's vertical centre-line; map its bottom point into the warp.
        camera_x = float(transform.warp_points(np.array([[w / 2.0, h - 1.0]]))[0, 0])
        return cls(cfg.lane_width_m / lane_px, cfg.view_length_m / h, h, camera_x)

    @property
    def y_bottom(self) -> float:
        return float(self.height - 1)

    def metric_coeffs(self, fit: LaneFit) -> tuple[float, float]:
        """(A, B) of the fit after converting both axes to metres."""
        a, b, _ = fit.coeffs
        return a * self.xm / self.ym**2, b * self.xm / self.ym

    def curvature_radius(self, fit: LaneFit, y_eval: float | None = None) -> float:
        """Radius of curvature in metres at warped row ``y_eval`` (default: next to the car).

        ``R = (1 + (2 A y + B)^2)^1.5 / |2 A|``; returns ``inf`` for a straight line.
        """
        a_m, b_m = self.metric_coeffs(fit)
        if abs(a_m) < 1e-12:
            return math.inf
        y_m = (self.y_bottom if y_eval is None else y_eval) * self.ym
        return float((1.0 + (2.0 * a_m * y_m + b_m) ** 2) ** 1.5 / abs(2.0 * a_m))

    def combined_radius(self, left: LaneFit | None, right: LaneFit | None) -> float | None:
        """Harmonic mean of the available radii (i.e. the mean curvature, inverted)."""
        radii = [self.curvature_radius(f) for f in (left, right) if f is not None]
        if not radii:
            return None
        mean_k = sum(1.0 / r for r in radii) / len(radii)
        return math.inf if mean_k == 0 else 1.0 / mean_k

    def offset(self, left: LaneFit, right: LaneFit, y_eval: float | None = None) -> float:
        """Vehicle offset from the lane centre in metres; positive = vehicle right of centre."""
        y = self.y_bottom if y_eval is None else y_eval
        centre = (left.x_at(y) + right.x_at(y)) / 2.0
        return float((self.camera_x - centre) * self.xm)

    def lane_widths(self, left: LaneFit, right: LaneFit, n: int = 10) -> np.ndarray:
        """Lane width in metres at ``n`` rows spread over the warped height."""
        ys = np.linspace(0, self.height - 1, n)
        return (right.x_at(ys) - left.x_at(ys)) * self.xm
