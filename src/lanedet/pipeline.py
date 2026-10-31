"""LanePipeline: ties preprocessing, warping, lane search, fitting, tracking and geometry together."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from lanedet.camera import Undistorter
from lanedet.config import Config
from lanedet.lanefit import (
    LaneFit,
    LaneGeometry,
    LanePixels,
    SlidingWindowFinder,
    fit_lanes,
    search_around_fits,
)
from lanedet.perspective import PerspectiveTransform, extend_up
from lanedet.preprocess import ThresholdMaskExtractor, to_bgr
from lanedet.tracker import LaneTracker

MaskExtractor = Callable[[np.ndarray], np.ndarray]
"""Any callable mapping a BGR frame to a 0/255 ``uint8`` lane mask of the same height and width."""


@dataclass
class FrameResult:
    """Everything the pipeline knows about one frame.

    ``left``/``right`` are warped-image fits (``None`` for the Hough baseline or a missing lane);
    ``left_image``/``right_image`` are ``(N, 2)`` polylines in camera-image pixels, used for drawing
    and evaluation by every detector.
    """

    left: LaneFit | None = None
    right: LaneFit | None = None
    left_image: np.ndarray | None = None
    right_image: np.ndarray | None = None
    curvature_m: float | None = None
    offset_m: float | None = None
    accepted: bool = False
    reason: str = ""
    search: str = ""  # "windows", "around_fit" or "hough"
    debug: dict[str, np.ndarray] = field(default_factory=dict)
    pixels: LanePixels | None = None

    @property
    def lanes_image(self) -> list[np.ndarray]:
        """Image-space polylines of the lanes that were found (left first)."""
        return [p for p in (self.left_image, self.right_image) if p is not None]

    @property
    def detected(self) -> bool:
        return self.left_image is not None and self.right_image is not None

    def to_dict(self) -> dict:
        """JSON-friendly summary (coefficients in warped pixels, metres for geometry)."""

        def coeffs(f: LaneFit | None) -> list[float] | None:
            return None if f is None else [float(c) for c in f.coeffs]

        curv = self.curvature_m
        return {
            "left_coeffs": coeffs(self.left),
            "right_coeffs": coeffs(self.right),
            "curvature_m": None if curv is None or not np.isfinite(curv) else float(curv),
            "straight": bool(curv is not None and not np.isfinite(curv)),
            "offset_m": self.offset_m,
            "accepted": self.accepted,
            "reason": self.reason,
            "search": self.search,
        }


class LanePipeline:
    """Classical lane detector: mask -> bird's-eye view -> sliding windows -> polyfit -> tracking.

    Args:
        config: pipeline configuration (defaults if ``None``).
        mask_extractor: replaces the thresholding stage (e.g. a CNN); must return a 0/255 mask.
        use_tracker: smooth over frames and apply sanity checks. ``False`` gives raw per-frame fits.
        debug: keep intermediate images in ``FrameResult.debug``.
    """

    def __init__(
        self,
        config: Config | None = None,
        mask_extractor: MaskExtractor | None = None,
        use_tracker: bool | None = None,
        debug: bool = False,
    ) -> None:
        self.config = (config or Config()).validate()
        self.mask_extractor: MaskExtractor = mask_extractor or ThresholdMaskExtractor(
            self.config.preprocess, self.config.roi
        )
        self.use_tracker = self.config.tracker.enabled if use_tracker is None else use_tracker
        self.debug = debug
        self.undistort = Undistorter.from_config(self.config.camera)
        self.finder = SlidingWindowFinder.from_config(self.config.window)
        self.size: tuple[int, int] | None = None
        self.transform: PerspectiveTransform | None = None
        self.geometry: LaneGeometry | None = None
        self.tracker: LaneTracker | None = None

    def _setup(self, size: tuple[int, int]) -> None:
        p = self.config.perspective
        self.size = size
        self.transform = PerspectiveTransform.from_fractions(p.src, p.dst, size)
        self.geometry = LaneGeometry.from_config(self.config.geometry, self.transform, p.dst)
        self.tracker = LaneTracker(self.geometry, self.config.tracker) if self.use_tracker else None

    def reset(self) -> None:
        """Forget temporal state (call between unrelated clips)."""
        if self.tracker is not None:
            self.tracker.reset()

    def _search(self, warped: np.ndarray) -> tuple[LanePixels, LaneFit | None, LaneFit | None]:
        min_fit = self.config.window.min_fit_pixels
        if self.tracker is not None and self.tracker.has_fit:
            prev = self.tracker.smoothed()
            assert prev is not None
            pixels = search_around_fits(warped, prev[0], prev[1], self.config.window.search_margin)
            left, right = fit_lanes(pixels, min_fit)
            if left is not None and right is not None:
                return pixels, left, right
        pixels = self.finder.find(warped)
        left, right = fit_lanes(pixels, min_fit)
        return pixels, left, right

    def _image_curve(self, fit: LaneFit, h: int) -> np.ndarray:
        """The fitted lane in camera-image pixels, extended upwards if configured."""
        assert self.transform is not None
        curve = self.transform.unwarp_curve(fit.coeffs)
        ext = self.config.perspective.extend_to
        return curve if ext is None else extend_up(curve, ext * (h - 1))

    def process(self, frame: np.ndarray) -> FrameResult:
        """Detect lanes in one BGR (or grayscale) ``uint8`` frame."""
        bgr = to_bgr(frame)
        h, w = bgr.shape[:2]
        if self.size != (w, h):
            self._setup((w, h))
        assert self.transform is not None and self.geometry is not None
        if self.undistort is not None:
            bgr = self.undistort(bgr)

        mask = self.mask_extractor(bgr)
        if mask.shape[:2] != (h, w):
            raise ValueError("mask extractor must return a mask the size of the frame")
        warped = self.transform.warp(mask, nearest=True)
        pixels, left, right = self._search(warped)

        accepted, reason = left is not None and right is not None, "raw"
        if self.tracker is not None:
            state = self.tracker.update(left, right)
            left, right, accepted, reason = state.left, state.right, state.accepted, state.reason

        result = FrameResult(
            left=left,
            right=right,
            left_image=None if left is None else self._image_curve(left, h),
            right_image=None if right is None else self._image_curve(right, h),
            curvature_m=self.geometry.combined_radius(left, right),
            offset_m=None if left is None or right is None else self.geometry.offset(left, right),
            accepted=accepted,
            reason=reason,
            search=pixels.method,
            pixels=pixels,
        )
        if self.debug:
            result.debug = {"mask": mask, "warped": warped}
        return result
