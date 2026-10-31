"""Colour + gradient thresholding that turns a frame into a binary lane mask.

Masks are ``uint8`` arrays with values 0 or 255.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from lanedet.config import PreprocessConfig, ROIConfig


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Return a 3-channel ``uint8`` BGR view of ``image`` (accepts gray, BGR or BGRA)."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("expected a non-empty image array")
    if image.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got {image.dtype}")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"unsupported image shape {image.shape}")


def _in_range(channel: np.ndarray, lo_hi: Sequence[int]) -> np.ndarray:
    lo, hi = lo_hi
    return (channel >= lo) & (channel <= hi)


def color_mask(
    bgr: np.ndarray,
    s_thresh: Sequence[int],
    l_thresh: Sequence[int],
    s_min_lightness: int = 0,
) -> np.ndarray:
    """Yellow paint via HLS saturation, white paint via HLS lightness."""
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    light, sat = hls[:, :, 1], hls[:, :, 2]
    yellow = _in_range(sat, s_thresh) & (light >= s_min_lightness)
    white = _in_range(light, l_thresh)
    return ((yellow | white) * 255).astype(np.uint8)


def gradient_mask(bgr: np.ndarray, thresh: Sequence[int], ksize: int = 3) -> np.ndarray:
    """Threshold the absolute x-gradient of the HLS lightness channel.

    The gradient is scaled by the kernel's largest possible response (not by the frame maximum), so a
    frame without lane edges does not get its noise stretched up to 255.
    """
    light = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)[:, :, 1]
    sobel = np.abs(cv2.Sobel(light, cv2.CV_64F, 1, 0, ksize=ksize))
    kx, ky = cv2.getDerivKernels(1, 0, ksize)
    max_response = 255.0 * (np.abs(kx).sum() / 2.0) * np.abs(ky).sum()
    scaled = np.clip(sobel * (255.0 / max_response), 0, 255)
    return (_in_range(scaled, thresh) * 255).astype(np.uint8)


def morphology(mask: np.ndarray, close: int = 0, open_: int = 0) -> np.ndarray:
    """Close (join split edges) then open (drop specks). Kernel size 0 skips a step."""
    out = mask
    if close > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    if open_ > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((open_, open_), np.uint8))
    return out


def roi_polygon(
    vertices_frac: Sequence[tuple[float, float]], width: int, height: int
) -> np.ndarray:
    """ROI vertices in pixels as an int32 ``(N, 2)`` array."""
    pts = np.array(vertices_frac, dtype=np.float64) * [width - 1, height - 1]
    return np.round(pts).astype(np.int32)


def apply_roi(mask: np.ndarray, vertices_frac: Sequence[tuple[float, float]]) -> np.ndarray:
    """Clear everything outside the ROI polygon."""
    h, w = mask.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    cv2.fillPoly(keep, [roi_polygon(vertices_frac, w, h)], 255)
    return cv2.bitwise_and(mask, mask, mask=keep)


class ThresholdMaskExtractor:
    """Classical mask stage: colour OR gradient, morphology, ROI.

    Any callable ``frame -> mask`` with the same contract can replace it in
    :class:`lanedet.pipeline.LanePipeline` (e.g. a CNN segmentation model).
    """

    def __init__(self, cfg: PreprocessConfig | None = None, roi: ROIConfig | None = None) -> None:
        self.cfg = cfg or PreprocessConfig()
        self.roi = roi or ROIConfig()

    def stages(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """All intermediate masks, for debugging and the demo notebook."""
        bgr = to_bgr(frame)
        c = self.cfg
        color = color_mask(bgr, c.s_thresh, c.l_thresh, c.s_min_lightness)
        grad = gradient_mask(bgr, c.sobel_thresh, c.sobel_kernel)
        combined = morphology(cv2.bitwise_or(color, grad), c.morph_close, c.morph_open)
        return {
            "color": color,
            "gradient": grad,
            "combined": combined,
            "mask": apply_roi(combined, self.roi.vertices),
        }

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.stages(frame)["mask"]
