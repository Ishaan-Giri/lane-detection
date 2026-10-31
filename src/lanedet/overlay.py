"""Drawing: lane overlay, text, and a debug panel showing intermediate stages."""

from __future__ import annotations

import math

import cv2
import numpy as np

from lanedet.config import OverlayConfig
from lanedet.lanefit import LaneFit, LanePixels
from lanedet.pipeline import FrameResult


def format_radius(radius: float | None) -> str:
    if radius is None:
        return "Radius: n/a"
    if not math.isfinite(radius) or radius > 5000:
        return "Radius: straight"
    return f"Radius: {radius:,.0f} m"


def format_offset(offset: float | None) -> str:
    if offset is None:
        return "Offset: n/a"
    side = "right" if offset > 0 else "left"
    return f"Offset: {abs(offset):.2f} m {side} of centre"


def draw_lanes(
    frame: np.ndarray, result: FrameResult, cfg: OverlayConfig | None = None
) -> np.ndarray:
    """Return a copy of ``frame`` with the lane area, lane lines and geometry text drawn on it."""
    cfg = cfg or OverlayConfig()
    out = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    if result.left_image is not None and result.right_image is not None:
        poly = np.vstack([result.left_image, result.right_image[::-1]])
        layer = out.copy()
        cv2.fillPoly(layer, [np.round(poly).astype(np.int32)], cfg.fill_color)
        out = cv2.addWeighted(layer, cfg.alpha, out, 1 - cfg.alpha, 0)
    for line in result.lanes_image:
        pts = np.round(line).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, cfg.line_color, max(2, w // 200), cv2.LINE_AA)
    if cfg.show_text:
        scale = max(w / 1280.0, 0.4)
        lines = [format_radius(result.curvature_m), format_offset(result.offset_m)]
        if not result.accepted and result.lanes_image:
            lines.append(f"tracking ({result.reason})")
        for i, text in enumerate(lines):
            org = (int(20 * scale), int((45 + 40 * i) * scale))
            cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(
                out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA
            )
    return out


def draw_search(
    warped_mask: np.ndarray,
    pixels: LanePixels | None,
    left: LaneFit | None,
    right: LaneFit | None,
) -> np.ndarray:
    """Bird's-eye debug image: lane pixels (red/blue), search windows (green), fits (yellow)."""
    out = cv2.cvtColor(warped_mask, cv2.COLOR_GRAY2BGR) // 3
    h, w = warped_mask.shape[:2]
    if pixels is not None:
        out[pixels.left_y, pixels.left_x] = (0, 0, 255)
        out[pixels.right_y, pixels.right_x] = (255, 80, 0)
        for x0, y0, x1, y1 in pixels.windows:
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)
    ys = np.linspace(0, h - 1, 50)
    for fit in (left, right):
        if fit is not None:
            pts = np.column_stack([fit.x_at(ys), ys]).round().astype(np.int32)
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 255, 255), 3)
    return out


def debug_panel(overlay: np.ndarray, result: FrameResult) -> np.ndarray:
    """Overlay on the left, mask and bird's-eye search stacked on the right (same total height)."""
    h, w = overlay.shape[:2]
    tiles = []
    mask = result.debug.get("mask")
    if mask is not None:
        tiles.append(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    warped = result.debug.get("warped")
    if warped is not None:
        tiles.append(draw_search(warped, result.pixels, result.left, result.right))
    if not tiles:
        return overlay
    th, tw = h // len(tiles), w // len(tiles)
    side = np.vstack([cv2.resize(t, (tw, th)) for t in tiles])
    if side.shape[0] < h:
        pad = np.zeros((h - side.shape[0], tw, 3), np.uint8)
        side = np.vstack([side, pad])
    return np.hstack([overlay, side])
