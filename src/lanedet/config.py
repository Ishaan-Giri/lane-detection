"""Pipeline configuration: typed dataclasses loaded from a YAML file.

All image-space points (ROI, perspective) are fractions of the frame width/height, so one config
works across resolutions. Window sizes are in warped-image pixels and are tuned for 1280x720.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Point = tuple[float, float]
Range = tuple[int, int]


@dataclass
class CameraConfig:
    """Optional camera calibration. Leave both as ``None`` to skip undistortion."""

    matrix: list[list[float]] | None = None  # 3x3 intrinsic matrix
    dist: list[float] | None = None  # distortion coefficients (k1, k2, p1, p2[, k3])


@dataclass
class PreprocessConfig:
    """Colour and gradient thresholds that turn a frame into a binary lane mask."""

    s_thresh: Range = (100, 255)  # HLS saturation: yellow paint
    s_min_lightness: int = 70  # ignore saturated-but-dark pixels (shadows are noisy in S)
    l_thresh: Range = (200, 255)  # HLS lightness: white paint
    sobel_thresh: Range = (40, 255)  # |d/dx| of L, scaled to 0..255 by the kernel's max response
    sobel_kernel: int = 3
    morph_close: int = 3  # kernel size, 0 disables
    morph_open: int = 0  # kernel size, 0 disables


@dataclass
class ROIConfig:
    """Polygon (fractions of width, height) outside which the mask is cleared."""

    vertices: list[Point] = field(
        default_factory=lambda: [(0.05, 1.0), (0.40, 0.52), (0.60, 0.52), (0.95, 1.0)]
    )


@dataclass
class PerspectiveConfig:
    """Bird's-eye-view mapping. Order: top-left, top-right, bottom-right, bottom-left.

    The bottom ``src`` corners should sit on the ego lane lines on a straight, flat road, so that the
    lane is ``dst`` bottom-right minus bottom-left wide in the warped image.
    """

    src: list[Point] = field(
        default_factory=lambda: [(0.425, 0.55), (0.575, 0.55), (0.80, 1.0), (0.20, 1.0)]
    )
    dst: list[Point] = field(
        default_factory=lambda: [(0.25, 0.0), (0.75, 0.0), (0.75, 1.0), (0.25, 1.0)]
    )
    # Image row (fraction of height) that reported lane lines are extended up to, as straight lines
    # through their far end. ``None`` stops them at the top of the trapezoid (or ROI, for Hough).
    extend_to: float | None = None


@dataclass
class WindowConfig:
    """Sliding-window search parameters (warped-image pixels)."""

    n_windows: int = 9
    margin: int = 100  # half-width of each window
    min_pix: int = 50  # pixels needed to recentre the next window
    search_margin: int = 80  # half-width of the band around the previous fit
    min_fit_pixels: int = 150  # fewer pixels than this -> no fit for that lane


@dataclass
class GeometryConfig:
    """Pixel-to-metre assumptions for curvature and offset."""

    lane_width_m: float = 3.7  # US highway lane width
    view_length_m: float = 30.0  # road length covered by the warped image height


@dataclass
class TrackerConfig:
    """Temporal smoothing, sanity checks and fallback."""

    enabled: bool = True
    n_history: int = 5
    method: str = "mean"  # "mean" over history or "ema"
    ema_alpha: float = 0.3
    min_lane_width_m: float = 2.8
    max_lane_width_m: float = 4.8
    max_width_std_m: float = 0.5  # lines must be roughly parallel
    max_jump_m: float = 1.0  # max lateral change vs the smoothed fit, per frame
    max_failures: int = 5  # consecutive rejections before the history is reset


@dataclass
class HoughConfig:
    """Parameters for the Hough-line baseline."""

    blur_kernel: int = 5
    canny: Range = (50, 150)
    rho: float = 2.0
    theta_deg: float = 1.0
    threshold: int = 30
    min_line_length: int = 20
    max_line_gap: int = 100
    min_abs_slope: float = 0.4  # |dy/dx|; flatter segments are ignored


@dataclass
class OverlayConfig:
    """Drawing options."""

    fill_color: tuple[int, int, int] = (0, 200, 0)  # BGR
    line_color: tuple[int, int, int] = (0, 0, 255)
    alpha: float = 0.35
    show_text: bool = True


@dataclass
class Config:
    """Top-level configuration."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)

    def validate(self) -> Config:
        """Raise ``ValueError`` if any value is out of range. Returns ``self`` for chaining."""
        p = self.preprocess
        for name in ("s_thresh", "l_thresh", "sobel_thresh"):
            lo, hi = getattr(p, name)
            if not 0 <= lo <= hi <= 255:
                raise ValueError(f"preprocess.{name} must satisfy 0 <= lo <= hi <= 255")
        if p.sobel_kernel not in (1, 3, 5, 7):
            raise ValueError("preprocess.sobel_kernel must be 1, 3, 5 or 7")
        if p.morph_close < 0 or p.morph_open < 0:
            raise ValueError("preprocess morphology kernel sizes must be >= 0")

        if len(self.roi.vertices) < 3:
            raise ValueError("roi.vertices needs at least 3 points")
        for name in ("src", "dst"):
            pts = getattr(self.perspective, name)
            if len(pts) != 4:
                raise ValueError(f"perspective.{name} needs exactly 4 points")
        for x, y in [*self.roi.vertices, *self.perspective.src, *self.perspective.dst]:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError("ROI and perspective points are fractions and must be in [0, 1]")
        ext = self.perspective.extend_to
        if ext is not None and not (isinstance(ext, (int, float)) and 0.0 <= ext < 1.0):
            raise ValueError("perspective.extend_to must be null or a fraction in [0, 1)")
        dst = self.perspective.dst
        if dst[2][0] <= dst[3][0]:
            raise ValueError("perspective.dst bottom-right x must be > bottom-left x")

        w = self.window
        if w.n_windows < 1 or w.margin < 1 or w.search_margin < 1:
            raise ValueError("window.n_windows, margin and search_margin must be >= 1")
        if w.min_pix < 0 or w.min_fit_pixels < 3:
            raise ValueError("window.min_pix must be >= 0 and min_fit_pixels >= 3")

        g = self.geometry
        if g.lane_width_m <= 0 or g.view_length_m <= 0:
            raise ValueError("geometry lengths must be positive")

        t = self.tracker
        if t.method not in ("mean", "ema"):
            raise ValueError("tracker.method must be 'mean' or 'ema'")
        if not 0.0 < t.ema_alpha <= 1.0:
            raise ValueError("tracker.ema_alpha must be in (0, 1]")
        if t.n_history < 1 or t.max_failures < 1:
            raise ValueError("tracker.n_history and max_failures must be >= 1")
        if not 0 < t.min_lane_width_m < t.max_lane_width_m:
            raise ValueError("tracker lane width range is invalid")

        h = self.hough
        if h.blur_kernel < 1 or h.blur_kernel % 2 == 0:
            raise ValueError("hough.blur_kernel must be a positive odd number")

        if not 0.0 <= self.overlay.alpha <= 1.0:
            raise ValueError("overlay.alpha must be in [0, 1]")

        cam = self.camera
        if (cam.matrix is None) != (cam.dist is None):
            raise ValueError("camera.matrix and camera.dist must be given together")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (tuples become lists), suitable for YAML/JSON."""
        return _to_plain(dataclasses.asdict(self))


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _coerce(default: Any, value: Any, where: str) -> Any:
    """Convert a YAML value to the shape of the field's default value."""
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(default):
            raise ValueError(f"{where} must be a list of {len(default)} values")
        return tuple(type(d)(v) for d, v in zip(default, value, strict=True))
    if isinstance(default, list) and default and isinstance(default[0], tuple):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be a list of points")
        try:
            return [(float(x), float(y)) for x, y in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where} must be a list of [x, y] points") from exc
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{where} must be true or false")
        return value
    if isinstance(default, (int, float)) and default is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where} must be a number")
        return type(default)(value)
    return value


def _build_section(cls: type, data: dict[str, Any], section: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"config section '{section}' must be a mapping")
    obj = cls()
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown key(s) in '{section}': {sorted(unknown)}")
    for key, value in data.items():
        setattr(obj, key, _coerce(getattr(obj, key), value, f"{section}.{key}"))
    return obj


def config_from_dict(data: dict[str, Any] | None) -> Config:
    """Build and validate a :class:`Config`, overriding defaults with ``data``."""
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    cfg = Config()
    sections = {f.name: f for f in dataclasses.fields(Config)}
    unknown = set(data) - set(sections)
    if unknown:
        raise ValueError(f"unknown config section(s): {sorted(unknown)}")
    for name, value in data.items():
        section_cls = type(getattr(cfg, name))
        setattr(cfg, name, _build_section(section_cls, value or {}, name))
    return cfg.validate()


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config. ``None`` returns the defaults."""
    if path is None:
        return Config().validate()
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return config_from_dict(data)
