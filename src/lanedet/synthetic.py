"""Synthetic road images with known lane geometry.

Lanes are defined in the bird's-eye (warped) image as ``x = A*y**2 + B*y + C`` and painted there, then
projected into the camera view with the inverse perspective transform. Because the ground truth is
exact, these images are used for unit tests, for a TuSimple-format sanity-check dataset, and for the
demo when no real footage is available. They are much easier than real roads: numbers measured on
them say the code works, not that the detector is good.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lanedet.config import Config
from lanedet.io import TuSimpleSample, write_tusimple_labels
from lanedet.metrics import sample_lane_x
from lanedet.perspective import PerspectiveTransform

YELLOW = (0, 200, 235)  # BGR
WHITE = (235, 235, 235)
SCENARIOS = ("straight", "curve", "shadow", "worn", "dropout")


def lane_coeffs(x_bottom: float, slope: float, curve: float, height: int) -> np.ndarray:
    """Coefficients (A, B, C) of a lane given its bottom x, bottom slope dx/dy and A."""
    y0 = height - 1.0
    a = curve
    b = slope - 2.0 * a * y0
    c = x_bottom - slope * y0 + a * y0**2
    return np.array([a, b, c])


def default_transform(size: tuple[int, int], config: Config | None = None) -> PerspectiveTransform:
    p = (config or Config()).perspective
    return PerspectiveTransform.from_fractions(p.src, p.dst, size)


def horizon_y(transform: PerspectiveTransform) -> float:
    """Row where the two trapezoid sides meet (the vanishing point)."""
    tl, tr, br, bl = transform.src.astype(np.float64)
    # Intersect line bl->tl with br->tr, parametrised by y.
    sl = (tl[0] - bl[0]) / (tl[1] - bl[1])
    sr = (tr[0] - br[0]) / (tr[1] - br[1])
    if abs(sl - sr) < 1e-9:
        return 0.0
    y = (br[0] - bl[0] + sl * bl[1] - sr * br[1]) / (sl - sr)
    return float(np.clip(y, 0, transform.size[1] - 1))


def h_samples_for(transform: PerspectiveTransform, step: int = 10) -> list[int]:
    """TuSimple-style label rows covering the warped region (top of trapezoid to the bottom)."""
    h = transform.size[1]
    top = int(np.ceil(transform.src[:, 1].min() / step) * step)
    return list(range(top, h - step + 1, step))


@dataclass
class RoadScene:
    """Parameters of one synthetic frame."""

    left: np.ndarray  # warped-image coefficients
    right: np.ndarray
    right_dashed: bool = True
    dash_phase: float = 0.0  # warped pixels, advances as the car moves
    contrast: float = 1.0  # paint opacity
    worn: float = 0.0  # fraction of paint blocks removed
    hide_right: bool = False
    shadow: bool = False
    noise: float = 5.0


def _paint_lane(
    canvas: np.ndarray,
    coeffs: np.ndarray,
    width_px: int,
    dashed: bool,
    phase: float,
    dash_px: float,
    period_px: float,
) -> None:
    h = canvas.shape[0]
    ys = np.arange(0, h, 2, dtype=np.float64)
    xs = np.polyval(coeffs, ys)
    pts = np.column_stack([xs, ys])
    if not dashed:
        cv2.polylines(canvas, [pts.round().astype(np.int32)], False, 255, width_px, cv2.LINE_AA)
        return
    on = ((ys + phase) % period_px) < dash_px
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        seg = pts[start:stop]
        if len(seg) >= 2:
            cv2.polylines(canvas, [seg.round().astype(np.int32)], False, 255, width_px, cv2.LINE_AA)


def render(
    scene: RoadScene,
    transform: PerspectiveTransform,
    rng: np.random.Generator,
    config: Config | None = None,
) -> np.ndarray:
    """Render a BGR road frame for ``scene``."""
    cfg = config or Config()
    w, h = transform.size
    hy = int(horizon_y(transform))

    img = np.empty((h, w, 3), np.float32)
    sky = np.linspace(0, 1, max(hy, 1))[:, None, None]
    img[:hy] = (1 - sky) * np.array([235, 205, 170]) + sky * np.array([210, 190, 165])
    blotches = cv2.resize(rng.normal(0, 6, (8, 16)).astype(np.float32), (w, h - hy))
    img[hy:] = (95.0 + blotches)[:, :, None]

    lane_px = (cfg.perspective.dst[2][0] - cfg.perspective.dst[3][0]) * w
    px_per_m_x = lane_px / cfg.geometry.lane_width_m
    px_per_m_y = h / cfg.geometry.view_length_m
    width_px = max(int(round(0.15 * px_per_m_x)), 2)
    dash_px, period_px = 3.0 * px_per_m_y, 12.0 * px_per_m_y  # 3 m dashes every 12 m

    for coeffs, color, dashed, hidden in (
        (scene.left, YELLOW, False, False),
        (scene.right, WHITE, scene.right_dashed, scene.hide_right),
    ):
        if hidden:
            continue
        canvas8 = np.zeros((h, w), np.uint8)
        _paint_lane(canvas8, coeffs, width_px, dashed, scene.dash_phase, dash_px, period_px)
        canvas = canvas8.astype(np.float32) / 255.0
        if scene.worn > 0:
            blocks = (rng.random((24, 1)) >= scene.worn).astype(np.float32)
            canvas *= cv2.resize(blocks, (w, h), interpolation=cv2.INTER_NEAREST)
        alpha = transform.unwarp(canvas)[:, :, None] * scene.contrast
        img = img * (1 - alpha) + np.array(color, np.float32) * alpha

    if scene.shadow:
        shade = np.zeros((h, w), np.float32)
        cx = rng.uniform(0.2, 0.8) * w
        cy = rng.uniform(hy + 0.15 * (h - hy), h * 0.95)
        axes = (int(rng.uniform(0.15, 0.3) * w), int(rng.uniform(0.04, 0.09) * h))
        cv2.ellipse(shade, (int(cx), int(cy)), axes, rng.uniform(-15, 15), 0, 360, 1.0, -1)
        shade = cv2.GaussianBlur(shade, (31, 31), 0)
        img *= (1 - 0.55 * shade)[:, :, None]

    img += rng.normal(0, scene.noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def ground_truth(
    scene: RoadScene, transform: PerspectiveTransform, h_samples: list[int]
) -> list[list[float]]:
    """TuSimple-style lanes (x at each row of ``h_samples``, ``-2`` where absent). Lanes are labelled
    continuously across dash gaps, as in TuSimple. A hidden lane is not labelled."""
    lanes = []
    for coeffs, hidden in ((scene.left, False), (scene.right, scene.hide_right)):
        if hidden:
            continue
        poly = transform.unwarp_curve(coeffs, n=240)
        xs = sample_lane_x(poly, h_samples, width=transform.size[0])
        lanes.append([round(float(x), 1) for x in xs])
    return lanes


def drive(
    n_frames: int,
    size: tuple[int, int] = (1280, 720),
    seed: int = 0,
    scenario: str = "curve",
    config: Config | None = None,
) -> Iterator[tuple[np.ndarray, RoadScene]]:
    """Yield ``(frame, scene)`` for a smooth synthetic drive.

    Scenarios: ``straight``, ``curve``, ``shadow`` (moving shadows), ``worn`` (faded, broken paint),
    ``dropout`` (the right line disappears for the last two frames), ``mixed`` (slowly varying curve
    with occasional shadows; used for the demo video).
    """
    if scenario not in (*SCENARIOS, "mixed"):
        raise ValueError(f"unknown scenario {scenario!r}")
    rng = np.random.default_rng(seed)
    transform = default_transform(size, config)
    w, h = size
    cfg = config or Config()
    lane_px = (cfg.perspective.dst[2][0] - cfg.perspective.dst[3][0]) * w
    left0 = cfg.perspective.dst[3][0] * w
    straight = scenario == "straight"
    curve0 = 0.0 if straight else rng.uniform(1.5e-4, 3.5e-4) * rng.choice([-1, 1])
    dcurve = 0.0 if straight else rng.uniform(-4e-6, 4e-6)
    shift0, dshift = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.6, 0.6)
    width = lane_px * rng.uniform(0.95, 1.05)
    speed = rng.uniform(20, 35)  # warped px per frame

    for t in range(n_frames):
        if scenario == "mixed":
            curve = 3e-4 * np.sin(2 * np.pi * t / 240.0)
            shift = 0.04 * w * np.sin(2 * np.pi * t / 400.0)
        else:
            curve, shift = curve0 + dcurve * t, shift0 + dshift * t
        xl = left0 + shift
        # Equal bottom slopes keep the lines parallel.
        scene = RoadScene(
            left=lane_coeffs(xl, 0.0, curve, h),
            right=lane_coeffs(xl + width, 0.0, curve, h),
            dash_phase=speed * t,
        )
        if scenario == "shadow" or (scenario == "mixed" and (t // 30) % 4 == 3):
            scene.shadow = True
        if scenario == "worn":
            scene.contrast, scene.worn = 0.45, 0.35
        if scenario == "dropout" and t >= n_frames - 2:
            scene.hide_right = True
        yield render(scene, transform, rng, cfg), scene


def make_synthetic_tusimple(
    out_dir: str | Path,
    n_clips: int = 25,
    frames_per_clip: int = 20,
    size: tuple[int, int] = (1280, 720),
    seed: int = 0,
    config: Config | None = None,
) -> Path:
    """Write a TuSimple-format dataset (``clips/<name>/<i>.jpg`` + ``label_synthetic.json``).

    As in TuSimple, only the last frame of each clip is labelled. Scenarios cycle through
    :data:`SCENARIOS`. Returns the label-file path.
    """
    out = Path(out_dir)
    transform = default_transform(size, config)
    rows = h_samples_for(transform)
    samples = []
    for k in range(n_clips):
        scenario = SCENARIOS[k % len(SCENARIOS)]
        clip_dir = out / "clips" / f"{scenario}_{k:03d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        last_scene = None
        for i, (frame, scene) in enumerate(
            drive(frames_per_clip, size, seed + k, scenario, config), start=1
        ):
            cv2.imwrite(str(clip_dir / f"{i}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            last_scene = scene
        assert last_scene is not None
        raw = f"clips/{clip_dir.name}/{frames_per_clip}.jpg"
        samples.append(TuSimpleSample(raw, ground_truth(last_scene, transform, rows), rows))
    label = out / "label_synthetic.json"
    write_tusimple_labels(samples, label)
    return label
