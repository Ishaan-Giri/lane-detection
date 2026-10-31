"""Run detector variants over a TuSimple-format dataset and report accuracy, FP, FN and FPS."""

from __future__ import annotations

import heapq
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from lanedet.config import Config
from lanedet.hough import HoughLaneDetector
from lanedet.io import load_tusimple_labels, read_image
from lanedet.metrics import aggregate, sample_lane_x, score_frame, select_ego_lanes
from lanedet.overlay import draw_lanes
from lanedet.pipeline import LanePipeline

VARIANTS = ("hough", "sliding", "smoothed")
VARIANT_LABELS = {
    "hough": "Hough-line baseline",
    "sliding": "Sliding-window polynomial (main)",
    "smoothed": "+ temporal smoothing",
}


def make_detector(variant: str, config: Config | None = None) -> HoughLaneDetector | LanePipeline:
    """``hough`` | ``sliding`` (per-frame, no tracker) | ``smoothed`` (tracker on)."""
    if variant == "hough":
        return HoughLaneDetector(config)
    if variant == "sliding":
        return LanePipeline(config, use_tracker=False)
    if variant == "smoothed":
        return LanePipeline(config, use_tracker=True)
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


@dataclass
class VariantResult:
    variant: str
    accuracy: float
    fp_rate: float
    fn_rate: float
    fps: float
    frames_scored: int
    frames_processed: int


def find_label_files(dataset: str | Path) -> list[Path]:
    """Label files in a TuSimple root: ``test_label.json`` if present, otherwise any ``*.json``."""
    root = Path(dataset)
    test = root / "test_label.json"
    if test.exists():
        return [test]
    files = sorted(root.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no label .json files in {root}")
    return files


def evaluate(
    dataset: str | Path,
    label_files: Sequence[str | Path] | None = None,
    variants: Sequence[str] = VARIANTS,
    config: Config | None = None,
    limit: int | None = None,
    context_frames: int = 19,
    ego_only: bool = True,
    pixel_thresh: float = 20.0,
    ratio: float = 0.85,
    save_worst: int = 0,
    save_dir: str | Path | None = None,
) -> dict:
    """Evaluate each variant on the same labelled frames.

    Temporal variants (``smoothed``) run on up to ``context_frames`` earlier frames of each clip before
    the labelled frame, and are reset between clips. FPS counts detector time only (no disk I/O),
    over every frame processed. With ``ego_only`` the ground truth is reduced to the two lane lines
    bordering the ego lane, because these detectors only look for those two.
    """
    root = Path(dataset)
    files = list(label_files) if label_files else find_label_files(root)
    samples = load_tusimple_labels(files)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        raise ValueError("no labelled samples to evaluate")

    results = []
    for variant in variants:
        det = make_detector(variant, config)
        temporal = variant == "smoothed"
        scores = []
        worst: list[tuple[float, int, np.ndarray]] = []
        elapsed, processed = 0.0, 0
        for idx, s in enumerate(samples):
            paths = s.frame_paths(root, context_frames if temporal else 0)
            det.reset()
            res = None
            frame = None
            for p in paths:
                frame = read_image(p)
                t0 = time.perf_counter()
                res = det.process(frame)
                elapsed += time.perf_counter() - t0
                processed += 1
            assert res is not None and frame is not None
            w = frame.shape[1]
            gt = select_ego_lanes(s.lanes, s.h_samples, w) if ego_only else s.lanes
            pred = [list(sample_lane_x(lane, s.h_samples, w)) for lane in res.lanes_image]
            fs = score_frame(pred, gt, pixel_thresh, ratio)
            scores.append(fs)
            if save_worst > 0 and save_dir is not None:
                img = _failure_image(frame, res, gt, s.h_samples, config)
                cv2.putText(
                    img,
                    f"{variant} acc={fs.accuracy:.2f} {s.raw_file}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                item = (-fs.accuracy, idx, img)
                if len(worst) < save_worst:
                    heapq.heappush(worst, item)
                elif item > worst[0]:
                    heapq.heapreplace(worst, item)
        summary = aggregate(scores)
        results.append(
            VariantResult(
                variant=variant,
                accuracy=summary.accuracy,
                fp_rate=summary.fp_rate,
                fn_rate=summary.fn_rate,
                fps=processed / elapsed if elapsed > 0 else 0.0,
                frames_scored=summary.n_frames,
                frames_processed=processed,
            )
        )
        if worst and save_dir is not None:
            out = Path(save_dir)
            out.mkdir(parents=True, exist_ok=True)
            for rank, (_, _, img) in enumerate(sorted(worst, reverse=True), start=1):
                cv2.imwrite(str(out / f"{variant}_worst_{rank:02d}.jpg"), img)

    return {
        "dataset": str(root),
        "label_files": [str(f) for f in files],
        "settings": {
            "frames": len(samples),
            "context_frames": context_frames,
            "ego_only": ego_only,
            "pixel_thresh": pixel_thresh,
            "ratio": ratio,
        },
        "results": [asdict(r) for r in results],
    }


def _failure_image(frame, res, gt, h_samples, config) -> np.ndarray:
    img = draw_lanes(frame, res, (config or Config()).overlay)
    for lane in gt:
        for x, y in zip(lane, h_samples, strict=True):
            if x >= 0:
                cv2.circle(img, (int(round(x)), int(y)), 4, (255, 0, 255), -1)
    return img


def markdown_table(report: dict) -> str:
    """Results as the comparison table used in ``docs/results.md``."""
    lines = [
        "| Variant | Accuracy | FP | FN | FPS |",
        "|---|---|---|---|---|",
    ]
    for r in report["results"]:
        label = VARIANT_LABELS.get(r["variant"], r["variant"])
        lines.append(
            f"| {label} | {r['accuracy'] * 100:.1f}% | {r['fp_rate']:.3f} | "
            f"{r['fn_rate']:.3f} | {r['fps']:.1f} |"
        )
    return "\n".join(lines)
