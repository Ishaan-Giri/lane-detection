"""Tune on TuSimple training clips, then evaluate once on the TuSimple test set.

Written to run as a Kaggle notebook with the TuSimple dataset attached (nothing is downloaded), but it
works on any machine where the dataset is laid out the same way:

    <input>/…/train_set/label_data_*.json + clips/…      (training labels and frames)
    <input>/…/test_label.json, test_set/clips/…          (test labels and frames)

Steps:
  1. Calibrate: fit a straight line to the near half of each ego lane in *every* training label and
     take the median position. That gives the perspective trapezoid (src points on the ego lane lines)
     and the ROI, for a few candidate horizon rows.
  2. Tune: score the per-frame pipeline on a fixed random subset of training frames for each
     candidate (horizon row x threshold preset). Keep the best by accuracy, then lower FP + FN.
  3. Test: run every variant once on the full test set with the chosen config. For reference,
     also run the per-frame pipeline with the untuned default config.

`scripts/kaggle_push.py` bundles the lanedet package into this script and pushes it to Kaggle.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from lanedet.config import Config, config_from_dict
from lanedet.evaluation import VARIANTS, evaluate, markdown_table
from lanedet.io import load_tusimple_labels, read_image, write_tusimple_labels
from lanedet.metrics import sample_lane_x, score_frame, select_ego_lanes
from lanedet.overlay import draw_lanes
from lanedet.pipeline import LanePipeline

HORIZONS = (0.42, 0.45, 0.50, 0.55)  # candidate top rows of the trapezoid, fraction of height
EXTEND_PERCENTILE = 5  # lane lines are extended up to this percentile of the labelled lane tops
N_EXAMPLES = 6  # random test frames saved as typical (not worst-case) examples
PRESETS = {
    "default": {},
    "strict": {"l_thresh": (220, 255), "sobel_thresh": (50, 255)},
    "loose": {"l_thresh": (180, 255), "sobel_thresh": (30, 255)},
}
ROI_PAD_X = (0.04, 0.12)  # ROI widening beyond the trapezoid at (top, bottom), fraction of width
ROI_PAD_TOP = 0.03  # ROI starts this much above the trapezoid top


def locate(base: Path) -> tuple[Path, list[Path]]:
    """Find test_label.json and the training label files, without walking into clips/."""
    test_label, train_labels = None, []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "clips"]
        for f in filenames:
            if f == "test_label.json":
                test_label = Path(dirpath) / f
            elif f.startswith("label_data_") and f.endswith(".json"):
                train_labels.append(Path(dirpath) / f)
    if test_label is None or not train_labels:
        raise FileNotFoundError(f"TuSimple labels not found under {base}")
    return test_label, sorted(train_labels)


def image_root(label_file: Path, raw_file: str) -> Path:
    """The directory that a label file's ``raw_file`` paths are relative to."""
    here = label_file.parent
    candidates = [here, here.parent, *sorted(p for p in here.iterdir() if p.is_dir())]
    for c in candidates:
        if (c / raw_file).exists():
            return c
    raise FileNotFoundError(f"cannot find {raw_file} near {label_file}")


def _near_line(lane, h_samples, y_min: float):
    """Straight-line fit x = k*y + c to the lane's valid points below ``y_min``."""
    pts = [(x, y) for x, y in zip(lane, h_samples, strict=True) if x >= 0 and y >= y_min]
    if len(pts) < 5:
        return None
    xs, ys = np.asarray(pts, dtype=np.float64).T
    return np.polyfit(ys, xs, 1)


def lane_top_row(samples, w: int, h: int) -> float:
    """Low percentile of the topmost labelled row of the ego lanes, as a fraction of height."""
    tops = []
    for s in samples:
        for lane in select_ego_lanes(s.lanes, s.h_samples, w):
            ys = [y for x, y in zip(lane, s.h_samples, strict=True) if x >= 0]
            if ys:
                tops.append(min(ys))
    return round(float(np.percentile(tops, EXTEND_PERCENTILE)) / (h - 1), 4)


def save_examples(root, label_file, cfg: Config, out: Path, n: int, seed: int) -> list[dict]:
    """Overlay the per-frame pipeline on ``n`` random test frames (magenta dots = ground truth)."""
    samples = load_tusimple_labels([label_file])
    pick = random.Random(seed).sample(range(len(samples)), min(n, len(samples)))
    out.mkdir(parents=True, exist_ok=True)
    pipe = LanePipeline(cfg, use_tracker=False)
    rows = []
    for i, idx in enumerate(sorted(pick), start=1):
        s = samples[idx]
        frame = read_image(Path(root) / s.raw_file)
        res = pipe.process(frame)
        w = frame.shape[1]
        gt = select_ego_lanes(s.lanes, s.h_samples, w)
        pred = [list(sample_lane_x(lane, s.h_samples, w)) for lane in res.lanes_image]
        fs = score_frame(pred, gt)
        img = draw_lanes(frame, res, cfg.overlay)
        for lane in gt:
            for x, y in zip(lane, s.h_samples, strict=True):
                if x >= 0:
                    cv2.circle(img, (int(round(x)), int(y)), 4, (255, 0, 255), -1)
        name = f"example_{i:02d}.jpg"
        cv2.imwrite(str(out / name), img)
        rows.append({"file": name, "raw_file": s.raw_file, "accuracy": fs.accuracy})
    return rows


def median_ego_lines(samples, w: int, h: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Median (k, c) of the left and right ego-lane lines over all labelled frames."""
    lefts, rights = [], []
    for s in samples:
        ego = select_ego_lanes(s.lanes, s.h_samples, w)
        if len(ego) != 2:
            continue
        lf = _near_line(ego[0], s.h_samples, 0.5 * h)
        rf = _near_line(ego[1], s.h_samples, 0.5 * h)
        if lf is not None and rf is not None:
            lefts.append(lf)
            rights.append(rf)
    if not lefts:
        raise ValueError("no frame with two usable ego lanes")
    return np.median(lefts, axis=0), np.median(rights, axis=0), len(lefts)


def trapezoid(left, right, w: int, h: int, top: float) -> tuple[list, list]:
    """Perspective src points on the median ego lines, and an ROI around them (fractions)."""
    y_top = top * (h - 1)
    y_bot = float(h - 1)
    # If a line leaves the frame before the bottom row, stop the trapezoid where it does.
    for (k, c), edge in ((left, 0.0), (right, w - 1.0)):
        x_bot = k * y_bot + c
        if (edge == 0.0 and x_bot < 0) or (edge > 0 and x_bot > edge):
            y_bot = min(y_bot, (edge - c) / k)

    def x_at(line, y):
        return float(np.polyval(line, y))

    fx = lambda x: min(max(x / w, 0.0), 1.0)  # noqa: E731
    fy = lambda y: min(max(y / (h - 1), 0.0), 1.0)  # noqa: E731
    lt, rt = x_at(left, y_top), x_at(right, y_top)
    lb, rb = x_at(left, y_bot), x_at(right, y_bot)
    src = [(fx(lt), fy(y_top)), (fx(rt), fy(y_top)), (fx(rb), fy(y_bot)), (fx(lb), fy(y_bot))]
    pt, pb = ROI_PAD_X
    roi = [
        (max(fx(lb) - pb, 0.0), 1.0),
        (max(fx(lt) - pt, 0.0), max(top - ROI_PAD_TOP, 0.0)),
        (min(fx(rt) + pt, 1.0), max(top - ROI_PAD_TOP, 0.0)),
        (min(fx(rb) + pb, 1.0), 1.0),
    ]
    rnd = lambda pts: [(round(x, 4), round(y, 4)) for x, y in pts]  # noqa: E731
    return rnd(src), rnd(roi)


def make_config(src, roi, preset: dict, extend_to: float | None) -> Config:
    d = Config().to_dict()
    d["perspective"]["extend_to"] = extend_to
    d["perspective"]["src"] = [list(p) for p in src]
    d["roi"]["vertices"] = [list(p) for p in roi]
    d["preprocess"].update({k: list(v) for k, v in preset.items()})
    return config_from_dict(d)


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", default="/kaggle/input", help="where the dataset is mounted")
    ap.add_argument("--output", default="/kaggle/working", help="where results are written")
    ap.add_argument("--dev-frames", type=int, default=400, help="training frames used for tuning")
    ap.add_argument(
        "--test-limit", type=int, help="score only the first N test frames (smoke test)"
    )
    ap.add_argument("--save-worst", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args(argv)  # Kaggle may pass extra arguments

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    test_label, train_labels = locate(Path(args.input))
    log(f"test labels: {test_label}")
    log(f"train labels: {[str(p) for p in train_labels]}")

    # 1. Calibrate on all training labels (label geometry only, no images).
    train = load_tusimple_labels(train_labels)
    train_root = image_root(train_labels[0], train[0].raw_file)
    w, h = read_image(train_root / train[0].raw_file).shape[1::-1]
    left, right, n_used = median_ego_lines(train, w, h)
    extend_to = lane_top_row(train, w, h)
    log(f"calibration: {len(train)} training labels, {n_used} with both ego lanes; frame {w}x{h}")
    log(f"lane lines extended up to row {extend_to:.3f} x height")

    # 2. Tune on a fixed random subset of training frames.
    dev = random.Random(args.seed).sample(train, min(args.dev_frames, len(train)))
    dev_file = out / "dev_labels.json"
    write_tusimple_labels(dev, dev_file)
    tuning = []
    for top in HORIZONS:
        src, roi = trapezoid(left, right, w, h, top)
        for name, preset in PRESETS.items():
            cfg = make_config(src, roi, preset, extend_to)
            r = evaluate(train_root, [dev_file], ["sliding"], cfg)["results"][0]
            row = {"horizon": top, "preset": name, "src": src, "roi": roi, **r}
            tuning.append(row)
            log(
                f"tune horizon={top:.2f} preset={name:8s} acc={r['accuracy']:.4f} "
                f"fp={r['fp_rate']:.3f} fn={r['fn_rate']:.3f}"
            )
    best = max(tuning, key=lambda r: (round(r["accuracy"], 4), -(r["fp_rate"] + r["fn_rate"])))
    log(f"chosen: horizon={best['horizon']} preset={best['preset']}")
    cfg = make_config(best["src"], best["roi"], PRESETS[best["preset"]], extend_to)
    (out / "tusimple.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    dev_file.unlink()

    # 3. Test set: once with the chosen config, and the per-frame pipeline with the default config.
    test_root = image_root(test_label, load_tusimple_labels([test_label])[0].raw_file)
    log(f"test set: {test_root}")
    t0 = time.time()
    report = evaluate(
        test_root,
        [test_label],
        VARIANTS,
        cfg,
        limit=args.test_limit,
        save_worst=args.save_worst,
        save_dir=out / "failures",
    )
    log(f"test evaluation took {time.time() - t0:.0f} s")
    print(markdown_table(report), flush=True)
    reference = evaluate(test_root, [test_label], ["sliding"], Config(), limit=args.test_limit)
    log("untuned default config, per-frame pipeline:")
    print(markdown_table(reference), flush=True)

    report.update(
        {
            "config": cfg.to_dict(),
            "tuning": {
                "dev_frames": len(dev),
                "seed": args.seed,
                "calibration_frames": n_used,
                "extend_to": extend_to,
                "train_label_files": [p.name for p in train_labels],
                "chosen": {"horizon": best["horizon"], "preset": best["preset"]},
                "presets": {k: {kk: list(vv) for kk, vv in v.items()} for k, v in PRESETS.items()},
                "candidates": tuning,
            },
            "untuned_default_sliding": reference["results"][0],
            "machine": {
                "cpu": cpu_name(),
                "cpus": os.cpu_count(),
                "python": platform.python_version(),
                "opencv": __import__("cv2").__version__,
                "numpy": np.__version__,
                "where": "Kaggle notebook" if Path("/kaggle").exists() else platform.node(),
            },
        }
    )
    report["examples"] = save_examples(
        test_root, test_label, cfg, out / "examples", N_EXAMPLES, args.seed
    )
    (out / "tusimple.json").write_text(json.dumps(report, indent=2))
    log(f"wrote {out / 'tusimple.json'}")
    return report


if __name__ == "__main__":
    main()
