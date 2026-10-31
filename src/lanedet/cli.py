"""Command-line interface: ``lanedet run``, ``lanedet eval`` and ``lanedet synth``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from lanedet.config import load_config
from lanedet.evaluation import VARIANTS, evaluate, make_detector, markdown_table
from lanedet.io import FrameWriter, iter_frames, video_fps
from lanedet.overlay import debug_panel, draw_lanes


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    detector = make_detector(args.variant, config)
    detector.debug = args.debug
    per_frame = []
    n, elapsed, detected = 0, 0.0, 0
    with FrameWriter(args.output, fps=video_fps(args.input)) as writer:
        for name, frame in iter_frames(args.input):
            if args.max_frames is not None and n >= args.max_frames:
                break
            t0 = time.perf_counter()
            result = detector.process(frame)
            elapsed += time.perf_counter() - t0
            out = draw_lanes(frame, result, config.overlay)
            if args.debug:
                out = debug_panel(out, result)
            writer.write(out, name)
            detected += result.detected
            per_frame.append({"frame": name, **result.to_dict()})
            n += 1
    if n == 0:
        print("no frames read from input", file=sys.stderr)
        return 1
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(per_frame, indent=1), encoding="utf-8")
    fps = n / elapsed if elapsed > 0 else 0.0
    print(f"{n} frames, both lanes on {detected}, {fps:.1f} FPS (detector only) -> {args.output}")
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    report = evaluate(
        args.dataset,
        label_files=args.labels,
        variants=variants,
        config=config,
        limit=args.limit,
        context_frames=args.context_frames,
        ego_only=not args.all_lanes,
        pixel_thresh=args.pixel_thresh,
        ratio=args.ratio,
        save_worst=args.save_worst,
        save_dir=args.save_dir,
    )
    print(markdown_table(report))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    from lanedet.synthetic import make_synthetic_tusimple

    config = load_config(args.config)
    label = make_synthetic_tusimple(
        args.output, args.clips, args.frames, (args.width, args.height), args.seed, config
    )
    print(f"wrote {args.clips} clips; labels in {label}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lanedet", description="Lane detection for road video.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="annotate an image, folder of images, or video")
    run.add_argument("--input", "-i", required=True, help="image, folder or video")
    run.add_argument("--output", "-o", required=True, help="video file, image file, or folder")
    run.add_argument("--config", "-c", help="YAML config (defaults if omitted)")
    run.add_argument("--variant", choices=VARIANTS, default="smoothed")
    run.add_argument("--debug", action="store_true", help="add mask and bird's-eye panels")
    run.add_argument("--json", help="also write per-frame coefficients/curvature/offset here")
    run.add_argument("--max-frames", type=int)
    run.set_defaults(func=_cmd_run)

    ev = sub.add_parser("eval", help="evaluate on a TuSimple-format dataset")
    ev.add_argument("--dataset", "-d", required=True, help="dataset root (contains clips/)")
    ev.add_argument("--labels", nargs="+", help="label .json files (default: auto-detect)")
    ev.add_argument("--config", "-c")
    ev.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    ev.add_argument("--limit", type=int, help="evaluate only the first N labelled frames")
    ev.add_argument("--context-frames", type=int, default=19, help="earlier frames for tracking")
    ev.add_argument("--all-lanes", action="store_true", help="score against every GT lane")
    ev.add_argument("--pixel-thresh", type=float, default=20.0)
    ev.add_argument("--ratio", type=float, default=0.85)
    ev.add_argument("--save-worst", type=int, default=0, help="save N worst frames per variant")
    ev.add_argument("--save-dir", default="results/failures")
    ev.add_argument("--output", "-o", help="write the JSON report here")
    ev.set_defaults(func=_cmd_eval)

    sy = sub.add_parser("synth", help="write a synthetic TuSimple-format dataset")
    sy.add_argument("--output", "-o", required=True)
    sy.add_argument("--clips", type=int, default=25)
    sy.add_argument("--frames", type=int, default=20)
    sy.add_argument("--width", type=int, default=1280)
    sy.add_argument("--height", type=int, default=720)
    sy.add_argument("--seed", type=int, default=0)
    sy.add_argument("--config", "-c")
    sy.set_defaults(func=_cmd_synth)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
