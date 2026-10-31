"""Make an annotated demo video (and optionally a GIF for the README).

Examples:
    python scripts/make_demo_video.py --input my_dashcam.mp4 --output docs/demo.mp4 --debug
    python scripts/make_demo_video.py --output docs/demo.mp4 --gif docs/images/demo.gif  # synthetic
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from lanedet.config import load_config
from lanedet.io import FrameWriter, iter_frames, video_fps
from lanedet.overlay import debug_panel, draw_lanes
from lanedet.pipeline import LanePipeline
from lanedet.synthetic import drive


def frames(args: argparse.Namespace) -> Iterator[np.ndarray]:
    if args.input:
        for _, f in iter_frames(args.input):
            yield f
    else:
        n = int(args.seconds * args.fps)
        for f, _ in drive(n, seed=args.seed, scenario="mixed"):
            yield f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", help="video/folder; omit for a synthetic drive")
    ap.add_argument("--output", default="docs/demo.mp4")
    ap.add_argument("--config")
    ap.add_argument("--debug", action="store_true", help="add mask and bird's-eye panels")
    ap.add_argument("--seconds", type=float, default=30.0, help="synthetic length")
    ap.add_argument("--fps", type=float, default=25.0, help="synthetic frame rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gif", help="also write a small GIF here")
    ap.add_argument("--gif-frames", type=int, default=75)
    ap.add_argument("--gif-width", type=int, default=640)
    args = ap.parse_args()

    config = load_config(args.config)
    pipe = LanePipeline(config, debug=args.debug)
    fps = video_fps(args.input, args.fps) if args.input else args.fps
    gif_frames = []
    with FrameWriter(args.output, fps=fps) as writer:
        for frame in frames(args):
            r = pipe.process(frame)
            out = draw_lanes(frame, r, config.overlay)
            if args.debug:
                out = debug_panel(out, r)
            writer.write(out)
            if args.gif and len(gif_frames) < args.gif_frames and writer.count % 2 == 0:
                h = int(out.shape[0] * args.gif_width / out.shape[1])
                small = cv2.resize(out, (args.gif_width, h), interpolation=cv2.INTER_AREA)
                gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        count = writer.count
    print(f"wrote {count} frames to {args.output}")

    if args.gif and gif_frames:
        from PIL import Image  # installed with matplotlib

        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        imgs = [Image.fromarray(f) for f in gif_frames]
        imgs[0].save(
            args.gif, save_all=True, append_images=imgs[1:], duration=int(2000 / fps), loop=0
        )
        print(f"wrote {args.gif}")


if __name__ == "__main__":
    main()
