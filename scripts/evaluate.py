"""Evaluate all detector variants and write results.json plus a markdown table.

Examples:
    # TuSimple test set (download separately; see README)
    python scripts/evaluate.py --dataset data/tusimple/test_set \
        --labels data/tusimple/test_label.json --output results/tusimple.json

    # Synthetic sanity check (generated on the fly, no download)
    python scripts/evaluate.py --synthetic --output results/synthetic.json
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

from lanedet.config import load_config
from lanedet.evaluation import VARIANTS, evaluate, markdown_table
from lanedet.synthetic import make_synthetic_tusimple


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="TuSimple-format root (contains clips/)")
    src.add_argument("--synthetic", action="store_true", help="generate and use synthetic data")
    ap.add_argument("--labels", nargs="+", help="label files (default: auto-detect)")
    ap.add_argument("--config", help="YAML config")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--clips", type=int, default=50, help="synthetic clips to generate")
    ap.add_argument("--seed", type=int, default=0, help="synthetic data seed")
    ap.add_argument("--save-worst", type=int, default=0)
    ap.add_argument("--save-dir", default="results/failures")
    ap.add_argument("--output", default="results/results.json")
    args = ap.parse_args()

    config = load_config(args.config)
    with tempfile.TemporaryDirectory() as tmp:
        dataset = args.dataset
        if args.synthetic:
            make_synthetic_tusimple(tmp, n_clips=args.clips, seed=args.seed, config=config)
            dataset = tmp
        report = evaluate(
            dataset,
            label_files=args.labels,
            variants=args.variants,
            config=config,
            limit=args.limit,
            save_worst=args.save_worst,
            save_dir=args.save_dir,
        )
    if args.synthetic:
        report["dataset"] = f"synthetic (clips={args.clips}, seed={args.seed})"
        report["label_files"] = []
    report["machine"] = {
        "cpu": cpu_name(),
        "python": platform.python_version(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(markdown_table(report))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
