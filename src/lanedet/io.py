"""Image/video/folder readers and writers, and the TuSimple label format."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def natural_key(path: str | Path) -> list:
    """Sort key so that ``2.jpg`` comes before ``10.jpg``."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def read_image(path: str | Path) -> np.ndarray:
    """Read a BGR image or raise ``FileNotFoundError``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def list_images(folder: str | Path) -> list[Path]:
    return sorted(
        (p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS), key=natural_key
    )


def input_kind(path: str | Path) -> str:
    """``"image"``, ``"folder"`` or ``"video"``."""
    p = Path(path)
    if p.is_dir():
        return "folder"
    if not p.exists():
        raise FileNotFoundError(f"input not found: {path}")
    ext = p.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    raise ValueError(f"unsupported input type: {path}")


def iter_frames(path: str | Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(name, BGR frame)`` from an image, a folder of images, or a video file."""
    kind = input_kind(path)
    p = Path(path)
    if kind == "image":
        yield p.name, read_image(p)
    elif kind == "folder":
        for f in list_images(p):
            yield f.name, read_image(f)
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise OSError(f"cannot open video: {path}")
        try:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield f"frame_{i:06d}.png", frame
                i += 1
        finally:
            cap.release()


def video_fps(path: str | Path, default: float = 25.0) -> float:
    """Frame rate of a video file, or ``default`` for images/unknown."""
    if Path(path).suffix.lower() not in VIDEO_EXTS:
        return default
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps if fps and fps > 0 else default


class FrameWriter:
    """Writes frames to a video file (by extension), a single image, or a folder of images."""

    def __init__(self, output: str | Path, fps: float = 25.0) -> None:
        self.output = Path(output)
        self.fps = fps
        ext = self.output.suffix.lower()
        self.kind = "video" if ext in VIDEO_EXTS else "image" if ext in IMAGE_EXTS else "folder"
        self._writer: cv2.VideoWriter | None = None
        self.count = 0
        if self.kind == "folder":
            self.output.mkdir(parents=True, exist_ok=True)
        else:
            self.output.parent.mkdir(parents=True, exist_ok=True)

    def write(self, frame: np.ndarray, name: str | None = None) -> None:
        if self.kind == "video":
            if self._writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(str(self.output), fourcc, self.fps, (w, h))
                if not self._writer.isOpened():
                    raise OSError(f"cannot open video writer for {self.output}")
            self._writer.write(frame)
        elif self.kind == "image":
            if self.count > 0:
                raise ValueError("an image output path can hold only one frame; use a folder")
            cv2.imwrite(str(self.output), frame)
        else:
            stem = Path(name).stem if name else f"frame_{self.count:06d}"
            cv2.imwrite(str(self.output / f"{stem}.png"), frame)
        self.count += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> FrameWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class TuSimpleSample:
    """One labelled frame: lane x positions at fixed rows ``h_samples`` (``-2`` = no lane)."""

    raw_file: str
    lanes: list[list[float]]
    h_samples: list[int]

    def frame_paths(self, root: str | Path, context: int = 0) -> list[Path]:
        """The labelled frame, preceded by up to ``context`` earlier frames from the same clip."""
        target = Path(root) / self.raw_file
        if context <= 0:
            return [target]
        clip = list_images(target.parent)
        idx = clip.index(target) if target in clip else len(clip) - 1
        return clip[max(0, idx - context) : idx + 1]


def load_tusimple_labels(paths: Sequence[str | Path]) -> list[TuSimpleSample]:
    """Read TuSimple JSON-lines label files (one JSON object per line)."""
    samples = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    samples.append(TuSimpleSample(d["raw_file"], d["lanes"], d["h_samples"]))
    return samples


def write_tusimple_labels(samples: Sequence[TuSimpleSample], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            row = {"raw_file": s.raw_file, "lanes": s.lanes, "h_samples": s.h_samples}
            fh.write(json.dumps(row) + "\n")
