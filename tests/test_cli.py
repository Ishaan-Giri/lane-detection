import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from lanedet.cli import main
from lanedet.config import Config, config_from_dict, load_config
from lanedet.io import FrameWriter, input_kind, iter_frames, video_fps

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_video(tmp_path, synthetic_video):
    path = tmp_path / "in.mp4"
    with FrameWriter(path, fps=10) as w:
        for f in synthetic_video[:6]:
            w.write(f)
    return path


def test_run_writes_video_and_json(tmp_path, tiny_video, capsys):
    out = tmp_path / "out.mp4"
    js = tmp_path / "frames.json"
    assert main(["run", "-i", str(tiny_video), "-o", str(out), "--json", str(js)]) == 0
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    cap.release()
    frames = json.loads(js.read_text())
    assert len(frames) == 6 and frames[-1]["offset_m"] is not None
    assert "6 frames" in capsys.readouterr().out


def test_run_on_image_and_folder(tmp_path, synthetic_video):
    img = tmp_path / "frame.jpg"
    cv2.imwrite(str(img), synthetic_video[0])
    assert main(["run", "-i", str(img), "-o", str(tmp_path / "o.png"), "--debug"]) == 0
    assert cv2.imread(str(tmp_path / "o.png")).shape[1] > 1280

    folder = tmp_path / "frames"
    folder.mkdir()
    for i, f in enumerate(synthetic_video[:3]):
        cv2.imwrite(str(folder / f"{i}.jpg"), f)
    out = tmp_path / "out_frames"
    assert main(["run", "-i", str(folder), "-o", str(out), "--variant", "hough"]) == 0
    assert len(list(out.glob("*.png"))) == 3


def test_run_missing_input(tmp_path, capsys):
    assert main(["run", "-i", str(tmp_path / "nope.mp4"), "-o", str(tmp_path / "x.mp4")]) == 2
    assert "not found" in capsys.readouterr().err


def test_synth_and_eval(tmp_path, capsys):
    data = tmp_path / "synth"
    assert main(["synth", "-o", str(data), "--clips", "5", "--frames", "4"]) == 0
    report_path = tmp_path / "results.json"
    args = [
        "eval",
        "-d",
        str(data),
        "-o",
        str(report_path),
        "--save-worst",
        "1",
        "--save-dir",
        str(tmp_path / "fail"),
    ]
    assert main(args) == 0
    report = json.loads(report_path.read_text())
    variants = {r["variant"]: r for r in report["results"]}
    assert set(variants) == {"hough", "sliding", "smoothed"}
    assert variants["sliding"]["accuracy"] > 0.8
    assert variants["smoothed"]["frames_processed"] == 20
    assert variants["sliding"]["frames_processed"] == 5
    assert len(list((tmp_path / "fail").glob("*.jpg"))) == 3
    assert "| Variant |" in capsys.readouterr().out


def test_default_yaml_matches_dataclass_defaults():
    assert load_config(ROOT / "configs" / "default.yaml").to_dict() == Config().to_dict()
    assert load_config(None).to_dict() == Config().to_dict()


@pytest.mark.parametrize(
    "bad",
    [
        {"nope": {}},
        {"preprocess": {"nope": 1}},
        {"preprocess": {"s_thresh": [10]}},
        {"preprocess": {"s_thresh": [200, 100]}},
        {"preprocess": {"sobel_kernel": 4}},
        {"perspective": {"src": [[0, 0], [1, 0], [1, 1]]}},
        {"roi": {"vertices": [[0, 0], [2, 0], [1, 1]]}},
        {"tracker": {"enabled": "yes"}},
        {"tracker": {"n_history": "five"}},
        {"camera": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
        [1, 2],
    ],
)
def test_bad_config_rejected(bad):
    with pytest.raises(ValueError):
        config_from_dict(bad)


def test_config_override():
    cfg = config_from_dict({"window": {"n_windows": 12}, "tracker": {"method": "ema"}})
    assert cfg.window.n_windows == 12 and cfg.tracker.method == "ema"
    assert cfg.window.margin == Config().window.margin


def test_io_helpers(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(ValueError):
        input_kind(tmp_path / "a.txt")
    assert video_fps(tmp_path / "a.jpg") == 25.0
    w = FrameWriter(tmp_path / "one.png")
    w.write(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(ValueError):
        w.write(np.zeros((4, 4, 3), np.uint8))
    assert [n for n, _ in iter_frames(tmp_path / "one.png")] == ["one.png"]
