import cv2
import numpy as np
import pytest

from lanedet.camera import Undistorter, calibrate_from_chessboard
from lanedet.config import CameraConfig, Config
from lanedet.pipeline import LanePipeline


def _chessboard_views(pattern=(7, 5), square=40, n=6):
    """Render a chessboard and view it through a few homographies."""
    cols, rows = pattern[0] + 1, pattern[1] + 1
    board = np.full((rows * square + 80, cols * square + 80), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = 40 + r * square, 40 + c * square
                board[y : y + square, x : x + square] = 0
    h, w = board.shape
    rng = np.random.default_rng(0)
    views = []
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    for _ in range(n):
        jitter = rng.uniform(-40, 40, (4, 2)).astype(np.float32)
        dst = src * 1.2 + 60 + jitter
        M = cv2.getPerspectiveTransform(src, dst)
        views.append(cv2.warpPerspective(board, M, (int(w * 1.5), int(h * 1.5)), borderValue=255))
    return views


def test_calibrate_from_synthetic_chessboard():
    matrix, dist, rms = calibrate_from_chessboard(_chessboard_views(), (7, 5))
    assert matrix.shape == (3, 3) and dist.ndim == 1
    # Random homographies are not views from one real camera, so only check it ran.
    assert np.isfinite(rms) and np.all(np.isfinite(matrix))


def test_calibrate_without_corners_fails():
    with pytest.raises(ValueError):
        calibrate_from_chessboard([np.zeros((100, 100), np.uint8)])


def test_identity_undistort_is_noop():
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    cam = CameraConfig(matrix=[[100, 0, 40], [0, 100, 30], [0, 0, 1]], dist=[0, 0, 0, 0, 0])
    und = Undistorter.from_config(cam)
    np.testing.assert_array_equal(und(img), img)
    assert Undistorter.from_config(CameraConfig()) is None


def test_pipeline_uses_undistortion(straight_frame):
    cfg = Config()
    cfg.camera = CameraConfig(matrix=[[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dist=[0] * 5)
    pipe = LanePipeline(cfg)
    assert pipe.undistort is not None
    assert pipe.process(straight_frame[0]).detected
