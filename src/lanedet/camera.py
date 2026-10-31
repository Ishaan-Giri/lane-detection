"""Optional lens undistortion and chessboard calibration."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from lanedet.config import CameraConfig


class Undistorter:
    """Removes lens distortion using a known camera matrix and distortion coefficients."""

    def __init__(self, matrix: np.ndarray, dist: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()

    @classmethod
    def from_config(cls, cfg: CameraConfig) -> Undistorter | None:
        """Return an undistorter, or ``None`` when the config has no calibration."""
        if cfg.matrix is None or cfg.dist is None:
            return None
        return cls(np.array(cfg.matrix), np.array(cfg.dist))

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.undistort(image, self.matrix, self.dist)


def calibrate_from_chessboard(
    images: Iterable[np.ndarray], pattern: tuple[int, int] = (9, 6)
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calibrate from chessboard photos.

    Args:
        images: BGR or grayscale images of a chessboard.
        pattern: inner-corner count (columns, rows).

    Returns:
        (camera matrix, distortion coefficients, RMS reprojection error).
    """
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    obj_points, img_points = [], []
    size = None
    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        found, corners = cv2.findChessboardCorners(gray, pattern)
        if found:
            obj_points.append(obj)
            img_points.append(corners)
    if not obj_points or size is None:
        raise ValueError("no chessboard corners found in any image")
    rms, matrix, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    return matrix, dist.ravel(), float(rms)
