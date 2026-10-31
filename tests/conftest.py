"""Synthetic-lane fixtures: road frames with known lane polynomials (no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest

from lanedet.config import Config
from lanedet.synthetic import RoadScene, default_transform, drive, lane_coeffs, render

SIZE = (1280, 720)
W, H = SIZE
LEFT_X = 0.25 * W
RIGHT_X = 0.75 * W


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def transform():
    return default_transform(SIZE)


@pytest.fixture
def make_frame(transform):
    """Render a frame. Returns ``(frame, scene)``."""

    def _make(curve: float = 0.0, shift: float = 0.0, seed: int = 0, **scene_kwargs):
        scene = RoadScene(
            left=lane_coeffs(LEFT_X + shift, 0.0, curve, H),
            right=lane_coeffs(RIGHT_X + shift, 0.0, curve, H),
            **scene_kwargs,
        )
        frame = render(scene, transform, np.random.default_rng(seed))
        return frame, scene

    return _make


@pytest.fixture
def straight_frame(make_frame):
    return make_frame(curve=0.0)


@pytest.fixture
def curved_frame(make_frame):
    return make_frame(curve=2.5e-4)


@pytest.fixture
def synthetic_video():
    """A short list of frames from a curved synthetic drive."""
    return [f for f, _ in drive(12, SIZE, seed=3, scenario="curve")]
