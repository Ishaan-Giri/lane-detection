import numpy as np
import pytest

from lanedet.perspective import PerspectiveTransform, extend_up


def test_round_trip_points(transform):
    rng = np.random.default_rng(0)
    pts = rng.uniform([300, 420], [1000, 719], size=(50, 2))
    back = transform.unwarp_points(transform.warp_points(pts))
    np.testing.assert_allclose(back, pts, atol=1e-6)


def test_src_maps_to_dst(transform):
    np.testing.assert_allclose(transform.warp_points(transform.src), transform.dst, atol=1e-3)


def test_image_round_trip(transform):
    img = np.zeros((720, 1280), np.uint8)
    img[500:700, 600:680] = 255
    back = transform.unwarp(transform.warp(img, nearest=True), nearest=True)
    inner = img[520:680, 610:670]
    assert (back[520:680, 610:670] == inner).mean() > 0.99


def test_straight_lanes_are_vertical_when_warped(transform):
    # Points on the trapezoid's left side should land on one warped column.
    tl, bl = transform.src[0], transform.src[3]
    side = np.linspace(tl, bl, 10)
    warped = transform.warp_points(side)
    assert np.ptp(warped[:, 0]) < 1e-3


def test_unwarp_curve_shape(transform):
    poly = transform.unwarp_curve(np.array([0, 0, 320.0]), n=30)
    assert poly.shape == (30, 2)
    assert poly[0, 1] < poly[-1, 1]  # top to bottom


def test_empty_points(transform):
    assert transform.warp_points(np.empty((0, 2))).shape == (0, 2)


def test_degenerate_points_rejected():
    pts = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)
    with pytest.raises(ValueError):
        PerspectiveTransform(pts, pts, (100, 100))


def test_bad_size_rejected():
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    with pytest.raises(ValueError):
        PerspectiveTransform(pts, pts, (0, 100))


def test_extend_up_continues_straight_line():
    ys = np.linspace(400, 719, 30)
    line = np.column_stack([0.5 * ys + 100, ys])
    out = extend_up(line, 250.0)
    assert out[0, 1] == pytest.approx(250.0)
    assert np.all(np.diff(out[:, 1]) > 0)
    np.testing.assert_allclose(out[:, 0], 0.5 * out[:, 1] + 100, atol=1e-6)


def test_extend_up_leaves_long_lines_alone():
    line = np.array([[10.0, 100.0], [20.0, 300.0], [30.0, 500.0]])
    np.testing.assert_array_equal(extend_up(line, 200.0), line)
    np.testing.assert_array_equal(extend_up(line[:1], 50.0), line[:1])
