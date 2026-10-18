import numpy as np
import pytest

from fair_trees.augment import colour_jitter, downscale, flip, rot90

BOXES = np.array([[5, 3, 20, 13], [30, 10, 58, 39]], np.float32)


def _img(h=40, w=60, c=4, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (h, w, c), dtype=np.uint8)


def _mask_box(mask):
    ys, xs = np.nonzero(mask)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


@pytest.mark.parametrize("seed", range(6))
def test_flip_twice_is_identity(seed):
    img = _img()
    once = flip(img, BOXES, np.random.default_rng(seed))
    twice = flip(*once, np.random.default_rng(seed))
    np.testing.assert_array_equal(twice[0], img)
    np.testing.assert_allclose(twice[1], BOXES)


@pytest.mark.parametrize("seed", range(6))
def test_rot90_four_times_is_identity(seed):
    out = (_img(), BOXES)
    for _ in range(4):
        out = rot90(*out, np.random.default_rng(seed))
    np.testing.assert_array_equal(out[0], _img())
    np.testing.assert_allclose(out[1], BOXES)


@pytest.mark.parametrize("fn", [flip, rot90])
@pytest.mark.parametrize("seed", range(8))
def test_boxes_follow_pixels(fn, seed):
    img = np.zeros((40, 60, 1), np.uint8)
    img[3:13, 5:20, 0] = 1
    out, boxes = fn(img, BOXES[:1], np.random.default_rng(seed))
    assert boxes.tolist()[0] == _mask_box(out[..., 0] == 1)
    h, w = out.shape[:2]
    assert (boxes >= 0).all() and (boxes[:, [0, 2]] <= w).all() and (boxes[:, [1, 3]] <= h).all()


def test_empty_boxes_pass_through():
    out, boxes = rot90(_img(), np.zeros((0, 4), np.float32), np.random.default_rng(1))
    assert boxes.shape == (0, 4) and out.ndim == 3


def test_downscale_keeps_shape_and_is_seeded():
    rgb = _img(c=3)
    a = downscale(rgb, np.random.default_rng(1))
    assert a.shape == rgb.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a, downscale(rgb, np.random.default_rng(1)))
    assert not np.array_equal(a, rgb)


def test_downscale_factor_one_is_identity():
    rgb = _img(c=3)
    np.testing.assert_array_equal(downscale(rgb, np.random.default_rng(0), (1.0, 1.0)), rgb)


def test_colour_jitter_seeded_and_zero_is_identity():
    rgb = _img(c=3)
    a = colour_jitter(rgb, np.random.default_rng(2))
    np.testing.assert_array_equal(a, colour_jitter(rgb, np.random.default_rng(2)))
    assert a.shape == rgb.shape and a.dtype == np.uint8 and not np.array_equal(a, rgb)
    same = colour_jitter(rgb, np.random.default_rng(2), brightness=0, contrast=0, saturation=0, hue=0)
    np.testing.assert_array_equal(same, rgb)
