import numpy as np
import pytest

from experiments.leakage import box_distance_m, near_footprints
from experiments.split import block_split

X0 = Y0 = 2**19


def test_box_distance_at_the_equator():
    a = np.array([[0, -10, 10, 0]], float)
    b = np.array([[110, -10, 120, 0], [0, -5, 5, 0]], float)
    np.testing.assert_allclose(box_distance_m(a, b), [[100, 0]], atol=1e-3)


def test_near_footprints():
    tiles = [(X0, Y0, 20)]
    plot = (0.0048, -0.0003, 0.0050, 0.0)  # about 500 m east of the EQ tile
    assert near_footprints(tiles, [plot], 1000.0).tolist() == [True]
    assert near_footprints(tiles, [plot], 300.0).tolist() == [False]
    assert near_footprints(tiles, [], 1000.0).tolist() == [False]


def test_near_footprints_no_tiles():
    assert near_footprints([], [(0.0, 0.0, 0.001, 0.001)], 1000.0).tolist() == []


def test_block_split_no_tiles():
    assert block_split([], block_size=2, val_ratio=0.2, seed=0) == []


def test_block_split_whole_blocks_and_zooms():
    tiles = [(x, y, 20) for x in range(X0, X0 + 16) for y in range(Y0, Y0 + 16)]
    tiles += [(x, y, 19) for x in range(X0 // 2, X0 // 2 + 8) for y in range(Y0 // 2, Y0 // 2 + 8)]
    s = block_split(tiles, block_size=2, val_ratio=0.2, seed=0, buffer_m=0.0)
    assert s == block_split(tiles, block_size=2, val_ratio=0.2, seed=0, buffer_m=0.0)
    ground = {}
    for (x, y, z), side in zip(tiles, s, strict=True):
        ground.setdefault((x >> (z - 19), y >> (z - 19)), set()).add(side)
    assert all(len(v) == 1 for v in ground.values())
    assert 0.15 <= s.count("val") / len(s) <= 0.3 and "drop" not in s


def test_block_split_buffer_drops_nearby_train():
    tiles = [(x, Y0, 20) for x in range(X0, X0 + 40)]  # a 1.5 km row of 38 m tiles on the equator
    s = block_split(tiles, block_size=4, val_ratio=0.1, seed=1, buffer_m=100.0)
    val_x = [x for (x, _, _), side in zip(tiles, s, strict=True) if side == "val"]
    assert val_x and "drop" in s
    for (x, _, _), side in zip(tiles, s, strict=True):
        gap = min(abs(x - vx) for vx in val_x)
        if side == "train":
            assert gap >= 4  # 3 tiles apart is 76 m < 100 m
        if side == "drop":
            assert 1 <= gap <= 3


def test_block_split_test_share():
    tiles = [(x, y, 20) for x in range(X0, X0 + 16) for y in range(Y0, Y0 + 16)]
    s = block_split(tiles, block_size=1, val_ratio=0.2, test_ratio=0.2, seed=2, buffer_m=0.0)
    assert 0.15 <= s.count("test") / len(s) <= 0.3 and 0.15 <= s.count("val") / len(s) <= 0.3


@pytest.mark.parametrize("kw", [{"val_ratio": 0.9}, {"val_ratio": 0.5, "test_ratio": 0.45}])
def test_block_split_train_keeps_a_block(kw):
    tiles = [(X0, Y0, 20), (X0 + 8, Y0, 20)]  # two blocks of one tile
    s = block_split(tiles, block_size=1, seed=0, buffer_m=0.0, **kw)
    assert "train" in s


@pytest.mark.parametrize("kw", [{"val_ratio": 0.6, "test_ratio": 0.5}, {"block_size": 0}, {"val_ratio": 0.0}])
def test_block_split_bad_arguments(kw):
    args = {"block_size": 2, "val_ratio": 0.2, "seed": 0} | kw
    with pytest.raises(ValueError):
        block_split([(X0, Y0, 20)], **args)
