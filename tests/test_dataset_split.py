import logging
from pathlib import Path

import pytest

from fair_trees.dataset import block_key, spatial_split

X0 = Y0 = 2**19


def _names(tiles):
    return [f"OAM-{x}-{y}-{z}.tif" for x, y, z in tiles]


def _grid(n, z=20, x0=X0, y0=Y0):
    return [(x0 + i, y0 + j, z) for j in range(n) for i in range(n)]


def test_block_key_shifts_to_reference_zoom():
    assert block_key(X0 + 5, Y0 + 9, 20, 19, 4) == ((X0 + 5) // 2 // 4, (Y0 + 9) // 2 // 4)
    assert block_key(X0 // 2, Y0 // 2, 19, 19, 4) == (X0 // 2 // 4, Y0 // 2 // 4)
    # a chip coarser than z_ref maps to the block of its north-west descendant
    assert block_key(X0 // 4, Y0 // 4, 18, 19, 4) == (X0 // 2 // 4, Y0 // 2 // 4)


def test_deterministic_disjoint_full_cover():
    names = _names(_grid(10))
    train, val = spatial_split(names, 0.2, 7)
    assert spatial_split(names, 0.2, 7) == (train, val)
    assert set(train).isdisjoint(val) and sorted(train + val) == sorted(names)
    assert 20 <= len(val) <= 36  # target round(0.2 * 100); whole blocks of at most 16 chips


def test_blocks_stay_whole():
    tiles = _grid(10)
    names = _names(tiles)
    _, val = spatial_split(names, 0.2, 3)
    val = set(val)
    sides = {}
    for n, (x, y, z) in zip(names, tiles, strict=True):
        sides.setdefault(block_key(x, y, z, 20, 4), set()).add(n in val)
    assert all(len(s) == 1 for s in sides.values())


def test_same_ground_across_zooms_on_one_side():
    tiles = []
    for x, y, _ in _grid(8):
        tiles.append((x, y, 20))
        tiles += [(2 * x + i, 2 * y + j, 21) for i in (0, 1) for j in (0, 1)]
    tiles += _grid(4, 19, X0 // 2, Y0 // 2)
    names = _names(tiles)
    _, val = spatial_split(names, 0.3, 1, block_size=2)
    val = set(val)
    sides = {}
    for n, (x, y, z) in zip(names, tiles, strict=True):
        sides.setdefault((x >> (z - 19), y >> (z - 19)), set()).add(n in val)
    assert all(len(s) == 1 for s in sides.values())
    assert val and len(val) < len(names)


def test_z_ref_defaults_to_lowest_zoom():
    names = _names(_grid(8, 20) + _grid(4, 19, X0 // 2, Y0 // 2))
    assert spatial_split(names, 0.3, 5) == spatial_split(names, 0.3, 5, z_ref=19)


def test_coarse_zooms_do_not_crash():
    names = _names([(X0 // 8, Y0 // 8, 17), (X0 // 4, Y0 // 4, 18)] + _grid(4))
    for z_ref in (None, 19):
        train, val = spatial_split(names, 0.25, 0, z_ref=z_ref)
        assert sorted(train + val) == sorted(names)


def test_unnamed_chips_fill_leftover_val():
    named = _names(_grid(3))  # one block: it cannot go to val without emptying train
    unnamed = [f"photo_{i}.tif" for i in range(5)]
    train, val = spatial_split(named + unnamed, 0.2, 0)
    assert len(val) == 3 and set(val) <= set(unnamed)  # round(0.2 * 14) = 3
    assert set(named) <= set(train)


def test_single_block_warns_and_keeps_train(caplog):
    with caplog.at_level(logging.WARNING, logger="fair_trees.dataset"):
        train, val = spatial_split(_names(_grid(2)), 0.5, 0)
    assert val == [] and len(train) == 4 and "val" in caplog.text


def test_returns_inputs_unchanged_in_input_order():
    paths = [Path("chips") / n for n in _names(_grid(4))]
    train, val = spatial_split(paths, 0.25, 0, block_size=1)
    assert all(isinstance(p, Path) for p in train + val)
    assert train == [p for p in paths if p in set(train)]
    assert val == [p for p in paths if p in set(val)]


@pytest.mark.parametrize("val_ratio, block_size", [(0.0, 4), (1.0, 4), (0.2, 0)])
def test_bad_arguments(val_ratio, block_size):
    with pytest.raises(ValueError):
        spatial_split(_names(_grid(2)), val_ratio, 0, block_size=block_size)
