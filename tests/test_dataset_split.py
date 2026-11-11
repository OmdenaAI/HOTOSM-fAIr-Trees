import logging
from itertools import combinations
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


def _share_ground(a, b):
    (xa, ya, za), (xb, yb, zb) = sorted((a, b), key=lambda t: t[2])
    return (xb >> (zb - za), yb >> (zb - za)) == (xa, ya)


def _coarse_and_z19(z):
    """A 2 x 2 of zoom-``z`` tiles at an odd offset, every z19 tile under them, and a few z19 tiles beside."""
    x0, y0 = (X0 >> (20 - z)) + 1, (Y0 >> (20 - z)) + 1
    coarse = [(x0 + i, y0 + j, z) for j in (0, 1) for i in (0, 1)]
    s = 1 << (19 - z)
    fine = [(x * s + i, y * s + j, 19) for x, y, _ in coarse for j in range(s) for i in range(s)]
    beside = [((x0 + 3) * s + i, y0 * s, 19) for i in range(3)]
    return coarse + fine + beside


def test_tile_coarser_than_z_ref_stays_with_the_tiles_it_contains():
    names = ["OAM-1-1-16.tif", "OAM-12-8-19.tif"]  # the z16 tile contains the z19 one
    train, val = spatial_split(names, 0.5, 0, z_ref=19)
    assert val == [] and train == names


@pytest.mark.parametrize("z", [17, 16])
@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
def test_tiles_coarser_than_z_ref_never_straddle(z, block_size):
    tiles = _coarse_and_z19(z)
    names = _names(tiles)
    held = 0
    for seed in range(20):
        _, val = spatial_split(names, 0.3, seed, block_size=block_size, z_ref=19)
        val = set(val)
        held += bool(val)
        side = [n in val for n in names]
        for i, j in combinations(range(len(tiles)), 2):
            if side[i] != side[j]:
                assert not _share_ground(tiles[i], tiles[j]), (seed, tiles[i], tiles[j])
    assert held


# Splits computed before tiles coarser than z_ref got their own handling. No tile here is coarser than
# z_ref, so these must never change: the fAIr data hash covers each chip's split. One character per chip,
# "v" for val and "." for train.
def _golden_mixed():
    t = [(X0 // 2 + i, Y0 // 2 + j, 19) for j in range(10) for i in range(10) if (i + 2 * j) % 5 == 0]
    t += [(X0 + i, Y0 + j, 20) for j in range(12) for i in range(12) if (3 * i + j) % 4 == 1]
    return t + [(2 * X0 + 3 + i, 2 * Y0 + 5 + j, 21) for j in range(6) for i in range(5)]


def _golden_single():
    return [(X0 + i, Y0 + j, 20) for j in range(9) for i in range(9) if (i * j) % 7 != 3]


_GOLDEN = {
    (
        "mixed",
        0,
        1,
        None,
    ): ".vv...............vvvvv...v.v.v..v.v........v..v...v.v..v....v....v....v....v....v....",
    (
        "mixed",
        0,
        2,
        None,
    ): "..v...vvv.v.....v..v.v..v..v..v...v..v..v..vvv.vv.vv.vv...............................",
    (
        "mixed",
        0,
        4,
        None,
    ): ".v.v.vvv.v.v...vv.vv..v..v..v..v..v..v..v..v..v..v..v..v..............................",
    (
        "mixed",
        0,
        4,
        19,
    ): ".v.v.vvv.v.v...vv.vv..v..v..v..v..v..v..v..v..v..v..v..v..............................",
    (
        "mixed",
        0,
        3,
        18,
    ): ".vv..vv.vv.vv..vv.vv.vv.vv.vv.vv.vv.vv.vv.vv.vv.vv.vv.vv..............................",
    (
        "mixed",
        1,
        1,
        None,
    ): "v....vv.........v...vv.v.v..vv...v..v.vvvvvv.v........v..vvvv.vvvv.vvvv...............",
    (
        "mixed",
        1,
        2,
        None,
    ): "....v......v....................v..v..v..v....v..v..v..v...............vvvvvvvvvvvvvvv",
    (
        "mixed",
        1,
        4,
        None,
    ): "v.v.v........v......vv.vv.vv.vv.vv.vv.vv.vv.............vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv",
    (
        "mixed",
        1,
        4,
        19,
    ): "v.v.v........v......vv.vv.vv.vv.vv.vv.vv.vv.............vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv",
    (
        "mixed",
        1,
        3,
        18,
    ): "v..vv.....v...v.....v..v..v..v..v..v..v..v..v..v..v..v..vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv",
    (
        "mixed",
        2,
        1,
        None,
    ): ".v.v.v....v.v.v.....v.v......v................v...v......vvvv.vvvv.vvvv...............",
    (
        "mixed",
        2,
        2,
        None,
    ): ".....v..vvvvvv.vv..v........................vvvvvvvvvvvv..............................",
    (
        "mixed",
        2,
        4,
        None,
    ): ".v.v.vvv.v.v.v.vvvvv..v..v..v..v..v..v..v..v..v..v..v..v..............................",
    (
        "mixed",
        2,
        4,
        19,
    ): ".v.v.vvv.v.v.v.vvvvv..v..v..v..v..v..v..v..v..v..v..v..v..............................",
    (
        "mixed",
        2,
        3,
        18,
    ): ".vvv.vvvvv.vvv.v.v...vv.vv.vv.vv.vv.vv.vv.vv.vv.vv.vv.vv..............................",
    ("single", 0, 1, None): "...................v.v.v..vv.v..v...vv.vv...v.v.................vvv..vv..",
    ("single", 0, 2, None): "....vv......vv.......v......vv..vv......vv.............v........vvv...vvv",
    ("single", 0, 4, None): "....................................vvv....vvvv.....vvv.....vvvv....vvvv.",
    ("single", 0, 3, 18): "....vvvvv...vvvvv....vvvv...vvvv.........................................",
    ("single", 1, 1, None): "..v...v.......v......v..v.v..........vv......v.vv.v.v....v..v.....v.vv...",
    ("single", 1, 2, None): "......vvv.....vvvvv.....vv........vv..v...v..vv........v........v........",
    ("single", 1, 4, None): "vvvv.....vvv.....vvvv....vvv...........v.......v.......v........v........",
    ("single", 1, 3, 18): "vvvv.....vvv.....vvvv....vvv....vvvv....vvv.....vvvv....vvvv.....vvv.....",
    ("single", 2, 1, None): "...vvv.v...vv...v.vv.................vv.v...v....v....v.v...v.v..........",
    ("single", 2, 2, None): "........v.......v.....vvv.....vv..vv......v..........vvv......vvv...vv...",
    ("single", 2, 4, None): "....vvvvv...vvvvv....vvvv...vvvv.......v.......v.......v........vvvvvvvvv",
    ("single", 2, 3, 18): "....................................vvvv...vvvvv....vvvv....vvvvv...vvvvv",
}


@pytest.mark.parametrize("key", sorted(_GOLDEN, key=repr))
def test_split_without_coarser_tiles_is_unchanged(key):
    tiles, seed, block_size, z_ref = key
    names = _names(_golden_mixed() if tiles == "mixed" else _golden_single())
    _, val = spatial_split(names, 0.25, seed, block_size=block_size, z_ref=z_ref)
    val = set(val)
    assert "".join("v" if n in val else "." for n in names) == _GOLDEN[key]


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
