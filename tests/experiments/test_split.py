import math
from itertools import combinations

import numpy as np
import pytest

from experiments.leakage import box_distance_m, near_footprints
from experiments.split import MAX_GROUPS, block_split, group_split

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


def _share_ground(a, b):
    (xa, ya, za), (xb, yb, zb) = sorted((a, b), key=lambda t: t[2])
    return (xb >> (zb - za), yb >> (zb - za)) == (xa, ya)


@pytest.mark.parametrize("z", [17, 16])
@pytest.mark.parametrize("block_size", [1, 2, 3, 4])
def test_block_split_tiles_coarser_than_z_ref_never_straddle(z, block_size):
    # a 2 x 2 of zoom-z tiles at an odd offset, every z19 tile under them, and a few z19 tiles beside
    x0, y0, s = (X0 >> (20 - z)) + 1, (Y0 >> (20 - z)) + 1, 1 << (19 - z)
    coarse = [(x0 + i, y0 + j, z) for j in (0, 1) for i in (0, 1)]
    tiles = coarse + [(x * s + i, y * s + j, 19) for x, y, _ in coarse for j in range(s) for i in range(s)]
    tiles += [((x0 + 3) * s + i, y0 * s, 19) for i in range(3)]
    held = 0
    for seed in range(20):
        sides = block_split(
            tiles, block_size=block_size, val_ratio=0.3, test_ratio=0.2, seed=seed, buffer_m=0.0
        )
        held += any(side != "train" for side in sides)
        for i, j in combinations(range(len(tiles)), 2):
            if sides[i] != sides[j]:
                assert not _share_ground(tiles[i], tiles[j]), (seed, tiles[i], tiles[j])
    assert held


def _row(scene, x0, n):
    return [(x, Y0, 20) for x in range(X0 + x0, X0 + x0 + n)], [scene] * n


def test_group_split_keeps_nearby_scenes_together():
    tiles, scenes = [], []
    for scene, x0, n in [("a", 0, 4), ("b", 5, 4), ("c", 100, 4), ("d", 200, 10)]:  # z20 tiles are ~38 m
        t, s = _row(scene, x0, n)
        tiles += t
        scenes += s
    sides = group_split(tiles, scenes, val_ratio=0.2, seed=0)
    by_scene = {}
    for scene, side in zip(scenes, sides, strict=True):
        by_scene.setdefault(scene, set()).add(side)
    # a and b are 38 m apart: one group; of the groups {a, b}, {c}, {d}, c alone is closest to 20% of the area
    assert by_scene == {"a": {"train"}, "b": {"train"}, "c": {"val"}, "d": {"train"}}
    assert sides == group_split(tiles, scenes, val_ratio=0.2, seed=0)


def test_group_split_one_group_has_no_val():
    tiles, scenes = _row("a", 0, 4)
    t, s = _row("b", 6, 4)
    assert group_split(tiles + t, scenes + s, val_ratio=0.2, seed=0) == ["train"] * 8


def test_group_split_no_tiles():
    assert group_split([], [], val_ratio=0.2, seed=0) == []


@pytest.mark.parametrize("val_ratio, scenes", [(0.0, ["a"]), (1.0, ["a"]), (0.2, ["a", "b"])])
def test_group_split_bad_arguments(val_ratio, scenes):
    with pytest.raises(ValueError):
        group_split([(X0, Y0, 20)], scenes, val_ratio=val_ratio, seed=0)


def test_group_split_refuses_more_groups_than_the_exact_search_handles():
    tiles, scenes = [], []
    for g in range(MAX_GROUPS + 1):  # scenes 100 tiles (~3.8 km) apart: one group each
        t, s = _row(f"s{g}", 100 * g, 1)
        tiles += t
        scenes += s
    with pytest.raises(ValueError, match="scene groups"):
        group_split(tiles, scenes, val_ratio=0.15, seed=0)


def test_group_split_weighs_groups_by_ground_area():
    # 12 tiles at 60 degrees cover the ground of 3 at the equator: by area c alone is 15% of the total,
    # while by tile count a (5 of 29) would be closest
    y60 = int((1 - math.asinh(math.tan(math.radians(60))) / math.pi) / 2 * 2**19)
    rows = {"a": (X0 // 2, Y0 // 2, 5), "c": (X0 // 2 + 300, y60, 12), "d": (X0 // 2 + 600, Y0 // 2, 12)}
    tiles, scenes = [], []
    for scene, (x0, y, n) in rows.items():
        tiles += [(x, y, 19) for x in range(x0, x0 + n)]
        scenes += [scene] * n
    sides = group_split(tiles, scenes, val_ratio=0.15, seed=0)
    assert {s for s, side in zip(scenes, sides, strict=True) if side == "val"} == {"c"}


def test_group_split_counts_the_ground_of_tiles_coarser_than_z_ref():
    # in z19 tiles: a is one z17 tile (16) with two z19 tiles inside it, b is 12 and c is 4 of 32 in all,
    # so c alone is 12.5%; counting the z17 tile as its north-west z19 tile would make a (2 of 18) closest
    y = Y0 // 2
    a = [(X0 // 8, y // 4, 17), (X0 // 2, y, 19), (X0 // 2 + 1, y + 1, 19)]
    b = [(X0 // 2 + 200 + i, y, 19) for i in range(12)]
    c = [(X0 // 2 + 400 + i, y, 19) for i in range(4)]
    scenes = ["a"] * 3 + ["b"] * 12 + ["c"] * 4
    sides = group_split(a + b + c, scenes, val_ratio=0.125, seed=0)
    assert {s for s, side in zip(scenes, sides, strict=True) if side == "val"} == {"c"}


# Splits computed before tiles coarser than z_ref got their own handling; no tile here is coarser than
# z_ref = 19, so these must not change. One character per tile: "." train, "v" val, "x" test, "d" drop.
_SIDE = {"train": ".", "val": "v", "test": "x", "drop": "d"}


def _golden_tiles():
    t = [(X0 // 2 + i, Y0 // 2 + j, 19) for j in range(10) for i in range(10) if (i + 2 * j) % 5 == 0]
    t += [(X0 + i, Y0 + j, 20) for j in range(12) for i in range(12) if (3 * i + j) % 4 == 1]
    return t + [(2 * X0 + 3 + i, 2 * Y0 + 5 + j, 21) for j in range(6) for i in range(5)]


_BLOCK_GOLDEN = {
    (0, 1, 0.0): ".xxv..v......v....vxvxx...v.v.xv.vvx..vvv...v..x...x.x..v....v....v....x....x....x....",
    (0, 2, 0.0): "v.x...xxv.xv.vv.v.vxvx.vx.vx.vx...x..x..x..xxvvxvvxvvxvvvvvvvvvvvvvvvvv...............",
    (0, 3, 0.0): "v.xx......x...x....xvx.vxxvxxvx...................x..x..vvvvvvvvvvvvvvv...............",
    (0, 2, 50.0): "..v..dvvd.v........vdv..vd.vd.vd.dv..v..vddvvddvddv..v....ddd...dd....d...............",
    (1, 1, 0.0): "v....xx.........x...xv.v.x..xv...v..x.xvxxxx.v........x..vvvv.vvvv.vvvv...............",
    (1, 2, 0.0): "...vx..v..vvv..vvv.v............x..x..x..x..v.vv.vv.vv.v...............xxxxxxxxxxxxxxx",
    (1, 3, 0.0): ".v..x.vvv........v.v..v........vxv.xvvxvvxv.xv.xvv.....................xxxxxxxxxxxxxxx",
    (1, 2, 50.0): "....v...d.................d..d..v..vd.vd.v..d...........dddddddddddddddvvvvvvvvvvvvvvv",
    (2, 1, 0.0): "vv.x.x....xvx.xvv..vx.vv.v.vvx........v..v..v.xv..xv...v.xxxx.xxxx.xxxx...............",
    (2, 2, 0.0): "v....vv.xxvxxx.vx..xv..v..v..v....v..v..v..vvxxvxxvxxvxxvvvvvvvvvvvvvvv...............",
    (2, 3, 0.0): "vx........xx.xxxx.x.v.xv..v..v.x..................x..x.xvvvvvvvvvvvvvvv...............",
    (2, 2, 50.0): ".....vd.vv.vvv.vv..v...........d..d....d.ddddvv.vv.vv.vv..............................",
}


@pytest.mark.parametrize("key", sorted(_BLOCK_GOLDEN))
def test_block_split_without_coarser_tiles_is_unchanged(key):
    seed, block_size, buffer_m = key
    test_ratio = 0.15 if buffer_m == 0 else 0.0
    sides = block_split(
        _golden_tiles(),
        block_size=block_size,
        val_ratio=0.2,
        test_ratio=test_ratio,
        seed=seed,
        buffer_m=buffer_m,
    )
    assert "".join(_SIDE[s] for s in sides) == _BLOCK_GOLDEN[key]


def _golden_scenes():
    tiles, scenes = [], []
    for scene, x0, n, z in (("a", 0, 4, 20), ("b", 300, 6, 19), ("c", 900, 3, 20), ("d", 1500, 8, 20)):
        xb, yb = (X0 >> (20 - z)) + (x0 >> (20 - z)), Y0 >> (20 - z)
        tiles += [(xb + i, yb, z) for i in range(n)]
        scenes += [scene] * n
    return tiles + [(2 * (X0 // 2 + 150) + 1, Y0, 20)], scenes + ["b"]  # a z20 tile on scene b's z19 ground


_GROUP_GOLDEN = {
    (0, 0.15): "..........vvv.........",
    (0, 0.3): "vvvv......vvv.........",
    (0, 0.5): "....vvvvvvvvv........v",
    (1, 0.15): "vvvv..................",
    (1, 0.3): "vvvv......vvv.........",
    (1, 0.5): "vvvvvvvvvv...........v",
    (2, 0.15): "..........vvv.........",
    (2, 0.3): ".............vvvvvvvv.",
    (2, 0.5): "vvvv......vvvvvvvvvvv.",
}


@pytest.mark.parametrize("key", sorted(_GROUP_GOLDEN))
def test_group_split_without_coarser_tiles_is_unchanged(key):
    seed, val_ratio = key
    sides = group_split(*_golden_scenes(), val_ratio=val_ratio, seed=seed)
    assert "".join(_SIDE[s] for s in sides) == _GROUP_GOLDEN[key]
