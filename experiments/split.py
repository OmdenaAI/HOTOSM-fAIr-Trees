"""Train/val splits with a buffer: the block split for contiguous scenes, the group split for scenes smaller
than the buffer.

Official splits are used where a dataset has one.
"""

import logging
from collections.abc import Sequence

import numpy as np

from experiments.leakage import box_distance_m, tile_boxes_3857
from fair_trees.dataset import block_key, reference_grid
from fair_trees.georef import tile_bounds_4326, true_gsd

logger = logging.getLogger(__name__)

MAX_GROUPS = 16  # group_split tries every subset of groups


def block_split(
    tiles: Sequence[tuple[int, int, int]],
    *,
    block_size: int,
    val_ratio: float,
    seed: int,
    z_ref: int = 19,
    test_ratio: float = 0.0,
    buffer_m: float = 1000.0,
) -> list[str]:
    """Per tile, one of ``"train"``, ``"val"``, ``"test"`` or ``"drop"``.

    Whole blocks go to one side, so a tile and its ground-overlapping tiles at other zooms never straddle; a
    tile coarser than ``z_ref`` lowers the block grid to its zoom (see ``reference_grid``). Train tiles within
    ``buffer_m`` of any val or test tile are dropped.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    if val_ratio <= 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError(
            f"need 0 < val_ratio, 0 <= test_ratio, val_ratio + test_ratio < 1: {val_ratio}, {test_ratio}"
        )
    if not len(tiles):
        return []

    z_ref, block_size = reference_grid([z for _, _, z in tiles], z_ref, block_size)
    blocks: dict[tuple[int, int], list[int]] = {}
    for i, (x, y, z) in enumerate(tiles):
        blocks.setdefault(block_key(x, y, z, z_ref, block_size), []).append(i)
    keys = sorted(blocks)
    order = np.random.default_rng(seed).permutation(len(keys))

    n = len(tiles)
    want = {"test": round(test_ratio * n), "val": round(val_ratio * n)}
    got = {"test": 0, "val": 0}
    sides = ["train"] * n
    for rank, k in enumerate(order):
        idx = blocks[keys[k]]
        if rank == len(order) - 1:
            break  # the last block always stays train
        side = next((s for s in ("test", "val") if got[s] < want[s]), None)
        if side is None:
            break
        got[side] += len(idx)
        for i in idx:
            sides[i] = side

    return _drop_near_held(tiles, sides, buffer_m)


def group_split(
    tiles: Sequence[tuple[int, int, int]],
    scenes: Sequence[str],
    *,
    val_ratio: float,
    seed: int,
    z_ref: int = 19,
    buffer_m: float = 1000.0,
) -> list[str]:
    """Per tile, one of ``"train"``, ``"val"`` or ``"drop"``, keeping groups of nearby scenes whole.

    For scenes smaller than the buffer, where any block split leaves most of train within ``buffer_m`` of
    val. Scenes closer than ``buffer_m`` form one group; val is the proper subset of groups whose share of
    the ground area (covered by distinct ``z_ref`` tiles, a coarser tile covering every one under it) is
    closest to ``val_ratio``. Groups lie at least ``buffer_m`` apart, so the buffer, still applied, drops
    nothing.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"need 0 < val_ratio < 1, got {val_ratio}")
    if len(scenes) != len(tiles):
        raise ValueError(f"{len(tiles)} tiles but {len(scenes)} scene names")
    if not len(tiles):
        return []

    names = sorted(set(scenes))
    scene_of = np.array([names.index(s) for s in scenes])
    boxes = tile_boxes_3857(tiles)
    extents = np.array(
        [
            [*boxes[scene_of == i, :2].min(axis=0), *boxes[scene_of == i, 2:].max(axis=0)]
            for i in range(len(names))
        ]
    )
    close = box_distance_m(extents, extents) < buffer_m
    close |= close.T  # the gap is scaled at the first box's latitude, so make it symmetric
    group = [-1] * len(names)
    n_groups = 0
    for start in range(len(names)):
        if group[start] >= 0:
            continue
        group[start] = n_groups
        stack = [start]
        while stack:
            for j in np.flatnonzero(close[stack.pop()]):
                if group[j] < 0:
                    group[j] = n_groups
                    stack.append(j)
        n_groups += 1
    tile_group = [group[i] for i in scene_of]
    if n_groups == 1:
        logger.warning(
            "all %d scenes lie within %.0f m of each other: one group, no val", len(names), buffer_m
        )
        return ["train"] * len(tiles)
    if n_groups > MAX_GROUPS:
        raise ValueError(f"{n_groups} scene groups; the exact val search handles at most {MAX_GROUPS}")

    # ground area, not a tile count: a z_ref tile at 68 degrees covers 13% of one at the equator
    area = np.zeros(n_groups)
    covered = set()
    for g, (x, y, z) in zip(tile_group, tiles, strict=True):
        if z >= z_ref:
            covered.add((g, block_key(x, y, z, z_ref, 1)))
        else:  # every z_ref tile under a coarser one
            s = 1 << (z_ref - z)
            covered.update((g, (x * s + i, y * s + j)) for j in range(s) for i in range(s))
    for g, (x, y) in covered:
        _, south, _, north = tile_bounds_4326(x, y, z_ref)
        area[g] += (256 * true_gsd(z_ref, (south + north) / 2)) ** 2
    order = np.random.default_rng(seed).permutation(n_groups)  # breaks ties between equally close subsets
    masks = np.arange(1, 2**n_groups - 1)  # every proper, non-empty subset
    bits = ((masks[:, None] >> np.arange(n_groups)) & 1).astype(np.uint8)
    share = bits @ area[order] / area.sum()
    best = int(np.argmin(np.abs(share - val_ratio)))
    val_groups = {int(order[b]) for b in np.flatnonzero(bits[best])}
    held = [names[i] for i in range(len(names)) if group[i] in val_groups]
    logger.info("group split: %d groups, val area share %.3f, val scenes %s", n_groups, share[best], held)

    sides = ["val" if g in val_groups else "train" for g in tile_group]
    return _drop_near_held(tiles, sides, buffer_m)


def _drop_near_held(tiles: Sequence[tuple[int, int, int]], sides: list[str], buffer_m: float) -> list[str]:
    """``sides`` with every train tile within ``buffer_m`` of a val or test tile set to ``"drop"``."""
    held = [i for i, s in enumerate(sides) if s != "train"]
    if buffer_m > 0 and held:
        train = [i for i, s in enumerate(sides) if s == "train"]
        boxes = tile_boxes_3857(tiles)
        near = (box_distance_m(boxes[train], boxes[held]) < buffer_m).any(axis=1)
        for i, is_near in zip(train, near, strict=True):
            if is_near:
                sides[i] = "drop"
    return sides
