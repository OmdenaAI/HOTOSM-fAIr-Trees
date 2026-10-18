"""The block split with a buffer, for contiguous scenes only.

Official splits are used where a dataset has one.
"""

import logging
from collections.abc import Sequence

import numpy as np

from experiments.leakage import box_distance_m, tile_boxes_3857
from fair_trees.dataset import block_key

logger = logging.getLogger(__name__)


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

    Whole blocks go to one side, so a tile and its ground-overlapping tiles at other zooms never straddle.
    Train tiles within ``buffer_m`` of any val or test tile are dropped.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be at least 1, got {block_size}")
    if val_ratio <= 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError(
            f"need 0 < val_ratio, 0 <= test_ratio, val_ratio + test_ratio < 1: {val_ratio}, {test_ratio}"
        )
    if not len(tiles):
        return []

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

    held = [i for i, s in enumerate(sides) if s != "train"]
    if buffer_m > 0 and held:
        train = [i for i, s in enumerate(sides) if s == "train"]
        boxes = tile_boxes_3857(tiles)
        near = (box_distance_m(boxes[train], boxes[held]) < buffer_m).any(axis=1)
        for i, is_near in zip(train, near, strict=True):
            if is_near:
                sides[i] = "drop"
    return sides
