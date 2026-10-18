"""Training data: regions, labels, masks, the crop sampler and the spatial train/val split.

Torch-free: numpy, Pillow, rasterio, pyproj and the stdlib only.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from fair_trees.georef import parse_oam_name

logger = logging.getLogger(__name__)


def block_key(x: int, y: int, z: int, z_ref: int, block_size: int) -> tuple[int, int]:
    """Block of a tile at the reference zoom: ``block_size`` x ``block_size`` reference tiles per block.

    A tile finer than ``z_ref`` maps to its ancestor; a coarser one maps to its north-west descendant [P].
    """
    if z >= z_ref:
        bx, by = x >> (z - z_ref), y >> (z - z_ref)
    else:
        bx, by = x << (z_ref - z), y << (z_ref - z)
    return bx // block_size, by // block_size


def spatial_split(
    chip_names: Sequence[str | Path],
    val_ratio: float,
    seed: int,
    *,
    block_size: int = 4,
    z_ref: int | None = None,
) -> tuple[list, list]:
    """Split chips into ``(train, val)`` by whole spatial blocks so neighbouring ground never straddles.

    Chips named ``OAM-x-y-z.<ext>`` are grouped by ``block_key`` at ``z_ref`` (default: the lowest zoom
    present), so one piece of ground at several zooms lands on one side. Blocks go to val in a seeded
    random order until about ``round(val_ratio * len(chip_names))`` chips are in val; train always keeps
    at least one block. Chips whose name does not parse fill any remaining val quota. Items come back
    unchanged and in input order.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    parsed = [parse_oam_name(item) for item in chip_names]
    named = [i for i, p in enumerate(parsed) if p is not None]
    unnamed = [i for i, p in enumerate(parsed) if p is None]
    if z_ref is None and named:
        z_ref = min(parsed[i][2] for i in named)

    blocks: dict[tuple[int, int], list[int]] = {}
    for i in named:
        x, y, z = parsed[i]
        blocks.setdefault(block_key(x, y, z, z_ref, block_size), []).append(i)
    keys = sorted(blocks)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))

    target = round(val_ratio * len(chip_names))
    val_idx: list[int] = []
    val_blocks = 0
    for k in order:
        # keep at least one block for train: a block goes to val only if another one remains
        if len(val_idx) < target and len(keys) - val_blocks >= 2:
            val_idx += blocks[keys[k]]
            val_blocks += 1

    for j in rng.permutation(len(unnamed)):
        if len(val_idx) >= target:
            break
        val_idx.append(unnamed[j])

    if not val_idx:
        logger.warning(
            "spatial_split: val is empty (%d chips in one block, no unnamed chips); "
            "training continues without val",
            len(chip_names),
        )

    val_set = set(val_idx)
    train = [item for i, item in enumerate(chip_names) if i not in val_set]
    val = [item for i, item in enumerate(chip_names) if i in val_set]
    return train, val
