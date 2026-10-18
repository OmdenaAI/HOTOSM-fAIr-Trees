"""Training data: regions, labels, masks, the crop sampler and the spatial train/val split.

Torch-free: numpy, Pillow, rasterio, pyproj and the stdlib only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
from affine import Affine
from pyproj import Transformer

from fair_trees.georef import mercator_y_to_lat, parse_oam_name, tile_bounds_3857, true_gsd
from fair_trees.preprocess import tile_groups

logger = logging.getLogger(__name__)

MIN_BOX_FRACTION = 0.4
_TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
_REQUIRED_KEYS = ("id", "source", "biome", "licence", "split", "footprint", "valid_area_m2")


@dataclass(frozen=True)
class Region:
    """One labelled area: its chips per zoom, label and ignore files, and the footprint that was mapped.

    ``footprint`` is an EPSG:4326 GeoJSON geometry (``None`` = everything is mapped).
    """

    id: str
    source: str
    biome: str
    licence: str
    split: str
    chips: Mapping[int, tuple[Path, ...]]
    labels_path: Path
    ignore_path: Path | None
    footprint: dict | None
    valid_area_m2: float


def load_geojson(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _is_tree_label(properties: dict) -> bool:
    """A missing label is a tree; a present one must be the number 1."""
    if "label" not in properties:
        return True
    try:
        return float(properties["label"]) == 1.0
    except (TypeError, ValueError):
        return False


def _invalid_reason(ring: np.ndarray) -> str | None:
    if len(ring) < 4:
        return f"{len(ring)} positions"
    if not np.isfinite(ring).all():
        return "non-finite coordinate"
    if (ring.max(axis=0) == ring.min(axis=0)).any():
        return "zero extent"
    return None


def label_rings(geojson: dict) -> list[np.ndarray]:
    """Exterior rings of the tree labels as (k, 2) float64 lon/lat arrays, one per polygon part.

    Features whose ``label`` is present and not 1 are dropped silently; empty, unsupported and invalid
    geometries are skipped with a warning.
    """
    rings = []
    for i, feature in enumerate(geojson.get("features", [])):
        if not _is_tree_label(feature.get("properties") or {}):
            continue
        geometry = feature.get("geometry")
        if not geometry or not geometry.get("coordinates"):
            logger.warning(f"label feature {i}: empty geometry, skipped")
            continue
        kind = geometry.get("type")
        if kind == "Polygon":
            parts = [geometry["coordinates"]]
        elif kind == "MultiPolygon":
            parts = geometry["coordinates"]
        else:
            logger.warning(f"label feature {i}: unsupported geometry type {kind}, skipped")
            continue
        for part in parts:
            ring = np.asarray(part[0] if part else [], np.float64).reshape(-1, 2)
            why = _invalid_reason(ring)
            if why:
                logger.warning(f"label feature {i}: invalid geometry ({why}), skipped")
                continue
            rings.append(ring)
    return rings


def labels_to_boxes(geojson: dict, transform_3857: Affine) -> np.ndarray:
    """Tight XYXY box per label ring in the pixel frame of ``transform_3857``, float32 (N, 4)."""
    inverse = ~transform_3857
    boxes = []
    for ring in label_rings(geojson):
        x, y = _TO_3857.transform(ring[:, 0], ring[:, 1])
        cols, rows = inverse @ (np.asarray(x), np.asarray(y))
        boxes.append([cols.min(), rows.min(), cols.max(), rows.max()])
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def _project_geometry(geometry: dict) -> dict:
    def walk(coords):
        if isinstance(coords[0], int | float):
            x, y = _TO_3857.transform(coords[0], coords[1])
            return [x, y]
        return [walk(c) for c in coords]

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def rasterize_mask(
    region: Region, transform_3857: Affine, shape: tuple[int, int], *, ignore: bool = True
) -> np.ndarray:
    """(H, W) bool, True = masked: outside the footprint or, when ``ignore``, inside an ignore polygon."""
    masked = np.zeros(shape, bool)
    if region.footprint is not None:
        inside = rasterio.features.rasterize(
            [(_project_geometry(region.footprint), 1)],
            out_shape=shape,
            transform=transform_3857,
            fill=0,
            dtype="uint8",
        )
        masked |= inside == 0
    if ignore and region.ignore_path is not None:
        shapes = [
            (_project_geometry(f["geometry"]), 1)
            for f in load_geojson(region.ignore_path).get("features", [])
            if f.get("geometry") and f["geometry"].get("type") in ("Polygon", "MultiPolygon")
        ]
        if shapes:
            burned = rasterio.features.rasterize(
                shapes, out_shape=shape, transform=transform_3857, fill=0, dtype="uint8"
            )
            masked |= burned == 1
    return masked


def boxes_in_window(
    boxes: np.ndarray,
    unmasked: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    min_fraction: float = MIN_BOX_FRACTION,
) -> np.ndarray:
    """Boxes (mosaic frame) kept for a window, clipped, in window coordinates, float32 (K, 4).

    A box is kept when at least ``min_fraction`` of its unclipped area is unmasked pixels (by pixel
    centre) inside the window. ``unmasked`` is the window-sized (height, width) boolean image.
    """
    shifted = np.asarray(boxes, np.float64).reshape(-1, 4) - np.array([x0, y0, x0, y0])
    area = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
    shifted, area = shifted[area > 0], area[area > 0]
    clipped = np.clip(shifted, 0, [width, height, width, height])
    integral = np.zeros((unmasked.shape[0] + 1, unmasked.shape[1] + 1), np.int64)
    integral[1:, 1:] = unmasked.astype(np.int64).cumsum(0).cumsum(1)
    # pixel j counts when x1 <= j + 0.5 < x2, i.e. j in [ceil(x1 - 0.5), ceil(x2 - 0.5))
    edges = np.ceil(clipped - 0.5).astype(np.int64)
    c1 = np.clip(edges[:, 0], 0, unmasked.shape[1])
    c2 = np.clip(edges[:, 2], c1, unmasked.shape[1])
    r1 = np.clip(edges[:, 1], 0, unmasked.shape[0])
    r2 = np.clip(edges[:, 3], r1, unmasked.shape[0])
    count = integral[r2, c2] - integral[r1, c2] - integral[r2, c1] + integral[r1, c1]
    return clipped[count / area >= min_fraction].astype(np.float32).reshape(-1, 4)


def load_regions(root: Path) -> list[Region]:
    """One region per ``root/*/region.json`` (base-training layout), sorted by id."""
    regions = []
    for meta_path in sorted(Path(root).glob("*/region.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        for key in _REQUIRED_KEYS:
            if key not in meta:
                raise ValueError(f"{d}: region.json lacks {key}")
        chips = {}
        for zdir in sorted(d.glob("z*")):
            if not zdir.is_dir():
                continue
            found = tuple(p for p in sorted((zdir / "chips").glob("OAM-*")) if parse_oam_name(p))
            if found:
                chips[int(zdir.name[1:])] = found
        ignore = d / "ignore.geojson"
        regions.append(
            Region(
                id=meta["id"],
                source=meta["source"],
                biome=meta["biome"],
                licence=meta["licence"],
                split=meta["split"],
                chips=chips,
                labels_path=d / "labels.geojson",
                ignore_path=ignore if ignore.is_file() else None,
                footprint=meta["footprint"],
                valid_area_m2=float(meta["valid_area_m2"]),
            )
        )
    return sorted(regions, key=lambda r: r.id)


def load_fair_dataset(chips: Sequence[Path], labels_path: Path, *, split: str = "train") -> list[Region]:
    """Regions for a mapper's own chips: one per zoom and 8-connected tile component."""
    named, xyzs = [], []
    for path in map(Path, chips):
        xyz = parse_oam_name(path)
        if xyz is None:
            logger.warning(f"{path.name}: not an OAM-x-y-z chip name, skipped")
            continue
        named.append(path)
        xyzs.append(xyz)
    regions = []
    for k, group in enumerate(tile_groups(xyzs)):
        z = xyzs[group[0]][2]
        paths = tuple(named[i] for i in group)
        with rasterio.open(paths[0]) as ds:
            tile_px = ds.width
        north = max(tile_bounds_3857(*xyzs[i])[3] for i in group)
        south = min(tile_bounds_3857(*xyzs[i])[1] for i in group)
        gsd = true_gsd(z, mercator_y_to_lat((north + south) / 2))
        regions.append(
            Region(
                id=f"fair-z{z}-{k:03d}",
                source="fair",
                biome="unknown",
                licence="unknown",
                split=split,
                chips={z: paths},
                labels_path=Path(labels_path),
                ignore_path=None,
                footprint=None,
                valid_area_m2=len(paths) * (tile_px * gsd) ** 2,
            )
        )
    return regions


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
