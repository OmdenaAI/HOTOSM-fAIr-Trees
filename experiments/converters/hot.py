"""HOT's Freetown trees (``hotosm/fAIr-guru-td-trees``) to regions, one per connected run of tiles in a block.

Only HOT ``train`` and ``validation`` are converted (as ``train`` and ``val``); HOT ``test`` stays out of
training. A tree outside every ``tree_group`` with a known crown diameter becomes a label square of that
side, centred on its point; one with an unknown diameter becomes a 4 m ignore square; trees inside a group
get nothing, as the group itself is masked. A region carries every feature that touches its tiles, so a
crown cut by a block edge is labelled on both sides. Labels come only from trees of the region's own split:
a crown from another split that reaches into a region is masked there instead (HOT's split has no buffer).
"""

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import mapping

from experiments.label_study.hot import UTM, HotData, known_diameter, split_units, write_tile
from fair_trees.georef import parse_oam_name, tile_bounds_4326, true_gsd
from fair_trees.preprocess import read_chip

logger = logging.getLogger(__name__)

SOURCE = "hot"
LICENCE = "CC-BY-4.0 imagery, ODbL-1.0 labels"
UNKNOWN_SIDE_M = 4.0
SPLITS = {"train": "train", "validation": "val"}  # HOT split -> region split; HOT test is never converted


@dataclass(frozen=True)
class _Region:
    id: str
    hot_split: str
    tiles: gpd.GeoDataFrame  # the region's rows of HotData.tiles
    footprint: shapely.Geometry  # EPSG:4326 union of the tile squares


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _tile_square(x: int, y: int, z: int) -> shapely.Geometry:
    """EPSG:4326 tile square whose edges equal its neighbours' exactly, so a union leaves no slivers."""
    west, _, _, north = tile_bounds_4326(x, y, z)
    east, _, _, south = tile_bounds_4326(x + 1, y + 1, z)
    return shapely.box(west, south, east, north)


def _regions(tiles: gpd.GeoDataFrame, block_tiles: int) -> list[_Region]:
    regions = []
    for hot_split, split in SPLITS.items():
        in_split = tiles[tiles["split"] == hot_split]
        position = pd.Index(in_split["tile_id"])
        for unit_id, tile_ids in split_units(in_split, split, block_tiles):
            rows = in_split.iloc[position.get_indexer(tile_ids)]
            squares = [_tile_square(x, y, z) for x, y, z in zip(rows["x"], rows["y"], rows["z"], strict=True)]
            regions.append(_Region(f"{SOURCE}-{unit_id}", hot_split, rows, shapely.union_all(squares)))
    return regions


def _crowns(data: HotData) -> tuple[gpd.GeoDataFrame, np.ndarray]:
    """Squares (EPSG:4326) for the trees outside every group, and the in-group flag of every tree.

    The squares carry ``split`` and ``known`` (diameter in (0, 30] m: a label; otherwise a 4 m ignore).
    """
    trees = data.trees.to_crs(UTM)
    in_group = np.zeros(len(trees), bool)
    in_group[data.groups.to_crs(UTM).sindex.query(trees.geometry, predicate="within")[0]] = True
    trees = trees[~in_group]
    diameter = known_diameter(trees["diameter_m"])
    squares = trees.geometry.buffer(diameter.fillna(UNKNOWN_SIDE_M).to_numpy() / 2, cap_style="square")
    crowns = gpd.GeoDataFrame(
        {"split": trees["split"].to_numpy(), "known": diameter.notna().to_numpy()},
        geometry=squares.to_crs("EPSG:4326").to_numpy(),
        crs="EPSG:4326",
    )
    return crowns, in_group


def _touching(features: gpd.GeoDataFrame, regions: list[_Region]) -> list[np.ndarray]:
    """Per region, the sorted positions of the features that intersect its footprint."""
    region_idx, feature_idx = features.sindex.query([r.footprint for r in regions], predicate="intersects")
    order = np.lexsort((feature_idx, region_idx))
    region_idx, feature_idx = region_idx[order], feature_idx[order]
    return np.split(feature_idx, np.searchsorted(region_idx, np.arange(1, len(regions))))


def _write_chips(
    regions: list[_Region], images: Callable[[str], Iterable[tuple[str, bytes]]], out_root: Path
) -> dict[str, float]:
    """Write each streamed tile into its region's chips as it arrives; return valid ground m^2 per region."""
    area_m2 = {r.id: 0.0 for r in regions}
    for hot_split in SPLITS:
        pending = {t: r.id for r in regions if r.hot_split == hot_split for t in r.tiles["tile_id"]}
        for tile_id, image in images(hot_split):
            region_id = pending.pop(tile_id, None)
            if region_id is None:
                raise ValueError(f"{tile_id}: not a HOT {hot_split} tile, or its image came twice")
            _, _, z = parse_oam_name(f"{tile_id}.png")
            chip_dir = out_root / region_id / f"z{z}" / "chips"
            chip_dir.mkdir(parents=True, exist_ok=True)
            _, valid, (x, y, z) = read_chip(write_tile(image, tile_id, chip_dir))
            _, south, _, north = tile_bounds_4326(x, y, z)
            area_m2[region_id] += int(valid.sum()) * true_gsd(z, (south + north) / 2) ** 2
        if pending:
            raise ValueError(f"no image for {len(pending)} HOT {hot_split} tiles, e.g. {min(pending)}")
    return area_m2


def _feature(geometry: shapely.Geometry, properties: dict) -> dict:
    return {"type": "Feature", "properties": properties, "geometry": mapping(geometry)}


def _write_region(
    out_root: Path,
    region: _Region,
    labels: list[dict],
    ignore: list[dict],
    valid_area_m2: float,
    biome_fn: Callable[[float, float], str] | None,
) -> None:
    region_dir = out_root / region.id
    _write_json(region_dir / "labels.geojson", {"type": "FeatureCollection", "features": labels})
    _write_json(region_dir / "ignore.geojson", {"type": "FeatureCollection", "features": ignore})
    centre = region.footprint.centroid
    meta = {
        "id": region.id,
        "source": SOURCE,
        "biome": biome_fn(centre.x, centre.y) if biome_fn else "unknown",
        "licence": LICENCE,
        "split": SPLITS[region.hot_split],
        "zooms": sorted({int(z) for z in region.tiles["z"]}),
        "footprint": mapping(region.footprint),
        "valid_area_m2": valid_area_m2,
    }
    _write_json(region_dir / "region.json", meta)


def _label_free_share(tiles: gpd.GeoDataFrame, crowns: gpd.GeoDataFrame) -> dict[str, float]:
    """Per split, the share of tiles that no label square of that split touches."""
    shares = {}
    for hot_split, split in SPLITS.items():
        in_split = tiles[tiles["split"] == hot_split]
        labels = crowns[crowns["known"] & (crowns["split"] == hot_split)]
        touched = np.unique(labels.sindex.query(in_split.geometry, predicate="intersects")[0])
        shares[split] = 1.0 - len(touched) / len(in_split) if len(in_split) else 0.0
    return shares


def convert(
    data: HotData,
    images: Callable[[str], Iterable[tuple[str, bytes]]],
    out_root: Path,
    *,
    biome_fn: Callable[[float, float], str] | None = None,
    block_tiles: int = 16,
) -> dict:
    """Write a region per block component and ``stats.json`` into an empty ``out_root``; return the stats.

    ``images(hot_split)`` yields ``(tile_id, image_bytes)`` for every tile of a HOT split; each tile is
    written as it arrives, so the images are never all in memory.
    """
    out_root = Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(f"{out_root}: output directory is not empty; old regions would mix with new ones")

    regions = _regions(data.tiles, block_tiles)
    crowns, in_group = _crowns(data)
    groups = data.groups.to_crs("EPSG:4326")
    area_m2 = _write_chips(regions, images, out_root)

    squares, known, tree_split = (
        crowns.geometry.to_numpy(),
        crowns["known"].to_numpy(),
        crowns["split"].to_numpy(),
    )
    group_shapes = groups.geometry.to_numpy()
    labelled, unknown, other_split, masked_groups = set(), set(), set(), set()
    for region, crown_idx, group_idx in zip(
        regions, _touching(crowns, regions), _touching(groups, regions), strict=True
    ):
        is_known = known[crown_idx]
        is_own = tree_split[crown_idx] == region.hot_split
        label_idx, other_idx = crown_idx[is_known & is_own], crown_idx[is_known & ~is_own]
        unknown_idx = crown_idx[~is_known]
        labelled.update(label_idx)
        unknown.update(unknown_idx)
        other_split.update(other_idx)
        masked_groups.update(group_idx)

        properties = {"label": 1, "source": SOURCE, "licence": LICENCE, "region": region.id}
        labels = [_feature(g, properties) for g in squares[label_idx]]
        ignore = [_feature(g, {"reason": "tree_group"}) for g in group_shapes[group_idx]]
        ignore += [_feature(g, {"reason": "unknown_diameter"}) for g in squares[unknown_idx]]
        ignore += [_feature(g, {"reason": "other_split"}) for g in squares[other_idx]]
        _write_region(out_root, region, labels, ignore, area_m2[region.id], biome_fn)

    split_m2 = dict.fromkeys(SPLITS.values(), 0.0)
    regions_per_split = dict.fromkeys(SPLITS.values(), 0)
    chips_per_split = dict.fromkeys(SPLITS.values(), 0)
    for region in regions:
        split = SPLITS[region.hot_split]
        split_m2[split] += area_m2[region.id]
        regions_per_split[split] += 1
        chips_per_split[split] += len(region.tiles)
    total_m2 = sum(split_m2.values())
    stats = {
        "regions": regions_per_split,
        "chips": chips_per_split,
        "boxes": len(labelled),
        "ignored_unknown_diameter": len(unknown),
        "masked_other_split": len(other_split),
        "trees_in_groups": int((in_group & data.trees["split"].isin(list(SPLITS)).to_numpy()).sum()),
        "groups": len(masked_groups),
        "label_free_tile_share": _label_free_share(data.tiles, crowns),
        # share of the valid ground area, as the NEON converter reports it
        "val_share": split_m2["val"] / total_m2 if total_m2 else 0.0,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    logger.info("converted HOT into %d regions: %s", len(regions), stats)
    return stats
