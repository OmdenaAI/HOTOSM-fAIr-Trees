"""HOT's Freetown trees (``hotosm/fAIr-guru-td-trees``) to regions, one per connected run of tiles in a block.

Only HOT ``train`` and ``validation`` are converted (as ``train`` and ``val``); HOT ``test`` stays out of
training. A tree outside every ``tree_group`` with a known crown diameter becomes a label square of that
side, centred on its point; one with an unknown diameter becomes a 4 m ignore square; trees inside a group
get nothing, as the group itself is a label (``label: 2``, a tree_group). A region carries every feature
that touches its tiles, so a crown cut by a block edge is labelled on both sides. Labels come only from
features of the region's own split (a group's split is that of the tile under its representative point):
a crown or group from another split that reaches into a region is masked there instead (HOT's split has no
buffer).
"""

import json
import logging
from collections import Counter
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
from fair_trees.params import CLASSES
from fair_trees.preprocess import read_chip

logger = logging.getLogger(__name__)

SOURCE = "hot"
LICENCE = "CC-BY-4.0 imagery, ODbL-1.0 labels"
UNKNOWN_SIDE_M = 4.0
SPLITS = {"train": "train", "validation": "val"}  # HOT split -> region split, for training
TEST_SPLITS = {"test": "test"}  # HOT test on its own: a held-out benchmark part


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


def _regions(tiles: gpd.GeoDataFrame, block_tiles: int, splits: dict[str, str]) -> list[_Region]:
    regions = []
    for hot_split, split in splits.items():
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
    regions: list[_Region],
    images: Callable[[str], Iterable[tuple[str, bytes]]],
    out_root: Path,
    splits: dict[str, str],
) -> dict[str, float]:
    """Write each streamed tile into its region's chips as it arrives; return valid ground m^2 per region."""
    area_m2 = {r.id: 0.0 for r in regions}
    for hot_split in splits:
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
    biome: str,
    split: str,
) -> None:
    region_dir = out_root / region.id
    _write_json(region_dir / "labels.geojson", {"type": "FeatureCollection", "features": labels})
    _write_json(region_dir / "ignore.geojson", {"type": "FeatureCollection", "features": ignore})
    meta = {
        "id": region.id,
        "source": SOURCE,
        "biome": biome,
        "licence": LICENCE,
        "split": split,
        "zooms": sorted({int(z) for z in region.tiles["z"]}),
        "footprint": mapping(region.footprint),
        "valid_area_m2": valid_area_m2,
        "classes": list(CLASSES),
    }
    _write_json(region_dir / "region.json", meta)


def _biomes(regions: list[_Region], biome_fn: Callable[[float, float], str] | None) -> dict[str, str]:
    """Per region id, the commonest biome over its tile centres, ignoring ``"unknown"`` answers.

    A footprint centroid can fall outside the ecoregion polygons (coastal blocks) although the tiles do not,
    so the lookup is per tile. A region with no tile on a polygon takes the commonest biome of the whole
    conversion; ``"unknown"`` remains only when no tile anywhere found one.
    """
    found = {}
    for region in regions:
        answers = []
        if biome_fn:
            for x, y, z in zip(region.tiles["x"], region.tiles["y"], region.tiles["z"], strict=True):
                west, south, east, north = tile_bounds_4326(x, y, z)
                answers.append(biome_fn((west + east) / 2, (south + north) / 2))
        found[region.id] = Counter(a for a in answers if a != "unknown")
    overall = sum(found.values(), Counter())
    fallback = overall.most_common(1)[0][0] if overall else "unknown"
    return {rid: counts.most_common(1)[0][0] if counts else fallback for rid, counts in found.items()}


def _group_splits(groups: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame) -> np.ndarray:
    """Per group, the HOT split of the tile under its representative point (None off every tile)."""
    points = gpd.GeoDataFrame(geometry=groups.geometry.representative_point(), crs=groups.crs)
    under = points.sjoin(tiles[["split", "geometry"]].to_crs(groups.crs), predicate="intersects")
    split = under[~under.index.duplicated()]["split"]  # a point on a tile edge: the first tile
    return split.reindex(groups.index).to_numpy(dtype=object)


def _label_free_share(
    tiles: gpd.GeoDataFrame, crowns: gpd.GeoDataFrame, splits: dict[str, str]
) -> dict[str, float]:
    """Per split, the share of tiles that no label square of that split touches."""
    shares = {}
    for hot_split, split in splits.items():
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
    splits: dict[str, str] = SPLITS,
) -> dict:
    """Write a region per block component and ``stats.json`` into an empty ``out_root``; return the stats.

    ``images(hot_split)`` yields ``(tile_id, image_bytes)`` for every tile of a HOT split; each tile is
    written as it arrives, so the images are never all in memory. ``splits`` maps the HOT splits to
    convert to their region split (``TEST_SPLITS`` for the held-out test part); crowns and groups of any
    other split that reach into a region are masked there.
    """
    out_root = Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(f"{out_root}: output directory is not empty; old regions would mix with new ones")

    regions = _regions(data.tiles, block_tiles, splits)
    crowns, in_group = _crowns(data)
    groups = data.groups.to_crs("EPSG:4326")
    area_m2 = _write_chips(regions, images, out_root, splits)
    biomes = _biomes(regions, biome_fn)

    squares, known, tree_split = (
        crowns.geometry.to_numpy(),
        crowns["known"].to_numpy(),
        crowns["split"].to_numpy(),
    )
    group_shapes = groups.geometry.to_numpy()
    group_split = _group_splits(groups, data.tiles)
    labelled, unknown, other_split, group_labels, other_groups = set(), set(), set(), set(), set()
    for region, crown_idx, group_idx in zip(
        regions, _touching(crowns, regions), _touching(groups, regions), strict=True
    ):
        is_known = known[crown_idx]
        is_own = tree_split[crown_idx] == region.hot_split
        label_idx, other_idx = crown_idx[is_known & is_own], crown_idx[is_known & ~is_own]
        unknown_idx = crown_idx[~is_known]
        own_group = group_split[group_idx] == region.hot_split
        own_group_idx, other_group_idx = group_idx[own_group], group_idx[~own_group]
        labelled.update(label_idx)
        unknown.update(unknown_idx)
        other_split.update(other_idx)
        group_labels.update(own_group_idx)
        other_groups.update(other_group_idx)

        properties = {"source": SOURCE, "licence": LICENCE, "region": region.id}
        labels = [_feature(g, {"label": 1, **properties}) for g in squares[label_idx]]
        labels += [_feature(g, {"label": 2, **properties}) for g in group_shapes[own_group_idx]]
        ignore = [_feature(g, {"reason": "other_split_group"}) for g in group_shapes[other_group_idx]]
        ignore += [_feature(g, {"reason": "unknown_diameter"}) for g in squares[unknown_idx]]
        ignore += [_feature(g, {"reason": "other_split"}) for g in squares[other_idx]]
        split = splits[region.hot_split]
        _write_region(out_root, region, labels, ignore, area_m2[region.id], biomes[region.id], split)

    split_m2 = dict.fromkeys(splits.values(), 0.0)
    regions_per_split = dict.fromkeys(splits.values(), 0)
    chips_per_split = dict.fromkeys(splits.values(), 0)
    for region in regions:
        split = splits[region.hot_split]
        split_m2[split] += area_m2[region.id]
        regions_per_split[split] += 1
        chips_per_split[split] += len(region.tiles)
    total_m2 = sum(split_m2.values())
    stats = {
        "regions": regions_per_split,
        "chips": chips_per_split,
        "regions_per_biome": dict(Counter(biomes.values())),
        "boxes": len(labelled),
        "ignored_unknown_diameter": len(unknown),
        "masked_other_split": len(other_split),
        "trees_in_groups": int((in_group & data.trees["split"].isin(list(splits)).to_numpy()).sum()),
        "group_labels": len(group_labels),
        "groups_masked_other_split": len(other_groups),
        "label_free_tile_share": _label_free_share(data.tiles, crowns, splits),
        # share of the valid ground area, as the NEON converter reports it
        "val_share": split_m2.get("val", 0.0) / total_m2 if total_m2 else 0.0,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    logger.info("converted HOT into %d regions: %s", len(regions), stats)
    return stats
