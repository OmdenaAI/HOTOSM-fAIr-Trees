"""Load HOT's Freetown tree labels and measure what masking ``tree_group`` polygons would cost."""

import json
import re
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from fair_trees.georef import parse_oam_name

REPO_ID = "hotosm/fAIr-guru-td-trees"
REVISION = "v0.1.0"
UTM = "EPSG:32628"


def parse_diameter(value) -> float:
    """Metres from strings like '3m', '2.412m', '3,134 m', '4'; NaN if missing or unclear."""
    if not isinstance(value, str):
        return np.nan
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(?:m|M|metres|meters)?\s*", value)
    return float(m.group(1).replace(",", ".")) if m else np.nan


@dataclass(frozen=True, eq=False)
class HotData:
    trees: gpd.GeoDataFrame  # EPSG:4326, one row per osm_id: osm_id, split, diameter_m, geometry (Point)
    groups: gpd.GeoDataFrame  # EPSG:4326, dissolved by osm_id: osm_id, geometry
    tiles: gpd.GeoDataFrame  # tile_id, x, y, z, split, geometry


def load_hot(labels: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame) -> HotData:
    """Collapse the per-(label, tile) rows into one tree per osm_id and one dissolved group per osm_id."""
    trees = labels[labels["class"] == "tree"].drop_duplicates("osm_id").copy()
    trees["diameter_m"] = trees["tags"].map(lambda t: parse_diameter(json.loads(t).get("diameter_crown")))
    trees = trees[["osm_id", "split", "diameter_m", "geometry"]].reset_index(drop=True)

    pieces = labels.loc[labels["class"] == "tree_group", ["osm_id", "geometry"]]
    groups = pieces.dissolve("osm_id").reset_index()

    tiles = tiles[["tile_id", "split", "geometry"]].copy()
    tiles[["x", "y", "z"]] = [parse_oam_name(f"{t}.png") for t in tiles["tile_id"]]
    return HotData(trees=trees, groups=groups, tiles=tiles[["tile_id", "x", "y", "z", "split", "geometry"]])


def read_hot(labels_path, tiles_path) -> HotData:
    return load_hot(gpd.read_parquet(labels_path), gpd.read_parquet(tiles_path))


def masking_cost(data: HotData, split: str = "train") -> dict:
    """How much of a split's ground and trees fall inside tree_group polygons (areas in UTM)."""
    tiles = data.tiles[data.tiles["split"] == split].to_crs(UTM)
    trees = data.trees[data.trees["split"] == split].to_crs(UTM)
    group_union = data.groups.to_crs(UTM).geometry.union_all()

    # disjoint parts of the union, so each tile only meets the few parts near it
    parts = gpd.GeoSeries(shapely.get_parts(group_union), crs=UTM)
    tile_idx, part_idx = parts.sindex.query(tiles.geometry, predicate="intersects")
    overlap = tiles.geometry.iloc[tile_idx].intersection(parts.iloc[part_idx], align=False).area
    covered = pd.Series(overlap.to_numpy()).groupby(tile_idx).sum().reindex(range(len(tiles)), fill_value=0.0)
    covered = covered.to_numpy()
    return {
        "group_area_share": float(covered.sum() / tiles.geometry.area.sum()),
        "tiles_total": len(tiles),
        "tiles_over_80pct_group": int((covered / tiles.geometry.area.to_numpy() > 0.8).sum()),
        "trees_total": len(trees),
        "trees_in_groups": len(np.unique(parts.sindex.query(trees.geometry, predicate="within")[0])),
    }
