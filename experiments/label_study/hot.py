"""Load HOT's Freetown tree labels and measure what masking ``tree_group`` polygons would cost."""

import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from PIL import Image
from rasterio.features import rasterize

from fair_trees.georef import mercator_y_to_lat, parse_oam_name, tile_bounds_3857
from fair_trees.preprocess import Mosaic, mosaic_chips, tile_groups

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


def split_units(tiles: gpd.GeoDataFrame, split: str, block_tiles: int = 16) -> list[tuple[str, list[str]]]:
    """Scoring units as ``(unit_id, tile_ids)``: connected tile runs per block of tiles."""
    tiles = tiles.sort_values("tile_id")
    units = []
    for (bx, by), block in tiles.groupby([tiles["x"] // block_tiles, tiles["y"] // block_tiles]):
        xyzs = list(zip(block["x"], block["y"], block["z"], strict=True))
        ids = block["tile_id"].tolist()
        for k, group in enumerate(tile_groups(xyzs)):
            units.append((f"{split}-{bx}-{by}-{k}", [ids[i] for i in group]))
    return sorted(units)


def unit_mosaic(images: Mapping[str, bytes], tile_ids: Sequence[str], workdir: Path) -> Mosaic:
    """Stitch a unit's tiles into one mosaic; ``workdir`` is created fresh to hold the PNG chips."""
    workdir.mkdir(parents=True)
    paths = []
    for tile_id in tile_ids:
        path = workdir / f"{tile_id}.png"
        image = Image.open(io.BytesIO(images[tile_id]))
        if image.format == "PNG" and image.mode in ("RGB", "RGBA"):
            path.write_bytes(images[tile_id])
        else:
            transparent = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
            image.convert("RGBA" if transparent else "RGB").save(path, format="PNG")
        paths.append(path)
    mosaics = mosaic_chips(paths, use_sidecar=False)
    if len(mosaics) != 1:
        raise ValueError(f"expected one mosaic for {len(tile_ids)} tiles, got {len(mosaics)}")
    return mosaics[0]


@dataclass(frozen=True, eq=False)
class UnitTruth:
    tree_points: np.ndarray  # (P, 2) mosaic px (x, y), trees outside groups, ordered by osm_id
    tree_diam_px: np.ndarray  # (P,) crown diameter in mosaic px, NaN where unknown
    group_boxes: np.ndarray  # (G, 4) mosaic px (x0, y0, x1, y1), one per group part clipped to the unit
    group_mask: np.ndarray  # (H, W) bool, pixels inside a group
    group_cover: float  # share of the unit's tile area inside groups


def unit_truth(data: HotData, mosaic: Mosaic, tile_ids: Sequence[str]) -> UnitTruth:
    """HOT's trees and tree groups in the pixels of a unit's mosaic."""
    to_px = ~mosaic.transform_3857
    squares = [shapely.box(*tile_bounds_3857(*parse_oam_name(f"{t}.png"))) for t in tile_ids]
    unit = shapely.union_all(squares)

    groups = data.groups.to_crs("EPSG:3857").geometry
    groups = groups[groups.sindex.query(unit, predicate="intersects")]
    trees = data.trees.to_crs("EPSG:3857")
    trees = trees[trees.geometry.within(unit)].sort_values("osm_id")
    if len(groups):
        trees = trees[~trees.geometry.within(groups.union_all())]

    xs, ys = to_px @ (trees.geometry.x.to_numpy(), trees.geometry.y.to_numpy())
    # a diameter in metres on the ground is 1 / cos(lat) longer in Web Mercator
    lat = mercator_y_to_lat(mosaic.transform_3857.f - mosaic.rgb.shape[0] / 2 * mosaic.transform_3857.a)
    diam_px = trees["diameter_m"].to_numpy(float) / math.cos(math.radians(lat)) / mosaic.transform_3857.a

    clipped = shapely.get_parts(groups.intersection(unit).to_numpy())
    clipped = clipped[~shapely.is_empty(clipped)]
    cover = float(shapely.union_all(clipped).area / unit.area)
    clipped = shapely.transform(clipped, lambda c: np.column_stack(to_px @ (c[:, 0], c[:, 1])))
    mask = np.zeros(mosaic.rgb.shape[:2], dtype=bool)
    if len(clipped):
        mask = rasterize(clipped, out_shape=mask.shape, fill=0, default_value=1, dtype="uint8").astype(bool)
    return UnitTruth(np.column_stack([xs, ys]), diam_px, shapely.bounds(clipped).reshape(-1, 4), mask, cover)
