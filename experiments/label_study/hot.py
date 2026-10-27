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

from experiments.label_study.audit import label_table, region_table
from fair_trees.georef import mercator_y_to_lat, parse_oam_name
from fair_trees.preprocess import Mosaic, mosaic_chips, tile_groups

REPO_ID = "hotosm/fAIr-guru-td-trees"
REVISION = "v0.1.0"
UTM = "EPSG:32628"
# diameter_crown typos reach 25,231 m; one such square covers the city and chains every tree into one cluster
MAX_DIAMETER_M = 30.0
TILE_PX = 256  # side of a HOT tile
KEPT_FORMATS = {"PNG": ".png", "JPEG": ".jpg"}  # tile formats written to the chips unchanged


def parse_diameter(value) -> float:
    """Metres from strings like '3m', '2.412m', '3,134 m', '4'; NaN if missing or unclear."""
    if not isinstance(value, str):
        return np.nan
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(?:m|M|metres|meters)?\s*", value)
    return float(m.group(1).replace(",", ".")) if m else np.nan


def known_diameter(diameter_m: pd.Series) -> pd.Series:
    """Crown diameters with those outside (0, MAX_DIAMETER_M] set to NaN, like a missing one."""
    return diameter_m.where((diameter_m > 0) & (diameter_m <= MAX_DIAMETER_M))


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


def hot_labels(data: HotData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HOT as audit tables: one region per split, trees as squares of their crown diameter.

    A diameter outside (0, MAX_DIAMETER_M] counts as unknown, like a missing one: the tree is kept, with NaN
    size. A group belongs to the split of the tile under its representative point.
    """
    trees = data.trees.to_crs(UTM)
    groups = data.groups.to_crs(UTM)
    tiles = data.tiles.to_crs(UTM)

    x, y, half = trees.geometry.x, trees.geometry.y, known_diameter(trees["diameter_m"]) / 2
    tree_rows = label_table("hot", trees["split"].to_numpy(), "tree", x - half, y - half, x + half, y + half)
    points = gpd.GeoDataFrame(geometry=groups.geometry.representative_point(), crs=UTM)
    under = points.sjoin(tiles[["split", "geometry"]], predicate="intersects")
    split = under[~under.index.duplicated()]["split"].reindex(groups.index)  # a point on a tile edge: first
    group_rows = label_table(
        "hot", split.to_numpy(), "group", *groups.geometry.bounds.to_numpy().T, groups.area
    )

    area = tiles.geometry.area.groupby(tiles["split"]).sum()
    return pd.concat([tree_rows, group_rows], ignore_index=True), region_table("hot", area.index, area)


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


def write_tile(image: bytes, tile_id: str, out_dir: Path) -> Path:
    """Write a tile as ``{tile_id}.png|.jpg``: RGB/RGBA PNG or JPEG unchanged, anything else as PNG."""
    img = Image.open(io.BytesIO(image))
    if img.format in KEPT_FORMATS and img.mode in ("RGB", "RGBA"):
        path = out_dir / f"{tile_id}{KEPT_FORMATS[img.format]}"
        path.write_bytes(image)
    else:
        path = out_dir / f"{tile_id}.png"
        transparent = img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info
        img.convert("RGBA" if transparent else "RGB").save(path, format="PNG")
    return path


def unit_mosaic(images: Mapping[str, bytes], tile_ids: Sequence[str], workdir: Path) -> Mosaic:
    """Stitch a unit's tiles into one mosaic; ``workdir`` is created fresh to hold the chips."""
    workdir.mkdir(parents=True)
    paths = [write_tile(images[tile_id], tile_id, workdir) for tile_id in tile_ids]
    mosaics = mosaic_chips(paths, use_sidecar=False)
    if len(mosaics) != 1:
        raise ValueError(f"expected one mosaic for {len(tile_ids)} tiles, got {len(mosaics)}")
    return mosaics[0]


@dataclass(frozen=True, eq=False)
class UnitTruth:
    tree_points: np.ndarray  # (P, 2) mosaic px (x, y), trees outside groups, ordered by osm_id
    tree_diam_px: np.ndarray  # (P,) crown diameter in mosaic px, NaN where unknown
    group_boxes: np.ndarray  # (G, 4) mosaic px (x0, y0, x1, y1), one per group part clipped to the unit
    group_area_m2: np.ndarray  # (G,) ground area of each clipped part
    group_edge: np.ndarray  # (G,) bool, the clipped part touches the unit's boundary
    group_mask: np.ndarray  # (H, W) bool, pixels inside a group
    group_cover: float  # share of the unit's tile area inside groups
    tile_cover: np.ndarray  # (H / TILE_PX, W / TILE_PX) float32, each tile's share in groups, 0 off the unit


def unit_truth(data: HotData, mosaic: Mosaic, tile_ids: Sequence[str]) -> UnitTruth:
    """HOT's trees and tree groups in the pixels of a unit's mosaic."""
    to_px = ~mosaic.transform_3857
    # tile squares from the mosaic's pixel grid share exact edges; tile_bounds_3857 leaves ~1e-9 m gaps
    # between neighbours, which would split every group crossing a tile seam and put a boundary there
    xyz = np.array([parse_oam_name(f"{t}.png") for t in tile_ids])
    cols, rows = (xyz[:, :2] - xyz[:, :2].min(axis=0)).T * TILE_PX
    squares = shapely.box(cols, rows, cols + TILE_PX, rows + TILE_PX)
    unit = shapely.union_all(
        shapely.transform(squares, lambda c: np.column_stack(mosaic.transform_3857 @ (c[:, 0], c[:, 1])))
    )

    groups = data.groups.to_crs("EPSG:3857").geometry
    groups = groups[groups.sindex.query(unit, predicate="intersects")]
    trees = data.trees.to_crs("EPSG:3857")
    trees = trees[trees.geometry.within(unit)].sort_values("osm_id")
    if len(groups):
        trees = trees[~trees.geometry.within(groups.union_all())]

    xs, ys = to_px @ (trees.geometry.x.to_numpy(), trees.geometry.y.to_numpy())
    # a diameter in metres on the ground is 1 / cos(lat) longer in Web Mercator
    lat = mercator_y_to_lat(mosaic.transform_3857.f - mosaic.rgb.shape[0] / 2 * mosaic.transform_3857.a)
    diameter = known_diameter(trees["diameter_m"]).to_numpy(float)
    diam_px = diameter / math.cos(math.radians(lat)) / mosaic.transform_3857.a

    clipped = shapely.get_parts(groups.intersection(unit).to_numpy())
    clipped = clipped[shapely.area(clipped) > 0]  # drop empties and the lines or points of mere edge contact
    cover = float(shapely.union_all(clipped).area / unit.area)
    # clipping puts vertices on the boundary up to rounding: a hundredth of a pixel counts as touching
    edge = shapely.dwithin(clipped, unit.boundary, 0.01 * mosaic.transform_3857.a)
    clipped = shapely.transform(clipped, lambda c: np.column_stack(to_px @ (c[:, 0], c[:, 1])))
    mask = np.zeros(mosaic.rgb.shape[:2], dtype=bool)
    if len(clipped):
        mask = rasterize(clipped, out_shape=mask.shape, fill=0, default_value=1, dtype="uint8").astype(bool)
    tiles = mask.reshape(mask.shape[0] // TILE_PX, TILE_PX, mask.shape[1] // TILE_PX, TILE_PX)
    return UnitTruth(
        tree_points=np.column_stack([xs, ys]),
        tree_diam_px=diam_px,
        group_boxes=shapely.bounds(clipped).reshape(-1, 4),
        group_area_m2=shapely.area(clipped) * mosaic.gsd_m**2,
        group_edge=edge,
        group_mask=mask,
        group_cover=cover,
        tile_cover=tiles.mean(axis=(1, 3), dtype=np.float32),
    )
