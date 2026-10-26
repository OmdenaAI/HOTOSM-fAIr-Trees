import json

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

from experiments.label_study.hot import load_hot, masking_cost, parse_diameter
from fair_trees.georef import tile_bounds_4326

TILE_A, TILE_B = "OAM-485900-499600-20", "OAM-485901-499600-20"


def tiles_frame():
    return gpd.GeoDataFrame(
        {"tile_id": [TILE_A, TILE_B], "split": ["train", "train"]},
        geometry=[box(*tile_bounds_4326(485900, 499600, 20)), box(*tile_bounds_4326(485901, 499600, 20))],
        crs="EPSG:4326",
    )


def labels_frame(rows):
    """rows: (tile_id, osm_id, class, tags, geometry)."""
    tile_id, osm_id, cls, tags, geom = zip(*rows, strict=True)
    return gpd.GeoDataFrame(
        {
            "tile_id": tile_id,
            "osm_id": osm_id,
            "class": cls,
            "tags": [json.dumps(t) for t in tags],
            "split": "train",
        },
        geometry=list(geom),
        crs="EPSG:4326",
    )


def test_parse_diameter():
    assert parse_diameter("3m") == 3.0 and parse_diameter("2.412m") == 2.412
    assert parse_diameter("3,134 m") == 3.134 and parse_diameter("4") == 4.0
    assert np.isnan(parse_diameter(None)) and np.isnan(parse_diameter("big"))


def test_tree_rows_repeated_per_tile_become_one_tree():
    west, south, east, north = tile_bounds_4326(485900, 499600, 20)
    tree = Point(east, (south + north) / 2)
    tags = {"diameter_crown": "3m"}
    rows = [(TILE_A, 7, "tree", tags, tree), (TILE_B, 7, "tree", tags, tree)]
    data = load_hot(labels_frame(rows), tiles_frame())
    assert len(data.trees) == 1 and data.trees.diameter_m.iloc[0] == 3.0


def test_group_pieces_dissolve_into_one_group():
    west, south, east, north = tile_bounds_4326(485900, 499600, 20)
    _, _, east_b, _ = tile_bounds_4326(485901, 499600, 20)
    mid = (south + north) / 2
    rows = [
        (TILE_A, 9, "tree_group", {}, box(west, south, east, mid)),
        (TILE_B, 9, "tree_group", {}, box(east, south, east_b, mid)),
    ]
    data = load_hot(labels_frame(rows), tiles_frame())
    assert len(data.groups) == 1 and data.groups.geometry.iloc[0].geom_type == "Polygon"


def test_masking_cost_counts_area_tiles_and_trees():
    west, south, east, north = tile_bounds_4326(485900, 499600, 20)
    rows = [
        (TILE_A, 9, "tree_group", {}, box(west, south, west + 0.9 * (east - west), north)),
        (TILE_A, 3, "tree", {"diameter_crown": "3m"}, Point(west + 0.4 * (east - west), (south + north) / 2)),
    ]
    cost = masking_cost(load_hot(labels_frame(rows), tiles_frame()))
    assert cost["tiles_total"] == 2 and cost["tiles_over_80pct_group"] == 1
    assert cost["trees_total"] == 1 and cost["trees_in_groups"] == 1
    assert 0.44 < cost["group_area_share"] < 0.46
