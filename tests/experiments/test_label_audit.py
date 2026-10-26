import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

gpd = pytest.importorskip("geopandas")
pd = pytest.importorskip("pandas")

from shapely.geometry import Point, box  # noqa: E402

from experiments.label_study.audit import (  # noqa: E402
    LABEL_COLUMNS,
    coco_labels,
    geo_labels,
    neon_labels,
    profile,
)
from experiments.label_study.hot import HotData, hot_labels  # noqa: E402
from fair_trees.georef import tile_bounds_4326  # noqa: E402


def labels_of(rows, cls="tree"):
    """rows: (region, x_m, y_m, w_m, h_m)."""
    region, x, y, w, h = (list(c) for c in zip(*rows, strict=True))
    frame = pd.DataFrame(
        {"source": "s", "region": region, "cls": cls, "x_m": x, "y_m": y, "w_m": w, "h_m": h}
    )
    frame["area_m2"] = frame.w_m * frame.h_m
    return frame[LABEL_COLUMNS]


def table_with_trees(rows):
    return labels_of([("r", *row) for row in rows])


def table_with_groups_long_sides(sides):
    return labels_of([("r", 100.0 * k, 0.0, side, 5.0) for k, side in enumerate(sides)], cls="group")


def one_region_of(area_m2):
    return pd.DataFrame({"source": ["s"], "region": ["r"], "area_m2": [area_m2]})


two_regions_with_boxes_at_the_same_local_coordinates = labels_of([("a", 0, 0, 2, 2), ("b", 0, 0, 2, 2)])
regions_two = pd.DataFrame({"source": "s", "region": ["a", "b"], "area_m2": [10_000.0, 10_000.0]})


def hot_data(group_tile=None):
    """Tiles A (train) and B (val), a tree without a diameter on A, a group on tile group_tile if given."""
    squares = [box(*tile_bounds_4326(485900 + dx, 499600, 20)) for dx in (0, 1)]
    tiles = gpd.GeoDataFrame(
        {"tile_id": ["A", "B"], "x": [485900, 485901], "y": 499600, "z": 20, "split": ["train", "val"]},
        geometry=squares,
        crs="EPSG:4326",
    )
    trees = gpd.GeoDataFrame(
        {"osm_id": [1], "split": ["train"], "diameter_m": [np.nan]},
        geometry=[squares[0].centroid],
        crs="EPSG:4326",
    )
    group = [] if group_tile is None else [squares[group_tile].centroid.buffer(1e-5)]
    groups = gpd.GeoDataFrame({"osm_id": [2] * len(group)}, geometry=group, crs="EPSG:4326")
    return HotData(trees=trees, groups=groups, tiles=tiles)


@pytest.fixture
def hot_data_one_tree_without_diameter():
    return hot_data()


def test_coco_labels_scale_boxes_and_split_classes():
    coco = {"images": [{"id": 1, "width": 100, "height": 50}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10], "area": 150},
                            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 50, 50]},
                            {"image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1]}]}  # fmt: skip
    labels, regions = coco_labels(coco, source="oamtcd", gsd_m=0.1, tree_ids={1}, group_ids={2})
    assert list(labels.columns) == LABEL_COLUMNS and labels.cls.tolist() == ["tree", "group"]
    assert labels.w_m.iloc[0] == pytest.approx(2.0) and labels.area_m2.iloc[0] == pytest.approx(1.5)
    assert regions.area_m2.iloc[0] == pytest.approx(50.0)


def coco_with(*anns):
    image = {"id": 1, "width": 10, "height": 8}
    return {
        "images": [image],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 8]} | a for a in anns],
    }


def test_coco_labels_measure_zero_area_crowds_from_their_segmentation():
    rle = {"area": 0, "iscrowd": 1, "segmentation": {"size": [8, 10], "counts": [2, 6, 72]}}
    square = {"area": 0, "iscrowd": 0, "segmentation": [[0, 0, 4, 0, 4, 4, 0, 4]]}
    labels, _ = coco_labels(coco_with(rle, square, {"area": 0}), source="oamtcd", gsd_m=0.1, tree_ids={1})
    assert labels.area_m2.tolist() == pytest.approx([0.06, 0.16, 0.8])  # RLE runs, shoelace, box


def test_coco_labels_decode_compressed_rle():
    # "0X12N" is runs [0, 40, 2, 38]: "X1" is 40 in two 5-bit chunks, "N" is -2 added to the run two back
    rle = {"area": 0, "iscrowd": 1, "segmentation": {"size": [8, 10], "counts": "0X12N"}}
    labels, _ = coco_labels(coco_with(rle), source="oamtcd", gsd_m=0.1, tree_ids={1})
    assert labels.area_m2.iloc[0] == pytest.approx(0.78)


def test_profile_touching_and_hot_rule_shares():
    # (x_m, y_m, w_m, h_m): three 2 m crowns in a touching row and one alone
    labels = table_with_trees([(0, 0, 2, 2), (2, 0, 2, 2), (4, 0, 2, 2), (20, 20, 2, 2)])
    p = profile(labels, one_region_of(10_000), gap_m=0.0, min_crowns=3)
    assert p["trees"] == 4 and p["crowns_per_ha"] == pytest.approx(4.0)
    assert p["touching_share"] == pytest.approx(0.75) and p["hot_rule_group_share"] == pytest.approx(0.75)
    assert math.isnan(p["groups_over_40m"])


def test_profile_group_sizes():
    p = profile(table_with_groups_long_sides([10, 30, 50]), one_region_of(10_000), gap_m=0.0, min_crowns=3)
    assert p["groups_over_20m"] == pytest.approx(2 / 3) and p["groups_over_40m"] == pytest.approx(1 / 3)


def test_clusters_never_cross_regions():
    p = profile(two_regions_with_boxes_at_the_same_local_coordinates, regions_two, gap_m=0.0, min_crowns=2)
    assert p["touching_share"] == 0.0


def test_hot_labels_keep_trees_without_diameter(hot_data_one_tree_without_diameter):
    labels, _ = hot_labels(hot_data_one_tree_without_diameter)
    assert (labels.cls == "tree").sum() == 1 and labels.w_m.isna().all()


def test_hot_labels_treat_implausible_diameters_as_unknown():
    data = hot_data()
    trees = pd.concat([data.trees] * 2, ignore_index=True).assign(diameter_m=[4.0, 25_000.0])
    labels, _ = hot_labels(HotData(trees=trees, groups=data.groups, tiles=data.tiles))
    assert labels.w_m.iloc[0] == pytest.approx(4.0)
    assert np.isnan(labels.w_m.iloc[1]) and np.isnan(labels.h_m.iloc[1])  # a typo, kept for the counts


def test_neon_labels(tmp_path):  # tiny 20x10 tif + VOC xml with one 10x5 px box
    tif, xml = tmp_path / "scene.tif", tmp_path / "scene.xml"
    meta = {"driver": "GTiff", "width": 20, "height": 10, "count": 3, "dtype": "uint8", "crs": "EPSG:32617"}
    with rasterio.open(tif, "w", transform=from_origin(400_000, 3_285_000, 0.1, 0.1), **meta) as dst:
        dst.write(np.zeros((3, 10, 20), np.uint8))
    xml.write_text(
        "<annotation><object><name>Tree</name><bndbox><xmin>2</xmin><ymin>3</ymin><xmax>12</xmax><ymax>8</ymax>"
        "</bndbox></object></annotation>"
    )
    labels, regions = neon_labels([(tif, xml)])
    assert labels.w_m.iloc[0] == pytest.approx(1.0) and regions.area_m2.iloc[0] == pytest.approx(2.0)


def test_geo_labels_take_polygon_bounds_and_area():
    crowns = gpd.GeoDataFrame(
        {"region": ["train"]}, geometry=[Point(500_010, 1_000_005).buffer(2)], crs="EPSG:32618"
    )
    aoi = gpd.GeoDataFrame(
        {"region": ["train"]}, geometry=[box(500_000, 1_000_000, 500_100, 1_000_050)], crs=32618
    )
    labels, regions = geo_labels(crowns, aoi, source="quebec")
    row = labels.iloc[0]
    assert (row.x_m, row.y_m, row.w_m, row.h_m) == pytest.approx((500_010, 1_000_005, 4, 4))
    assert row.area_m2 == pytest.approx(math.pi * 4, rel=0.01)
    assert regions.area_m2.iloc[0] == pytest.approx(5000)


def test_hot_labels_put_a_group_in_its_tile_split():
    labels, regions = hot_labels(hot_data(group_tile=1))
    assert labels.loc[labels.cls == "group", "region"].tolist() == ["val"]
    assert regions.region.tolist() == ["train", "val"]
    assert regions.area_m2.iloc[0] == pytest.approx(37.7**2, rel=0.01)  # a z20 tile at 8.4 N is ~37.7 m
