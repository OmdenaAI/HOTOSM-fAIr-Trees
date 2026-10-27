import io
import math

import numpy as np
import pytest
from PIL import Image

gpd = pytest.importorskip("geopandas")

from shapely.geometry import MultiPolygon, Point, box  # noqa: E402

from experiments.label_study.hot import HotData, split_units, unit_mosaic, unit_truth, write_tile  # noqa: E402
from fair_trees.georef import mercator_y_to_lat, tile_bounds_4326  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Dataset has no geotransform")

X0, Y0 = 485900, 499600
TILE_IDS = [f"OAM-{X0 + dx}-{Y0 + dy}-20" for dy in (0, 1) for dx in (0, 1)]


def tiles_at(xys):
    ids = [f"OAM-{x}-{y}-20" for x, y in xys]
    return gpd.GeoDataFrame(
        {"tile_id": ids, "x": [x for x, _ in xys], "y": [y for _, y in xys], "z": 20, "split": "test"},
        geometry=[box(*tile_bounds_4326(x, y, 20)) for x, y in xys],
        crs="EPSG:4326",
    )


def png_bytes(mode="RGB", fmt="PNG"):
    buf = io.BytesIO()
    Image.new(mode, (256, 256), (90, 120, 60, 255)[: len(mode)]).save(buf, format=fmt)
    return buf.getvalue()


IMAGES = {tile_id: png_bytes() for tile_id in TILE_IDS}


def hot_data(with_labels=True):
    west, south, east, north = tile_bounds_4326(X0, Y0, 20)
    w, h = east - west, north - south
    w2, s2, e2, n2 = tile_bounds_4326(X0 + 1, Y0 + 1, 20)
    trees = [
        (1, Point((w2 + e2) / 2, (s2 + n2) / 2), 3.0),  # centre of the bottom-right tile
        (2, Point(west + 0.25 * w, south + 0.5 * h), 2.0),  # inside the group
        (3, Point(east + 0.5 * w, north - 0.5 * h), np.nan),  # centre of the top-right tile, no diameter
    ]
    if not with_labels:
        trees, group = [], []
    else:
        group = [box(west - w, south, west + 0.5 * w, north + 0.5 * h)]
    return HotData(
        trees=gpd.GeoDataFrame(
            {
                "osm_id": [t[0] for t in trees],
                "split": "test",
                "diameter_m": [t[2] for t in trees],
            },
            geometry=[t[1] for t in trees],
            crs="EPSG:4326",
        ),
        groups=gpd.GeoDataFrame({"osm_id": range(100, 100 + len(group))}, geometry=group, crs="EPSG:4326"),
        tiles=tiles_at([(X0 + dx, Y0 + dy) for dy in (0, 1) for dx in (0, 1)]),
    )


def test_units_follow_blocks_and_connectivity():
    # z20 tiles (485900, 499600) and (485902, 499600) share block (121475, 124900) but do not touch;
    # (485904, 499600) is in the next block
    units = split_units(
        tiles_at([(485900, 499600), (485902, 499600), (485904, 499600)]), "test", block_tiles=4
    )
    assert [u for u, _ in units] == ["test-121475-124900-0", "test-121475-124900-1", "test-121476-124900-0"]


def test_unit_mosaic_stitches_the_tiles(tmp_path):
    m = unit_mosaic(IMAGES, TILE_IDS, tmp_path / "u")
    assert m.rgb.shape == (512, 512, 3) and m.z == 20


def test_jpeg_tiles_are_written_unchanged_as_jpg(tmp_path):
    jpegs = {tile_id: png_bytes(fmt="JPEG") for tile_id in TILE_IDS}
    m = unit_mosaic(jpegs, TILE_IDS, tmp_path / "u")
    written = sorted((tmp_path / "u").iterdir())
    assert [p.name for p in written] == sorted(f"{t}.jpg" for t in TILE_IDS)
    assert all(p.read_bytes() == jpegs[p.stem] for p in written) and m.rgb.shape == (512, 512, 3)


def test_truth_points_land_on_their_pixel_and_skip_trees_in_groups(tmp_path):
    m = unit_mosaic(IMAGES, TILE_IDS, tmp_path / "u")
    t = unit_truth(hot_data(), m, TILE_IDS)
    assert len(t.tree_points) == 2  # the tree inside the group is left out
    assert np.allclose(t.tree_points[0], [384, 384], atol=0.5)
    lat = mercator_y_to_lat(m.transform_3857.f - 256 * m.transform_3857.a)
    assert np.isclose(t.tree_diam_px[0], 3.0 / math.cos(math.radians(lat)) / m.transform_3857.a, rtol=0.02)
    assert np.isnan(t.tree_diam_px[1])  # Review Focus: no diameter, still a tree


def test_group_box_is_clipped_to_the_unit(tmp_path):
    m = unit_mosaic(IMAGES, TILE_IDS, tmp_path / "u")
    t = unit_truth(hot_data(), m, TILE_IDS)
    assert t.group_boxes.shape == (1, 4) and t.group_boxes.min() >= 0 and t.group_boxes.max() <= 512
    assert t.group_mask[64, 64] and not t.group_mask[400, 400]
    assert 0.12 < t.group_cover < 0.13
    # the group crosses the unit's west edge: the clipped part is the west half of the top-left tile
    assert t.group_edge.tolist() == [True]
    assert np.allclose(t.group_area_m2, [128 * 256 * m.gsd_m**2], rtol=1e-3)
    assert t.tile_cover.dtype == np.float32 and np.allclose(t.tile_cover, [[0.5, 0], [0, 0]], atol=0.01)


def test_tile_cover_is_zero_off_the_unit(tmp_path):
    tile_ids = TILE_IDS[:3]  # an L of three tiles; the bottom-right square of the mosaic is not in the unit
    west, south, east, north = tile_bounds_4326(X0, Y0, 20)
    data = with_groups([box(west, south - 2 * (north - south), east + (east - west), north)])  # all four
    t = unit_truth(data, unit_mosaic(IMAGES, tile_ids, tmp_path / "u"), tile_ids)
    assert np.allclose(t.tile_cover, [[1, 1], [1, 0]], atol=0.01)


def test_unit_without_labels(tmp_path):
    t = unit_truth(hot_data(with_labels=False), unit_mosaic(IMAGES, TILE_IDS, tmp_path / "u"), TILE_IDS)
    assert t.tree_points.shape == (0, 2) and t.group_boxes.shape == (0, 4) and t.group_cover == 0.0
    assert t.group_area_m2.shape == (0,) and t.group_edge.shape == (0,) and not t.tile_cover.any()


def test_crown_diameters_outside_0_to_30_m_are_unknown(tmp_path):
    data = hot_data()
    trees = data.trees.assign(diameter_m=[40.0, 2.0, 0.0])  # 40 m on tree 1, 0 m on tree 3
    t = unit_truth(
        HotData(trees, data.groups, data.tiles), unit_mosaic(IMAGES, TILE_IDS, tmp_path / "u"), TILE_IDS
    )
    assert np.isnan(t.tree_diam_px).all()


def with_groups(geoms):
    data = hot_data(with_labels=False)
    groups = gpd.GeoDataFrame({"osm_id": range(len(geoms))}, geometry=geoms, crs="EPSG:4326")
    return HotData(trees=data.trees, groups=groups, tiles=data.tiles)


def test_group_touching_only_the_unit_edge_leaves_no_trace(tmp_path):
    west, south, east, north = tile_bounds_4326(X0, Y0, 20)
    data = with_groups([box(west - (east - west), south, west, north)])
    t = unit_truth(data, unit_mosaic(IMAGES, TILE_IDS, tmp_path / "u"), TILE_IDS)
    assert t.group_boxes.shape == (0, 4) and not t.group_mask.any() and t.group_cover == 0.0


def test_group_clip_with_a_polygon_and_a_line_keeps_the_polygon(tmp_path):
    west, south, east, north = tile_bounds_4326(X0, Y0, 20)
    w, h = east - west, north - south
    inside = box(west + 0.25 * w, south + 0.25 * h, west + 0.5 * w, south + 0.5 * h)
    touching = box(west - w, south, west, north)  # meets the unit along its west edge only
    data = with_groups([MultiPolygon([inside, touching])])
    t = unit_truth(data, unit_mosaic(IMAGES, TILE_IDS, tmp_path / "u"), TILE_IDS)
    assert t.group_boxes.shape == (1, 4)
    assert np.allclose(t.group_boxes[0], [64, 128, 128, 192], atol=0.5)


def test_interior_group_does_not_touch_the_edge(tmp_path):
    # it crosses the seam between the top two tiles and stays one part
    west, south, east, north = tile_bounds_4326(X0, Y0, 20)
    w, h = east - west, north - south
    data = with_groups([box(west + 0.25 * w, south + 0.25 * h, east + 0.5 * w, south + 0.5 * h)])
    m = unit_mosaic(IMAGES, TILE_IDS, tmp_path / "u")
    t = unit_truth(data, m, TILE_IDS)
    assert t.group_edge.tolist() == [False]
    assert np.allclose(t.group_area_m2, [320 * 64 * m.gsd_m**2], rtol=1e-3)


def test_write_tile_keeps_png_and_reencodes_other_modes(tmp_path):
    assert write_tile(png_bytes(), "OAM-1-2-20", tmp_path).name == "OAM-1-2-20.png"
    grey = write_tile(png_bytes("L"), "OAM-1-3-20", tmp_path)
    assert Image.open(grey).mode == "RGB"
