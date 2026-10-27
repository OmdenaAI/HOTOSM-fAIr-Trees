import io
import json

import numpy as np
import pytest
from PIL import Image

gpd = pytest.importorskip("geopandas")

from shapely.geometry import Point, box, shape  # noqa: E402

from experiments.converters.hot import convert  # noqa: E402
from experiments.label_study.hot import UTM, HotData  # noqa: E402
from fair_trees.dataset import load_regions  # noqa: E402
from fair_trees.georef import tile_bounds_4326, true_gsd  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Dataset has no geotransform")

X0, Y0 = 485900, 499600
# one region per tile (block_tiles=1): A, B and E train, C validation, D test
TILES = {"A": (0, 0, "train"), "B": (1, 0, "train"), "C": (2, 0, "validation"), "D": (3, 0, "test")}
TILES["E"] = (0, 1, "train")  # below A, no trees: label-free
REGION = {
    "A": "hot-train-485900-499600-0",
    "B": "hot-train-485901-499600-0",
    "C": "hot-val-485902-499600-0",
    "E": "hot-train-485900-499601-0",
}
# osm_id: (tile, fraction east, fraction south, diameter_m, split); a z20 tile here is ~37.8 m wide
TREES = {
    1: ("A", 0.5, 0.5, 3.0, "train"),
    2: ("A", 0.97, 0.3, 6.0, "train"),  # ~1.1 m from B: its 6 m square crosses into B
    3: ("B", 0.3, 0.3, np.nan, "train"),
    4: ("B", 0.3, 0.7, 0.0, "train"),
    5: ("B", 0.7, 0.7, 31.0, "train"),
    6: ("C", 0.75, 0.75, 5.0, "validation"),  # inside the C group
    7: ("C", 0.5, 0.25, 4.0, "validation"),
    8: ("C", 0.03, 0.5, 6.0, "validation"),  # ~1.1 m from B: a val crown reaching into a train region
    9: ("D", 0.5, 0.5, 3.0, "test"),
}
GROUPS = {"C": (0.5, 0.5, 0.9, 0.9), "D": (0.2, 0.2, 0.4, 0.4)}  # fractions east0, south0, east1, south1


def tile_xy(name):
    dx, dy, _ = TILES[name]
    return X0 + dx, Y0 + dy


def at(name, fx, fy):
    west, south, east, north = tile_bounds_4326(*tile_xy(name), 20)
    return west + fx * (east - west), north - fy * (north - south)


def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (90, 120, 60)).save(buf, format="PNG")
    return buf.getvalue()


def hot_data():
    names = list(TILES)
    tiles = gpd.GeoDataFrame(
        {
            "tile_id": [f"OAM-{tile_xy(n)[0]}-{tile_xy(n)[1]}-20" for n in names],
            "x": [tile_xy(n)[0] for n in names],
            "y": [tile_xy(n)[1] for n in names],
            "z": 20,
            "split": [TILES[n][2] for n in names],
        },
        geometry=[box(*tile_bounds_4326(*tile_xy(n), 20)) for n in names],
        crs="EPSG:4326",
    )
    trees = gpd.GeoDataFrame(
        {
            "osm_id": list(TREES),
            "split": [t[4] for t in TREES.values()],
            "diameter_m": [t[3] for t in TREES.values()],
        },
        geometry=[Point(at(t[0], t[1], t[2])) for t in TREES.values()],
        crs="EPSG:4326",
    )
    groups = gpd.GeoDataFrame(
        {"osm_id": [100, 101]},
        geometry=[box(*at(n, f[0], f[3]), *at(n, f[2], f[1])) for n, f in GROUPS.items()],
        crs="EPSG:4326",
    )
    return HotData(trees=trees, groups=groups, tiles=tiles)


class Images:
    """``images(hot_split)`` over the fixture tiles, recording which splits were asked for."""

    def __init__(self, data):
        self.data = data
        self.asked = []

    def __call__(self, hot_split):
        self.asked.append(hot_split)
        for tile_id in self.data.tiles.loc[self.data.tiles["split"] == hot_split, "tile_id"]:
            yield tile_id, png_bytes()


def run(tmp_path, **kwargs):
    data = hot_data()
    images = Images(data)
    stats = convert(data, images, tmp_path / "out", block_tiles=1, **kwargs)
    return tmp_path / "out", stats, images


def features(out, name, kind):
    return json.loads((out / REGION[name] / f"{kind}.geojson").read_text())["features"]


def utm_shapes(feats):
    return gpd.GeoSeries([shape(f["geometry"]) for f in feats], crs="EPSG:4326").to_crs(UTM)


def utm_point(osm_id):
    return gpd.GeoSeries([Point(at(*TREES[osm_id][:3]))], crs="EPSG:4326").to_crs(UTM).iloc[0]


def square_for(feats, osm_id):
    """The feature centred nearest to the tree, as a UTM polygon."""
    shapes = utm_shapes(feats)
    return shapes.iloc[int(shapes.centroid.distance(utm_point(osm_id)).argmin())]


def assert_square(poly, osm_id, side):
    x0, y0, x1, y1 = poly.bounds
    assert x1 - x0 == pytest.approx(side, abs=0.01) and y1 - y0 == pytest.approx(side, abs=0.01)
    assert poly.centroid.distance(utm_point(osm_id)) < 0.01


def test_regions_follow_blocks_and_skip_test(tmp_path):
    out, _, images = run(tmp_path)
    ids = sorted(p.name for p in out.iterdir() if p.is_dir())
    assert ids == sorted(REGION.values())
    assert not any(i.startswith("hot-test-") for i in ids)
    assert sorted(images.asked) == ["train", "validation"]
    regions = load_regions(out)
    assert [r.id for r in regions] == ids
    assert all(len(r.chips[20]) == 1 and r.ignore_path is not None for r in regions)


def test_chips_are_the_region_tiles(tmp_path):
    out, _, _ = run(tmp_path)
    for name, region_id in REGION.items():
        x, y = tile_xy(name)
        assert [p.name for p in (out / region_id / "z20" / "chips").iterdir()] == [f"OAM-{x}-{y}-20.png"]


def test_tree_box_is_a_ground_square_of_its_diameter(tmp_path):
    out, _, _ = run(tmp_path)
    labels = features(out, "A", "labels")
    assert len(labels) == 2
    assert labels[0]["properties"] == {
        "label": 1,
        "source": "hot",
        "licence": "CC-BY-4.0 imagery, ODbL-1.0 labels",
        "region": REGION["A"],
    }
    assert_square(square_for(labels, 1), 1, 3.0)
    assert_square(square_for(labels, 2), 2, 6.0)


def test_box_crossing_a_block_edge_is_in_both_regions(tmp_path):
    out, _, _ = run(tmp_path)
    (in_b,) = features(out, "B", "labels")  # tree 2's point is in A
    assert in_b["properties"]["region"] == REGION["B"]
    assert_square(utm_shapes([in_b]).iloc[0], 2, 6.0)
    assert_square(square_for(features(out, "A", "labels"), 2), 2, 6.0)


def test_unknown_diameter_tree_is_a_4_m_ignore_square(tmp_path):
    out, _, _ = run(tmp_path)
    unknown = [f for f in features(out, "B", "ignore") if f["properties"] == {"reason": "unknown_diameter"}]
    assert len(unknown) == 3
    for osm_id in (3, 4, 5):  # NaN, 0 m and 31 m
        assert_square(square_for(unknown, osm_id), osm_id, 4.0)


def test_crown_from_another_split_is_masked_not_labelled(tmp_path):
    out, _, _ = run(tmp_path)
    other = [f for f in features(out, "B", "ignore") if f["properties"] == {"reason": "other_split"}]
    assert len(other) == 1
    assert_square(utm_shapes(other).iloc[0], 8, 6.0)
    assert_square(square_for(features(out, "C", "labels"), 8), 8, 6.0)


def test_tree_inside_a_group_gets_no_box_and_no_square(tmp_path):
    out, _, _ = run(tmp_path)
    tree = utm_point(6)
    labels, ignore = features(out, "C", "labels"), features(out, "C", "ignore")
    assert len(labels) == 2  # trees 7 and 8
    assert utm_shapes(labels).distance(tree).min() > 1.0
    (group,) = ignore
    assert group["properties"] == {"reason": "tree_group"}
    assert utm_shapes(ignore).iloc[0].contains(tree)


def test_groups_are_ignore_polygons(tmp_path):
    out, _, _ = run(tmp_path)
    (group,) = features(out, "C", "ignore")
    assert group["geometry"]["type"] == "Polygon"
    expected = hot_data().groups.geometry.iloc[0]
    assert shape(group["geometry"]).symmetric_difference(expected).area < 1e-12
    # the test tile's group is never written
    assert all(f["properties"]["reason"] != "tree_group" for n in "ABE" for f in features(out, n, "ignore"))


def test_region_json_and_stats(tmp_path):
    centres = []

    def biome_fn(lon, lat):
        centres.append((lon, lat))
        return "tropical"

    out, stats, _ = run(tmp_path, biome_fn=biome_fn)
    meta = json.loads((out / REGION["C"] / "region.json").read_text())
    assert set(meta) == {"id", "source", "biome", "licence", "split", "zooms", "footprint", "valid_area_m2"}
    assert meta["id"] == REGION["C"] and meta["split"] == "val" and meta["zooms"] == [20]
    assert meta["source"] == "hot" and meta["biome"] == "tropical"
    assert meta["licence"] == "CC-BY-4.0 imagery, ODbL-1.0 labels"
    west, south, east, north = tile_bounds_4326(*tile_xy("C"), 20)
    assert shape(meta["footprint"]).symmetric_difference(box(west, south, east, north)).area < 1e-14
    assert meta["valid_area_m2"] == pytest.approx(256**2 * true_gsd(20, (south + north) / 2) ** 2)
    assert ((west + east) / 2, (south + north) / 2) in [pytest.approx(c) for c in centres]
    assert json.loads((out / REGION["A"] / "region.json").read_text())["split"] == "train"

    assert stats == json.loads((out / "stats.json").read_text())
    assert stats["regions"] == {"train": 3, "val": 1}
    assert stats["chips"] == {"train": 3, "val": 1}
    assert stats["boxes"] == 4  # trees 1, 2, 7, 8
    assert stats["ignored_unknown_diameter"] == 3
    assert stats["masked_other_split"] == 1
    assert stats["trees_in_groups"] == 1
    assert stats["groups"] == 1
    assert stats["label_free_tile_share"] == {"train": pytest.approx(1 / 3), "val": 0.0}
    assert stats["val_share"] == pytest.approx(0.25, abs=1e-3)


def test_region_footprint_is_the_union_of_its_tiles(tmp_path):
    data = hot_data()
    out = tmp_path / "out"
    convert(data, Images(data), out, block_tiles=16)  # A, B and E share a block and touch
    (train,) = [r for r in load_regions(out) if r.split == "train"]
    assert len(train.chips[20]) == 3
    tiles = [box(*tile_bounds_4326(*tile_xy(n), 20)) for n in "ABE"]
    footprint = shape(train.footprint)
    assert footprint.geom_type == "Polygon"
    assert footprint.symmetric_difference(tiles[0].union(tiles[1]).union(tiles[2])).area < 1e-14


def test_convert_refuses_a_non_empty_out_root(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "old-region").mkdir()
    data = hot_data()
    with pytest.raises(ValueError, match="not empty"):
        convert(data, Images(data), out)


def test_convert_refuses_an_image_for_an_unknown_tile(tmp_path):
    data = hot_data()

    def images(hot_split):
        yield "OAM-1-1-20", png_bytes()

    with pytest.raises(ValueError, match="OAM-1-1-20"):
        convert(data, images, tmp_path / "out")


def biome_by_tile(on_polygon):
    """A biome_fn that answers "forest" at the centres of the named tiles and "unknown" elsewhere."""
    centres = {
        n: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for n in TILES for b in [tile_bounds_4326(*tile_xy(n), 20)]
    }

    def biome_fn(lon, lat):
        return "forest" if any((lon, lat) == pytest.approx(centres[n]) for n in on_polygon) else "unknown"

    return biome_fn


def biomes(out):
    return {n: json.loads((out / r / "region.json").read_text())["biome"] for n, r in REGION.items()}


def test_biome_is_the_commonest_answer_over_a_regions_tiles(tmp_path):
    # block_tiles=2: A, B and E form one train region whose tiles are mostly on-polygon
    data = hot_data()
    stats = convert(data, Images(data), tmp_path / "out", block_tiles=2, biome_fn=biome_by_tile("AB"))
    metas = [json.loads(p.read_text()) for p in (tmp_path / "out").glob("*/region.json")]
    assert {m["split"]: m["biome"] for m in metas if m["split"] == "train"} == {"train": "forest"}
    assert stats["regions_per_biome"] == {
        "forest": 2
    }  # the train region, and val C by the conversion-wide majority


def test_region_off_polygon_takes_the_conversion_wide_majority(tmp_path):
    out, stats, _ = run(tmp_path, biome_fn=biome_by_tile("AB"))
    assert biomes(out) == {"A": "forest", "B": "forest", "C": "forest", "E": "forest"}
    assert stats["regions_per_biome"] == {"forest": 4}


def test_regions_per_biome_counts_regions(tmp_path):
    out, stats, _ = run(tmp_path, biome_fn=biome_by_tile("ABE"))
    assert stats["regions_per_biome"] == {"forest": 4}
    assert json.loads((out / "stats.json").read_text())["regions_per_biome"] == {"forest": 4}


def test_no_biome_fn_gives_unknown(tmp_path):
    out, stats, _ = run(tmp_path)
    assert set(biomes(out).values()) == {"unknown"}
    assert stats["regions_per_biome"] == {"unknown": 4}
