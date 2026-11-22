import csv
import io
import json
import math

import numpy as np
import pytest

gpd = pytest.importorskip("geopandas")

from PIL import Image  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

from experiments.gold_crowns import (  # noqa: E402
    MISSING_GREY,
    STRATA,
    UNSIZED,
    assemble,
    crop_window,
    sample_gold,
    stratum,
    window_tiles,
    write_gold,
)
from experiments.label_study.hot import TILE_PX, HotData  # noqa: E402
from fair_trees.georef import tile_bounds_4326  # noqa: E402

X, Y, Z = 485900, 499600, 20  # Freetown


def test_stratum_bins_presets_and_unsized():
    assert [stratum(d) for d in (1.0, 1.5, 2.0, 2.231, 2.412, 2.5, 4.0, 6.0, 30.0)] == [
        "<1.5",
        "1.5-2.5",
        "1.5-2.5",
        "1.5-2.5 preset",
        "1.5-2.5 preset",
        "2.5-4",
        "4-6",
        ">=6",
        ">=6",
    ]
    assert [stratum(d) for d in (math.nan, 0.0, 31.0)] == [UNSIZED] * 3


def test_crop_window_is_20_m_centred_on_the_point():
    west, south, east, north = tile_bounds_4326(X, Y, Z)
    lon, lat = west + 0.5 * (east - west), south + 0.5 * (north - south)
    w = crop_window(lon, lat, Z)
    assert w.m_per_px == pytest.approx(0.1477, abs=1e-3)
    assert w.size * w.m_per_px == pytest.approx(20.0, abs=w.m_per_px)
    assert w.point[0] == pytest.approx(w.size / 2, abs=1) and w.point[1] == pytest.approx(w.size / 2, abs=1)
    assert window_tiles(w) == [(X, Y)]


def test_crop_window_near_a_corner_spans_four_tiles():
    west, south, east, north = tile_bounds_4326(X, Y, Z)
    w = crop_window(east - 1e-7, south + 1e-7, Z)  # bottom-right corner of the tile
    assert window_tiles(w) == [(X, Y), (X + 1, Y), (X, Y + 1), (X + 1, Y + 1)]


def test_assemble_stitches_tiles_and_greys_missing_ones():
    west, south, east, north = tile_bounds_4326(X, Y, Z)
    w = crop_window(east - 1e-7, (south + north) / 2, Z)  # right edge: this tile and its east neighbour
    tiles = {(X, Y): np.full((TILE_PX, TILE_PX, 3), 10, np.uint8)}
    rgb, grey = assemble(w, lambda x, y: tiles.get((x, y)))
    assert rgb.shape == (w.size, w.size, 3)
    half = w.size // 2
    assert (rgb[:, : half - 1] == 10).all() and (rgb[:, half + 1 :] == MISSING_GREY).all()
    assert grey == pytest.approx(0.5, abs=2 / w.size)


def _png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((TILE_PX, TILE_PX, 3), value, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _data(diameters, splits, group_over_first=False) -> HotData:
    west, south, east, north = tile_bounds_4326(X, Y, Z)
    n = len(diameters)
    points = [Point(west + (k + 0.5) / n * (east - west), (south + north) / 2) for k in range(n)]
    trees = gpd.GeoDataFrame(
        {"osm_id": np.arange(n), "split": splits, "diameter_m": diameters}, geometry=points, crs="EPSG:4326"
    )
    groups = gpd.GeoDataFrame(
        {"osm_id": [999]} if group_over_first else {"osm_id": []},
        geometry=[points[0].buffer(1e-7)] if group_over_first else [],
        crs="EPSG:4326",
    )
    tiles = gpd.GeoDataFrame(
        {"tile_id": [f"OAM-{X}-{Y}-{Z}"], "x": [X], "y": [Y], "z": [Z], "split": ["train"]},
        geometry=[box(west, south, east, north)],
        crs="EPSG:4326",
    )
    return HotData(trees=trees, groups=groups, tiles=tiles)


def test_sample_gold_caps_each_split_and_stratum_skips_groups_and_pools_unsized():
    diameters = [1.0] * 50 + [2.231] * 5 + [5.0] * 3 + [math.nan] * 45 + [1.0] * 4
    splits = ["train"] * 103 + ["test"] * 4
    data = _data(diameters, splits, group_over_first=True)
    s = sample_gold(data, seed=1)
    counts = s.groupby(["split", "stratum"]).size().to_dict()
    n_small = dict((name, n) for name, _, _, n in STRATA)["<1.5"]
    assert counts == {
        ("train", "<1.5"): n_small,
        ("train", "1.5-2.5 preset"): 5,
        ("train", "4-6"): 3,
        ("train", UNSIZED): 40,
        ("test", "<1.5"): 4,
    }
    assert 0 not in set(s["osm_id"])  # inside a tree_group
    assert s.equals(sample_gold(data, seed=1)) and not s.equals(sample_gold(data, seed=2))


def test_write_gold_keeps_tags_out_of_the_annotate_folder(tmp_path):
    data = _data([1.0, 2.0, 5.0, math.nan], ["train", "train", "validation", "test"])
    sample = sample_gold(data)
    images = [(f"OAM-{X}-{Y}-{Z}", _png(60)), ("OAM-1-1-20", _png(0))]  # the second is not under any crop
    stats = write_gold(sample, data, images, tmp_path / "gold", n_second=2)
    assert stats["trees"] == 4 and stats["second"] == 2 and stats["tiles_missing"] == 0

    annotate = tmp_path / "gold" / "annotate"
    items = json.loads((annotate / "items.json").read_text())
    second = json.loads((annotate / "items_second.json").read_text())
    assert items["set"] == "items" and second["set"] == "items_second" and len(second["items"]) == 2
    assert {i["id"] for i in second["items"]} <= {i["id"] for i in items["items"]}
    for item in items["items"]:
        assert set(item) == {"id", "file", "size_px", "m_per_px", "point"}
        crop = np.asarray(Image.open(annotate / item["file"]))
        assert crop.shape == (item["size_px"], item["size_px"], 3)
    assert (annotate / "gold_annotator.html").is_file()

    with (tmp_path / "gold" / "key" / "manifest.csv").open() as f:
        key = list(csv.DictReader(f))
    assert sorted(r["gold_id"] for r in key) == sorted(i["id"] for i in items["items"])
    assert sorted(r["stratum"] for r in key) == sorted(["<1.5", "1.5-2.5", "4-6", UNSIZED])
    assert sum(r["second"] == "True" for r in key) == 2
    # trees at the tile's left and right edges reach into missing neighbours
    assert stats["crops_with_grey"] >= 1


def test_write_gold_refuses_an_existing_folder(tmp_path):
    data = _data([1.0], ["train"])
    (tmp_path / "gold" / "annotate" / "crops").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        write_gold(sample_gold(data), data, [], tmp_path / "gold")
