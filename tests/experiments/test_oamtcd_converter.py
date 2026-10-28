import io
import json

import numpy as np
import pytest
import shapely
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from experiments.converters import oamtcd
from fair_trees.dataset import load_regions

pytest.importorskip("geopandas")

CRS = "EPSG:3395"
X0, Y0 = 100_000.0, 100_000.0  # EPSG:3395 metres, ~0.9 degrees north and east: z19 and z20 both exist
IDS = {"tree": 2, "canopy": 1}
FAR_HOT = (-13.30, 8.40, -13.10, 8.50)  # Freetown-sized box, far from the test tiles

# column-major COCO RLE of [[0,0,0,0,1],[0,1,1,0,0],[0,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0],[0,0,0,0,1]],
# compressed by pycocotools.mask.encode
SMALL_MASK = np.array(
    [[0, 0, 0, 0, 1], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 1]],
    bool,
)
SMALL_RLE = {"size": [6, 5], "counts": "73305NL0"}
# a 300 x 200 mask, ones in rows 10:250 and columns 20:180 except a hole at rows 100:120, columns 50:60
BIG_RLE = {
    "size": [300, 200],
    "counts": (
        "jk5`7l10000000000000000000000000000000000000000000000000000000000ZKhNX1X1hNhNX1X1hNhNX1X1h"
        "NhNX1X1hNhNX1X1hNhNX1X1hNhNX1X1hNhNX1X1hNhNX1X1hNhNX1X1^3000000000000000000000000000000000"
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
        "0000000000000000000000000Vk5"
    ),
}


def test_decode_rle_compressed_and_list_counts():
    assert (oamtcd.decode_rle(SMALL_RLE) == SMALL_MASK).all()
    assert (oamtcd.decode_rle({"size": [6, 5], "counts": [7, 3, 3, 3, 8, 1, 4, 1]}) == SMALL_MASK).all()


def test_decode_rle_long_compressed_string():
    expected = np.zeros((300, 200), bool)
    expected[10:250, 20:180] = True
    expected[100:120, 50:60] = False
    assert (oamtcd.decode_rle(BIG_RLE) == expected).all()


def _ann(category, bbox, segmentation=None):
    x, y, w, h = bbox
    seg = segmentation if segmentation is not None else [[x, y, x + w, y, x + w, y + h, x, y + h]]
    return {"category_id": category, "bbox": [x, y, w, h], "segmentation": seg, "area": w * h, "iscrowd": 0}


def test_category_ids_picks_the_numerous_small_class_as_tree():
    rows = [[_ann(7, (0, 0, 20, 20)) for _ in range(5)] + [_ann(3, (0, 0, 400, 400))]]
    assert oamtcd.category_ids(rows) == {"tree": 7, "canopy": 3}


def test_category_ids_refuses_when_canopy_is_not_larger():
    rows = [[_ann(7, (0, 0, 400, 400)) for _ in range(5)] + [_ann(3, (0, 0, 20, 20))]]
    with pytest.raises(ValueError, match="canopy"):
        oamtcd.category_ids(rows)


TRANSFORM = from_bounds(X0, Y0, X0 + 51.2, Y0 + 51.2, 512, 512)


def _shape(feature):
    return shapely.geometry.shape(feature["geometry"])


def _lonlat(px, py):
    x, y = TRANSFORM @ (px, py)
    west, south, _, _ = transform_bounds(CRS, "EPSG:4326", x, y, x, y)
    return shapely.Point(west, south)


def test_tile_features_places_tree_boxes():
    labels, ignore, counts = oamtcd.tile_features([_ann(2, (100, 200, 40, 30))], TRANSFORM, CRS, IDS)
    assert ignore == [] and counts == {"trees": 1, "large_trees": 0, "canopy": 0, "ignore_px": 0.0}
    west, south, east, north = _shape(labels[0]).bounds
    x1, y2 = TRANSFORM @ (100, 230)
    x2, y1 = TRANSFORM @ (140, 200)
    expected = transform_bounds(CRS, "EPSG:4326", x1, y2, x2, y1)
    assert np.allclose((west, south, east, north), expected, atol=1e-9)


def test_tile_features_large_tree_becomes_ignore():
    labels, ignore, counts = oamtcd.tile_features([_ann(2, (0, 0, 350, 40))], TRANSFORM, CRS, IDS)
    assert labels == [] and [f["properties"]["reason"] for f in ignore] == ["large_tree"]
    assert counts["large_trees"] == 1


def test_tile_features_large_tree_limit_is_in_ground_metres():
    # 350 px at 6.3 cm on the ground (a 10 cm Mercator pixel at 51 degrees) is a 22 m crown: a label
    labels, ignore, _ = oamtcd.tile_features([_ann(2, (0, 0, 350, 40))], TRANSFORM, CRS, IDS, gsd_m=0.063)
    assert len(labels) == 1 and ignore == []


def test_tile_features_canopy_keeps_its_trees_unmasked():
    anns = [_ann(1, (0, 0, 300, 300)), _ann(2, (50, 50, 20, 20))]
    labels, ignore, _ = oamtcd.tile_features(anns, TRANSFORM, CRS, IDS)
    (canopy,) = ignore
    assert canopy["properties"]["reason"] == "canopy" and len(labels) == 1
    assert not _shape(canopy).contains(_lonlat(60, 60))  # the tree's centre
    assert _shape(canopy).contains(_lonlat(200, 200))


def test_tile_features_canopy_covered_by_trees_writes_nothing():
    anns = [_ann(1, (50, 50, 20, 20)), _ann(2, (50, 50, 20, 20))]
    labels, ignore, counts = oamtcd.tile_features(anns, TRANSFORM, CRS, IDS)
    assert len(labels) == 1 and ignore == [] and counts["canopy"] == 0


def _rle_list(mask):
    """Uncompressed COCO RLE: column-major run lengths, starting with zeros."""
    flat = mask.flatten(order="F")
    edges = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    runs = np.diff(np.concatenate([[0], edges, [flat.size]])).tolist()
    return {"size": list(mask.shape), "counts": ([0] if flat[0] else []) + runs}


def test_tile_features_rle_canopy():
    mask = np.zeros((512, 512), bool)
    mask[100:150, 100:110] = True
    assert (oamtcd.decode_rle(_rle_list(mask)) == mask).all()
    anns = [
        {
            "category_id": 1,
            "bbox": [100, 100, 10, 50],
            "segmentation": _rle_list(mask),
            "area": 0,
            "iscrowd": 1,
        }
    ]
    _, ignore, counts = oamtcd.tile_features(anns, TRANSFORM, CRS, IDS)
    assert counts["canopy"] == 1 and _shape(ignore[0]).contains(_lonlat(105, 125))


def test_tile_features_simplifies_rle_canopy_outlines():
    yy, xx = np.mgrid[:512, :512]
    disk = (yy - 256) ** 2 + (xx - 256) ** 2 < 200**2  # its pixel outline is a staircase of ~1,600 corners
    anns = [{"category_id": 1, "bbox": [56, 56, 400, 400], "segmentation": _rle_list(disk), "iscrowd": 1}]
    _, ignore, counts = oamtcd.tile_features(anns, TRANSFORM, CRS, IDS)
    assert shapely.get_num_coordinates(_shape(ignore[0])) < 200
    assert counts["ignore_px"] == pytest.approx(disk.sum(), rel=0.01)


def test_tile_features_reports_masked_pixels():
    anns = [
        _ann(1, (0, 0, 300, 300)),
        _ann(2, (50, 50, 20, 20)),
        _ann(2, (400, 0, 350, 10)),
    ]  # + a large tree
    _, _, counts = oamtcd.tile_features(anns, TRANSFORM, CRS, IDS)
    assert counts["ignore_px"] == pytest.approx(300 * 300 - 20 * 20 + 350 * 10)


def test_near_hot_uses_ground_metres():
    hot = (0.0, 0.0, 0.01, 0.01)
    lon_1km = 0.01 + 0.5 / 111.32  # 500 m east of the box
    lon_2km = 0.01 + 2.0 / 111.32
    boxes = []
    for lon in (lon_1km, lon_2km):
        boxes.append(transform_bounds("EPSG:4326", "EPSG:3857", lon, 0.0, lon + 0.001, 0.001))
    assert oamtcd.near_hot(np.array(boxes), hot).tolist() == [True, False]


SIDE = 1024  # 102.4 m: wider than a z19 tile, so every zoom keeps chips over the tiler's nodata limit


def _png(rng):
    buf = io.BytesIO()
    Image.fromarray(rng.integers(1, 255, (SIDE, SIDE, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _row(image_id, *, dataset="tcd", hf_split="train", fold=1, anns=None, x0=X0):
    rng = np.random.default_rng(image_id)
    return {
        "dataset": dataset,
        "hf_split": hf_split,
        "image_id": image_id,
        "image": _png(rng),
        "width": SIDE,
        "height": SIDE,
        "bounds": [x0, Y0, x0 + SIDE / 10, Y0 + SIDE / 10],
        "crs": CRS,
        "validation_fold": fold,
        "coco_annotations": json.dumps(anns if anns is not None else [_ann(2, (100, 100, 40, 40))]),
    }


def test_convert_writes_regions_and_stats(tmp_path):
    near_x0 = X0 + 5000.0
    hot_box = transform_bounds(CRS, "EPSG:4326", near_x0 + 700, Y0, near_x0 + 800, Y0 + 50)  # ~600 m away
    rows = [
        _row(1, fold=0),
        _row(2, fold=2, x0=X0 + 1000),
        _row(3, hf_split="test", fold=-1, x0=X0 + 2000),
        _row(1, dataset="tcd-nc", fold=3, x0=X0 + 3000),
        _row(5, fold=4, anns=[], x0=X0 + 4000),
        _row(6, fold=1, x0=near_x0),
    ]
    out = tmp_path / "oamtcd"

    stats = oamtcd.convert(rows, out, ids=IDS, hot_bbox_4326=hot_box, work_dir=tmp_path / "work")

    regions = {r.id: r for r in load_regions(out)}
    assert set(regions) == {"oamtcd-by-1", "oamtcd-by-2", "oamtcd-by-3", "oamtcd-nc-1", "oamtcd-by-5"}
    assert [regions[k].split for k in ("oamtcd-by-1", "oamtcd-by-2", "oamtcd-by-3")] == [
        "val",
        "train",
        "test",
    ]
    assert regions["oamtcd-nc-1"].licence == "CC-BY-NC-4.0" and regions["oamtcd-by-2"].licence == "CC-BY-4.0"
    assert all(r.source == "oamtcd" and set(r.chips) == {19, 20} for r in regions.values())
    assert json.loads(regions["oamtcd-by-5"].labels_path.read_text())["features"] == []
    assert stats["dropped_near_hot"] == 1 and stats["tree_boxes"] == 4 and stats["categories"] == IDS
    assert json.loads((out / "stats.json").read_text()) == stats
    assert not (tmp_path / "work" / "oamtcd-by-1.tif").exists()


def test_convert_drops_mostly_masked_train_tiles_only(tmp_path):
    canopy = [_ann(1, (0, 0, SIDE, int(SIDE * 0.9)))]  # 90% of the tile
    rows = [_row(1, fold=2, anns=canopy), _row(2, hf_split="test", fold=-1, anns=canopy, x0=X0 + 1000)]

    stats = oamtcd.convert(rows, tmp_path / "oamtcd", ids=IDS, hot_bbox_4326=FAR_HOT, work_dir=tmp_path / "w")

    (region,) = load_regions(tmp_path / "oamtcd")
    assert region.id == "oamtcd-by-2" and stats["dropped_mostly_masked"] == 1
    meta = json.loads((region.labels_path.parent / "region.json").read_text())
    assert meta["ignore_share"] == pytest.approx(0.9, abs=0.01)
    assert stats["regions_over_max_masked"] == {"train": 0, "val": 0, "test": 1}


def test_convert_refuses_a_repeated_image_id(tmp_path):
    rows = [_row(1), _row(1, hf_split="test", fold=-1, x0=X0 + 1000)]
    with pytest.raises(ValueError, match="oamtcd-by-1"):
        oamtcd.convert(rows, tmp_path / "oamtcd", ids=IDS, hot_bbox_4326=FAR_HOT, work_dir=tmp_path / "w")


def test_convert_refuses_a_non_empty_output(tmp_path):
    out = tmp_path / "oamtcd"
    out.mkdir()
    (out / "old").mkdir()
    with pytest.raises(ValueError, match="not empty"):
        oamtcd.convert([_row(1)], out, ids=IDS, hot_bbox_4326=FAR_HOT)


def test_convert_refuses_bounds_in_the_wrong_order(tmp_path):
    row = _row(1) | {"bounds": [X0, X0 + 102.4, Y0, Y0 + 102.4]}  # minx, maxx, miny, maxy
    with pytest.raises(ValueError, match="bounds"):
        oamtcd.convert([row], tmp_path / "oamtcd", ids=IDS, hot_bbox_4326=FAR_HOT)
