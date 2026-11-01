import json
import logging

import numpy as np
import pytest
from affine import Affine

from fair_trees.dataset import (
    boxes_in_window,
    label_rings,
    labels_to_boxes,
    load_fair_dataset,
    load_regions,
    rasterize_mask,
)
from fair_trees.params import CLASSES
from fair_trees.preprocess import tile_groups

X0 = Y0 = 2**19


@pytest.fixture
def origin(geo):
    """z20 pixel frame whose (0, 0) is the EQ tile's north-west corner."""
    return Affine(geo.res20, 0, 0, 0, -geo.res20, 0)


def test_tile_groups():
    xyz = [(5, 0, 20), (0, 0, 20), (1, 1, 20), (0, 0, 19)]
    assert tile_groups(xyz) == [[3], [1, 2], [0]]


def test_labels_to_boxes_round_trip(geo, origin):
    fc = geo.feature_collection([geo.box_polygon((10, 20, 60, 90))], label=1)
    np.testing.assert_allclose(labels_to_boxes(fc, origin), [[10, 20, 60, 90]], atol=1e-6)
    shifted = origin @ Affine.translation(5, 5)
    np.testing.assert_allclose(labels_to_boxes(fc, shifted), [[5, 15, 55, 85]], atol=1e-6)
    assert labels_to_boxes(geo.feature_collection([]), origin).shape == (0, 4)


def test_only_label_one_kept_and_missing_label_is_tree(geo):
    poly = geo.box_polygon((0, 0, 10, 10))
    props = [{"label": 1}, {"label": 2}, {"label": "1"}, {}, None, {"label": "tree"}]
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": poly, "properties": p} for p in props],
    }
    assert len(label_rings(fc)) == 4  # 1, "1", missing, None


def test_label_rings_by_class(geo, origin):
    polys = [geo.box_polygon((k * 20, 0, k * 20 + 10, 10)) for k in range(5)]
    props = [{}, {"label": 1}, {"label": "1"}, {"label": 2}, {"label": 3}]
    pairs = zip(polys, props, strict=True)
    features = [{"type": "Feature", "geometry": g, "properties": p} for g, p in pairs]
    fc = {"type": "FeatureCollection", "features": features}
    assert len(label_rings(fc)) == 3
    np.testing.assert_allclose(labels_to_boxes(fc, origin, label=2), [[60, 0, 70, 10]], atol=1e-6)
    assert label_rings(fc, label=3) == []


def test_region_classes_default_and_read(write_region):
    write_region("a")
    write_region("b", classes=["tree", "tree_group"])
    a, b = load_regions(write_region("c", classes=["tree"]).parent)[:2]
    assert a.classes == ("tree",)
    assert b.classes == ("tree", "tree_group")
    d = write_region("d", classes=["tree", "shrub"])
    with pytest.raises(ValueError, match="shrub"):
        load_regions(d.parent)


def test_fair_dataset_classes_follow_labels(write_oam_chip, tmp_path, geo):
    chips = [write_oam_chip(X0, Y0)]
    labels = tmp_path / "labels.geojson"
    fc = geo.feature_collection([geo.box_polygon((10, 10, 20, 20))])
    labels.write_text(json.dumps(fc))
    assert load_fair_dataset(chips, labels)[0].classes == ("tree",)
    fc["features"] += geo.feature_collection([geo.box_polygon((30, 30, 90, 90))], label=2)["features"]
    labels.write_text(json.dumps(fc))
    assert load_fair_dataset(chips, labels)[0].classes == CLASSES


def test_multipolygon_split_into_parts(geo, origin):
    a = geo.box_polygon((0, 0, 10, 10))["coordinates"]
    b = geo.box_polygon((50, 50, 70, 80))["coordinates"]
    feature = {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": [a, b]},
        "properties": {},
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    np.testing.assert_allclose(labels_to_boxes(fc, origin), [[0, 0, 10, 10], [50, 50, 70, 80]], atol=1e-6)


def test_multipolygon_group_is_one_box(geo, origin):
    # canopy minus the crowns inside can fall apart into pieces: still one group
    a = geo.box_polygon((0, 0, 10, 10))["coordinates"]
    b = geo.box_polygon((50, 50, 70, 80))["coordinates"]
    feature = {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon", "coordinates": [a, b]},
        "properties": {"label": 2},
    }
    fc = {"type": "FeatureCollection", "features": [feature]}
    np.testing.assert_allclose(labels_to_boxes(fc, origin, label=2), [[0, 0, 70, 80]], atol=1e-6)


def test_altitude_is_dropped(geo, origin):
    poly = geo.box_polygon((10, 20, 60, 90))
    with_alt = {"type": "Polygon", "coordinates": [[[*p, 123.0] for p in poly["coordinates"][0]]]}
    flat, alt = (labels_to_boxes(geo.feature_collection([g], label=1), origin) for g in (poly, with_alt))
    np.testing.assert_array_equal(alt, flat)


@pytest.mark.parametrize(
    "geometry, reason",
    [
        (None, "empty"),
        ({"type": "Polygon", "coordinates": [[[0], [1e-5], [2e-5], [1e-5], [0]]]}, "invalid"),
        ({"type": "Polygon", "coordinates": []}, "empty"),
        ({"type": "Polygon", "coordinates": [[[0, 0], [0, 0], [0, 0], [0, 0]]]}, "invalid"),
        ({"type": "Polygon", "coordinates": [[[0, 0], [1e-5, 0], [0, 0]]]}, "invalid"),
        ({"type": "Polygon", "coordinates": [[[0, 0], [float("nan"), 0], [1e-5, 1e-5], [0, 0]]]}, "invalid"),
        ({"type": "Point", "coordinates": [0, 0]}, "Point"),
    ],
)
def test_bad_geometries_skipped_with_warning(geometry, reason, caplog):
    fc = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": geometry, "properties": {}}],
    }
    with caplog.at_level(logging.WARNING, logger="fair_trees.dataset"):
        assert label_rings(fc) == []
    assert reason in caplog.text


def test_boxes_in_window_forty_percent_rule():
    boxes = np.array(
        [[10, 50, 60, 100], [0, 120, 40, 160], [100, 100, 140, 130], [150, 150, 200, 200]], np.float32
    )
    unmasked = np.ones((200, 200), bool)
    unmasked[122:182, 122:182] = False  # the fourth box, shifted by -28, lies here
    kept = boxes_in_window(boxes, unmasked, 28, 28, 200, 200)
    np.testing.assert_allclose(kept, [[0, 22, 32, 72], [72, 72, 112, 102]])
    edge = np.array([[-30, 0, 20, 10]], np.float32)  # 20 of 50 px inside: exactly 40 %, kept
    assert len(boxes_in_window(edge, np.ones((50, 50), bool), 0, 0, 50, 50)) == 1
    assert boxes_in_window(np.zeros((0, 4), np.float32), unmasked, 0, 0, 200, 200).shape == (0, 4)


def test_rasterize_mask_footprint_and_ignore(write_region, origin):
    d = write_region(tiles=((0, 0),), ignore_px=[(0, 0, 64, 64)], footprint_px=(0, 0, 256, 128))
    [region] = load_regions(d.parent)
    m = rasterize_mask(region, origin, (256, 256))
    assert m[:64, :64].all() and m[128:, :].all()
    assert not m[64:128, 64:].any()
    m = rasterize_mask(region, origin, (256, 256), ignore=False)
    assert not m[:64, :64].any() and m[128:, :].all()


def test_rasterize_mask_without_footprint_or_ignore(write_region, origin):
    [region] = load_regions(write_region().parent)
    assert not rasterize_mask(region, origin, (32, 48)).any()


def test_load_regions(write_region):
    write_region("a", tiles=((0, 0), (1, 0)), boxes_px=[(10, 10, 20, 20)])
    write_region("a", tiles=((0, 0),), z=19)
    d = write_region("b", tiles=((5, 5),), split="val", biome="b2", ignore_px=[(0, 0, 5, 5)])
    a, b = load_regions(d.parent)
    assert (a.id, b.id) == ("a", "b")
    assert sorted(a.chips) == [19, 20] and len(a.chips[20]) == 2 and len(a.chips[19]) == 1
    assert a.labels_path.name == "labels.geojson" and a.ignore_path is None
    assert (b.split, b.biome, b.ignore_path.name) == ("val", "b2", "ignore.geojson")


def test_load_regions_rejects_bad_metadata(write_region):
    d = write_region("a")
    meta = json.loads((d / "region.json").read_text())
    del meta["split"]
    (d / "region.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="split"):
        load_regions(d.parent)


def test_load_regions_ignores_sidecar_files(write_region):
    d = write_region("a")
    [chip] = (d / "z20" / "chips").glob("OAM-*.tif")
    chip.with_name(chip.name + ".aux.xml").write_text("<PAMDataset/>")
    [region] = load_regions(d.parent)
    assert region.chips[20] == (chip,)


def test_load_fair_dataset_groups_tiles(write_oam_chip, tmp_path, geo, caplog):
    chips = [write_oam_chip(X0, Y0), write_oam_chip(X0 + 1, Y0), write_oam_chip(X0 + 5, Y0)]
    (tmp_path / "photo.tif").write_bytes(chips[0].read_bytes())
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps(geo.feature_collection([geo.box_polygon((10, 10, 20, 20))])))
    with caplog.at_level(logging.WARNING, logger="fair_trees.dataset"):
        regions = load_fair_dataset([*chips, tmp_path / "photo.tif"], labels, split="val")
    assert [r.id for r in regions] == ["fair-z20-000", "fair-z20-001"]
    assert [len(r.chips[20]) for r in regions] == [2, 1]
    assert all(r.split == "val" and r.source == "fair" and r.labels_path == labels for r in regions)
    assert regions[0].valid_area_m2 == pytest.approx(2 * (256 * geo.res20) ** 2, rel=1e-3)
    assert "photo.tif" in caplog.text
