import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from experiments.converters.neon import convert, eval_plot_footprints, voc_boxes
from fair_trees.dataset import load_regions

SCENE = "2018_TEST_1_400000_3285000_image"


def _tif(path, east, north, size, crs="EPSG:32617"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(1).integers(1, 256, (3, size, size), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=3,
        dtype="uint8",
        crs=crs,
        transform=from_origin(east, north, 0.1, 0.1),
    ) as dst:
        dst.write(data)


def _voc(path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    objs = "".join(
        f"<object><name>Tree</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax>"
        "</bndbox></object>"
        for a, b, c, d in boxes
    )
    path.write_text(f"<annotation><filename>{path.stem}.tif</filename>{objs}</annotation>")


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "NeonTreeEvaluation"
    _tif(root / "training" / "RGB" / f"{SCENE}.tif", 400000, 3285150, 1500)
    _voc(
        root / "annotations" / f"{SCENE}.xml",
        [(100, 100, 200, 220), (700, 650, 760, 700), (1300, 50, 1400, 120)],
    )
    _tif(root / "evaluation" / "RGB" / "TEST_001.tif", 420000, 3300040, 400)  # ~25 km away
    _voc(root / "annotations" / "TEST_001.xml", [(10, 10, 50, 50)])
    return root


def test_voc_boxes(raw):
    np.testing.assert_array_equal(voc_boxes(raw / "annotations" / "TEST_001.xml"), [[10, 10, 50, 50]])


def test_convert_writes_regions(raw, tmp_path):
    out = tmp_path / "neon"
    stats = convert(
        raw, out, block_size=1, val_ratio=0.3, buffer_m=0.0, biome_fn=lambda lon, lat: "Test biome"
    )
    regions = load_regions(out)
    assert {r.id for r in regions} == {f"{SCENE}-train", f"{SCENE}-val"}
    for r in regions:
        assert (r.source, r.biome, r.licence) == ("neon", "Test biome", "CC-BY-4.0")
        assert sorted(r.chips) == [19, 20] and r.valid_area_m2 > 0 and r.footprint["type"] == "Polygon"
        labels = json.loads(r.labels_path.read_text())
        assert len(labels["features"]) == 3
        assert labels["features"][0]["properties"] == {
            "label": 1,
            "source": "neon",
            "licence": "CC-BY-4.0",
            "region": r.id,
        }
    assert stats["excluded_near_eval_plots"] == 0 and stats["chips"]["20"]["train"] > 0
    assert json.loads((out / "stats.json").read_text()) == stats


def test_chips_near_eval_plots_are_excluded(raw, tmp_path):
    _tif(raw / "evaluation" / "RGB" / "TEST_002.tif", 400200, 3285100, 400)  # 50 m east of the training scene
    _voc(raw / "annotations" / "TEST_002.xml", [(10, 10, 50, 50)])
    stats = convert(
        raw, tmp_path / "neon", block_size=1, val_ratio=0.3, buffer_m=1000.0, biome_fn=lambda lon, lat: "x"
    )
    assert stats["excluded_near_eval_plots"] > 0
    assert not list((tmp_path / "neon").glob("*/z*/chips/*.tif"))


def test_eval_plot_without_crs_is_an_error(raw):
    _tif(raw / "evaluation" / "RGB" / "TEST_003.tif", 0, 0, 100, crs=None)
    with pytest.raises(ValueError, match="CRS"):
        eval_plot_footprints(raw)


def test_resolve_biome_fn(tmp_path):
    pytest.importorskip("geopandas")
    from experiments.biome import resolve_biome_fn

    path = tmp_path / "eco.geojson"
    ring = [[-83, 29], [-81, 29], [-81, 30], [-83, 30], [-83, 29]]
    feature = {
        "type": "Feature",
        "properties": {"BIOME_NAME": "Temperate Conifer Forests"},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}))
    fn = resolve_biome_fn(path)
    assert fn(-82.0, 29.7) == "Temperate Conifer Forests" and fn(10.0, 10.0) == "unknown"
