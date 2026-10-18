import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from experiments.tiler import tile_range, tile_scene, zooms_for
from fair_trees.georef import tile_bounds_3857
from fair_trees.preprocess import read_chip


@pytest.fixture
def scene(tmp_path):
    """150 m x 150 m at 10 cm in UTM 17N near 29.7 N; the west third is nodata."""
    path = tmp_path / "scene.tif"
    data = np.random.default_rng(0).integers(1, 256, (3, 1500, 1500), dtype=np.uint8)
    data[:, :, :500] = 0
    profile = dict(
        driver="GTiff",
        width=1500,
        height=1500,
        count=3,
        dtype="uint8",
        crs="EPSG:32617",
        transform=from_origin(400000, 3285150, 0.1, 0.1),
        nodata=0,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def test_zooms_never_upsample():
    assert zooms_for(0.1, 29.7) == [19, 20]
    assert zooms_for(0.1, 50.0) == [19]  # z20 is finer than 10 cm above ~47.9 degrees
    assert zooms_for(0.05, 0.0) == [19, 20, 21]


def test_tile_range_covers_bounds():
    x0, y0, x1, y1 = tile_range(tile_bounds_3857(2**19, 2**19, 20), 20)
    assert (x0, y0) == (2**19, 2**19) and (x1, y1) == (2**19, 2**19)


def test_tile_scene_profile_and_alignment(scene, tmp_path):
    out = tile_scene(scene, tmp_path / "chips", source_gsd=0.1)
    assert sorted(out) == [19, 20] and out[20]
    for z, paths in out.items():
        for p in paths:
            _, x, y, zz = p.stem.split("-")
            with rasterio.open(p) as ds:
                assert int(zz) == z and ds.crs.to_epsg() == 3857
                assert (ds.width, ds.height, ds.count, ds.dtypes) == (256, 256, 3, ("uint8",) * 3)
                assert ds.profile["compress"].lower() == "jpeg"
                np.testing.assert_allclose(tuple(ds.bounds), tile_bounds_3857(int(x), int(y), z), atol=1e-6)
            _, valid, _ = read_chip(p, use_sidecar=False)
            assert valid.mean() >= 0.2 - 1e-9  # chips more than 80 % nodata are dropped


def test_nodata_survives_jpeg(scene, tmp_path):
    out = tile_scene(scene, tmp_path / "chips", source_gsd=0.1)
    partial = [p for p in out[20] if read_chip(p, use_sidecar=False)[1].mean() < 0.99]
    assert partial
    for p in partial:
        rgb, valid, _ = read_chip(p, use_sidecar=False)
        assert not valid.all() and (rgb[~valid] == 0).all()


def test_tile_scene_is_deterministic(scene, tmp_path):
    a = tile_scene(scene, tmp_path / "a", source_gsd=0.1)
    b = tile_scene(scene, tmp_path / "b", source_gsd=0.1)
    assert {z: [p.name for p in v] for z, v in a.items()} == {z: [p.name for p in v] for z, v in b.items()}
