from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine

from fair_trees.georef import (
    ORIGIN_SHIFT,
    mercator_y_to_lat,
    mosaic_px_to_4326_rings,
    parse_oam_name,
    read_sidecar_bounds,
    tile_bounds_3857,
    tile_bounds_4326,
    true_gsd,
)

O = ORIGIN_SHIFT  # noqa: E741
X0 = Y0 = 2**19
# Never import from conftest: nested conftests share the module name.
MERCATOR_SRS = 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84"],AUTHORITY["EPSG","3857"]]'


@pytest.mark.parametrize(
    "name, expected",
    [
        ("OAM-123-456-19.tif", (123, 456, 19)),
        ("OAM-1-2-3.png", (1, 2, 3)),
        (Path("dir") / "OAM-1-2-3.TIF", (1, 2, 3)),
        ("OAM-8-0-3.tif", None),
        ("OAM-1-2.tif", None),
        ("chip.tif", None),
        ("OAM-1-2-3.tif.aux.xml", None),
    ],
)
def test_parse_oam_name(name, expected):
    assert parse_oam_name(name) == expected


def test_tile_bounds_3857():
    assert tile_bounds_3857(0, 0, 0) == pytest.approx((-O, -O, O, O))
    assert tile_bounds_3857(1, 1, 1) == pytest.approx((0.0, -O, O, 0.0))


def test_tile_bounds_4326():
    assert tile_bounds_4326(0, 0, 1) == pytest.approx((-180.0, 0.0, 0.0, 85.0511287798066), abs=1e-9)
    assert mercator_y_to_lat(0.0) == 0.0


def test_true_gsd():
    assert true_gsd(20, 0.0) == pytest.approx(0.149291, abs=1e-6)
    assert true_gsd(20, 30.0) == pytest.approx(0.129290, abs=1e-6)
    assert true_gsd(19, 0.0) == pytest.approx(2 * true_gsd(20, 0.0))


def test_sidecar_bounds_equal_name_bounds(write_oam_chip, write_sidecar):
    path = write_oam_chip()
    expected = tile_bounds_4326(X0, Y0, 20)
    write_sidecar(path, expected, 256, 256)
    half_px = (expected[2] - expected[0]) / 256 / 2
    assert read_sidecar_bounds(path) == pytest.approx(expected, abs=half_px)


def test_sidecar_errors(write_oam_chip, write_sidecar, tmp_path):
    path = write_oam_chip()
    with pytest.raises(FileNotFoundError):
        read_sidecar_bounds(path)
    write_sidecar(path, tile_bounds_4326(X0, Y0, 20), 256, 256, rotation=1e-9)
    with pytest.raises(ValueError, match="rotat"):
        read_sidecar_bounds(path)
    write_sidecar(path, tile_bounds_4326(X0, Y0, 20), 256, 256, srs=MERCATOR_SRS)
    with pytest.raises(ValueError, match="4326"):
        read_sidecar_bounds(path)


def test_rings_closed_ccw_and_georeferenced():
    minx, miny, maxx, maxy = tile_bounds_3857(X0, Y0, 20)
    res = (maxx - minx) / 256
    mosaic = SimpleNamespace(transform_3857=Affine(res, 0.0, minx, 0.0, -res, maxy))
    [ring] = mosaic_px_to_4326_rings(np.array([[0, 0, 256, 256]], np.float32), mosaic)
    w, s, e, n = tile_bounds_4326(X0, Y0, 20)
    pts = np.array(ring)
    assert pts == pytest.approx(np.array([[w, s], [e, s], [e, n], [w, n], [w, s]]), abs=1e-9)
    signed_area = 0.5 * np.sum(pts[:-1, 0] * pts[1:, 1] - pts[1:, 0] * pts[:-1, 1])
    assert len(ring) == 5 and ring[0] == ring[-1] and signed_area > 0


def test_rings_empty():
    mosaic = SimpleNamespace(transform_3857=Affine.identity())
    assert mosaic_px_to_4326_rings(np.zeros((0, 4), np.float32), mosaic) == []
