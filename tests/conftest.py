"""Shared fixtures. Chips carry real OAM names so name-derived georeferencing is exercised."""

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from fair_trees.georef import tile_bounds_3857

EQ_TILE = (2**19, 2**19, 20)  # z20 tile whose north edge is the equator and west edge is lon 0

WGS84_SRS = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]'
)
MERCATOR_SRS = 'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84"],AUTHORITY["EPSG","3857"]]'


def write_sidecar_file(chip_path, bounds_4326, width, height, *, srs=WGS84_SRS, rotation=0.0):
    """Write a GDAL PAM sidecar (<chip>.aux.xml) with a linear EPSG:4326 geotransform."""
    w, s, e, n = bounds_4326
    gt = (w, (e - w) / width, rotation, n, 0.0, -(n - s) / height)
    srs_xml = f'  <SRS dataAxisToSRSAxisMapping="2,1">{srs}</SRS>\n' if srs else ""
    text = (
        "<PAMDataset>\n"
        + srs_xml
        + "  <GeoTransform> "
        + ", ".join(f"{v:.16e}" for v in gt)
        + "</GeoTransform>\n</PAMDataset>\n"
    )
    sidecar = Path(f"{chip_path}.aux.xml")
    sidecar.write_text(text)
    return sidecar


@pytest.fixture
def write_sidecar():
    return write_sidecar_file


@pytest.fixture
def write_oam_chip(tmp_path):
    def _write(
        x=EQ_TILE[0],
        y=EQ_TILE[1],
        z=EQ_TILE[2],
        *,
        size=256,
        count=3,
        fill=None,
        alpha=None,
        dtype="uint8",
        directory=None,
        name=None,
    ):
        directory = Path(directory or tmp_path)
        path = directory / (name or f"OAM-{x}-{y}-{z}.tif")
        if fill is None:
            data = np.random.default_rng(x * 31 + y).integers(1, 256, size=(count, size, size)).astype(dtype)
        else:
            data = np.full((count, size, size), fill, dtype=dtype)
        if alpha is not None:
            data[3] = alpha
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=size,
            width=size,
            count=count,
            dtype=dtype,
            crs="EPSG:3857",
            transform=from_bounds(*tile_bounds_3857(x, y, z), size, size),
        ) as dst:
            dst.write(data)
        return path

    return _write
