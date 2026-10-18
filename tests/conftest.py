"""Shared fixtures. Chips carry real OAM names so name-derived georeferencing is exercised."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds

from fair_trees.georef import ORIGIN_SHIFT, tile_bounds_3857

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


class FakeSession:
    """Stands in for onnxruntime.InferenceSession: background logits everywhere, optional planted logits."""

    def __init__(self, planted=None):
        self.planted = dict(planted or {})
        self.calls = []

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def run(self, output_names, feeds):
        from fair_trees.decode import make_anchors

        batch = feeds["images"]
        self.calls.append(tuple(batch.shape))
        n = sum(len(a) for a in make_anchors(batch.shape[2], batch.shape[3]))
        cls = np.full((1, n, 1), -10.0, np.float32)
        for idx, logit in self.planted.items():
            cls[0, idx, 0] = logit
        return [cls, np.zeros((1, n, 4), np.float32)]


@pytest.fixture
def fake_session():
    return FakeSession


RES20 = 2 * ORIGIN_SHIFT / 2**20 / 256  # EPSG:3857 metres per z20 pixel
_TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)


def px_to_4326(points_px):
    """z20 pixel offsets from the EQ tile's north-west corner (the EPSG:3857 origin) -> [lon, lat]."""
    lon, lat = _TO_4326.transform([p[0] * RES20 for p in points_px], [-p[1] * RES20 for p in points_px])
    return [[float(a), float(b)] for a, b in zip(lon, lat, strict=True)]


def box_polygon(box_px):
    x1, y1, x2, y2 = box_px
    ring = px_to_4326([(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)])  # counter-clockwise
    return {"type": "Polygon", "coordinates": [ring]}


def feature_collection(geometries, **properties):
    features = [{"type": "Feature", "geometry": g, "properties": dict(properties)} for g in geometries]
    return {"type": "FeatureCollection", "features": features}


@pytest.fixture
def geo():
    return SimpleNamespace(
        res20=RES20, px_to_4326=px_to_4326, box_polygon=box_polygon, feature_collection=feature_collection
    )


@pytest.fixture
def write_region(tmp_path, write_oam_chip):
    """Write one region in the base-training layout. Geometry is given in z20 pixels from the EQ origin,
    so the same boxes describe the same ground at every zoom."""

    def _write(
        region_id="r1",
        tiles=((0, 0),),
        z=20,
        *,
        boxes_px=(),
        ignore_px=(),
        footprint_px=None,
        source="src",
        biome="b1",
        split="train",
        root=None,
        fill=None,
        alpha=None,
        area=None,
    ):
        root = Path(root or tmp_path / "regions")
        d = root / region_id
        chip_dir = d / f"z{z}" / "chips"
        chip_dir.mkdir(parents=True, exist_ok=True)
        base = int(EQ_TILE[0] * 2.0 ** (z - 20))
        for dx, dy in tiles:
            write_oam_chip(
                base + dx,
                base + dy,
                z,
                directory=chip_dir,
                fill=fill,
                alpha=alpha,
                count=3 if alpha is None else 4,
            )
        labels = feature_collection([box_polygon(b) for b in boxes_px], label=1)
        (d / "labels.geojson").write_text(json.dumps(labels))
        if ignore_px:
            (d / "ignore.geojson").write_text(
                json.dumps(feature_collection([box_polygon(b) for b in ignore_px]))
            )
        tile_m = 2 * ORIGIN_SHIFT / 2**z
        meta = {
            "id": region_id,
            "source": source,
            "biome": biome,
            "licence": "CC-BY-4.0",
            "split": split,
            "zooms": sorted(int(p.name[1:]) for p in d.glob("z*") if p.is_dir()),
            "footprint": None if footprint_px is None else box_polygon(footprint_px),
            "valid_area_m2": float(area if area is not None else len(tiles) * tile_m**2),
        }
        (d / "region.json").write_text(json.dumps(meta))
        return d

    return _write
