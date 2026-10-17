"""OAM tile names, Web Mercator tile geometry, true ground sample distance, sidecars and rings."""

import math
import re
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

EARTH_RADIUS_M = 6378137.0
ORIGIN_SHIFT = math.pi * EARTH_RADIUS_M
Z0_GSD = 2 * ORIGIN_SHIFT / 256
MAX_ZOOM = 30

_OAM_NAME = re.compile(r"^OAM-(\d+)-(\d+)-(\d{1,2})\.(tif|tiff|png|jpg|jpeg)$", re.IGNORECASE)


def parse_oam_name(name: str | Path) -> tuple[int, int, int] | None:
    """Return (x, y, z) from an ``OAM-x-y-z.<ext>`` file name, or None if it is not a valid tile name."""
    match = _OAM_NAME.match(Path(name).name)
    if match is None:
        return None
    x, y, z = (int(g) for g in match.groups()[:3])
    if z > MAX_ZOOM or not (0 <= x < 2**z and 0 <= y < 2**z):
        return None
    return x, y, z


def tile_bounds_3857(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) of an XYZ tile in EPSG:3857."""
    size = 2 * ORIGIN_SHIFT / 2**z
    minx = -ORIGIN_SHIFT + x * size
    maxy = ORIGIN_SHIFT - y * size
    return minx, maxy - size, minx + size, maxy


def mercator_y_to_lat(y_m: float) -> float:
    """Latitude in degrees for a Web Mercator northing."""
    return math.degrees(math.atan(math.sinh(y_m / EARTH_RADIUS_M)))


def tile_bounds_4326(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """(west, south, east, north) of an XYZ tile in degrees."""
    minx, miny, maxx, maxy = tile_bounds_3857(x, y, z)
    return (
        math.degrees(minx / EARTH_RADIUS_M),
        mercator_y_to_lat(miny),
        math.degrees(maxx / EARTH_RADIUS_M),
        mercator_y_to_lat(maxy),
    )


def true_gsd(z: int, lat_deg: float) -> float:
    """Ground metres per pixel of a 256 px tile at zoom z and latitude lat_deg."""
    return Z0_GSD * math.cos(math.radians(lat_deg)) / 2**z


def read_sidecar_bounds(chip_path: Path) -> tuple[float, float, float, float]:
    """(west, south, east, north) from the chip's GDAL ``.aux.xml`` sidecar (EPSG:4326, unrotated)."""
    sidecar = Path(f"{chip_path}.aux.xml")
    if not sidecar.is_file():
        raise FileNotFoundError(f"sidecar not found: {sidecar}")
    try:
        root = ET.parse(sidecar).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"unreadable sidecar {sidecar}: {exc}") from exc
    gt_text = root.findtext("GeoTransform")
    if gt_text is None:
        raise ValueError(f"no GeoTransform in {sidecar}")
    gt = [float(v) for v in gt_text.split(",")]
    if len(gt) != 6:
        raise ValueError(f"GeoTransform in {sidecar} has {len(gt)} values, expected 6")
    if gt[2] != 0 or gt[4] != 0:
        raise ValueError(f"rotated geotransform in {sidecar}: {gt}")
    srs = root.findtext("SRS")
    if srs is not None and ('AUTHORITY["EPSG","4326"]' not in srs or srs.lstrip().startswith("PROJCS")):
        raise ValueError(f"sidecar SRS is not EPSG:4326 in {sidecar}")
    with rasterio.open(chip_path) as src:
        width, height = src.width, src.height
    return gt[0], gt[3] + gt[5] * height, gt[0] + gt[1] * width, gt[3]


@cache
def _to_4326() -> Transformer:
    return Transformer.from_crs(3857, 4326, always_xy=True)


def mosaic_px_to_4326_rings(boxes: np.ndarray, mosaic) -> list[list[list[float]]]:
    """Closed counter-clockwise EPSG:4326 rings for (N, 4) pixel boxes (x1, y1, x2, y2)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return []
    t = mosaic.transform_3857
    x1, y1, x2, y2 = boxes.T
    west_m, north_m = t.a * x1 + t.b * y1 + t.c, t.d * x1 + t.e * y1 + t.f
    east_m, south_m = t.a * x2 + t.b * y2 + t.c, t.d * x2 + t.e * y2 + t.f
    west, south = _to_4326().transform(west_m, south_m)
    east, north = _to_4326().transform(east_m, north_m)
    return [
        [[w, s], [e, s], [e, n], [w, n], [w, s]]
        for w, s, e, n in zip(west.tolist(), south.tolist(), east.tolist(), north.tolist(), strict=True)
    ]
