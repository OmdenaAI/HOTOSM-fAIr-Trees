"""Cut scenes into Web Mercator JPEG GeoTIFF chips that carry an internal nodata mask."""

import logging
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import MaskFlags, Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject, transform_bounds

from fair_trees.georef import ORIGIN_SHIFT, tile_bounds_3857, true_gsd

logger = logging.getLogger(__name__)

ZOOMS = (19, 20, 21)
JPEG_QUALITY = 90  # [?]
MAX_NODATA = 0.8
CHIP_SIZE = 256
_EDGE_TOL = 1e-9  # in tiles: keeps bounds lying exactly on a tile edge out of the neighbour


def zooms_for(source_gsd: float, lat_deg: float, zooms: Sequence[int] = ZOOMS) -> list[int]:
    """Zooms whose ground resolution is no finer than the source (chips are never upsampled)."""
    return [z for z in zooms if true_gsd(z, lat_deg) >= source_gsd * (1 - 1e-9)]


def tile_range(bounds_3857: tuple[float, float, float, float], z: int) -> tuple[int, int, int, int]:
    """Inclusive (xmin, ymin, xmax, ymax) of the tiles touching the bounds."""
    minx, miny, maxx, maxy = bounds_3857
    tile_m = 2 * ORIGIN_SHIFT / 2**z
    return (
        math.floor((minx + ORIGIN_SHIFT) / tile_m + _EDGE_TOL),
        math.floor((ORIGIN_SHIFT - maxy) / tile_m + _EDGE_TOL),
        math.ceil((maxx + ORIGIN_SHIFT) / tile_m - _EDGE_TOL) - 1,
        math.ceil((ORIGIN_SHIFT - miny) / tile_m - _EDGE_TOL) - 1,
    )


def _warp(
    src: rasterio.DatasetReader,
    source: np.ndarray,
    transform: Affine,
    resampling: Resampling,
    src_nodata: int | None,
) -> np.ndarray:
    dst = np.zeros((source.shape[0], CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)
    reproject(
        source=source,
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src_nodata,
        dst_transform=transform,
        dst_crs="EPSG:3857",
        dst_nodata=0,
        resampling=resampling,
    )
    return dst


def _write_chip(path: Path, rgb: np.ndarray, valid: np.ndarray, transform: Affine, quality: int) -> None:
    profile = dict(
        driver="GTiff",
        width=CHIP_SIZE,
        height=CHIP_SIZE,
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
        compress="JPEG",
        jpeg_quality=quality,
        photometric="YCBCR",
        tiled=True,
        blockxsize=CHIP_SIZE,
        blockysize=CHIP_SIZE,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # JPEG cannot store exact zeros, so the internal mask is what carries nodata.
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(rgb)
            dst.write_mask((valid * 255).astype(np.uint8))


def tile_scene(
    src_path: Path,
    out_dir: Path,
    *,
    source_gsd: float,
    zooms: Sequence[int] = ZOOMS,
    quality: int = JPEG_QUALITY,
    max_nodata: float = MAX_NODATA,
) -> dict[int, list[Path]]:
    """Write `out_dir/z{z}/OAM-x-y-z.tif` chips for every zoom the source resolution supports."""
    out: dict[int, list[Path]] = {}
    with rasterio.open(src_path) as src:
        bounds = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
        _, south, _, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        rgb_src = src.read([1, 2, 3])
        mask_src = src.dataset_mask()[None]
        if all(flags == [MaskFlags.all_valid] for flags in src.mask_flag_enums):
            # no nodata value, mask or alpha: black collars come through as (0, 0, 0) pixels
            mask_src = np.where(rgb_src.max(axis=0) > 0, 255, 0).astype(np.uint8)[None]
        for z in zooms_for(source_gsd, (south + north) / 2, zooms):
            xmin, ymin, xmax, ymax = tile_range(bounds, z)
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    transform = from_bounds(*tile_bounds_3857(x, y, z), CHIP_SIZE, CHIP_SIZE)
                    rgb = _warp(src, rgb_src, transform, Resampling.average, src.nodata)
                    valid = _warp(src, mask_src, transform, Resampling.nearest, 0)[0] > 0
                    if 1 - valid.mean() > max_nodata:
                        continue
                    rgb[:, ~valid] = 0
                    path = out_dir / f"z{z}" / f"OAM-{x}-{y}-{z}.tif"
                    _write_chip(path, rgb, valid, transform, quality)
                    out.setdefault(z, []).append(path)
    logger.info("tiled %s: %s", src_path, {z: len(v) for z, v in out.items()})
    return out
