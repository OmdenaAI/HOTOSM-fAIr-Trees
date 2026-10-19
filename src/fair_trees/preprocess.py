"""Window preprocessing shared by training and ONNX serving: normalise, resize, pad."""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from rasterio.errors import RasterioIOError

from fair_trees.georef import (
    mercator_y_to_lat,
    parse_oam_name,
    read_sidecar_bounds,
    tile_bounds_3857,
    tile_bounds_4326,
    true_gsd,
)
from fair_trees.params import IMAGE_MEAN, IMAGE_STD, MAX_MOSAIC_PIXELS, MAX_SIZE, MIN_SIZE, SIZE_DIVISIBLE

logger = logging.getLogger(__name__)

CHIP_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class WindowScale:
    """Resized (unpadded) size and the factors that map window pixels to resized pixels."""

    resized_width: int
    resized_height: int
    scale_x: float
    scale_y: float


def preprocess_window(rgb_hwc: np.ndarray) -> tuple[np.ndarray, WindowScale]:
    """Normalise, resize (short side 800, long side <= 1333) and zero-pad to a multiple of 32.

    Returns a C-contiguous float32 batch of shape (1, 3, Hp, Wp).
    """
    if rgb_hwc.ndim != 3 or rgb_hwc.shape[2] != 3 or rgb_hwc.dtype != np.uint8:
        raise ValueError(f"expected uint8 array of shape (H, W, 3), got {rgb_hwc.shape} {rgb_hwc.dtype}")
    h, w = rgb_hwc.shape[:2]

    mean = np.asarray(IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(IMAGE_STD, dtype=np.float32)
    x = (rgb_hwc.astype(np.float32) / 255.0 - mean) / std

    scale = min(MIN_SIZE / min(h, w), MAX_SIZE / max(h, w))
    rh, rw = math.floor(h * scale), math.floor(w * scale)

    hp = math.ceil(rh / SIZE_DIVISIBLE) * SIZE_DIVISIBLE
    wp = math.ceil(rw / SIZE_DIVISIBLE) * SIZE_DIVISIBLE
    batch = np.zeros((1, 3, hp, wp), np.float32)
    for c in range(3):
        channel = np.ascontiguousarray(x[:, :, c], dtype=np.float32)
        resized = Image.fromarray(channel).resize((rw, rh), Image.Resampling.BILINEAR)
        batch[0, c, :rh, :rw] = np.asarray(resized, dtype=np.float32)
    return batch, WindowScale(rw, rh, rw / w, rh / h)


@dataclass(frozen=True, eq=False)
class Mosaic:
    """Stitched chips: uint8 RGB (invalid pixels 0), validity mask and EPSG:3857 geotransform."""

    rgb: np.ndarray
    valid: np.ndarray
    transform_3857: Affine
    z: int
    gsd_m: float


@dataclass(frozen=True)
class Window:
    """Pixel window inside a mosaic."""

    x: int
    y: int
    width: int
    height: int


def read_chip(path: Path, *, use_sidecar: bool = True) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Read an OAM-x-y-z chip as (rgb HWC uint8, valid mask, (x, y, z))."""
    path = Path(path)
    xyz = parse_oam_name(path)
    if xyz is None:
        raise ValueError(f"{path.name}: not an OAM-x-y-z chip name")
    with rasterio.open(path) as ds:
        if ds.count < 3:
            raise ValueError(f"{path.name}: needs 3 bands, got {ds.count}")
        if any(dtype != "uint8" for dtype in ds.dtypes):
            raise ValueError(f"{path.name}: needs uint8 bands, got {ds.dtypes[0]}")
        data = ds.read()
        valid = (ds.dataset_mask() > 0) & (data[:3].max(axis=0) > 0)
        if ds.count == 4:
            valid &= data[3] > 0  # band 4 is alpha whatever its colour interpretation
        width = ds.width
        height = ds.height
    if height != width:
        raise ValueError(f"{path.name}: chip is not square ({width}x{height})")
    rgb = np.ascontiguousarray(data[:3].transpose(1, 2, 0))
    rgb[~valid] = 0
    if use_sidecar and Path(f"{path}.aux.xml").is_file():
        found = read_sidecar_bounds(path)
        expected = tile_bounds_4326(*xyz)
        tolerance = (expected[2] - expected[0]) / width / 2
        if any(abs(a - b) > tolerance for a, b in zip(found, expected, strict=True)):
            raise ValueError(f"{path.name}: sidecar bounds disagree with the OAM name")
    return rgb, valid, xyz


def _components(coords: Sequence[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Group (x, y) tiles into 8-connected components."""
    remaining = set(coords)
    groups = []
    while remaining:
        stack = [remaining.pop()]
        group = []
        while stack:
            x, y = stack.pop()
            group.append((x, y))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbour = (x + dx, y + dy)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
        groups.append(group)
    return groups


def _build_mosaic(tiles: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], z: int) -> Mosaic:
    xmin = min(x for x, _ in tiles)
    ymin = min(y for _, y in tiles)
    shapes = {rgb.shape[:2] for rgb, _ in tiles.values()}
    if len(shapes) != 1 or len(set(next(iter(shapes)))) != 1:
        raise ValueError(f"mixed tile sizes in one mosaic at zoom {z}: {sorted(shapes)}")
    t = next(iter(shapes))[0]
    height = (max(y for _, y in tiles) - ymin + 1) * t
    width = (max(x for x, _ in tiles) - xmin + 1) * t
    if height * width > MAX_MOSAIC_PIXELS:
        raise ValueError(
            f"{len(tiles)} tiles at zoom {z} span a {width}x{height} mosaic, "
            f"more than the {MAX_MOSAIC_PIXELS} pixel limit; send fewer or more compact tiles"
        )
    rgb = np.zeros((height, width, 3), np.uint8)
    valid = np.zeros((height, width), bool)
    for (x, y), (chip_rgb, chip_valid) in tiles.items():
        r, c = (y - ymin) * t, (x - xmin) * t
        rgb[r : r + t, c : c + t] = chip_rgb
        valid[r : r + t, c : c + t] = chip_valid
    minx, _, maxx, maxy = tile_bounds_3857(xmin, ymin, z)
    res = (maxx - minx) / t
    gsd = true_gsd(z, mercator_y_to_lat(maxy - height * res / 2))
    return Mosaic(rgb, valid, Affine(res, 0, minx, 0, -res, maxy), z, gsd)


def mosaic_chips(
    paths: Sequence[Path], *, use_sidecar: bool = True, skipped: list[tuple[str, str]] | None = None
) -> list[Mosaic]:
    """Stitch chips into one mosaic per (zoom, 8-connected tile component), sorted by (z, y, x).

    Unreadable chips are skipped with a warning; when `skipped` is given, `(name, reason)` is appended.
    """
    by_zoom: dict[int, dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = {}
    for path in map(Path, paths):
        if path.suffix.lower() not in CHIP_SUFFIXES:
            continue
        try:
            rgb, valid, (x, y, z) = read_chip(path, use_sidecar=use_sidecar)
        except (ValueError, RasterioIOError) as exc:
            logger.warning("skipping %s: %s", path.name, exc)
            if skipped is not None:
                skipped.append((path.name, str(exc)))
            continue
        tiles = by_zoom.setdefault(z, {})
        if (x, y) in tiles:
            logger.warning("ignoring %s: duplicate of tile %d-%d-%d already read", path.name, x, y, z)
            continue
        tiles[(x, y)] = (rgb, valid)
    groups = [
        (z, min(y for _, y in group), min(x for x, _ in group), group)
        for z, tiles in by_zoom.items()
        for group in _components(list(tiles))
    ]
    groups.sort(key=lambda item: item[:3])
    return [_build_mosaic({k: by_zoom[z][k] for k in group}, z) for z, _, _, group in groups]


def array_mosaic(rgb: np.ndarray, gsd_m: float, valid: np.ndarray | None = None) -> Mosaic:
    """Wrap a plain RGB array as a mosaic (identity transform, z = -1)."""
    valid = np.ones(rgb.shape[:2], bool) if valid is None else valid.astype(bool)
    rgb = rgb.copy()
    rgb[~valid] = 0
    return Mosaic(rgb, valid, Affine.identity(), -1, gsd_m)


def _starts(length: int, size: int, stride: int) -> list[int]:
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def window_grid(height: int, width: int, gsd_m: float, window_m: float, overlap: float) -> list[Window]:
    """Row-major windows of window_m metres with the given overlap, shifted back at the edges."""
    win = max(1, round(window_m / gsd_m))
    ww, wh = min(win, width), min(win, height)
    stride_x = max(1, round(ww * (1 - overlap)))
    stride_y = max(1, round(wh * (1 - overlap)))
    return [Window(x, y, ww, wh) for y in _starts(height, wh, stride_y) for x in _starts(width, ww, stride_x)]


def is_blank(mosaic: Mosaic, window: Window, min_valid_fraction: float) -> bool:
    """True when the valid-pixel fraction of the window is below the threshold."""
    patch = mosaic.valid[window.y : window.y + window.height, window.x : window.x + window.width]
    return bool(patch.mean() < min_valid_fraction)


def preprocess(image_path: Path, chip_size: int) -> tuple[np.ndarray, WindowScale]:
    """Single-chip contract: read one chip of chip_size x chip_size and preprocess it."""
    rgb, _, _ = read_chip(image_path)
    if rgb.shape[:2] != (chip_size, chip_size):
        got = f"{rgb.shape[1]}x{rgb.shape[0]}"
        raise ValueError(f"{Path(image_path).name}: chip is {got}, expected {chip_size}x{chip_size}")
    return preprocess_window(rgb)
