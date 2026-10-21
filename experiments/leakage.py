"""Metre-based distances between Web Mercator boxes, and the check that tiles sit near held-out footprints."""

import math
from collections.abc import Sequence

import numpy as np
from pyproj import Transformer

from fair_trees.georef import mercator_y_to_lat, tile_bounds_3857

_TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)


def tile_boxes_3857(tiles: Sequence[tuple[int, int, int]]) -> np.ndarray:
    """(n, 4) array of ``(minx, miny, maxx, maxy)`` in EPSG:3857, one row per ``(x, y, z)`` tile."""
    if not len(tiles):
        return np.zeros((0, 4))
    return np.array([tile_bounds_3857(x, y, z) for x, y, z in tiles], float)


def box_distance_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, m) gap between EPSG:3857 boxes in ground metres; 0 when boxes touch or overlap.

    The Web Mercator gap is scaled by ``cos(lat)`` at each ``a`` box's centre.
    """
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    dx = np.maximum(0.0, np.maximum(b[None, :, 0] - a[:, None, 2], a[:, None, 0] - b[None, :, 2]))
    dy = np.maximum(0.0, np.maximum(b[None, :, 1] - a[:, None, 3], a[:, None, 1] - b[None, :, 3]))
    scale = np.array([math.cos(math.radians(mercator_y_to_lat((r[1] + r[3]) / 2))) for r in a])
    return np.hypot(dx, dy) * scale[:, None]


def near_footprints(
    tiles: Sequence[tuple[int, int, int]],
    footprints_4326: Sequence[tuple[float, float, float, float]],
    buffer_m: float,
) -> np.ndarray:
    """Bool per tile: ``True`` if within ``buffer_m`` of any ``(west, south, east, north)`` footprint."""
    if not len(tiles) or not len(footprints_4326):
        return np.zeros(len(tiles), bool)
    boxes = []
    for west, south, east, north in footprints_4326:
        minx, miny = _TO_3857.transform(west, south)
        maxx, maxy = _TO_3857.transform(east, north)
        boxes.append((minx, miny, maxx, maxy))
    return (box_distance_m(tile_boxes_3857(tiles), np.array(boxes)) < buffer_m).any(axis=1)
