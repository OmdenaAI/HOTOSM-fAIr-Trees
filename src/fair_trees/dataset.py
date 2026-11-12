"""Training data: regions, labels, masks, the crop sampler and the spatial train/val split.

Torch-free: numpy, Pillow, rasterio, pyproj and the stdlib only.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rasterio
import rasterio.features
from affine import Affine
from pyproj import Transformer

from fair_trees.augment import DOWNSCALE_P, colour_jitter, downscale, flip, rot90
from fair_trees.georef import ORIGIN_SHIFT, mercator_y_to_lat, parse_oam_name, tile_bounds_3857, true_gsd
from fair_trees.params import CLASSES, TrainParams
from fair_trees.preprocess import Window, read_chip, tile_groups

logger = logging.getLogger(__name__)

MIN_BOX_FRACTION = 0.4
MAX_MASKED = 0.8  # a crop with more masked area than this is redrawn
CROP_RETRIES = 20
LOWRES_CELLS = 16  # low-resolution mask cells per tile side
# mapper diameter_crown typos reach 25,231 m; a crown over this is read as an unknown diameter
MAX_DIAMETER_M = 30.0
DEFAULT_CROWN_M = 4.0  # square of a tree Point without a usable diameter: the median of HOT's diameter_crown
# a file's tree Points without a usable diameter are masked when at least this share has one, else labelled
MASK_UNSIZED_FROM = 0.5
_DIAMETER = re.compile(r"\s*(\d+(?:[.,]\d+)?)\s*(?:m|M|metres|meters)?\s*")
_TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
_TO_4326 = Transformer.from_crs(3857, 4326, always_xy=True)
_REQUIRED_KEYS = ("id", "source", "biome", "licence", "split", "footprint", "valid_area_m2")


@dataclass(frozen=True)
class Region:
    """One labelled area: its chips per zoom, label and ignore files, and the footprint that was mapped.

    ``footprint`` is an EPSG:4326 GeoJSON geometry (``None`` = everything is mapped); ``classes`` names the
    classes its labels cover (a class not named is unlabelled there, not absent).
    """

    id: str
    source: str
    biome: str
    licence: str
    split: str
    chips: Mapping[int, tuple[Path, ...]]
    labels_path: Path
    ignore_path: Path | None
    footprint: dict | None
    valid_area_m2: float
    classes: tuple[str, ...] = ("tree",)


def load_geojson(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _osm_tags(properties: dict) -> dict:
    """A feature's OSM tags: its ``tags`` (an object, or a JSON string), else the properties themselves."""
    tags = properties.get("tags", properties)
    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except ValueError:
            return {}
    return tags if isinstance(tags, dict) else {}


def _label_of(properties: dict) -> int | None:
    """The class number of a label feature (1 tree, 2 tree_group).

    Its OSM ``natural`` tag decides when that names a class: fAIr numbers ``label`` by the order a dataset
    lists its classes, so a tree_group-only dataset arrives as ``label: 1``. Otherwise the ``label`` number
    does: missing is a tree, and anything but 1 or 2 is None.
    """
    natural = _osm_tags(properties).get("natural")
    if isinstance(natural, str) and natural in CLASSES:
        return CLASSES.index(natural) + 1
    if "label" not in properties:
        return 1
    try:
        value = float(properties["label"])
    except (TypeError, ValueError):
        return None
    return int(value) if value in (1.0, 2.0) else None


def _invalid_reason(ring: np.ndarray) -> str | None:
    if len(ring) < 4:
        return f"{len(ring)} positions"
    if not np.isfinite(ring).all():
        return "non-finite coordinate"
    if (ring.max(axis=0) == ring.min(axis=0)).any():
        return "zero extent"
    return None


def _exterior_ring(part: list) -> tuple[np.ndarray, str | None]:
    """A polygon part's exterior ring as (k, 2) lon/lat with any altitude dropped, and why it is invalid."""
    try:
        ring = np.asarray(part[0] if part else [], np.float64)
    except (TypeError, ValueError):
        return np.empty((0, 2)), "malformed positions"
    if ring.size == 0:
        ring = ring.reshape(0, 2)
    if ring.ndim != 2 or ring.shape[1] < 2:
        return np.empty((0, 2)), "positions with fewer than 2 numbers"
    ring = ring[:, :2]
    return ring, _invalid_reason(ring)


def parse_diameter(value) -> float:
    """Metres from a ``diameter_crown`` value such as '3m', '2.412m', '3,134 m', '4' or 4; NaN if missing or
    unclear."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return math.nan
    m = _DIAMETER.fullmatch(value)
    return float(m.group(1).replace(",", ".")) if m else math.nan


def crown_diameter_m(properties: dict) -> float | None:
    """A tree's ``diameter_crown`` in metres from its OSM tags (``_osm_tags``); None when missing, unclear
    or outside (0, MAX_DIAMETER_M]."""
    diameter = parse_diameter(_osm_tags(properties).get("diameter_crown"))
    return diameter if 0 < diameter <= MAX_DIAMETER_M else None


def _point_crown_ring(position: list, diameter_m: float) -> tuple[np.ndarray, str | None]:
    """The lon/lat ring of a square crown ``diameter_m`` on the ground, centred on a Point's position."""
    try:
        lon, lat = (float(v) for v in position[:2])
    except (TypeError, ValueError):
        return np.empty((0, 2)), "malformed position"
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return np.empty((0, 2)), "non-finite coordinate"
    x, y = _TO_3857.transform(lon, lat)
    half = diameter_m / 2 / math.cos(math.radians(lat))  # ground metres are 1 / cos(lat) longer in 3857
    xs = [x - half, x + half, x + half, x - half, x - half]
    ys = [y - half, y - half, y + half, y + half, y - half]
    return np.column_stack(_TO_4326.transform(xs, ys)), None


def _tree_points(geojson: dict) -> Iterator[tuple[dict, float | None]]:
    """The geometry and usable crown diameter (None without one) of each tree Point of a labels file."""
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        properties = feature.get("properties") or {}
        if geometry.get("type") == "Point" and _label_of(properties) == 1:
            yield geometry, crown_diameter_m(properties)


def tree_point_counts(geojson: dict) -> tuple[int, int]:
    """How many of a labels file's tree Points have a usable ``diameter_crown``, and how many do not."""
    diameters = [diameter for _, diameter in _tree_points(geojson)]
    unsized = sum(diameter is None for diameter in diameters)
    return len(diameters) - unsized, unsized


def masks_unsized_points(geojson: dict) -> bool:
    """Whether a labels file's tree Points without a usable ``diameter_crown`` are masked, not labelled.

    Masked when at least ``MASK_UNSIZED_FROM`` of the tree Points have a diameter: those are enough to
    learn from, so no size is guessed. Labelled as ``DEFAULT_CROWN_M`` crowns when fewer do, as OSM trees
    mostly come without one and masking them would leave almost nothing to learn.
    """
    sized, unsized = tree_point_counts(geojson)
    return sized >= MASK_UNSIZED_FROM * (sized + unsized)


def masked_crown_rings(geojson: dict) -> list[np.ndarray]:
    """A ``DEFAULT_CROWN_M`` square on each tree Point without a usable ``diameter_crown`` when the file
    masks those (``masks_unsized_points``), else nothing. A masked crown acts as an ignore polygon: its
    pixels are blacked out in training and predictions on it are not scored."""
    if not masks_unsized_points(geojson):
        return []
    squares = (
        _point_crown_ring(geometry.get("coordinates") or [], DEFAULT_CROWN_M)
        for geometry, diameter in _tree_points(geojson)
        if diameter is None
    )
    return [ring for ring, why in squares if not why]


def label_rings(geojson: dict, label: int = 1) -> list[np.ndarray]:
    """Exterior rings of the labels of class ``label`` (1 tree, 2 tree_group) as (k, 2) float64 lon/lat
    arrays, one per polygon part for trees. A tree_group is one group however many parts it has (canopy
    minus the crowns inside can fall apart), so its parts' rings are stacked into one array per feature.
    A tree Point (OSM maps trees as points) is a square crown of its ``diameter_crown`` metres; one
    without a usable diameter is a ``DEFAULT_CROWN_M`` square, unless the file masks those
    (``masks_unsized_points``): then it is no label but a masked crown (``masked_crown_rings``).

    Features of another class are dropped silently; empty, unsupported and invalid geometries are skipped
    with a warning.
    """
    rings = []
    mask_unsized = label == 1 and masks_unsized_points(geojson)
    for i, feature in enumerate(geojson.get("features", [])):
        properties = feature.get("properties") or {}
        if _label_of(properties) != label:
            continue
        geometry = feature.get("geometry")
        if not geometry or not geometry.get("coordinates"):
            logger.warning(f"label feature {i}: empty geometry, skipped")
            continue
        kind = geometry.get("type")
        if kind == "Point":
            if label != 1:
                logger.warning(f"label feature {i}: a {CLASSES[label - 1]} Point has no extent, skipped")
                continue
            diameter = crown_diameter_m(properties)
            if diameter is None and mask_unsized:
                continue
            ring, why = _point_crown_ring(geometry["coordinates"], diameter or DEFAULT_CROWN_M)
            if why:
                logger.warning(f"label feature {i}: invalid geometry ({why}), skipped")
                continue
            rings.append(ring)
            continue
        if kind == "Polygon":
            parts = [geometry["coordinates"]]
        elif kind == "MultiPolygon":
            parts = geometry["coordinates"]
        else:
            logger.warning(f"label feature {i}: unsupported geometry type {kind}, skipped")
            continue
        feature_rings = []
        for part in parts:
            ring, why = _exterior_ring(part)
            if why:
                logger.warning(f"label feature {i}: invalid geometry ({why}), skipped")
                continue
            feature_rings.append(ring)
        if label == 2 and feature_rings:
            feature_rings = [np.concatenate(feature_rings)]
        rings.extend(feature_rings)
    return rings


def _rings_to_boxes(rings: Sequence[np.ndarray], transform_3857: Affine) -> np.ndarray:
    inverse = ~transform_3857
    boxes = []
    for ring in rings:
        x, y = _TO_3857.transform(ring[:, 0], ring[:, 1])
        cols, rows = inverse @ (np.asarray(x), np.asarray(y))
        boxes.append([cols.min(), rows.min(), cols.max(), rows.max()])
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def labels_to_boxes(geojson: dict, transform_3857: Affine, label: int = 1) -> np.ndarray:
    """Tight XYXY box per ring of class ``label`` in the pixel frame of ``transform_3857``, float32 (N, 4)."""
    return _rings_to_boxes(label_rings(geojson, label), transform_3857)


def masked_crown_boxes(geojson: dict, transform_3857: Affine) -> np.ndarray:
    """XYXY box per masked crown (``masked_crown_rings``) in the pixel frame of ``transform_3857``, float32
    (N, 4)."""
    return _rings_to_boxes(masked_crown_rings(geojson), transform_3857)


def _project_geometry(geometry: dict) -> dict:
    def walk(coords):
        if isinstance(coords[0], int | float):
            x, y = _TO_3857.transform(coords[0], coords[1])
            return [x, y]
        return [walk(c) for c in coords]

    return {"type": geometry["type"], "coordinates": walk(geometry["coordinates"])}


def rasterize_mask(
    region: Region, transform_3857: Affine, shape: tuple[int, int], *, ignore: bool = True
) -> np.ndarray:
    """(H, W) bool, True = masked: outside the footprint or, when ``ignore``, inside an ignore polygon."""
    masked = np.zeros(shape, bool)
    if region.footprint is not None:
        inside = rasterio.features.rasterize(
            [(_project_geometry(region.footprint), 1)],
            out_shape=shape,
            transform=transform_3857,
            fill=0,
            dtype="uint8",
        )
        masked |= inside == 0
    if ignore and region.ignore_path is not None:
        shapes = [
            (_project_geometry(f["geometry"]), 1)
            for f in load_geojson(region.ignore_path).get("features", [])
            if f.get("geometry") and f["geometry"].get("type") in ("Polygon", "MultiPolygon")
        ]
        if shapes:
            burned = rasterio.features.rasterize(
                shapes, out_shape=shape, transform=transform_3857, fill=0, dtype="uint8"
            )
            masked |= burned == 1
    return masked


def rasterize_labels(
    region: Region, transform_3857: Affine, shape: tuple[int, int], label: int
) -> np.ndarray:
    """(H, W) bool, True inside the polygons of the region's class-``label`` labels (holes stay False, as
    for ignore polygons)."""
    shapes = [
        (_project_geometry(f["geometry"]), 1)
        for f in load_geojson(region.labels_path).get("features", [])
        if _label_of(f.get("properties") or {}) == label
        and f.get("geometry")
        and f["geometry"].get("type") in ("Polygon", "MultiPolygon")
        and f["geometry"].get("coordinates")
    ]
    if not shapes:
        return np.zeros(shape, bool)
    burned = rasterio.features.rasterize(
        shapes, out_shape=shape, transform=transform_3857, fill=0, dtype="uint8"
    )
    return burned == 1


def boxes_in_window(
    boxes: np.ndarray,
    unmasked: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    min_fraction: float = MIN_BOX_FRACTION,
) -> np.ndarray:
    """Boxes (mosaic frame) kept for a window, clipped, in window coordinates, float32 (K, 4).

    A box is kept when at least ``min_fraction`` of its unclipped area is unmasked pixels (by pixel
    centre) inside the window. ``unmasked`` is the window-sized (height, width) boolean image.
    """
    clipped, kept = window_boxes(boxes, unmasked, x0, y0, width, height, min_fraction)
    return clipped[kept]


def window_boxes(
    boxes: np.ndarray, unmasked: np.ndarray, x0: int, y0: int, width: int, height: int, min_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Every box of positive area clipped to the window, float32 (M, 4), and whether the 40 % rule keeps
    it."""
    shifted = np.asarray(boxes, np.float64).reshape(-1, 4) - np.array([x0, y0, x0, y0])
    area = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
    shifted, area = shifted[area > 0], area[area > 0]
    clipped = np.clip(shifted, 0, [width, height, width, height])
    integral = np.zeros((unmasked.shape[0] + 1, unmasked.shape[1] + 1), np.int64)
    integral[1:, 1:] = unmasked.astype(np.int64).cumsum(0).cumsum(1)
    # pixel j counts when x1 <= j + 0.5 < x2, i.e. j in [ceil(x1 - 0.5), ceil(x2 - 0.5))
    edges = np.ceil(clipped - 0.5).astype(np.int64)
    c1 = np.clip(edges[:, 0], 0, unmasked.shape[1])
    c2 = np.clip(edges[:, 2], c1, unmasked.shape[1])
    r1 = np.clip(edges[:, 1], 0, unmasked.shape[0])
    r2 = np.clip(edges[:, 3], r1, unmasked.shape[0])
    count = integral[r2, c2] - integral[r1, c2] - integral[r2, c1] + integral[r1, c1]
    return clipped.astype(np.float32).reshape(-1, 4), count / area >= min_fraction


def load_regions(root: Path) -> list[Region]:
    """One region per ``root/*/region.json`` (base-training layout), sorted by id."""
    regions = []
    for meta_path in sorted(Path(root).glob("*/region.json")):
        d = meta_path.parent
        meta = json.loads(meta_path.read_text())
        for key in _REQUIRED_KEYS:
            if key not in meta:
                raise ValueError(f"{d}: region.json lacks {key}")
        chips = {}
        for zdir in sorted(d.glob("z*")):
            if not zdir.is_dir():
                continue
            found = tuple(p for p in sorted((zdir / "chips").glob("OAM-*")) if parse_oam_name(p))
            if found:
                chips[int(zdir.name[1:])] = found
        ignore = d / "ignore.geojson"
        classes = tuple(meta.get("classes", ("tree",)))
        unknown = sorted(set(classes) - set(CLASSES))
        if unknown:
            raise ValueError(f"{d}: unknown classes {unknown} in region.json, expected a subset of {CLASSES}")
        regions.append(
            Region(
                id=meta["id"],
                source=meta["source"],
                biome=meta["biome"],
                licence=meta["licence"],
                split=meta["split"],
                chips=chips,
                labels_path=d / "labels.geojson",
                ignore_path=ignore if ignore.is_file() else None,
                footprint=meta["footprint"],
                valid_area_m2=float(meta["valid_area_m2"]),
                classes=classes,
            )
        )
    return sorted(regions, key=lambda r: r.id)


def load_fair_dataset(chips: Sequence[Path], labels_path: Path, *, split: str = "train") -> list[Region]:
    """Regions for a mapper's own chips: one per zoom and 8-connected tile component.

    The regions label the classes the labels file holds a feature of (tree when it holds none). A fAIr
    dataset has one geometry type, so tree points come without tree_groups and tree_group polygons
    without trees: a class that was not mapped is unlabelled, not absent.
    """
    labels = load_geojson(labels_path)
    classes = tuple(name for k, name in enumerate(CLASSES) if label_rings(labels, k + 1)) or ("tree",)
    sized, unsized = tree_point_counts(labels)
    if unsized:
        how = (
            f"masked as {DEFAULT_CROWN_M:g} m squares, as most trees have a size"
            if masks_unsized_points(labels)
            else f"labelled as {DEFAULT_CROWN_M:g} m crowns, as too few trees have a size to learn from"
        )
        logger.warning(f"{unsized} of {sized + unsized} tree points have no usable diameter_crown: {how}")
    named, xyzs = [], []
    for path in map(Path, chips):
        xyz = parse_oam_name(path)
        if xyz is None:
            logger.warning(f"{path.name}: not an OAM-x-y-z chip name, skipped")
            continue
        named.append(path)
        xyzs.append(xyz)
    regions = []
    for k, group in enumerate(tile_groups(xyzs)):
        z = xyzs[group[0]][2]
        paths = tuple(named[i] for i in group)
        with rasterio.open(paths[0]) as ds:
            tile_px = ds.width
        north = max(tile_bounds_3857(*xyzs[i])[3] for i in group)
        south = min(tile_bounds_3857(*xyzs[i])[1] for i in group)
        gsd = true_gsd(z, mercator_y_to_lat((north + south) / 2))
        regions.append(
            Region(
                id=f"fair-z{z}-{k:03d}",
                source="fair",
                biome="unknown",
                licence="unknown",
                split=split,
                chips={z: paths},
                labels_path=Path(labels_path),
                ignore_path=None,
                footprint=None,
                valid_area_m2=len(paths) * (tile_px * gsd) ** 2,
                classes=classes,
            )
        )
    return regions


def block_key(x: int, y: int, z: int, z_ref: int, block_size: int) -> tuple[int, int]:
    """Block of a tile at the reference zoom: ``block_size`` x ``block_size`` reference tiles per block.

    A tile finer than ``z_ref`` maps to its ancestor; a coarser one maps to its north-west descendant [P].
    """
    if z >= z_ref:
        bx, by = x >> (z - z_ref), y >> (z - z_ref)
    else:
        bx, by = x << (z_ref - z), y << (z_ref - z)
    return bx // block_size, by // block_size


def reference_grid(zooms: Sequence[int], z_ref: int | None, block_size: int) -> tuple[int, int]:
    """``(z_ref, block_size)`` of a block grid on which no tile of ``zooms`` straddles two blocks.

    ``z_ref`` defaults to the lowest zoom. A tile coarser than ``z_ref`` covers several reference tiles but
    ``block_key`` names only its north-west one, so the reference zoom drops to the coarsest tile and the
    block grows to whole tiles of that zoom of at least the requested ground size. Otherwise both come back
    unchanged.
    """
    z_min = min(zooms)
    if z_ref is None or z_ref <= z_min:
        return (z_min if z_ref is None else z_ref), block_size
    return z_min, -(-block_size // (1 << (z_ref - z_min)))


def spatial_split(
    chip_names: Sequence[str | Path],
    val_ratio: float,
    seed: int,
    *,
    block_size: int = 4,
    z_ref: int | None = None,
) -> tuple[list, list]:
    """Split chips into ``(train, val)`` by whole spatial blocks so neighbouring ground never straddles.

    Chips named ``OAM-x-y-z.<ext>`` are grouped by ``block_key`` at ``z_ref`` (default: the lowest zoom
    present), so one piece of ground at several zooms lands on one side; a chip coarser than ``z_ref``
    lowers the grid to its zoom (see ``reference_grid``). Blocks go to val in a seeded random order until
    about ``round(val_ratio * len(chip_names))`` chips are in val; train always keeps at least one block.
    Chips whose name does not parse fill any remaining val quota. Items come back unchanged and in input
    order.
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    parsed = [parse_oam_name(item) for item in chip_names]
    named = [i for i, p in enumerate(parsed) if p is not None]
    unnamed = [i for i, p in enumerate(parsed) if p is None]
    if named:
        z_ref, block_size = reference_grid([parsed[i][2] for i in named], z_ref, block_size)

    blocks: dict[tuple[int, int], list[int]] = {}
    for i in named:
        x, y, z = parsed[i]
        blocks.setdefault(block_key(x, y, z, z_ref, block_size), []).append(i)
    keys = sorted(blocks)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))

    target = round(val_ratio * len(chip_names))
    val_idx: list[int] = []
    val_blocks = 0
    for k in order:
        # keep at least one block for train: a block goes to val only if another one remains
        if len(val_idx) < target and len(keys) - val_blocks >= 2:
            val_idx += blocks[keys[k]]
            val_blocks += 1

    for j in rng.permutation(len(unnamed)):
        if len(val_idx) >= target:
            break
        val_idx.append(unnamed[j])

    if not val_idx:
        logger.warning(
            "spatial_split: val is empty (%d chips in one block, no unnamed chips); "
            "training continues without val",
            len(chip_names),
        )

    val_set = set(val_idx)
    train = [item for i, item in enumerate(chip_names) if i not in val_set]
    val = [item for i, item in enumerate(chip_names) if i in val_set]
    return train, val


@dataclass(frozen=True)
class SampleSpec:
    """One planned training crop: where it is, how big it is, and the seed of its augmentation."""

    region_id: str
    zoom: int
    centre_px: tuple[float, float]  # region pixel frame
    size_m: float
    aug_seed: int


@dataclass(frozen=True, eq=False)
class RegionGeom:
    """A region at one zoom: its tile grid, pixel frame, label boxes and low-resolution mask.

    The pixel frame starts at the north-west corner of tile ``(xmin, ymin)``; absent tiles are holes.
    ``boxes`` are the tree labels, ``group_boxes`` the tree_group labels (empty unless the region labels
    groups) and ``masked_boxes`` the masked crowns (``masked_crown_boxes``). ``lowres_masked`` has one cell
    per ``cell_px`` x ``cell_px`` pixels, True = absent tile or masked.
    """

    tiles: Mapping[tuple[int, int], Path]
    xmin: int
    ymin: int
    ncols: int
    nrows: int
    tile_px: int
    transform_3857: Affine
    gsd_m: float
    boxes: np.ndarray
    group_boxes: np.ndarray
    masked_boxes: np.ndarray
    lowres_masked: np.ndarray
    cell_px: int

    @property
    def width(self) -> int:
        return self.ncols * self.tile_px

    @property
    def height(self) -> int:
        return self.nrows * self.tile_px


def _build_geom(region: Region, zoom: int) -> RegionGeom:
    tiles = {}
    for path in region.chips[zoom]:
        x, y, _ = parse_oam_name(path)
        tiles[(x, y)] = Path(path)
    xs = [x for x, _ in tiles]
    ys = [y for _, y in tiles]
    xmin, ymin = min(xs), min(ys)
    ncols, nrows = max(xs) - xmin + 1, max(ys) - ymin + 1
    with rasterio.open(region.chips[zoom][0]) as ds:
        tile_px = ds.width
    minx, _, _, maxy = tile_bounds_3857(xmin, ymin, zoom)
    res = 2 * ORIGIN_SHIFT / 2**zoom / tile_px
    transform_3857 = Affine(res, 0, minx, 0, -res, maxy)
    gsd_m = true_gsd(zoom, mercator_y_to_lat(maxy - nrows * tile_px * res / 2))
    labels = load_geojson(region.labels_path)
    boxes = labels_to_boxes(labels, transform_3857)
    masked_boxes = masked_crown_boxes(labels, transform_3857)
    group_boxes = np.zeros((0, 4), np.float32)
    if "tree_group" in region.classes:
        group_boxes = labels_to_boxes(labels, transform_3857, label=2)

    cells = LOWRES_CELLS if tile_px % LOWRES_CELLS == 0 else 1
    cell_px = tile_px // cells
    shape = (nrows * cells, ncols * cells)
    lowres = rasterize_mask(region, transform_3857 @ Affine.scale(cell_px), shape)
    lowres |= burn_boxes(masked_boxes / cell_px, shape)
    for r in range(nrows):
        for c in range(ncols):
            if (xmin + c, ymin + r) not in tiles:
                lowres[r * cells : (r + 1) * cells, c * cells : (c + 1) * cells] = True
    return RegionGeom(
        tiles=tiles,
        xmin=xmin,
        ymin=ymin,
        ncols=ncols,
        nrows=nrows,
        tile_px=tile_px,
        transform_3857=transform_3857,
        gsd_m=gsd_m,
        boxes=boxes,
        group_boxes=group_boxes,
        masked_boxes=masked_boxes,
        lowres_masked=lowres,
        cell_px=cell_px,
    )


def _window(geom: RegionGeom, centre_px: tuple[float, float], size_m: float) -> Window:
    """Square crop of ``size_m`` metres around ``centre_px``, clamped inside the region frame."""
    n = max(1, int(round(size_m / geom.gsd_m)))
    cx, cy = centre_px
    x0 = min(max(int(round(cx - n / 2)), 0), geom.width - n)
    y0 = min(max(int(round(cy - n / 2)), 0), geom.height - n)
    return Window(x0, y0, n, n)


def _masked_fraction(geom: RegionGeom, w: Window) -> float:
    c0, c1 = w.x // geom.cell_px, math.ceil((w.x + w.width) / geom.cell_px)
    r0, r1 = w.y // geom.cell_px, math.ceil((w.y + w.height) / geom.cell_px)
    return float(geom.lowres_masked[r0:r1, c0:c1].mean())


def _is_empty(geom: RegionGeom, w: Window) -> bool:
    cx = (geom.boxes[:, 0] + geom.boxes[:, 2]) / 2
    cy = (geom.boxes[:, 1] + geom.boxes[:, 3]) / 2
    inside = (cx >= w.x) & (cx < w.x + w.width) & (cy >= w.y) & (cy < w.y + w.height)
    return not inside.any()


def _tiles_present(geom: RegionGeom, w: Window) -> bool:
    cols = range(w.x // geom.tile_px, (w.x + w.width - 1) // geom.tile_px + 1)
    rows = range(w.y // geom.tile_px, (w.y + w.height - 1) // geom.tile_px + 1)
    return all((geom.xmin + c, geom.ymin + r) in geom.tiles for r in rows for c in cols)


class RegionCache:
    """Per-process cache of region geometry and decoded chips.

    It pickles without its caches, so each DataLoader worker rebuilds what it reads.
    """

    def __init__(self, regions: Sequence[Region], *, max_chips: int = 256):
        self._regions = {r.id: r for r in regions}
        self.max_chips = max_chips
        self._geoms: dict[tuple[str, int], RegionGeom] = {}
        self._chips: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def __getstate__(self) -> dict:
        return {"_regions": self._regions, "max_chips": self.max_chips}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._geoms = {}
        self._chips = OrderedDict()

    def region(self, region_id: str) -> Region:
        return self._regions[region_id]

    def geom(self, region_id: str, zoom: int) -> RegionGeom:
        key = (region_id, zoom)
        if key not in self._geoms:
            self._geoms[key] = _build_geom(self._regions[region_id], zoom)
        return self._geoms[key]

    def chip(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """``(rgb, valid)`` of a chip; least-recently-used entries are evicted beyond ``max_chips``."""
        path = Path(path)
        if path in self._chips:
            self._chips.move_to_end(path)
            return self._chips[path]
        rgb, valid, _ = read_chip(path, use_sidecar=False)
        self._chips[path] = (rgb, valid)
        while len(self._chips) > self.max_chips:
            self._chips.popitem(last=False)
        return rgb, valid

    def crop_window(self, spec: SampleSpec) -> Window:
        return _window(self.geom(spec.region_id, spec.zoom), spec.centre_px, spec.size_m)

    def masked_fraction(self, spec: SampleSpec) -> float:
        """Masked share of the crop on the low-resolution mask (absent tiles count as masked)."""
        return _masked_fraction(self.geom(spec.region_id, spec.zoom), self.crop_window(spec))

    def is_empty(self, spec: SampleSpec) -> bool:
        """True when no label box centre lies inside the crop."""
        return _is_empty(self.geom(spec.region_id, spec.zoom), self.crop_window(spec))


def _on_present_tiles(geom: RegionGeom, boxes: np.ndarray) -> np.ndarray:
    """Per box, whether it overlaps a present tile of the region with positive area; a non-finite box
    (pyproj maps coordinates outside lon/lat range to inf) does not."""
    present = np.zeros((geom.nrows + 1, geom.ncols + 1), np.int64)
    for x, y in geom.tiles:
        present[y - geom.ymin + 1, x - geom.xmin + 1] = 1
    integral = present.cumsum(0).cumsum(1)
    # zeroed before the integer cast, where NaN would become an out-of-bounds index; zero area never counts
    finite = np.where(np.isfinite(boxes).all(axis=1, keepdims=True), boxes, 0)
    clipped = np.clip(finite, 0, [geom.width, geom.height, geom.width, geom.height])
    positive = (clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])
    c1, r1 = (np.floor(clipped[:, :2] / geom.tile_px).astype(np.int64)).T
    c2, r2 = (np.ceil(clipped[:, 2:] / geom.tile_px).astype(np.int64)).T
    count = integral[r2, c2] - integral[r1, c2] - integral[r2, c1] + integral[r1, c1]
    return positive & (count > 0)


def any_label_on_chips(regions: Sequence[Region], classes: Sequence[str], cache: RegionCache) -> bool:
    """Whether a label of ``classes`` overlaps a chip of the regions at any zoom, as training sees the
    labels: tree Points as crowns, masked crowns (``masked_crown_rings``) as no label, tree_groups only in
    a region that labels them. Ignore polygons, the footprint and the 40 % rule are not applied, so a
    label they would drop still counts."""
    for region in regions:
        for zoom in region.chips:
            geom = cache.geom(region.id, zoom)
            by_class = {"tree": geom.boxes, "tree_group": geom.group_boxes}
            if any(_on_present_tiles(geom, by_class[name]).any() for name in classes):
                return True
    return False


def total_chips(regions: Sequence[Region]) -> int:
    return sum(len(paths) for r in regions for paths in r.chips.values())


def _probabilities(weights: Sequence[float]) -> np.ndarray:
    """Normalised weights; uniform when they sum to 0."""
    w = np.asarray(weights, np.float64)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1 / len(w))


def _fallback_centre(geom: RegionGeom, rng: np.random.Generator) -> tuple[float, float]:
    """Centre of a random present tile, preferring tiles that are not mostly masked."""
    cells = geom.tile_px // geom.cell_px
    present = sorted(geom.tiles)
    good = [
        (x, y)
        for x, y in present
        if geom.lowres_masked[
            (y - geom.ymin) * cells : (y - geom.ymin + 1) * cells,
            (x - geom.xmin) * cells : (x - geom.xmin + 1) * cells,
        ].mean()
        <= MAX_MASKED
    ]
    candidates = good or present
    x, y = candidates[int(rng.integers(len(candidates)))]
    return ((x - geom.xmin + 0.5) * geom.tile_px, (y - geom.ymin + 0.5) * geom.tile_px)


def _draw(rng: np.random.Generator, items: Sequence, weights: Sequence[float]):
    return items[rng.choice(len(items), p=_probabilities(weights))]


def plan_epoch(
    regions: Sequence[Region],
    cfg: TrainParams,
    seed: int,
    epoch: int,
    *,
    cache: RegionCache | None = None,
) -> list[SampleSpec]:
    """The epoch's crops, a pure function of ``(regions, cfg, seed, epoch)``.

    Each sample draws a source (``p ~ area ** source_alpha``), a biome within it (``biome_alpha``), a
    region within that (``p ~ valid area``), a zoom (``zoom_weights``), a size and a centre. A centre
    that puts the crop on an absent tile, over ``MAX_MASKED`` masked or past the empty-crop cap is
    redrawn, up to ``CROP_RETRIES`` times, before the crop falls back to one whole present tile.
    """
    if cache is None:
        cache = RegionCache(regions)
    rng = np.random.default_rng([seed, epoch])
    n = cfg.samples_per_epoch or total_chips(regions)

    eligible = []
    for region in sorted(regions, key=lambda r: r.id):
        if any(cfg.zoom_weights.get(z, 0) > 0 for z in region.chips):
            eligible.append(region)
        else:
            logger.warning(f"region {region.id}: no zoom with a positive zoom weight, skipped")
    if not eligible:
        raise ValueError("no region has a zoom with a positive zoom weight")

    counts: Counter[str] = Counter()
    drawn: Counter[str] = Counter()
    empties: Counter[str] = Counter()
    specs = []
    for _ in range(n):
        cap = cfg.region_repeat_cap
        open_regions = [r for r in eligible if cap is None or counts[r.id] < cap]
        if not open_regions:
            raise ValueError("region_repeat_cap too small for samples_per_epoch")

        sources = sorted({r.source for r in open_regions})
        areas = [sum(r.valid_area_m2 for r in open_regions if r.source == s) for s in sources]
        source = _draw(rng, sources, np.power(areas, cfg.source_alpha))
        in_source = [r for r in open_regions if r.source == source]
        biomes = sorted({r.biome for r in in_source})
        areas = [sum(r.valid_area_m2 for r in in_source if r.biome == b) for b in biomes]
        biome = _draw(rng, biomes, np.power(areas, cfg.biome_alpha))
        in_biome = [r for r in in_source if r.biome == biome]
        region = _draw(rng, in_biome, [r.valid_area_m2 for r in in_biome])
        zooms = sorted(region.chips)
        zoom = _draw(rng, zooms, [cfg.zoom_weights.get(z, 0) for z in zooms])
        geom = cache.geom(region.id, zoom)

        size_m = rng.uniform(*cfg.augment.crop_m)
        size_m = min(size_m, min(geom.width, geom.height) * geom.gsd_m)
        n_px = max(1, round(size_m / geom.gsd_m))
        centre = None
        for _ in range(CROP_RETRIES):
            cx = rng.uniform(n_px / 2, geom.width - n_px / 2)
            cy = rng.uniform(n_px / 2, geom.height - n_px / 2)
            w = _window(geom, (cx, cy), size_m)
            if not _tiles_present(geom, w) or _masked_fraction(geom, w) > MAX_MASKED:
                continue
            capped = cfg.empty_crop_cap is not None and _is_empty(geom, w)
            if capped and empties[source] + 1 > cfg.empty_crop_cap * (drawn[source] + 1):
                continue
            centre = (float(cx), float(cy))
            break
        if centre is None:
            centre = _fallback_centre(geom, rng)
            size_m = geom.tile_px * geom.gsd_m

        spec = SampleSpec(region.id, zoom, centre, float(size_m), int(rng.integers(0, 2**31 - 1)))
        specs.append(spec)
        counts[region.id] += 1
        drawn[source] += 1
        empties[source] += _is_empty(geom, _window(geom, centre, size_m))
    return specs


class Sample(NamedTuple):
    """One training crop and its targets, in crop pixels."""

    rgb: np.ndarray  # uint8 (n, n, 3), masked pixels at 0
    boxes: np.ndarray  # float32 (K, 4) XYXY, trees first, then groups
    labels: np.ndarray  # int64 (K,), index into CLASSES
    unsupervised: np.ndarray  # bool (len(CLASSES), n, n): True = no negatives for that class there


def _unsupervised(region: Region, crop_t: Affine, n: int, dropped_groups: np.ndarray) -> np.ndarray:
    """Per class, the crop pixels where it is not supervised.

    A class the region does not label is unsupervised everywhere; where the region labels tree_group, tree
    is unsupervised inside the group polygons, and tree_group inside the groups the 40 % rule dropped.
    """
    out = np.zeros((len(CLASSES), n, n), bool)
    for c, name in enumerate(CLASSES):
        if name not in region.classes:
            out[c] = True
    if "tree_group" in region.classes:
        out[0] |= rasterize_labels(region, crop_t, (n, n), 2)
        out[1] |= burn_boxes(dropped_groups, (n, n))
    return out


def _reaching(boxes: np.ndarray, w: Window) -> np.ndarray:
    """The boxes that reach into the window, in window coordinates."""
    x1, y1, x2, y2 = boxes.T
    reach = (x2 > w.x) & (x1 < w.x + w.width) & (y2 > w.y) & (y1 < w.y + w.height)
    return boxes[reach] - np.array([w.x, w.y, w.x, w.y], boxes.dtype)


def burn_boxes(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(H, W) bool, True on the pixels whose centre lies inside any of the boxes (as ``boxes_in_window``)."""
    out = np.zeros(shape, bool)
    edges = np.ceil(np.asarray(boxes, np.float64).reshape(-1, 4) - 0.5).astype(np.int64)
    for x1, y1, x2, y2 in np.clip(edges, 0, [shape[1], shape[0], shape[1], shape[0]]):
        out[y1:y2, x1:x2] = True
    return out


def load_sample(spec: SampleSpec, cache: RegionCache, cfg: TrainParams) -> Sample:
    """One training crop, its class-labelled boxes and the pixels where each class is unsupervised.

    A pure function of ``(spec, cfg)`` and the files on disk: all randomness comes from ``spec.aug_seed``.
    The cache's arrays are only read, never written.
    """
    geom = cache.geom(spec.region_id, spec.zoom)
    w = cache.crop_window(spec)
    n = w.width
    rgb = np.zeros((n, n, 3), np.uint8)
    valid = np.zeros((n, n), bool)
    tp = geom.tile_px
    for r in range(w.y // tp, (w.y + n - 1) // tp + 1):
        for c in range(w.x // tp, (w.x + n - 1) // tp + 1):
            path = geom.tiles.get((geom.xmin + c, geom.ymin + r))
            if path is None:
                continue
            chip_rgb, chip_valid = cache.chip(path)
            x0, x1 = max(w.x, c * tp), min(w.x + n, (c + 1) * tp)
            y0, y1 = max(w.y, r * tp), min(w.y + n, (r + 1) * tp)
            src = (slice(y0 - r * tp, y1 - r * tp), slice(x0 - c * tp, x1 - c * tp))
            dst = (slice(y0 - w.y, y1 - w.y), slice(x0 - w.x, x1 - w.x))
            rgb[dst] = chip_rgb[src]
            valid[dst] = chip_valid[src]

    region = cache.region(spec.region_id)
    crop_t = geom.transform_3857 @ Affine.translation(w.x, w.y)
    masked = (
        ~valid | rasterize_mask(region, crop_t, (n, n)) | burn_boxes(_reaching(geom.masked_boxes, w), (n, n))
    )
    trees = boxes_in_window(geom.boxes, ~masked, w.x, w.y, n, n)
    groups, kept = window_boxes(geom.group_boxes, ~masked, w.x, w.y, n, n, MIN_BOX_FRACTION)
    unsupervised = _unsupervised(region, crop_t, n, groups[~kept])
    boxes = np.concatenate([trees, groups[kept]])
    labels = np.repeat(np.arange(2, dtype=np.int64), [len(trees), int(kept.sum())])

    rng = np.random.default_rng(spec.aug_seed)
    aug = cfg.augment
    if aug.downscale and rng.random() < DOWNSCALE_P:
        rgb = downscale(rgb, rng)
    for enabled, transform in ((aug.flips, flip), (aug.rot90, rot90)):
        if enabled:
            layers = np.dstack([rgb, masked.astype(np.uint8), *unsupervised.astype(np.uint8)])
            stack, boxes = transform(layers, boxes, rng)
            rgb, masked = stack[..., :3], stack[..., 3].astype(bool)
            unsupervised = np.moveaxis(stack[..., 4:], -1, 0).astype(bool)
    if aug.colour_jitter:
        rgb = colour_jitter(np.ascontiguousarray(rgb), rng)
    rgb = np.array(rgb)  # contiguous and writable
    rgb[masked] = 0  # after jitter, so masked pixels stay exactly 0
    return Sample(rgb, boxes.astype(np.float32), labels, np.ascontiguousarray(unsupervised))


def sampler_stats(specs: Sequence[SampleSpec], cache: RegionCache) -> dict[str, float]:
    """Draws per source, biome and zoom, region repeat max and mean, and the empty-crop share per source."""
    sources: Counter[str] = Counter()
    biomes: Counter[str] = Counter()
    zooms: Counter[int] = Counter()
    repeats: Counter[str] = Counter()
    empty: Counter[str] = Counter()
    for spec in specs:
        region = cache.region(spec.region_id)
        sources[region.source] += 1
        biomes[region.biome] += 1
        zooms[spec.zoom] += 1
        repeats[spec.region_id] += 1
        empty[region.source] += cache.is_empty(spec)
    stats = {f"sampler/draws/source/{s}": float(c) for s, c in sorted(sources.items())}
    stats |= {f"sampler/draws/biome/{b}": float(c) for b, c in sorted(biomes.items())}
    stats |= {f"sampler/draws/zoom/{z}": float(c) for z, c in sorted(zooms.items())}
    if repeats:
        stats["sampler/region_repeat_max"] = float(max(repeats.values()))
        stats["sampler/region_repeat_mean"] = float(np.mean(list(repeats.values())))
    stats |= {f"sampler/empty_share/{s}": empty[s] / c for s, c in sorted(sources.items())}
    return stats
