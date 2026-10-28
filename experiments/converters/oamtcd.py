"""OAM-TCD (``restor/tcd`` CC-BY-4.0, ``restor/tcd-nc`` CC-BY-NC-4.0) to regions, one per tile.

Each 10 cm tile is written as a GeoTIFF in its own CRS and cut into Web Mercator chips by the tiler; tiles
are not contiguous, so a region never spans two. Tree instances become label boxes (their COCO ``bbox``); a
tree box wider or taller than ``MAX_TREE_SIDE_M`` becomes an ignore area instead. Canopy instances (tree
groups) become ignore areas with the tree boxes they overlap cut out, so masking never removes a tree
label. The Hugging Face ``test`` split is kept as ``test``; validation fold 0 is ``val`` and folds 1-4 are
``train``. Tiles within ``HOT_BUFFER_M`` of HOT's Freetown tiles are dropped so HOT val and test stay clean.
``restor/tcd-sa`` (CC-BY-SA) is never converted.
"""

import io
import json
import logging
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.affinity
from affine import Affine
from PIL import Image
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import box, mapping

from experiments.leakage import box_distance_m
from experiments.tiler import tile_scene
from fair_trees.georef import tile_bounds_4326, true_gsd
from fair_trees.preprocess import read_chip

logger = logging.getLogger(__name__)

SOURCE = "oamtcd"
SOURCE_GSD = 0.1
LICENCES = {"tcd": "CC-BY-4.0", "tcd-nc": "CC-BY-NC-4.0"}
ID_PREFIX = {"tcd": "by", "tcd-nc": "nc"}
MAX_TREE_SIDE_M = 30.0  # HOT's largest crown diameter
HOT_BUFFER_M = 1000.0
_SPLITS = ("train", "val", "test")
_TO_3857 = Transformer.from_crs(4326, 3857, always_xy=True)


def _rle_counts(s: str) -> list[int]:
    """Run lengths of a compressed COCO RLE string (the inverse of pycocotools' ``rleToString``)."""
    counts: list[int] = []
    p = 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_rle(rle: Mapping) -> np.ndarray:
    """(h, w) bool mask of a COCO RLE whose ``counts`` is a compressed string or a list (column-major)."""
    h, w = (int(v) for v in rle["size"])
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode()
    if isinstance(counts, str):
        counts = _rle_counts(counts)
    flat = np.zeros(h * w, bool)
    pos = 0
    for i, n in enumerate(counts):
        if i % 2:
            flat[pos : pos + n] = True
        pos += n
    return flat.reshape(w, h).T


def category_ids(annotations: Iterable[Sequence[Mapping]]) -> dict[str, int]:
    """``{"tree": id, "canopy": id}``: the more numerous category is the tree, and canopy must be larger.

    Sizes come from ``bbox`` (crowd RLE masks carry ``area`` 0).
    """
    counts: Counter[int] = Counter()
    sizes: dict[int, list[float]] = {}
    for anns in annotations:
        for a in anns:
            cid = int(a["category_id"])
            counts[cid] += 1
            sizes.setdefault(cid, []).append(float(a["bbox"][2]) * float(a["bbox"][3]))
    if len(counts) != 2:
        raise ValueError(f"expected two categories (tree, canopy), found {dict(counts)}")
    (tree, n_tree), (canopy, n_canopy) = counts.most_common()
    medians = {cid: float(np.median(s)) for cid, s in sizes.items()}
    if n_tree < 2 * n_canopy or medians[canopy] <= medians[tree]:
        raise ValueError(
            f"cannot tell tree from canopy: counts {dict(counts)}, median bbox areas {medians} "
            "(expected trees at least twice as many as canopy, canopy larger)"
        )
    return {"tree": tree, "canopy": canopy}


def _canopy_pixels(segmentation) -> shapely.Geometry:
    """Canopy outline in pixel coordinates from COCO polygons or an RLE mask."""
    if isinstance(segmentation, Mapping):
        mask = decode_rle(segmentation)
        parts = [
            shapely.geometry.shape(g)
            for g, v in rasterio.features.shapes(mask.astype(np.uint8), mask=mask)
            if v == 1
        ]
    else:
        parts = [
            shapely.Polygon(np.asarray(ring, float).reshape(-1, 2)).buffer(0)
            for ring in segmentation
            if len(ring) >= 6
        ]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def _to_4326(geom: shapely.Geometry, transform: Affine, crs) -> dict:
    """Pixel-space geometry -> EPSG:4326 GeoJSON geometry, through the tile transform and CRS."""
    t = transform
    projected = shapely.affinity.affine_transform(geom, [t.a, t.b, t.d, t.e, t.c, t.f])
    return transform_geom(crs, "EPSG:4326", mapping(projected))


def tile_features(
    annotations: Sequence[Mapping],
    transform: Affine,
    crs,
    ids: Mapping[str, int],
    *,
    gsd_m: float = SOURCE_GSD,
) -> tuple[list[dict], list[dict], dict]:
    """Label features (tree boxes) and ignore features (large trees, canopy minus trees), in EPSG:4326.

    ``gsd_m`` is the tile's ground metres per pixel, for the ``MAX_TREE_SIDE_M`` limit.
    """
    trees, large, canopy_px = [], [], []
    for a in annotations:
        cid = int(a["category_id"])
        if cid == ids["tree"]:
            x, y, w, h = (float(v) for v in a["bbox"])
            if w <= 0 or h <= 0:
                continue
            side_m = max(w, h) * gsd_m
            (large if side_m > MAX_TREE_SIDE_M else trees).append(box(x, y, x + w, y + h))
        elif cid == ids["canopy"]:
            canopy_px.append(_canopy_pixels(a["segmentation"]))
    tree_union = shapely.union_all(trees) if trees else shapely.Polygon()
    labels = [_feature(_to_4326(g, transform, crs), {"label": 1, "source": SOURCE}) for g in trees]
    ignore = [_feature(_to_4326(g, transform, crs), {"reason": "large_tree"}) for g in large]
    n_canopy = 0
    for geom in canopy_px:
        rest = geom.difference(tree_union) if trees else geom
        if rest.is_empty or rest.area <= 0:
            continue
        ignore.append(_feature(_to_4326(rest, transform, crs), {"reason": "canopy"}))
        n_canopy += 1
    return labels, ignore, {"trees": len(trees), "large_trees": len(large), "canopy": n_canopy}


def _feature(geometry: dict, properties: dict) -> dict:
    return {"type": "Feature", "properties": properties, "geometry": geometry}


def near_hot(
    bounds_3857: np.ndarray, hot_bbox_4326: tuple[float, float, float, float], buffer_m: float = HOT_BUFFER_M
) -> np.ndarray:
    """Bool per EPSG:3857 box: within ``buffer_m`` ground metres of HOT's (west, south, east, north) box."""
    west, south, east, north = hot_bbox_4326
    minx, miny = _TO_3857.transform(west, south)
    maxx, maxy = _TO_3857.transform(east, north)
    return (
        box_distance_m(np.asarray(bounds_3857, float), np.array([[minx, miny, maxx, maxy]]))[:, 0] < buffer_m
    )


def _split(row: Mapping) -> str:
    if row["hf_split"] == "test":
        return "test"
    return "val" if int(row["validation_fold"]) == 0 else "train"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _write_tif(path: Path, image: bytes, transform: Affine, crs) -> None:
    rgb = np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    with rasterio.open(
        path, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8", crs=crs, transform=transform
    ) as dst:
        dst.write(rgb.transpose(2, 0, 1))


def convert(
    rows: Iterable[Mapping],
    out_root: Path,
    *,
    ids: Mapping[str, int],
    hot_bbox_4326: tuple[float, float, float, float],
    biome_fn: Callable[[float, float], str] | None = None,
    zooms: Sequence[int] = (19, 20),
    work_dir: Path | None = None,
) -> dict:
    """Write one region per kept tile and ``stats.json`` into an empty ``out_root``; return the stats.

    Each row: ``dataset`` (``"tcd"`` or ``"tcd-nc"``), ``hf_split``, ``image_id``, ``image`` (encoded bytes),
    ``width``, ``height``, ``bounds`` (minx, miny, maxx, maxy in ``crs``), ``crs``, ``validation_fold`` and
    ``coco_annotations`` (a JSON list).
    """
    out_root = Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(f"{out_root}: output directory is not empty; old regions would mix with new ones")
    work_dir = Path(work_dir) if work_dir else out_root.parent / f"{out_root.name}-work"

    regions: dict[str, dict[str, int]] = {s: {} for s in _SPLITS}
    chips: dict[str, dict[str, int]] = {str(z): dict.fromkeys(_SPLITS, 0) for z in zooms}
    totals = Counter()
    area_m2 = dict.fromkeys(_SPLITS, 0.0)
    invalid_shares: list[float] = []
    for row in rows:
        licence = LICENCES[row["dataset"]]
        region_id = f"{SOURCE}-{ID_PREFIX[row['dataset']]}-{row['image_id']}"
        crs = CRS.from_user_input(row["crs"])
        minx, miny, maxx, maxy = (float(v) for v in row["bounds"])
        width, height = int(row["width"]), int(row["height"])
        px = (maxx - minx) / width
        if not (maxx > minx and maxy > miny and 0.05 <= px <= 0.5 and 0.05 <= (maxy - miny) / height <= 0.5):
            raise ValueError(
                f"{region_id}: bounds {row['bounds']} give no ~10 cm pixel (want minx, miny, maxx, maxy)"
            )
        if near_hot(np.array([transform_bounds(crs, "EPSG:3857", minx, miny, maxx, maxy)]), hot_bbox_4326)[0]:
            totals["dropped_near_hot"] += 1
            continue

        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        west, south, east, north = transform_bounds(crs, "EPSG:4326", minx, miny, maxx, maxy)
        lon, lat = (west + east) / 2, (south + north) / 2
        source_gsd = px * math.cos(math.radians(lat))  # a Mercator metre shrinks by cos(lat) on the ground
        tif = work_dir / f"{region_id}.tif"
        _write_tif(tif, row["image"], transform, crs)
        tiled = tile_scene(tif, work_dir / region_id, source_gsd=source_gsd, zooms=zooms)
        tif.unlink()
        if not tiled:
            totals["no_chips"] += 1
            continue

        split = _split(row)
        region_dir = out_root / region_id
        lowest = min(tiled)
        valid_area = 0.0
        for z, paths in sorted(tiled.items()):
            for path in paths:
                _, valid, (x, y, cz) = read_chip(path)  # closes the file before the move
                invalid_shares.append(1.0 - float(valid.mean()))
                if cz == lowest:
                    _, s, _, n = tile_bounds_4326(x, y, cz)
                    valid_area += int(valid.sum()) * true_gsd(cz, (s + n) / 2) ** 2
                dest = region_dir / f"z{z}" / "chips" / path.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(path, dest)
            chips[str(z)][split] += len(paths)
        shutil.rmtree(work_dir / region_id, ignore_errors=True)

        anns = json.loads(row["coco_annotations"] or "[]")
        labels, ignore, counts = tile_features(anns, transform, crs, ids, gsd_m=source_gsd)
        for f in labels:
            f["properties"] |= {"licence": licence, "region": region_id}
        _write_json(region_dir / "labels.geojson", {"type": "FeatureCollection", "features": labels})
        _write_json(region_dir / "ignore.geojson", {"type": "FeatureCollection", "features": ignore})
        footprint = transform_geom(crs, "EPSG:4326", mapping(box(minx, miny, maxx, maxy)))
        meta = {
            "id": region_id,
            "source": SOURCE,
            "biome": biome_fn(lon, lat) if biome_fn else "unknown",
            "licence": licence,
            "split": split,
            "zooms": sorted(tiled),
            "footprint": footprint,
            "valid_area_m2": valid_area,
        }
        _write_json(region_dir / "region.json", meta)
        regions[split][licence] = regions[split].get(licence, 0) + 1
        area_m2[split] += valid_area
        totals["tree_boxes"] += counts["trees"]
        totals["large_trees"] += counts["large_trees"]
        totals["canopy_masks"] += counts["canopy"]

    total_m2 = sum(area_m2.values())
    stats = {
        "regions": regions,
        "chips": chips,
        "tree_boxes": totals["tree_boxes"],
        "large_trees": totals["large_trees"],
        "canopy_masks": totals["canopy_masks"],
        "dropped_near_hot": totals["dropped_near_hot"],
        "no_chips": totals["no_chips"],
        "categories": dict(ids),
        "masked_pct": round(float(np.mean(invalid_shares)) * 100, 1) if invalid_shares else 0.0,
        "val_share": area_m2["val"] / total_m2 if total_m2 else 0.0,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    logger.info("converted %d OAM-TCD tiles: %s", sum(sum(v.values()) for v in regions.values()), stats)
    return stats
