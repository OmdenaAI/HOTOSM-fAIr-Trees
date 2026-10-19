"""NeonTreeEvaluation training rasters to regions, keeping every chip away from the eval plots.

Each annotated training raster is cut into chips per zoom; chips within ``buffer_m`` of an eval plot are
excluded, the rest are group split (rasters closer than ``buffer_m`` stay on one side), and each
``(scene, split)`` pair becomes one region.
"""

import json
import logging
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import transform, transform_bounds, transform_geom

from experiments.leakage import near_footprints
from experiments.split import group_split
from experiments.tiler import tile_scene
from fair_trees.georef import parse_oam_name, tile_bounds_4326, true_gsd
from fair_trees.preprocess import read_chip

logger = logging.getLogger(__name__)

SOURCE = "neon"
SOURCE_GSD = 0.1
LICENCE = "CC-BY-4.0"  # as stated on Zenodo record 5914554 [V]
_SPLITS = ("train", "val")
_SITE = re.compile(r"_([A-Z]{4})_")


@dataclass(frozen=True)
class _Scene:
    labels: list[dict]  # EPSG:4326 polygons
    footprint: dict
    biome: str


@dataclass(frozen=True)
class _Chip:
    scene: str
    xyz: tuple[int, int, int]
    path: Path


def voc_boxes(xml_path: Path) -> np.ndarray:
    """(N, 4) float32 ``xmin, ymin, xmax, ymax`` pixel boxes from a Pascal VOC file."""
    boxes = [
        [float(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")]
        for obj in ET.parse(xml_path).getroot().iter("object")
    ]
    return np.asarray(boxes, np.float32).reshape(-1, 4)


def find_training_rasters(raw_root: Path) -> list[tuple[Path, Path]]:
    """``(tif, xml)`` pairs for the training rasters that have annotations."""
    pairs = []
    for tif in sorted((Path(raw_root) / "training" / "RGB").glob("*.tif")):
        xml = Path(raw_root) / "annotations" / f"{tif.stem}.xml"
        if not xml.is_file():
            logger.info("%s: no annotations, skipped", tif.name)
            continue
        pairs.append((tif, xml))
    return pairs


def eval_plot_footprints(raw_root: Path) -> dict[str, tuple[float, float, float, float]]:
    """EPSG:4326 ``(west, south, east, north)`` per eval plot, keyed by plot name.

    An eval plot is an ``evaluation/RGB`` image with an annotation, the same set the benchmark scores;
    the archive's unannotated plot images are never scored, so they cannot leak and are ignored.
    """
    footprints = {}
    for tif in sorted((Path(raw_root) / "evaluation" / "RGB").glob("*.tif")):
        if not (Path(raw_root) / "annotations" / f"{tif.stem}.xml").is_file():
            continue
        with rasterio.open(tif) as ds:
            if ds.crs is None:
                raise ValueError(
                    f"{tif.name}: eval plot has no CRS; the 1 km exclusion needs georeferenced plots"
                )
            footprints[tif.stem] = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    return footprints


def _box_polygon(box: Sequence[float], affine: Affine) -> dict:
    """Closed counter-clockwise ring of a pixel box, in the CRS of ``affine``."""
    x0, y0, x1, y1 = map(float, box)
    ring = [affine @ corner for corner in ((x0, y1), (x1, y1), (x1, y0), (x0, y0))]
    signed_area = sum(ax * by - bx * ay for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1], strict=True))
    if signed_area < 0:
        ring.reverse()
    return {"type": "Polygon", "coordinates": [ring + ring[:1]]}


def _read_scene(tif: Path, xml: Path, biome_fn: Callable[[float, float], str] | None) -> _Scene:
    with rasterio.open(tif) as ds:
        polygons = [_box_polygon(box, ds.transform) for box in voc_boxes(xml)]
        polygons.append(_box_polygon((0, 0, ds.width, ds.height), ds.transform))
        *labels, footprint = transform_geom(ds.crs, "EPSG:4326", polygons)
        cx, cy = ds.transform @ (ds.width / 2, ds.height / 2)
        (lon,), (lat,) = transform(ds.crs, "EPSG:4326", [cx], [cy])
    return _Scene(labels, footprint, biome_fn(lon, lat) if biome_fn else "unknown")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _write_region(out_root: Path, stem: str, split: str, scene: _Scene, chips: list[_Chip]) -> list[float]:
    """Move the chips into the region folder and write its files.

    Returns each chip's invalid share and the region's valid ground area in m^2.
    """
    region_id = f"{stem}-{split}"
    region_dir = out_root / region_id
    lowest = min(c.xyz[2] for c in chips)
    valid_area_m2 = 0.0
    invalid_shares = []
    for chip in chips:
        _, valid, (x, y, z) = read_chip(chip.path)  # closes the file before the move
        invalid_shares.append(1.0 - float(valid.mean()))
        if z == lowest:
            _, south, _, north = tile_bounds_4326(x, y, z)
            valid_area_m2 += int(valid.sum()) * true_gsd(z, (south + north) / 2) ** 2
        dest = region_dir / f"z{z}" / "chips" / chip.path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(chip.path, dest)

    properties = {"label": 1, "source": SOURCE, "licence": LICENCE, "region": region_id}
    features = [{"type": "Feature", "properties": properties, "geometry": g} for g in scene.labels]
    _write_json(region_dir / "labels.geojson", {"type": "FeatureCollection", "features": features})
    _write_json(region_dir / "ignore.geojson", {"type": "FeatureCollection", "features": []})
    meta = {
        "id": region_id,
        "source": SOURCE,
        "biome": scene.biome,
        "licence": LICENCE,
        "split": split,
        "zooms": sorted({c.xyz[2] for c in chips}),
        "footprint": scene.footprint,
        "valid_area_m2": valid_area_m2,
    }
    _write_json(region_dir / "region.json", meta)
    return invalid_shares, valid_area_m2


def convert(
    raw_root: Path,
    out_root: Path,
    *,
    zooms: Sequence[int] = (19, 20),
    val_ratio: float = 0.15,
    seed: int = 42,
    buffer_m: float = 1000.0,
    eval_buffer_m: float = 1000.0,
    biome_fn: Callable[[float, float], str] | None = None,
    work_dir: Path | None = None,
) -> dict:
    """Write a region per ``(scene, split)`` and ``stats.json`` into an empty ``out_root``; return the stats.

    ``eval_buffer_m`` excludes chips near the eval plots; ``buffer_m`` is the train/val buffer of the split.
    """
    raw_root, out_root = Path(raw_root), Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise ValueError(f"{out_root}: output directory is not empty; old regions would mix with new ones")
    work_dir = Path(work_dir) if work_dir else out_root.parent / f"{out_root.name}-work"
    footprints = eval_plot_footprints(raw_root)

    scenes: dict[str, _Scene] = {}
    chips: list[_Chip] = []
    skipped_no_crs = []
    for tif, xml in find_training_rasters(raw_root):
        if _SITE.search(tif.stem) is None:
            logger.warning("%s: no NEON site code in the name, skipped", tif.name)
            continue
        with rasterio.open(tif) as ds:
            has_crs = ds.crs is not None
        if not has_crs:
            # the name gives the NEON tile corner, not a crop's offset, so the raster cannot be placed
            # (or kept away from the eval plots) without guessing
            logger.warning("%s: training raster has no CRS, skipped", tif.name)
            skipped_no_crs.append(tif.name)
            continue
        scenes[tif.stem] = _read_scene(tif, xml, biome_fn)
        tiled = tile_scene(tif, work_dir / tif.stem, source_gsd=SOURCE_GSD, zooms=zooms)
        chips += [_Chip(tif.stem, parse_oam_name(p), p) for z in sorted(tiled) for p in tiled[z]]

    near = near_footprints([c.xyz for c in chips], list(footprints.values()), eval_buffer_m)
    kept = [c for c, is_near in zip(chips, near, strict=True) if not is_near]

    # the rasters (0.1-0.6 km) are smaller than the buffer, so a block split would drop most of train
    sides = group_split(
        [c.xyz for c in kept], [c.scene for c in kept], val_ratio=val_ratio, seed=seed, buffer_m=buffer_m
    )
    by_region: dict[tuple[str, str], list[_Chip]] = {}
    dropped = 0
    for chip, side in zip(kept, sides, strict=True):
        if side == "drop":
            dropped += 1
        else:
            by_region.setdefault((chip.scene, side), []).append(chip)

    counts = {str(z): dict.fromkeys(_SPLITS, 0) for z in zooms}
    invalid_shares: list[float] = []
    area_m2 = dict.fromkeys(_SPLITS, 0.0)
    for (stem, split), group in sorted(by_region.items()):
        shares, valid_area_m2 = _write_region(out_root, stem, split, scenes[stem], group)
        invalid_shares += shares
        area_m2[split] += valid_area_m2
        for chip in group:
            counts[str(chip.xyz[2])][split] += 1
    for stem in scenes:
        if (work_dir / stem).exists():
            shutil.rmtree(work_dir / stem)  # only excluded and dropped chips are left

    total_m2 = sum(area_m2.values())
    stats = {
        "chips": counts,
        "boxes": sum(len(scenes[stem].labels) for stem in {stem for stem, _ in by_region}),
        "excluded_near_eval_plots": int(near.sum()),
        "dropped_by_buffer": dropped,
        "masked_pct": round(float(np.mean(invalid_shares)) * 100, 1) if invalid_shares else 0.0,
        "skipped_no_crs": skipped_no_crs,
        "val_scenes": sorted(stem for stem, split in by_region if split == "val"),
        # share of the valid ground area; chip counts mislead when a zoom exists only at some latitudes
        "val_share": area_m2["val"] / total_m2 if total_m2 else 0.0,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    logger.info("converted %d NEON scenes into %d regions: %s", len(scenes), len(by_region), stats)
    return stats
