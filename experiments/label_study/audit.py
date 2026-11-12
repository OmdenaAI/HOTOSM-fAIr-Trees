"""Reduce each training source's labels to one table and profile how it labels dense canopy.

Every source becomes a label table (one row per box, ``cls`` "tree" or "group", sizes in metres, centres in
a metric frame local to the region) and a region table (the annotated ground area per region).
"""

import math
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from experiments.converters.neon import voc_boxes
from experiments.label_study.merge import cluster_boxes

LABEL_COLUMNS = ["source", "region", "cls", "x_m", "y_m", "w_m", "h_m", "area_m2"]
REGION_COLUMNS = ["source", "region", "area_m2"]


def label_table(source: str, region, cls, x0, y0, x1, y1, area_m2=None) -> pd.DataFrame:
    """Rows from box corners in metres; ``area_m2`` defaults to the box area."""
    x0, y0, x1, y1 = (np.asarray(v, float) for v in (x0, y0, x1, y1))
    w, h = x1 - x0, y1 - y0
    area = w * h if area_m2 is None else np.asarray(area_m2, float)
    columns = [source, region, cls, (x0 + x1) / 2, (y0 + y1) / 2, w, h, area]
    return pd.DataFrame(dict(zip(LABEL_COLUMNS, columns, strict=True)))


def region_table(source: str, region, area_m2) -> pd.DataFrame:
    columns = [source, region, np.asarray(area_m2, float)]
    return pd.DataFrame(dict(zip(REGION_COLUMNS, columns, strict=True)))


def _rle_counts(text: str) -> list[int]:
    """Run lengths from a COCO compressed RLE string (pycocotools' ``rleFrString``)."""
    counts: list[int] = []
    pos = 0
    while pos < len(text):
        value, shift, more = 0, 0, True
        while more:
            c = ord(text[pos]) - 48
            value |= (c & 0x1F) << shift
            more = bool(c & 0x20)
            pos += 1
            shift += 5
            if not more and c & 0x10:
                value |= -1 << shift  # sign-extend the last 5-bit chunk
        if len(counts) > 2:
            value += counts[-2]  # runs after the third are stored as deltas
        counts.append(value)
    return counts


def _shoelace(flat: Sequence[float]) -> float:
    x, y = np.asarray(flat, float).reshape(-1, 2).T
    return abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1))) / 2


def _area_px(ann: Mapping) -> float | None:
    """Pixel area: ``area`` if positive, else the segmentation's area, else None.

    OAM-TCD's crowd (RLE) masks carry area 0.
    """
    if (ann.get("area") or 0) > 0:
        return float(ann["area"])
    seg = ann.get("segmentation")
    if isinstance(seg, dict):  # RLE: runs alternate background, foreground, starting with background
        counts = _rle_counts(seg["counts"]) if isinstance(seg["counts"], str) else seg["counts"]
        return float(sum(counts[1::2]))
    if seg:
        return sum(_shoelace(polygon) for polygon in seg)
    return None


def coco_labels(
    coco: Mapping,
    *,
    source: str,
    gsd_m: float,
    tree_ids: Collection[int],
    group_ids: Collection[int] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One region per COCO image; annotations of categories in neither id set are dropped."""
    images = coco["images"]
    regions = region_table(source, [im["id"] for im in images], [im["width"] * im["height"] for im in images])
    regions["area_m2"] *= gsd_m**2

    classes = dict.fromkeys(tree_ids, "tree") | dict.fromkeys(group_ids, "group")
    anns = [a for a in coco["annotations"] if a["category_id"] in classes]
    x, y, w, h = np.asarray([a["bbox"] for a in anns], float).reshape(-1, 4).T * gsd_m
    px = [_area_px(a) for a in anns]
    area = [bw * bh if p is None else p * gsd_m**2 for p, bw, bh in zip(px, w, h, strict=True)]
    cls = [classes[a["category_id"]] for a in anns]
    labels = label_table(source, [a["image_id"] for a in anns], cls, x, y, x + w, y + h, area)
    return labels, regions


def neon_labels(pairs: Sequence[tuple[Path, Path]], gsd_m: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One region per ``(tif, xml)`` training raster; boxes and raster size scaled by ``gsd_m``."""
    labels, regions = [], []
    for tif, xml in pairs:
        with rasterio.open(tif) as ds:
            regions.append(region_table("neon", [tif.stem], [ds.width * ds.height * gsd_m**2]))
        boxes = voc_boxes(xml).astype(float) * gsd_m
        labels.append(label_table("neon", tif.stem, "tree", *boxes.T))
    return pd.concat(labels, ignore_index=True), pd.concat(regions, ignore_index=True)


def geo_labels(
    gdf: gpd.GeoDataFrame, regions: gpd.GeoDataFrame, *, source: str, cls: str = "tree"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Polygons with a ``region`` column, in a metric CRS; boxes are their bounds, areas their own."""
    for frame in (gdf, regions):
        if frame.crs is None or not frame.crs.is_projected:
            raise ValueError(f"{source}: geometries must be in a metric CRS, got {frame.crs}")
    bounds = gdf.geometry.bounds
    labels = label_table(source, gdf["region"].to_numpy(), cls, *bounds.to_numpy().T, gdf.geometry.area)
    return labels, region_table(source, regions["region"].to_numpy(), regions.geometry.area)


def _percentile(values, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else math.nan


def _share(flags) -> float:
    return float(np.mean(flags)) if len(flags) else math.nan


def _cluster_size_per_box(boxes: pd.DataFrame, gap_m: float) -> np.ndarray:
    """Size of the cluster each box belongs to, clustering ``x_m, y_m, w_m, h_m`` boxes at ``gap_m``."""
    corners = np.column_stack(
        [
            boxes.x_m - boxes.w_m / 2,
            boxes.y_m - boxes.h_m / 2,
            boxes.x_m + boxes.w_m / 2,
            boxes.y_m + boxes.h_m / 2,
        ]
    )
    ids = cluster_boxes(corners, gap_m)
    return np.bincount(ids)[ids]


def profile(labels: pd.DataFrame, regions: pd.DataFrame, *, gap_m: float, min_crowns: int) -> dict:
    """Crown sizes, density, crown contact and group sizes of one source.

    Clustering runs per region on the tree boxes with a size; shares are over those trees.
    """
    trees = labels[labels.cls == "tree"]
    groups = labels[labels.cls == "group"]
    sized = trees.dropna(subset=["w_m", "h_m"])
    crown = np.sqrt(sized.w_m * sized.h_m).to_numpy()
    at_zero, at_gap = [], []
    for _, part in sized.groupby("region", sort=False):
        at_zero.append(_cluster_size_per_box(part, 0.0))
        at_gap.append(_cluster_size_per_box(part, gap_m))
    at_zero = np.concatenate(at_zero) if at_zero else np.zeros(0, np.int64)
    at_gap = np.concatenate(at_gap) if at_gap else np.zeros(0, np.int64)
    ground_m2 = float(regions.area_m2.sum())

    median_crown = _percentile(crown, 50)
    long_side = np.maximum(groups.w_m, groups.h_m).to_numpy()
    crowns_equiv = groups.area_m2.to_numpy() / (math.pi * (median_crown / 2) ** 2)
    has_groups = len(groups) > 0
    return {
        "regions": len(regions),
        "trees": len(trees),
        "groups": len(groups),
        "crown_size_m_p5": _percentile(crown, 5),
        "crown_size_m_median": median_crown,
        "crown_size_m_p95": _percentile(crown, 95),
        "crowns_per_ha": len(trees) / (ground_m2 / 10_000) if ground_m2 else math.nan,
        "touching_share": _share(at_zero >= 2),
        "hot_rule_group_share": _share(at_gap >= min_crowns),
        "tree_cluster_p90": _percentile(at_zero, 90),
        "group_area_share": float(groups.area_m2.sum()) / ground_m2 if has_groups and ground_m2 else math.nan,
        "group_long_m_median": _percentile(long_side, 50),
        "group_long_m_p90": _percentile(long_side, 90),
        "groups_over_20m": _share(long_side > 20),
        "groups_over_40m": _share(long_side > 40),
        "group_crowns_equiv_p10": _percentile(crowns_equiv, 10),
        "group_crowns_equiv_median": _percentile(crowns_equiv, 50),
    }


def audit_table(
    parts: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]], *, gap_m: float, min_crowns: int
) -> pd.DataFrame:
    """One profile row per source, indexed by source name."""
    rows = {
        name: profile(labels, regions, gap_m=gap_m, min_crowns=min_crowns)
        for name, (labels, regions) in parts.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("source")
