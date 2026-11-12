"""Mask the crowns the release model finds and the labels miss, in copies of a staged region tree.

The release model (the teacher) runs over each train region; a confident detection that overlaps no label
becomes a ``reason: "teacher"`` ignore square, so training is neither rewarded nor punished there. Scope
``"all"`` masks such detections everywhere, ``"label_free"`` only on tiles no label overlaps (the detection's
centre decides its tile). Val regions are copied unchanged, so val scores stay comparable.
"""

import dataclasses
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

from fair_trees.dataset import MAX_MASKED, Region, labels_to_boxes, load_geojson, load_regions, rasterize_mask
from fair_trees.georef import mosaic_px_to_4326_rings
from fair_trees.params import InferenceParams
from fair_trees.preprocess import mosaic_chips, read_chip
from fair_trees.serve import RunFn, predict_mosaic

logger = logging.getLogger(__name__)

SCOPES = {"all": "hot-teacher-all", "label_free": "hot-teacher-empty"}  # scope -> shard set name


def _tile_span(lo: np.ndarray, hi: np.ndarray, tile_px: int) -> tuple[np.ndarray, np.ndarray]:
    """First and last tile index each [lo, hi) interval overlaps with positive length (unclipped)."""
    return np.floor(lo / tile_px).astype(int), (np.ceil(hi / tile_px) - 1).astype(int)


def _tile_shares(mask: np.ndarray, tile_px: int) -> np.ndarray:
    """(rows, cols) share of True pixels per tile of a mosaic-sized bool image."""
    rows, cols = mask.shape[0] // tile_px, mask.shape[1] // tile_px
    return mask[: rows * tile_px, : cols * tile_px].reshape(rows, tile_px, cols, tile_px).mean(axis=(1, 3))


def label_free_tiles(label_boxes: np.ndarray, valid: np.ndarray, tile_px: int) -> np.ndarray:
    """(rows, cols) bool over the mosaic's tile grid: a present tile that no label box overlaps."""
    free = _tile_shares(valid, tile_px) > 0
    rows, cols = free.shape
    boxes = np.asarray(label_boxes, np.float64).reshape(-1, 4)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    c0, c1 = _tile_span(boxes[:, 0], boxes[:, 2], tile_px)
    r0, r1 = _tile_span(boxes[:, 1], boxes[:, 3], tile_px)
    inside = (c1 >= 0) & (c0 < cols) & (r1 >= 0) & (r0 < rows)
    for a, b, c, d in zip(r0[inside], r1[inside], c0[inside], c1[inside], strict=True):
        free[max(a, 0) : b + 1, max(c, 0) : d + 1] = False
    return free


def _burn(boxes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bool image of ``shape``, True on the pixels whose centre lies inside a box (as ``rasterize_mask``)."""
    out = np.zeros(shape, bool)
    edges = np.ceil(np.asarray(boxes, np.float64).reshape(-1, 4) - 0.5).astype(int)
    edges = np.clip(edges, 0, [shape[1], shape[0], shape[1], shape[0]])
    for x1, y1, x2, y2 in edges:
        out[y1:y2, x1:x2] = True
    return out


def _centre_masked(boxes: np.ndarray, masked: np.ndarray) -> np.ndarray:
    """Per box, whether the pixel holding its centre is already masked."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    h, w = masked.shape
    cx = np.clip(np.floor((boxes[:, 0] + boxes[:, 2]) / 2), 0, w - 1).astype(int)
    cy = np.clip(np.floor((boxes[:, 1] + boxes[:, 3]) / 2), 0, h - 1).astype(int)
    return masked[cy, cx]


def _on_labels(boxes: np.ndarray, label_boxes: np.ndarray) -> np.ndarray:
    """Per box, whether it overlaps any label box with positive area."""
    if len(boxes) == 0 or len(label_boxes) == 0:
        return np.zeros(len(boxes), bool)
    w = np.minimum(boxes[:, None, 2], label_boxes[None, :, 2]) - np.maximum(
        boxes[:, None, 0], label_boxes[None, :, 0]
    )
    h = np.minimum(boxes[:, None, 3], label_boxes[None, :, 3]) - np.maximum(
        boxes[:, None, 1], label_boxes[None, :, 1]
    )
    return ((w > 0) & (h > 0)).any(axis=1)


def select_teacher(
    boxes: np.ndarray, label_boxes: np.ndarray, free: np.ndarray, tile_px: int, scope: str
) -> np.ndarray:
    """Keep mask over teacher boxes: off all labels; for ``"label_free"``, centred on a label-free tile."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}, expected one of {sorted(SCOPES)}")
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    keep = ~_on_labels(boxes, np.asarray(label_boxes, np.float64).reshape(-1, 4))
    if scope == "label_free":
        rows, cols = free.shape
        c = np.clip(np.floor((boxes[:, 0] + boxes[:, 2]) / 2 / tile_px), 0, cols - 1).astype(int)
        r = np.clip(np.floor((boxes[:, 1] + boxes[:, 3]) / 2 / tile_px), 0, rows - 1).astype(int)
        keep &= free[r, c]
    return keep


def _zero_counts() -> dict:
    return {
        "teacher": 0,
        "on_labels": 0,
        "already_masked": 0,
        **dict.fromkeys(SCOPES, 0),
        "area_m2": dict.fromkeys(SCOPES, 0.0),
        "tiles": 0,
        "tiles_over_max_masked": dict.fromkeys(("existing", *SCOPES), 0),
    }


def _add_counts(total: dict, counts: dict) -> None:
    for key, value in counts.items():
        if isinstance(value, dict):
            for k, v in value.items():
                total[key][k] += v
        else:
            total[key] += value


def teacher_features(
    region: Region, run_fn: RunFn, params: InferenceParams, *, zoom: int = 20
) -> tuple[dict[str, list[dict]], dict]:
    """Per scope, the ignore features for one region, and its counts.

    A teacher box whose centre is already masked (outside the footprint, an absent pixel or an existing
    ignore polygon such as a tree group) adds nothing and is left out, counted in ``already_masked``.
    ``tiles_over_max_masked`` counts present tiles masked over ``MAX_MASKED``, which the sampler avoids,
    before (``existing``) and after each scope's teacher squares.
    """
    z = zoom if zoom in region.chips else min(region.chips)
    tile_px = read_chip(region.chips[z][0], use_sidecar=False)[0].shape[1]
    labels = load_geojson(region.labels_path)
    features: dict[str, list[dict]] = {scope: [] for scope in SCOPES}
    counts = _zero_counts()
    for mosaic in mosaic_chips(region.chips[z], use_sidecar=False):
        det = predict_mosaic(run_fn, mosaic, params)
        label_boxes = labels_to_boxes(labels, mosaic.transform_3857)
        free = label_free_tiles(label_boxes, mosaic.valid, tile_px)
        shape = mosaic.valid.shape
        existing = ~mosaic.valid | rasterize_mask(region, mosaic.transform_3857, shape)
        on_labels = _on_labels(det.boxes.astype(np.float64), label_boxes)
        already = _centre_masked(det.boxes, existing) & ~on_labels
        present = _tile_shares(mosaic.valid, tile_px) > 0
        counts["teacher"] += len(det.boxes)
        counts["on_labels"] += int(on_labels.sum())
        counts["already_masked"] += int(already.sum())
        counts["tiles"] += int(present.sum())
        over = (_tile_shares(existing, tile_px) > MAX_MASKED) & present
        counts["tiles_over_max_masked"]["existing"] += int(over.sum())
        for scope in SCOPES:
            keep = select_teacher(det.boxes, label_boxes, free, tile_px, scope) & ~already
            kept = det.boxes[keep]
            over = (_tile_shares(existing | _burn(kept, shape), tile_px) > MAX_MASKED) & present
            counts["tiles_over_max_masked"][scope] += int(over.sum())
            rings = mosaic_px_to_4326_rings(kept, mosaic)
            features[scope] += [
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                    "properties": {"reason": "teacher", "score": round(float(s), 3)},
                }
                for ring, s in zip(rings, det.scores[keep], strict=True)
            ]
            counts[scope] += int(keep.sum())
            sides = (kept[:, 2:] - kept[:, :2]).astype(np.float64)
            counts["area_m2"][scope] += float((sides[:, 0] * sides[:, 1]).sum()) * mosaic.gsd_m**2
    return features, counts


def _write_ignore(src_dir: Path, dst_dir: Path, added: list[dict]) -> None:
    """``dst_dir/ignore.geojson`` = the source's features plus ``added``, as a new file (never via a link)."""
    src, dst = src_dir / "ignore.geojson", dst_dir / "ignore.geojson"
    collection = (
        json.loads(src.read_text()) if src.is_file() else {"type": "FeatureCollection", "features": []}
    )
    collection["features"] = collection.get("features", []) + added
    if dst.exists():
        dst.unlink()  # a hard link to the source: writing through it would change the source too
    dst.write_text(json.dumps(collection))


def write_teacher_sets(
    src_root: Path, out_root: Path, run_fn: RunFn, params: InferenceParams, *, zoom: int = 20
) -> dict:
    """Write one region tree per scope under ``out_root/<set name>`` (files hard-linked from ``src_root``),
    with the teacher features added to each train region's ignore; also writes ``teacher.json`` per set."""
    src_root, out_root = Path(src_root), Path(out_root)
    regions = load_regions(src_root)
    for name in SCOPES.values():
        if (out_root / name).exists():
            raise FileExistsError(f"{out_root / name} exists; remove it to rebuild the set")
        for region in regions:
            src_dir = region.labels_path.parent
            shutil.copytree(src_dir, out_root / name / src_dir.name, copy_function=os.link)

    train = [r for r in regions if r.split == "train"]
    totals = _zero_counts()
    for i, region in enumerate(train, 1):
        features, counts = teacher_features(region, run_fn, params, zoom=zoom)
        for scope, name in SCOPES.items():
            src_dir = region.labels_path.parent
            _write_ignore(src_dir, out_root / name / src_dir.name, features[scope])
        _add_counts(totals, counts)
        logger.info(
            f"{region.id} ({i}/{len(train)}): {counts['teacher']} teacher boxes, {counts['all']} kept"
        )

    valid_m2 = sum(r.valid_area_m2 for r in train)
    stats = {
        "params": dataclasses.asdict(params),
        "zoom": zoom,
        "regions": len(train),
        "teacher": totals["teacher"],
        "on_labels": totals["on_labels"],
        "already_masked": totals["already_masked"],
        "tiles": totals["tiles"],
        "tiles_over_max_masked_before": totals["tiles_over_max_masked"]["existing"],
        "sets": {
            name: {
                "scope": scope,
                "kept": totals[scope],
                "kept_area_share": totals["area_m2"][scope] / valid_m2 if valid_m2 else 0.0,
                "tiles_over_max_masked": totals["tiles_over_max_masked"][scope],
            }
            for scope, name in SCOPES.items()
        },
    }
    for name, kept in stats["sets"].items():
        set_stats = {k: v for k, v in stats.items() if k != "sets"} | kept
        (out_root / name / "teacher.json").write_text(json.dumps(set_stats, indent=2) + "\n")
    return stats
