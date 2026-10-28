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

from fair_trees.dataset import Region, labels_to_boxes, load_geojson, load_regions
from fair_trees.georef import mosaic_px_to_4326_rings
from fair_trees.params import InferenceParams
from fair_trees.preprocess import mosaic_chips, read_chip
from fair_trees.serve import RunFn, predict_mosaic

logger = logging.getLogger(__name__)

SCOPES = {"all": "hot-teacher-all", "label_free": "hot-teacher-empty"}  # scope -> shard set name


def _tile_span(lo: np.ndarray, hi: np.ndarray, tile_px: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """First and last tile index each [lo, hi) interval overlaps with positive length, clipped to [0, n)."""
    first = np.clip(np.floor(lo / tile_px), 0, n - 1).astype(int)
    last = np.clip(np.ceil(hi / tile_px) - 1, 0, n - 1).astype(int)
    return first, last


def label_free_tiles(label_boxes: np.ndarray, valid: np.ndarray, tile_px: int) -> np.ndarray:
    """(rows, cols) bool over the mosaic's tile grid: a present tile that no label box overlaps."""
    rows, cols = valid.shape[0] // tile_px, valid.shape[1] // tile_px
    free = valid[: rows * tile_px, : cols * tile_px].reshape(rows, tile_px, cols, tile_px).any(axis=(1, 3))
    boxes = np.asarray(label_boxes, np.float64).reshape(-1, 4)
    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])]
    c0, c1 = _tile_span(boxes[:, 0], boxes[:, 2], tile_px, cols)
    r0, r1 = _tile_span(boxes[:, 1], boxes[:, 3], tile_px, rows)
    for a, b, c, d in zip(r0, r1, c0, c1, strict=True):
        free[a : b + 1, c : d + 1] = False
    return free


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


def teacher_features(
    region: Region, run_fn: RunFn, params: InferenceParams, *, zoom: int = 20
) -> tuple[dict[str, list[dict]], dict]:
    """Per scope, the ignore features for one region, and its counts (boxes and kept area in m2)."""
    z = zoom if zoom in region.chips else min(region.chips)
    tile_px = read_chip(region.chips[z][0], use_sidecar=False)[0].shape[1]
    labels = load_geojson(region.labels_path)
    features: dict[str, list[dict]] = {scope: [] for scope in SCOPES}
    counts = {"teacher": 0, "on_labels": 0, **dict.fromkeys(SCOPES, 0), "area_m2": dict.fromkeys(SCOPES, 0.0)}
    for mosaic in mosaic_chips(region.chips[z], use_sidecar=False):
        det = predict_mosaic(run_fn, mosaic, params)
        label_boxes = labels_to_boxes(labels, mosaic.transform_3857)
        free = label_free_tiles(label_boxes, mosaic.valid, tile_px)
        counts["teacher"] += len(det.boxes)
        counts["on_labels"] += int(_on_labels(det.boxes.astype(np.float64), label_boxes).sum())
        for scope in SCOPES:
            keep = select_teacher(det.boxes, label_boxes, free, tile_px, scope)
            kept = det.boxes[keep]
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
    totals = {"teacher": 0, "on_labels": 0, **dict.fromkeys(SCOPES, 0), "area_m2": dict.fromkeys(SCOPES, 0.0)}
    for i, region in enumerate(train, 1):
        features, counts = teacher_features(region, run_fn, params, zoom=zoom)
        for scope, name in SCOPES.items():
            src_dir = region.labels_path.parent
            _write_ignore(src_dir, out_root / name / src_dir.name, features[scope])
        for key in ("teacher", "on_labels", *SCOPES):
            totals[key] += counts[key]
        for scope in SCOPES:
            totals["area_m2"][scope] += counts["area_m2"][scope]
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
        "sets": {
            name: {
                "scope": scope,
                "kept": totals[scope],
                "kept_area_share": totals["area_m2"][scope] / valid_m2 if valid_m2 else 0.0,
            }
            for scope, name in SCOPES.items()
        },
    }
    for name, kept in stats["sets"].items():
        set_stats = {k: v for k, v in stats.items() if k != "sets"} | kept
        (out_root / name / "teacher.json").write_text(json.dumps(set_stats, indent=2) + "\n")
    return stats
