"""Run the release model once per HOT unit, cache its detections, sweep merge settings on val, score test."""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.label_study.hot import UnitTruth
from experiments.label_study.merge import apply_clusters, cluster_boxes
from experiments.label_study.score import UnitScore, match_points, score_groups, score_trees, summarise
from fair_trees.decode import Detections, filter_score
from fair_trees.evaluate import Counts, prf
from fair_trees.preprocess import Mosaic

GRID = {
    "score": (0.1, 0.2, 0.3, 0.4, 0.5),
    "gap_m": (0.0, 0.5, 1.0, 2.0),
    "min_crowns": (2, 3, 4, 5, 6, 8),
}
DENSE_COVER = 0.3


@dataclass(frozen=True, eq=False)
class UnitRecord:
    """One unit's cached detections and ground truth, enough to score any merge setting."""

    unit_id: str
    split: str
    dense: bool
    gsd_m: float
    boxes: np.ndarray
    scores: np.ndarray
    in_group: np.ndarray
    tree_points: np.ndarray
    tree_diam_px: np.ndarray
    group_boxes: np.ndarray


def make_record(unit_id: str, split: str, mosaic: Mosaic, truth: UnitTruth, dets: Detections) -> UnitRecord:
    mask = truth.group_mask
    centres = (dets.boxes[:, :2] + dets.boxes[:, 2:]) / 2
    cols = np.clip(centres[:, 0].astype(np.int64), 0, mask.shape[1] - 1)
    rows = np.clip(centres[:, 1].astype(np.int64), 0, mask.shape[0] - 1)
    return UnitRecord(
        unit_id=unit_id,
        split=split,
        dense=bool(truth.group_cover >= DENSE_COVER),
        gsd_m=float(mosaic.gsd_m),
        boxes=dets.boxes,
        scores=dets.scores,
        in_group=mask[rows, cols],
        tree_points=truth.tree_points,
        tree_diam_px=truth.tree_diam_px,
        group_boxes=truth.group_boxes,
    )


def save_record(rec: UnitRecord, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / f"{rec.unit_id}.npz"
    tmp = cache_dir / f"{rec.unit_id}.npz.tmp"
    with open(tmp, "wb") as f:  # a file handle keeps np.savez from appending ".npz"
        np.savez(
            f,
            unit_id=rec.unit_id,
            split=rec.split,
            dense=rec.dense,
            gsd_m=rec.gsd_m,
            boxes=rec.boxes,
            scores=rec.scores,
            in_group=rec.in_group,
            tree_points=rec.tree_points,
            tree_diam_px=rec.tree_diam_px,
            group_boxes=rec.group_boxes,
        )
    os.replace(tmp, final)
    return final


def _load_record(path: Path) -> UnitRecord:
    with np.load(path) as z:
        return UnitRecord(
            unit_id=str(z["unit_id"]),
            split=str(z["split"]),
            dense=bool(z["dense"]),
            gsd_m=float(z["gsd_m"]),
            boxes=z["boxes"],
            scores=z["scores"],
            in_group=z["in_group"],
            tree_points=z["tree_points"],
            tree_diam_px=z["tree_diam_px"],
            group_boxes=z["group_boxes"],
        )


def load_records(cache_dir: Path, split: str) -> list[UnitRecord]:
    records = [_load_record(p) for p in Path(cache_dir).glob("*.npz")]
    return sorted((r for r in records if r.split == split), key=lambda r: r.unit_id)


def run_units(
    predict: Callable[[Mosaic], Detections],
    units: Iterable[tuple[str, str, Callable[[], tuple[Mosaic, UnitTruth]]]],
    cache_dir: Path,
) -> int:
    """Predict and cache every unit without a finished record; returns how many were predicted now."""
    done = 0
    for unit_id, split, build in units:
        if (cache_dir / f"{unit_id}.npz").exists():
            continue
        mosaic, truth = build()
        save_record(make_record(unit_id, split, mosaic, truth, predict(mosaic)), cache_dir)
        done += 1
    return done


def _kept(rec: UnitRecord, score: float) -> tuple[Detections, np.ndarray]:
    """Detections above the score threshold, with their in_group flags."""
    keep = rec.scores > score
    return filter_score(Detections(rec.boxes, rec.scores), score), rec.in_group[keep]


def _score_clustered(rec: UnitRecord, dets: Detections, in_group, labels, min_crowns: int) -> UnitScore:
    merged = apply_clusters(dets, labels, min_crowns)
    return UnitScore(
        unit_id=rec.unit_id,
        dense=rec.dense,
        trees=score_trees(merged.trees.boxes, in_group[merged.tree_index], rec.tree_points),
        groups_40=score_groups(merged.groups.boxes, rec.group_boxes, 0.4),
        groups_25=score_groups(merged.groups.boxes, rec.group_boxes, 0.25),
    )


def score_record(rec: UnitRecord, score: float, gap_m: float, min_crowns: int) -> UnitScore:
    dets, in_group = _kept(rec, score)
    labels = cluster_boxes(dets.boxes, gap_m / rec.gsd_m)
    return _score_clustered(rec, dets, in_group, labels, min_crowns)


def _pooled(counts: Sequence[Counts]) -> Counts:
    total = Counts(0, 0, 0)
    for c in counts:
        total = total + c
    return total


def sweep(records: Sequence[UnitRecord], grid: Mapping = GRID) -> pd.DataFrame:
    """Pooled tree and group scores for every (score, gap_m, min_crowns) setting."""
    rows = []
    for score in grid["score"]:
        kept = [_kept(r, score) for r in records]
        for gap_m in grid["gap_m"]:
            labels = [
                cluster_boxes(d.boxes, gap_m / r.gsd_m) for r, (d, _) in zip(records, kept, strict=True)
            ]
            for min_crowns in grid["min_crowns"]:
                scores = [
                    _score_clustered(r, d, g, lab, min_crowns)
                    for r, (d, g), lab in zip(records, kept, labels, strict=True)
                ]
                tree_p, tree_r, tree_f1 = prf(_pooled([s.trees for s in scores]))
                group_p, group_r, group_f1 = prf(_pooled([s.groups_40 for s in scores]))
                _, _, group25_f1 = prf(_pooled([s.groups_25 for s in scores]))
                rows.append(
                    {
                        "score": score,
                        "gap_m": gap_m,
                        "min_crowns": min_crowns,
                        "tree_precision": tree_p,
                        "tree_recall": tree_r,
                        "tree_f1": tree_f1,
                        "group_precision": group_p,
                        "group_recall": group_r,
                        "group_f1": group_f1,
                        "group25_f1": group25_f1,
                        "mean_f1": (tree_f1 + group_f1) / 2,
                    }
                )
    return pd.DataFrame(rows)


def choose(table: pd.DataFrame) -> dict:
    """Highest mean F1; ties go to larger min_crowns, then smaller gap_m, then higher score."""
    best = table.sort_values(
        ["mean_f1", "min_crowns", "gap_m", "score"], ascending=[False, False, True, False], kind="stable"
    ).iloc[0]
    return {
        "score": float(best["score"]),
        "gap_m": float(best["gap_m"]),
        "min_crowns": int(best["min_crowns"]),
    }


def _diameter_ratio(records: Sequence[UnitRecord], setting: Mapping) -> dict:
    """Predicted tree box side sqrt(w*h) over HOT crown diameter, for matched points with a diameter."""
    ratios = []
    for rec in records:
        dets, in_group = _kept(rec, setting["score"])
        labels = cluster_boxes(dets.boxes, setting["gap_m"] / rec.gsd_m)
        merged = apply_clusters(dets, labels, setting["min_crowns"])
        boxes = merged.trees.boxes[~in_group[merged.tree_index]].astype(np.float64)
        side = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
        for i, j in match_points(boxes, rec.tree_points):
            if not np.isnan(rec.tree_diam_px[j]):
                ratios.append(side[i] / rec.tree_diam_px[j])
    q25, median, q75 = np.percentile(ratios, [25, 50, 75]) if ratios else (np.nan,) * 3
    return {
        "pairs": len(ratios),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "points_without_diameter": int(sum(np.isnan(r.tree_diam_px).sum() for r in records)),
    }


def score_test(records: Sequence[UnitRecord], setting: Mapping) -> dict:
    scores = [score_record(r, setting["score"], setting["gap_m"], setting["min_crowns"]) for r in records]
    return summarise(scores) | {"setting": dict(setting), "diameter_ratio": _diameter_ratio(records, setting)}
