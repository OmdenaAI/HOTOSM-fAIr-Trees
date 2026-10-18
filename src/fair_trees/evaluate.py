"""Detection metrics (IoU, Hungarian matching, counts, P/R/F1, AP, AOI-F1, point recall) and unit evaluation.

numpy only at import time; scipy is imported lazily inside ``match_hungarian``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from fair_trees.decode import Detections, filter_score
from fair_trees.params import InferenceParams
from fair_trees.preprocess import Mosaic
from fair_trees.serve import RunFn, predict_mosaic


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes, float64 of shape (N, M); zero union gives 0."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def match_hungarian(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.4) -> list[tuple[int, int]]:
    """Optimal one-to-one (pred, gt) pairs maximising IoU, kept iff IoU > threshold."""
    if len(pred) == 0 or len(gt) == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    iou = iou_matrix(pred, gt)
    rows, cols = linear_sum_assignment(iou, maximize=True)
    pairs = [(int(r), int(c)) for r, c in zip(rows, cols, strict=True) if iou[r, c] > iou_threshold]
    return sorted(pairs)


@dataclass(frozen=True)
class Counts:
    tp: int
    fp: int
    fn: int

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


def prf(c: Counts) -> tuple[float, float, float]:
    """Precision, recall, F1; any zero denominator gives 0.0."""
    p = c.tp / (c.tp + c.fp) if c.tp + c.fp > 0 else 0.0
    r = c.tp / (c.tp + c.fn) if c.tp + c.fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


def count_matches(pred: np.ndarray, gt: np.ndarray, iou_threshold: float = 0.4) -> Counts:
    tp = len(match_hungarian(pred, gt, iou_threshold))
    return Counts(tp=tp, fp=len(pred) - tp, fn=len(gt) - tp)


def average_precision(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    """Area under the precision envelope, integrated over recall."""
    if len(recalls) == 0:
        return 0.0
    order = np.argsort(np.asarray(recalls, dtype=np.float64), kind="stable")
    r = np.asarray(recalls, dtype=np.float64)[order]
    p = np.asarray(precisions, dtype=np.float64)[order]
    r = np.concatenate([[0.0], r])
    p = np.concatenate([[p[0]], p])
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def aoi_f1(units: Sequence[Counts]) -> float:
    """Mean per-unit F1 weighted by ground-truth count (tp + fn)."""
    total = sum(u.tp + u.fn for u in units)
    if total == 0:
        return 0.0
    return sum((u.tp + u.fn) * prf(u)[2] for u in units) / total


def point_in_box_recall(points: np.ndarray, boxes: np.ndarray) -> float:
    """Share of (x, y) points inside any xyxy box, edges inclusive."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return 0.0
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x, y = points[:, None, 0], points[:, None, 1]
    inside = (
        (x >= boxes[None, :, 0])
        & (x <= boxes[None, :, 2])
        & (y >= boxes[None, :, 1])
        & (y <= boxes[None, :, 3])
    )
    return float(inside.any(axis=1).mean())


SWEEP_THRESHOLDS = np.round(np.linspace(0.05, 0.95, 19), 2)
AP_THRESHOLDS = np.linspace(0.01, 1.0, 101)
DECODE_THRESHOLD = 0.01


@dataclass
class EvalUnit:
    """One evaluation tile: a mosaic, its ground-truth boxes (mosaic pixels) and optional crowd mask."""

    unit_id: str
    mosaic: Mosaic
    gt_boxes: np.ndarray
    ignore_mask: np.ndarray | None = None
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass
class EvalReport:
    confidence_threshold: float
    iou_threshold: float
    precision: float
    recall: float
    f1: float
    aoi_f1: float
    ap40: float
    f1_max: float
    conf_at_max: float
    sweep: list[dict]
    units: list[dict]
    breakdown: dict

    def to_dict(self) -> dict:
        """JSON-native view: python scalars, lists and str-keyed dicts only."""
        return {
            "confidence_threshold": float(self.confidence_threshold),
            "iou_threshold": float(self.iou_threshold),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1": float(self.f1),
            "aoi_f1": float(self.aoi_f1),
            "ap40": float(self.ap40),
            "f1_max": float(self.f1_max),
            "conf_at_max": float(self.conf_at_max),
            "sweep": [dict(s) for s in self.sweep],
            "units": [{**u, "meta": dict(u["meta"])} for u in self.units],
            "breakdown": {k: {v: dict(m) for v, m in vs.items()} for k, vs in self.breakdown.items()},
        }

    def unit_counts(self) -> list[Counts]:
        return [Counts(u["tp"], u["fp"], u["fn"]) for u in self.units]


def _drop_ignored(dets: Detections, mask: np.ndarray) -> Detections:
    """Drop detections whose centre pixel lies inside the crowd mask."""
    if len(dets.boxes) == 0:
        return dets
    h, w = mask.shape
    cx = np.clip(np.floor((dets.boxes[:, 0] + dets.boxes[:, 2]) / 2).astype(int), 0, w - 1)
    cy = np.clip(np.floor((dets.boxes[:, 1] + dets.boxes[:, 3]) / 2).astype(int), 0, h - 1)
    keep = ~mask[cy, cx]
    return Detections(dets.boxes[keep], dets.scores[keep])


def _pooled(
    decoded: Sequence[Detections], units: Sequence[EvalUnit], t: float, iou_threshold: float
) -> list[Counts]:
    return [
        count_matches(filter_score(d, t).boxes, u.gt_boxes, iou_threshold)
        for d, u in zip(decoded, units, strict=True)
    ]


def _sum(counts: Sequence[Counts]) -> Counts:
    total = Counts(0, 0, 0)
    for c in counts:
        total = total + c
    return total


def evaluate_units(
    run_fn: RunFn,
    units: Sequence[EvalUnit],
    params: InferenceParams,
    *,
    thresholds: Sequence[float] = SWEEP_THRESHOLDS,
    iou_threshold: float = 0.4,
    ignore: bool = True,
    decode_threshold: float = DECODE_THRESHOLD,
) -> EvalReport:
    """Decode each unit once at a low threshold, then score at the operating point and a sweep."""
    low_params = dataclasses.replace(params, confidence_threshold=decode_threshold)
    decoded = []
    for u in units:
        dets = predict_mosaic(run_fn, u.mosaic, low_params)
        if ignore and u.ignore_mask is not None:
            dets = _drop_ignored(dets, np.asarray(u.ignore_mask, dtype=bool))
        decoded.append(dets)

    t0 = params.confidence_threshold
    per_unit = _pooled(decoded, units, t0, iou_threshold)
    p, r, f1 = prf(_sum(per_unit))
    unit_rows = [
        {"unit_id": u.unit_id, "meta": dict(u.meta), "tp": c.tp, "fp": c.fp, "fn": c.fn}
        for u, c in zip(units, per_unit, strict=True)
    ]

    groups: dict[str, dict[str, list[Counts]]] = {}
    for u, c in zip(units, per_unit, strict=True):
        for key, value in u.meta.items():
            groups.setdefault(key, {}).setdefault(value, []).append(c)
    breakdown = {}
    for key, by_value in groups.items():
        breakdown[key] = {}
        for value, cs in by_value.items():
            bp, br, bf = prf(_sum(cs))
            breakdown[key][value] = {"precision": bp, "recall": br, "f1": bf, "n_units": len(cs)}

    sweep = []
    for t in thresholds:
        sp, sr, sf = prf(_sum(_pooled(decoded, units, float(t), iou_threshold)))
        sweep.append({"threshold": float(t), "precision": sp, "recall": sr, "f1": sf})
    if sweep:
        f1_max = max(s["f1"] for s in sweep)
        conf_at_max = min(s["threshold"] for s in sweep if s["f1"] == f1_max)
    else:
        f1_max, conf_at_max = 0.0, 0.0

    curve = [prf(_sum(_pooled(decoded, units, float(t), iou_threshold))) for t in AP_THRESHOLDS]
    ap40 = average_precision([c[1] for c in curve], [c[0] for c in curve]) if units else 0.0

    return EvalReport(
        confidence_threshold=float(t0),
        iou_threshold=float(iou_threshold),
        precision=p,
        recall=r,
        f1=f1,
        aoi_f1=aoi_f1(per_unit),
        ap40=ap40,
        f1_max=f1_max,
        conf_at_max=conf_at_max if sweep else (float(thresholds[0]) if len(thresholds) else 0.0),
        sweep=sweep,
        units=unit_rows,
        breakdown=breakdown,
    )
