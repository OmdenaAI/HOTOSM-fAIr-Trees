"""Detection metrics (IoU, Hungarian matching, counts, P/R/F1, AP, AOI-F1, point recall) and unit evaluation.

numpy only at import time; scipy is imported lazily inside ``match_hungarian``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from fair_trees.dataset import (
    MIN_BOX_FRACTION,
    Region,
    burn_boxes,
    labels_to_boxes,
    load_geojson,
    rasterize_labels,
    rasterize_mask,
    window_boxes,
)
from fair_trees.decode import Detections, filter_group_parts, filter_score, merge_group_detections
from fair_trees.params import CLASSES, InferenceParams
from fair_trees.preprocess import Mosaic, mosaic_chips
from fair_trees.serve import RunFn, predict_group_parts, predict_mosaic_classes


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
PICK_THRESHOLDS = np.round(np.linspace(0.1, 0.3, 9), 3)  # where a fine-tune picks its operating point
MIN_PICK_GT = 200  # fewer val boxes of a class than this: the prior, not a pick
PRIOR_THRESHOLD = {1: 0.15, 2: 0.2}  # by the model's class count: where HOT val F1 peaked for each


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
    pick_sweep: list[dict] = field(default_factory=list)  # P/R/F1 at each of PICK_THRESHOLDS

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
            "units": [{**u, "meta": {str(k): str(v) for k, v in u["meta"].items()}} for u in self.units],
            "breakdown": {
                str(k): {str(v): dict(m) for v, m in vs.items()} for k, vs in self.breakdown.items()
            },
            "pick_sweep": [dict(s) for s in self.pick_sweep],
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
    class_index: int = 0,
) -> EvalReport:
    """Decode each unit's class ``class_index`` once at a low threshold, then score at the operating point
    and a sweep.

    Trees are filtered after the cross-window merge (greedy NMS keeps the same boxes either way). Groups are
    filtered per window before their cross-window join, as serving filters them, since a low-score cut box
    would otherwise widen a confident one.
    """
    low_params = dataclasses.replace(
        params, confidence_threshold=min(decode_threshold, params.confidence_threshold), class_thresholds=()
    )
    masks = [
        np.asarray(u.ignore_mask, dtype=bool) if ignore and u.ignore_mask is not None else None for u in units
    ]
    if class_index == 1:
        parts = [predict_group_parts(run_fn, u.mosaic, low_params) for u in units]

        def at(i: int, t: float) -> Detections:
            dets = merge_group_detections(
                filter_group_parts(parts[i], t), params.iou_threshold, params.group_merge
            )
            return dets if masks[i] is None else _drop_ignored(dets, masks[i])
    else:
        decoded = []
        for u, mask in zip(units, masks, strict=True):
            dets = predict_mosaic_classes(run_fn, u.mosaic, low_params, (class_index,))[0]
            decoded.append(dets if mask is None else _drop_ignored(dets, mask))

        def at(i: int, t: float) -> Detections:
            return filter_score(decoded[i], t)

    def _pooled(t: float) -> list[Counts]:
        return [count_matches(at(i, t).boxes, u.gt_boxes, iou_threshold) for i, u in enumerate(units)]

    t0 = params.confidence_threshold
    per_unit = _pooled(t0)
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

    def _row(t: float) -> dict:
        sp, sr, sf = prf(_sum(_pooled(t)))
        return {"threshold": t, "precision": sp, "recall": sr, "f1": sf}

    sweep = [_row(float(t)) for t in thresholds]
    pick_sweep = [_row(float(t)) for t in PICK_THRESHOLDS]
    if sweep:
        f1_max = max(s["f1"] for s in sweep)
        conf_at_max = min(s["threshold"] for s in sweep if s["f1"] == f1_max)
    else:
        f1_max, conf_at_max = 0.0, 0.0

    curve = [prf(_sum(_pooled(float(t)))) for t in AP_THRESHOLDS]
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
        pick_sweep=pick_sweep,
    )


def pick_threshold(report: EvalReport, prior: float, *, min_gt: int = MIN_PICK_GT) -> float:
    """The operating point a fine-tune serves at: the lowest of ``PICK_THRESHOLDS`` with the best pooled F1,
    or ``prior`` when the report holds fewer than ``min_gt`` ground-truth boxes (a small val set's best
    threshold is mostly noise)."""
    n_gt = sum(u["tp"] + u["fn"] for u in report.units)
    if n_gt < min_gt or not report.pick_sweep:
        return float(prior)
    best = max(s["f1"] for s in report.pick_sweep)
    return min(s["threshold"] for s in report.pick_sweep if s["f1"] == best)


def f1_at(report: EvalReport, threshold: float) -> dict:
    """The pick-sweep row (threshold, precision, recall, f1) at ``threshold``; KeyError off the grid."""
    for s in report.pick_sweep:
        if abs(s["threshold"] - threshold) < 1e-9:
            return s
    raise KeyError(f"{threshold} is not one of the pick thresholds {list(PICK_THRESHOLDS)}")


def region_units(
    regions: Sequence[Region], *, zooms: Sequence[int] = (20, 19), label: int = 1
) -> list[EvalUnit]:
    """One unit per mosaic of each region labelling class ``label`` (1 tree, 2 tree_group), at the first of
    ``zooms`` the region has (else its lowest).

    Ignore polygons only mask predictions; labels are dropped only where the footprint or chip validity
    masks them. In a region labelling tree_group, tree predictions inside a group polygon are ignored, and
    so are group predictions inside a group the 40 % rule drops at the unit's edge.
    """
    name = CLASSES[label - 1]
    units = []
    for region in regions:
        if name not in region.classes:
            continue
        z = next((z for z in zooms if z in region.chips), None)
        if z is None:
            z = min(region.chips)
        mosaics = mosaic_chips(region.chips[z], use_sidecar=False)
        labels = load_geojson(region.labels_path)
        has_mask = region.footprint is not None or region.ignore_path is not None
        groups = "tree_group" in region.classes
        for k, m in enumerate(mosaics):
            h, w = m.rgb.shape[:2]
            boxes = labels_to_boxes(labels, m.transform_3857, label)
            unmasked = m.valid & ~rasterize_mask(region, m.transform_3857, (h, w), ignore=False)
            clipped, kept = window_boxes(boxes, unmasked, 0, 0, w, h, MIN_BOX_FRACTION)
            mask = rasterize_mask(region, m.transform_3857, (h, w)) if has_mask else None
            if groups:
                extra = (
                    rasterize_labels(region, m.transform_3857, (h, w), 2)
                    if label == 1
                    else burn_boxes(clipped[~kept], (h, w))
                )
                mask = extra if mask is None else mask | extra
            units.append(
                EvalUnit(
                    unit_id=region.id if len(mosaics) == 1 else f"{region.id}#{k}",
                    mosaic=m,
                    gt_boxes=clipped[kept],
                    ignore_mask=mask,
                    meta={"source": region.source, "biome": region.biome, "zoom": str(z)},
                )
            )
    return units


def evaluate_regions(
    run_fn: RunFn,
    regions: Sequence[Region],
    params: InferenceParams,
    *,
    zooms: Sequence[int] = (20, 19),
    thresholds: np.ndarray = SWEEP_THRESHOLDS,
    iou_threshold: float = 0.4,
    ignore: bool = True,
    label: int = 1,
) -> EvalReport:
    """Evaluate class ``label`` on region mosaics with crowd ignore; see ``region_units`` for the units."""
    return evaluate_units(
        run_fn,
        region_units(regions, zooms=zooms, label=label),
        params,
        thresholds=thresholds,
        iou_threshold=iou_threshold,
        ignore=ignore,
        class_index=label - 1,
    )


def _stack(units: Sequence[Counts]) -> np.ndarray:
    return np.array([(c.tp, c.fp, c.fn) for c in units], dtype=np.int64).reshape(-1, 3)


def _f1_from_sums(sums: np.ndarray) -> np.ndarray:
    """Vectorised F1 of (n, 3) tp/fp/fn sums; zero denominators give 0, as in ``prf``."""
    tp = sums[:, 0].astype(np.float64)
    denom = 2.0 * tp + sums[:, 1] + sums[:, 2]
    return np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)


def _resample_sums(stacked: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return stacked[idx].sum(axis=1)


def _quantile_bounds(values: np.ndarray, level: float) -> tuple[float, float]:
    lo, hi = np.quantile(values, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def bootstrap_ci(
    units: Sequence[Counts], *, n: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile CI of pooled F1 from resampling units with replacement (seeded)."""
    if len(units) == 0:
        return 0.0, 0.0
    stacked = _stack(units)
    idx = np.random.default_rng(seed).integers(0, len(units), size=(n, len(units)))
    return _quantile_bounds(_f1_from_sums(_resample_sums(stacked, idx)), level)


def paired_bootstrap_diff(
    a: Sequence[Counts], b: Sequence[Counts], *, n: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """(point, lo, hi) for pooled F1(a) - F1(b), resampling the same units for both sides."""
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs the same units, got {len(a)} and {len(b)}")
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    sa, sb = _stack(a), _stack(b)
    point = float(
        _f1_from_sums(sa.sum(axis=0, keepdims=True))[0] - _f1_from_sums(sb.sum(axis=0, keepdims=True))[0]
    )
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    diffs = _f1_from_sums(_resample_sums(sa, idx)) - _f1_from_sums(_resample_sums(sb, idx))
    lo, hi = _quantile_bounds(diffs, level)
    return point, lo, hi
