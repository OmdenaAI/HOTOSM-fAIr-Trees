"""Detection metrics: IoU, Hungarian matching, counts, P/R/F1, AP, AOI-F1, point recall.

numpy only at import time; scipy is imported lazily inside ``match_hungarian``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


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
