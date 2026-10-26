"""Score predicted tree boxes against HOT tree points and tree_group boxes against HOT group boxes."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fair_trees.evaluate import Counts, bootstrap_ci, count_matches, prf

NOT_CONTAINED = 1e6


def match_points(boxes: np.ndarray, points: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one (box, point) pairs, a point inside its box (edges inclusive).

    Maximises the number of pairs; among those, prefers points nearest the box centre.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(boxes) == 0 or len(points) == 0:
        return []
    from scipy.optimize import linear_sum_assignment

    x, y = points[None, :, 0], points[None, :, 1]
    inside = (
        (x >= boxes[:, None, 0])
        & (x <= boxes[:, None, 2])
        & (y >= boxes[:, None, 1])
        & (y <= boxes[:, None, 3])
    )
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    dist = np.linalg.norm(centres[:, None, :] - points[None, :, :], axis=2)
    diagonal = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)
    cost = np.where(inside, dist / (diagonal[:, None] + 1), NOT_CONTAINED)
    rows, cols = linear_sum_assignment(cost)
    return sorted((int(r), int(c)) for r, c in zip(rows, cols, strict=True) if inside[r, c])


def score_trees(boxes: np.ndarray, in_group: np.ndarray, points: np.ndarray) -> Counts:
    """Tree boxes against HOT points; boxes flagged ``in_group`` are dropped, not counted as FP."""
    kept = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)[~np.asarray(in_group, dtype=bool)]
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    tp = len(match_points(kept, points))
    return Counts(tp=tp, fp=len(kept) - tp, fn=len(points) - tp)


def score_groups(boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float) -> Counts:
    return count_matches(boxes, gt_boxes, iou_threshold)


@dataclass(frozen=True)
class UnitScore:
    unit_id: str
    dense: bool
    trees: Counts
    groups_40: Counts
    groups_25: Counts


def _pool(units: Sequence[Counts]) -> Counts:
    total = Counts(0, 0, 0)
    for c in units:
        total = total + c
    return total


def _summary(units: Sequence[Counts]) -> dict:
    precision, recall, f1 = prf(_pool(units))
    return {"precision": precision, "recall": recall, "f1": f1, "f1_ci": bootstrap_ci(units)}


def summarise(scores: Sequence[UnitScore]) -> dict:
    """Pooled precision/recall/F1 with a bootstrap CI over units, for all units and the dense subset."""
    out = {}
    for name, subset in (("all", list(scores)), ("dense", [s for s in scores if s.dense])):
        out[name] = {"units": len(subset)} | {
            key: _summary([getattr(s, key) for s in subset]) for key in ("trees", "groups_40", "groups_25")
        }
    return out
