"""Score predicted tree boxes against HOT tree points and tree_group boxes against HOT group boxes."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fair_trees.evaluate import Counts, bootstrap_ci, count_matches, prf

NOT_CONTAINED = 1e6


def match_points(boxes: np.ndarray, points: np.ndarray) -> list[tuple[int, int]]:
    """One-to-one (box, point) pairs, a point inside its box (edges inclusive).

    Maximises the number of pairs; among those, prefers points nearest the box centre. Each connected
    component of the containment graph is assigned on its own, so no box x point matrix is built.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    box_idx, point_idx = _containment(boxes, points)
    if len(box_idx) == 0:
        return []
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    centres = (boxes[box_idx, :2] + boxes[box_idx, 2:]) / 2
    dist = np.linalg.norm(centres - points[point_idx], axis=1)
    diagonal = np.linalg.norm(boxes[box_idx, 2:] - boxes[box_idx, :2], axis=1)
    edge_cost = dist / (diagonal + 1)

    # nodes: boxes 0..B-1, then points B..B+P-1
    n = len(boxes) + len(points)
    graph = coo_matrix((np.ones(len(box_idx)), (box_idx, len(boxes) + point_idx)), shape=(n, n))
    _, component = connected_components(graph, directed=False)
    order = np.argsort(component[box_idx], kind="stable")
    starts = np.flatnonzero(np.diff(component[box_idx][order])) + 1
    pairs = []
    for edges in np.split(order, starts):
        rows, r = np.unique(box_idx[edges], return_inverse=True)
        cols, c = np.unique(point_idx[edges], return_inverse=True)
        cost = np.full((len(rows), len(cols)), NOT_CONTAINED)
        cost[r, c] = edge_cost[edges]
        for i, j in zip(*linear_sum_assignment(cost), strict=True):
            if cost[i, j] < NOT_CONTAINED:
                pairs.append((int(rows[i]), int(cols[j])))
    return sorted(pairs)


def _containment(boxes: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every (box, point) index pair with the point inside the box, from points sorted by x."""
    order = np.argsort(points[:, 0], kind="stable")
    xs = points[order, 0]
    lo = np.searchsorted(xs, boxes[:, 0], side="left")
    counts = np.maximum(np.searchsorted(xs, boxes[:, 2], side="right") - lo, 0)
    box_idx = np.repeat(np.arange(len(boxes)), counts)
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    point_idx = order[np.repeat(lo, counts) + offsets]
    y = points[point_idx, 1]
    inside = (y >= boxes[box_idx, 1]) & (y <= boxes[box_idx, 3])
    return box_idx[inside], point_idx[inside]


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
