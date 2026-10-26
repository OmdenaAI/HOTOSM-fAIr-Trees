"""Score predicted tree boxes against HOT tree points and tree_group boxes against HOT group boxes."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fair_trees.evaluate import Counts, bootstrap_ci, match_hungarian, prf

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


def count_in(
    pairs: Sequence[tuple[int, int]], n_pred: int, n_gt: int, pred_in: np.ndarray, gt_in: np.ndarray
) -> Counts:
    """Counts in a region: TP are pairs whose ground truth is in it, FN and FP its unmatched items.

    ``pred_in`` and ``gt_in`` are (n_pred,) and (n_gt,) bool; an item left out of every region it could
    count in (a dropped prediction, an ignored box) is simply never ``True``.
    """
    matched_pred, matched_gt = np.zeros(n_pred, bool), np.zeros(n_gt, bool)
    for i, j in pairs:
        matched_pred[i], matched_gt[j] = True, True
    return Counts(
        tp=int((matched_gt & gt_in).sum()),
        fp=int((~matched_pred & pred_in).sum()),
        fn=int((~matched_gt & gt_in).sum()),
    )


@dataclass(frozen=True, eq=False)
class Matching:
    """One matching of predictions to ground truth, and which items count (are not dropped or ignored)."""

    pairs: list[tuple[int, int]]
    pred_counted: np.ndarray  # (N,) bool
    gt_counted: np.ndarray  # (M,) bool

    def counts(self, pred_in: np.ndarray | bool = True, gt_in: np.ndarray | bool = True) -> Counts:
        """Counts over the counted items, restricted to a region when ``pred_in`` / ``gt_in`` are masks."""
        return count_in(
            self.pairs,
            len(self.pred_counted),
            len(self.gt_counted),
            self.pred_counted & pred_in,
            self.gt_counted & gt_in,
        )


def centres(boxes: np.ndarray) -> np.ndarray:
    """(N, 2) centres of xyxy boxes."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return (boxes[:, :2] + boxes[:, 2:]) / 2


def match_trees(boxes: np.ndarray, in_group: np.ndarray, points: np.ndarray) -> Matching:
    """Tree boxes against HOT points; boxes flagged ``in_group`` are dropped, not counted as FP."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    in_group = np.asarray(in_group, dtype=bool)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    kept = np.flatnonzero(~in_group)
    pairs = [(int(kept[i]), j) for i, j in match_points(boxes[kept], points)]
    return Matching(pairs, ~in_group, np.ones(len(points), bool))


def match_groups(
    boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float, ignore: np.ndarray | None = None
) -> Matching:
    """Group boxes against the ground-truth boxes not ignored (Hungarian, IoU > threshold).

    An unmatched box with its centre inside an ignored box is dropped; ignored boxes are never missed.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    ignore = np.zeros(len(gt_boxes), bool) if ignore is None else np.asarray(ignore, dtype=bool)
    scored = np.flatnonzero(~ignore)
    pairs = [(i, int(scored[j])) for i, j in match_hungarian(boxes, gt_boxes[scored], iou_threshold)]
    matched = np.zeros(len(boxes), bool)
    matched[[i for i, _ in pairs]] = True
    x, y = centres(boxes).T[:, :, None]  # (N, 1) each, against the (1, K) sides of the ignored boxes
    x0, y0, x1, y1 = gt_boxes[ignore].T[:, None, :]
    on_ignored = ((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)).any(axis=1)
    return Matching(pairs, matched | ~on_ignored, ~ignore)


def score_trees(boxes: np.ndarray, in_group: np.ndarray, points: np.ndarray) -> Counts:
    return match_trees(boxes, in_group, points).counts()


def score_groups(
    boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float, ignore: np.ndarray | None = None
) -> Counts:
    return match_groups(boxes, gt_boxes, iou_threshold, ignore).counts()


@dataclass(frozen=True)
class UnitScore:
    """One unit's counts: over the whole unit, and over its dense tiles (empty when it has none)."""

    unit_id: str
    trees: Counts
    groups_40: Counts  # edge slivers ignored, as groups_25
    groups_25: Counts
    groups_40_all: Counts  # every ground-truth group box counted
    dense_tiles: int
    dense_trees: Counts
    dense_groups_40: Counts


def _pool(units: Sequence[Counts]) -> Counts:
    total = Counts(0, 0, 0)
    for c in units:
        total = total + c
    return total


def _summary(units: Sequence[Counts]) -> dict:
    """Pooled P/R/F1 with a bootstrap CI over units; None for each when there are no units."""
    if not units:
        return dict.fromkeys(("precision", "recall", "f1", "f1_ci"))
    precision, recall, f1 = prf(_pool(units))
    return {"precision": precision, "recall": recall, "f1": f1, "f1_ci": bootstrap_ci(units)}


def summarise(scores: Sequence[UnitScore]) -> dict:
    """Pooled scores for all units, and for the dense tiles of the units that have any."""
    whole = ("trees", "groups_40", "groups_25", "groups_40_all")
    dense = [s for s in scores if s.dense_tiles]
    return {
        "all": {"units": len(scores)} | {key: _summary([getattr(s, key) for s in scores]) for key in whole},
        "dense": {
            "units": len(dense),
            "tiles": sum(s.dense_tiles for s in dense),
            "trees": _summary([s.dense_trees for s in dense]),
            "groups_40": _summary([s.dense_groups_40 for s in dense]),
        },
    }
