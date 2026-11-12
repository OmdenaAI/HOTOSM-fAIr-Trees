"""Merge touching crown boxes into tree boxes and tree_group boxes (numpy only)."""

from dataclasses import dataclass

import numpy as np

from fair_trees.decode import Detections

_GAP_TOLERANCE = 1e-9  # absorbs float rounding, e.g. 2.6 - 2 > 0.6


@dataclass(frozen=True, eq=False)
class Merged:
    """Crowns kept as trees (with their input index) and one box per large cluster."""

    trees: Detections
    tree_index: np.ndarray
    groups: Detections
    group_sizes: np.ndarray


def _find(parent: np.ndarray, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]  # path halving
        i = parent[i]
    return i


def cluster_boxes(boxes: np.ndarray, gap: float) -> np.ndarray:
    """Cluster id per box; two boxes link when their x and y gaps are both <= gap, transitively.

    Ids run 0..K-1 in order of each cluster's first box.
    """
    n = len(boxes)
    gap += _GAP_TOLERANCE
    order = np.argsort(boxes[:, 0], kind="stable")
    b = np.asarray(boxes, np.float64)[order]
    ends = np.searchsorted(b[:, 0], b[:, 2] + gap, side="right")
    parent = np.arange(n)
    for i in range(n):
        rest = b[i + 1 : ends[i]]
        if not len(rest):
            continue
        near_x = rest[:, 0] - np.minimum(b[i, 2], rest[:, 2]) <= gap
        near_y = np.maximum(b[i, 1], rest[:, 1]) - np.minimum(b[i, 3], rest[:, 3]) <= gap
        for j in np.flatnonzero(near_x & near_y) + i + 1:
            ri, rj = _find(parent, i), _find(parent, j)
            if ri != rj:
                parent[rj] = ri
    roots = np.empty(n, np.int64)
    for k in range(n):
        roots[order[k]] = _find(parent, k)
    # np.unique numbers roots by value; reorder to first appearance in input order
    _, first, inverse = np.unique(roots, return_index=True, return_inverse=True)
    rank = np.empty(len(first), np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[inverse.reshape(-1)]


def apply_clusters(dets: Detections, labels: np.ndarray, min_crowns: int) -> Merged:
    """Clusters of at least min_crowns boxes become one group box; every other box stays a tree."""
    sizes = np.bincount(labels).astype(np.int64)
    is_group = sizes >= min_crowns
    group_ids = np.flatnonzero(is_group)
    tree_index = np.flatnonzero(~is_group[labels]).astype(np.int64)
    boxes = np.zeros((len(group_ids), 4), np.float32)
    scores = np.zeros(len(group_ids), np.float32)
    for g, cid in enumerate(group_ids):
        members = dets.boxes[labels == cid]
        boxes[g] = [*members[:, :2].min(axis=0), *members[:, 2:].max(axis=0)]
        scores[g] = dets.scores[labels == cid].max()
    trees = Detections(dets.boxes[tree_index], dets.scores[tree_index])
    return Merged(trees, tree_index, Detections(boxes, scores), sizes[group_ids])


def merge_crowns(dets: Detections, gap: float, min_crowns: int) -> Merged:
    return apply_clusters(dets, cluster_boxes(dets.boxes, gap), min_crowns)
