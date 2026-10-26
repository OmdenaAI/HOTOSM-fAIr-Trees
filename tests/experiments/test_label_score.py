import numpy as np
import pytest

from experiments.label_study.score import UnitScore, match_points, score_groups, score_trees, summarise
from fair_trees.evaluate import Counts


def test_one_box_cannot_claim_two_points():
    pairs = match_points(np.array([[0, 0, 10, 10]], float), np.array([[2, 2], [5, 5]], float))
    assert pairs == [(0, 1)]  # the point nearest the centre


def test_matching_maximises_pairs():
    boxes = np.array([[0, 0, 10, 10], [4, 4, 6, 6]], float)
    assert sorted(match_points(boxes, np.array([[5, 5], [1, 1]], float))) == [(0, 1), (1, 0)]


def brute_force_match(boxes, points):
    """match_points on the full box x point cost matrix."""
    from scipy.optimize import linear_sum_assignment

    x, y = points[None, :, 0], points[None, :, 1]
    inside = (x >= boxes[:, None, 0]) & (x <= boxes[:, None, 2]) & (y >= boxes[:, None, 1])
    inside &= y <= boxes[:, None, 3]
    centres = (boxes[:, :2] + boxes[:, 2:]) / 2
    dist = np.linalg.norm(centres[:, None, :] - points[None, :, :], axis=2)
    diagonal = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)
    rows, cols = linear_sum_assignment(np.where(inside, dist / (diagonal[:, None] + 1), 1e6))
    return sorted((int(r), int(c)) for r, c in zip(rows, cols, strict=True) if inside[r, c])


def test_matching_equals_the_full_cost_matrix():
    rng = np.random.default_rng(0)
    corner = rng.uniform(0, 300, (200, 2))
    boxes = np.hstack([corner, corner + rng.uniform(5, 30, (200, 2))])
    points = rng.uniform(0, 330, (200, 2))
    pairs = match_points(boxes, points)
    assert len(pairs) > 50 and pairs == brute_force_match(boxes, points)


def test_boxes_in_groups_are_ignored_not_false_positives():
    boxes = np.array([[0, 0, 4, 4], [10, 10, 14, 14]], float)
    c = score_trees(boxes, np.array([False, True]), np.array([[2, 2]], float))
    assert c == Counts(tp=1, fp=0, fn=0)


def test_empty_sides():
    assert score_trees(np.zeros((0, 4)), np.zeros(0, bool), np.array([[1, 1]], float)) == Counts(0, 0, 1)
    one_box = np.array([[0, 0, 1, 1]], float)
    assert score_trees(one_box, np.array([False]), np.zeros((0, 2))) == Counts(0, 1, 0)
    assert score_groups(np.zeros((0, 4)), np.zeros((0, 4)), 0.4) == Counts(0, 0, 0)


def test_group_iou_thresholds():
    pred, gt = np.array([[0, 0, 10, 10]], float), np.array([[0, 0, 10, 6]], float)  # IoU 0.6
    assert score_groups(pred, gt, 0.4).tp == 1
    pred2 = np.array([[0, 0, 10, 10]], float)
    gt2 = np.array([[0, 0, 10, 3]], float)  # IoU 0.3
    assert score_groups(pred2, gt2, 0.4).tp == 0 and score_groups(pred2, gt2, 0.25).tp == 1


def test_summarise_pools_and_splits_dense():
    s = [
        UnitScore("a", False, Counts(8, 2, 0), Counts(1, 0, 1), Counts(2, 0, 0)),
        UnitScore("b", True, Counts(2, 2, 6), Counts(0, 1, 1), Counts(1, 0, 0)),
    ]
    out = summarise(s)
    assert out["all"]["units"] == 2 and out["dense"]["units"] == 1
    assert out["all"]["trees"]["precision"] == pytest.approx(10 / 14)
    assert out["dense"]["trees"]["recall"] == pytest.approx(0.25)
