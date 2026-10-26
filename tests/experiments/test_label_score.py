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
