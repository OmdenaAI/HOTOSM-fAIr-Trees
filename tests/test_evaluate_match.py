import numpy as np
import pytest

pytest.importorskip("scipy")

from fair_trees.evaluate import (  # noqa: E402
    Counts,
    aoi_f1,
    average_precision,
    count_matches,
    iou_matrix,
    match_hungarian,
    point_in_box_recall,
    prf,
)


def test_iou_matrix_and_empties():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 0, 0]], np.float32)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0, 0.0]])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 4)


def test_hungarian_beats_greedy():
    gt = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], np.float32)
    pred = np.array([[2, 0, 12, 10], [-1, 0, 9, 10]], np.float32)  # greedy by score would match only 1
    assert sorted(match_hungarian(pred, gt)) == [(0, 1), (1, 0)]
    assert count_matches(pred, gt) == Counts(tp=2, fp=0, fn=0)


def test_iou_exactly_threshold_does_not_match():
    gt = np.array([[0, 0, 10, 10]], np.float32)
    pred = np.array([[0, 0, 10, 4]], np.float32)  # IoU = 0.4 exactly
    assert match_hungarian(pred, gt) == []
    assert count_matches(pred, gt) == Counts(tp=0, fp=1, fn=1)


def test_empty_sides():
    gt = np.array([[0, 0, 10, 10]], np.float32)
    assert count_matches(np.zeros((0, 4), np.float32), gt) == Counts(0, 0, 1)
    assert count_matches(gt, np.zeros((0, 4), np.float32)) == Counts(0, 1, 0)


def test_prf_zero_conventions():
    assert prf(Counts(0, 0, 0)) == (0.0, 0.0, 0.0)
    assert prf(Counts(0, 3, 0)) == (0.0, 0.0, 0.0)
    p, r, f = prf(Counts(10, 10, 20))
    assert (p, r) == (0.5, pytest.approx(1 / 3)) and f == pytest.approx(0.4)
    assert Counts(1, 2, 3) + Counts(1, 1, 1) == Counts(2, 3, 4)


def test_average_precision_envelope():
    assert average_precision([0.5, 1.0], [1.0, 0.5]) == pytest.approx(0.75)
    assert average_precision([1.0, 0.25, 0.5], [0.5, 0.4, 1.0]) == pytest.approx(0.75)
    assert average_precision([0.0], [0.0]) == 0.0


def test_aoi_f1_weights_by_gt_count():
    assert aoi_f1([Counts(10, 0, 0), Counts(10, 10, 20)]) == pytest.approx((10 * 1.0 + 30 * 0.4) / 40)
    assert aoi_f1([Counts(0, 5, 0)]) == 0.0
    assert aoi_f1([]) == 0.0


def test_point_in_box_recall():
    boxes = np.array([[0, 0, 10, 10]], np.float32)
    assert point_in_box_recall(np.array([[5, 5], [50, 50], [10, 10]]), boxes) == pytest.approx(2 / 3)
    assert point_in_box_recall(np.zeros((0, 2)), boxes) == 0.0
