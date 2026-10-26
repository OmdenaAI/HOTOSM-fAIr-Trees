import numpy as np
import pandas as pd
import pytest

from experiments.label_study.hot import UnitTruth
from experiments.label_study.study import (
    UnitRecord,
    choose,
    load_records,
    make_record,
    run_units,
    save_record,
    score_record,
    score_test,
    sweep,
)
from fair_trees.decode import Detections
from fair_trees.preprocess import Mosaic

GSD = 0.15


def _record(unit_id, boxes, scores, points, diam, groups, in_group=None, split="test", dense=False):
    boxes = np.array(boxes, np.float32).reshape(-1, 4)
    return UnitRecord(
        unit_id=unit_id,
        split=split,
        dense=dense,
        gsd_m=GSD,
        boxes=boxes,
        scores=np.array(scores, np.float32),
        in_group=np.zeros(len(boxes), bool) if in_group is None else np.array(in_group, bool),
        tree_points=np.array(points, np.float64).reshape(-1, 2),
        tree_diam_px=np.array(diam, np.float64),
        group_boxes=np.array(groups, np.float64).reshape(-1, 4),
    )


def _two_touching():
    return Detections(
        np.array([[0, 0, 10, 10], [10, 0, 20, 10]], np.float32), np.array([0.9, 0.8], np.float32)
    )


def _build(offset):
    def build():
        mask = np.zeros((20, 40), bool)
        mask[:, 30:] = True
        mosaic = Mosaic(np.zeros((20, 40, 3), np.uint8), np.ones((20, 40), bool), None, 18, GSD)
        truth = UnitTruth(
            np.array([[5.0 + offset, 5.0]]), np.array([4.0]), np.array([[30.0, 0, 40, 20]]), mask, 0.25
        )
        return mosaic, truth

    return build


def test_make_record_flags_boxes_in_groups_and_dense():
    mosaic, truth = _build(0)()
    truth = UnitTruth(truth.tree_points, truth.tree_diam_px, truth.group_boxes, truth.group_mask, 0.3)
    dets = Detections(
        np.array([[0, 0, 10, 10], [28, 4, 38, 14]], np.float32), np.array([0.9, 0.9], np.float32)
    )
    rec = make_record("u", "val", mosaic, truth, dets)
    assert rec.in_group.tolist() == [False, True] and rec.dense and rec.gsd_m == GSD and rec.split == "val"


def test_record_round_trip_and_tmp_files_are_ignored(tmp_path):
    rec_a = _record("test-0-0-0", [[0, 0, 10, 10]], [0.9], [[5, 5]], [4.0], [[0, 0, 5, 5]], dense=True)
    rec_b = _record("test-0-0-1", [[0, 0, 4, 4]], [0.9], [[2, 2]], [3.0], [], split="val")
    save_record(rec_a, tmp_path)
    save_record(rec_b, tmp_path)
    (tmp_path / "test-0-0-2.npz.tmp").write_bytes(b"partial")
    loaded = load_records(tmp_path, "test")
    assert [r.unit_id for r in loaded] == ["test-0-0-0"]
    assert np.array_equal(loaded[0].boxes, rec_a.boxes) and loaded[0].dense == rec_a.dense
    assert loaded[0].gsd_m == GSD and loaded[0].split == "test"


def test_run_units_skips_finished_units(tmp_path):
    calls = []

    def predict(m):
        calls.append(1)
        return _two_touching()

    units = [("test-0-0-0", "test", _build(0)), ("test-0-0-1", "test", _build(1))]
    assert run_units(predict, units, tmp_path) == 2
    assert run_units(predict, units, tmp_path) == 0 and len(calls) == 2


def test_score_record_uses_metres_for_the_gap():
    # two boxes 0.6 m apart at gsd 0.15 m (4 px): separate at gap 0.5 m, one group at gap 1 m
    rec = _record("u", [[0, 0, 10, 10], [14, 0, 24, 10]], [0.9, 0.9], [], [], [[0, 0, 24, 10]])
    assert score_record(rec, 0.1, 0.5, 2).groups_40.fp == 0
    merged = score_record(rec, 0.1, 1.0, 2).groups_40
    assert merged.tp + merged.fp == 1


def test_score_threshold_filters_before_merging():
    # three touching boxes in a row; the middle one scores 0.2
    rec = _record("u", [[0, 0, 10, 10], [10, 0, 20, 10], [20, 0, 30, 10]], [0.9, 0.2, 0.9], [], [], [])
    low, high = score_record(rec, 0.1, 0.0, 3).groups_40, score_record(rec, 0.5, 0.0, 3).groups_40
    assert low.tp + low.fp == 1 and high.tp + high.fp == 0


def test_sweep_covers_the_grid_and_choose_breaks_ties():
    rec = _record("u", [[0, 0, 10, 10]], [0.9], [[5, 5]], [4.0], [])
    table = sweep([rec], {"score": (0.1, 0.2), "gap_m": (0.0,), "min_crowns": (2, 3)})
    assert len(table) == 4
    tie = pd.DataFrame(
        {"score": [0.1, 0.2], "gap_m": [0.0, 0.0], "min_crowns": [3, 3], "mean_f1": [0.5, 0.5]}
    )
    assert choose(tie) == {"score": 0.2, "gap_m": 0.0, "min_crowns": 3}


def test_score_test_diameter_ratio():
    rec = _record("u", [[0, 0, 6, 6]], [0.9], [[3, 3], [20, 20]], [4.0, np.nan], [])
    out = score_test([rec], {"score": 0.1, "gap_m": 0.0, "min_crowns": 3})
    ratio = out["diameter_ratio"]
    assert ratio["pairs"] == 1 and ratio["median"] == pytest.approx(1.5)
    assert ratio["points_without_diameter"] == 1 and out["setting"]["min_crowns"] == 3
