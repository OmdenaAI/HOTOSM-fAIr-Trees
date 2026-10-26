import numpy as np
import pandas as pd
import pytest

from experiments.label_study.hot import UnitTruth
from experiments.label_study.study import (
    SLIVER_M2,
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
from fair_trees.evaluate import Counts
from fair_trees.preprocess import Mosaic

GSD = 0.15


def _record(
    unit_id,
    boxes,
    scores,
    points,
    diam,
    groups,
    in_group=None,
    split="test",
    area=None,
    edge=None,
    cover=None,
):
    boxes = np.array(boxes, np.float32).reshape(-1, 4)
    groups = np.array(groups, np.float64).reshape(-1, 4)
    return UnitRecord(
        unit_id=unit_id,
        split=split,
        gsd_m=GSD,
        boxes=boxes,
        scores=np.array(scores, np.float32),
        in_group=np.zeros(len(boxes), bool) if in_group is None else np.array(in_group, bool),
        tree_points=np.array(points, np.float64).reshape(-1, 2),
        tree_diam_px=np.array(diam, np.float64),
        group_boxes=groups,
        group_area_m2=np.full(len(groups), 1000.0) if area is None else np.array(area, np.float64),
        group_edge=np.zeros(len(groups), bool) if edge is None else np.array(edge, bool),
        tile_cover=np.zeros((1, 1), np.float32) if cover is None else np.array(cover, np.float32),
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
            tree_points=np.array([[5.0 + offset, 5.0]]),
            tree_diam_px=np.array([4.0]),
            group_boxes=np.array([[30.0, 0, 40, 20]]),
            group_area_m2=np.array([200 * GSD**2]),
            group_edge=np.array([True]),
            group_mask=mask,
            group_cover=0.25,
            tile_cover=np.array([[0.25]], np.float32),
        )
        return mosaic, truth

    return build


def test_make_record_flags_boxes_in_groups():
    mosaic, truth = _build(0)()
    dets = Detections(
        np.array([[0, 0, 10, 10], [28, 4, 38, 14]], np.float32), np.array([0.9, 0.9], np.float32)
    )
    rec = make_record("u", "val", mosaic, truth, dets)
    assert rec.in_group.tolist() == [False, True] and rec.gsd_m == GSD and rec.split == "val"
    assert rec.group_edge.tolist() == [True] and rec.tile_cover.tolist() == [[0.25]]
    assert np.array_equal(rec.group_area_m2, truth.group_area_m2)


def test_record_round_trip_and_tmp_files_are_ignored(tmp_path):
    rec_a = _record(
        "test-0-0-0", [[0, 0, 10, 10]], [0.9], [[5, 5]], [4.0], [[0, 0, 5, 5]], edge=[True], cover=[[0.5, 0]]
    )
    rec_b = _record("test-0-0-1", [[0, 0, 4, 4]], [0.9], [[2, 2]], [3.0], [], split="val")
    save_record(rec_a, tmp_path)
    save_record(rec_b, tmp_path)
    (tmp_path / "test-0-0-2.npz.tmp").write_bytes(b"partial")
    loaded = load_records(tmp_path, "test")
    assert [r.unit_id for r in loaded] == ["test-0-0-0"]
    assert np.array_equal(loaded[0].boxes, rec_a.boxes) and loaded[0].group_edge.tolist() == [True]
    assert np.array_equal(loaded[0].tile_cover, rec_a.tile_cover) and loaded[0].group_area_m2.tolist() == [
        1000
    ]
    assert loaded[0].gsd_m == GSD and loaded[0].split == "test"


def test_a_record_from_another_version_does_not_load(tmp_path):
    path = save_record(_record("test-0-0-0", [], [], [], [], []), tmp_path)
    with np.load(path) as z:
        fields = dict(z)
    with open(path, "wb") as f:
        np.savez(f, **(fields | {"version": 1}))
    with pytest.raises(ValueError, match="test-0-0-0.npz"):
        load_records(tmp_path, "test")


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


def test_edge_slivers_are_ignored_in_the_group_score():
    # an edge sliver, an interior part as small and an edge part above SLIVER_M2; no predictions
    small = SLIVER_M2 / 2
    groups = [[0, 0, 5, 5], [20, 0, 25, 5], [40, 0, 100, 60]]
    rec = _record("u", [], [], [], [], groups, area=[small, small, 2 * SLIVER_M2], edge=[True, False, True])
    out = score_record(rec, 0.1, 0.0, 2)
    assert out.groups_40 == Counts(0, 0, 2) and out.groups_25 == Counts(0, 0, 2)
    assert out.groups_40_all == Counts(0, 0, 3)


def test_dense_counts_keep_only_the_dense_tile():
    # two 256 px tiles side by side, only the left one dense; per tile a matched tree, a spare box and a group
    boxes = [[90, 90, 110, 110], [150, 150, 160, 160], [390, 90, 410, 110], [450, 150, 460, 160]]
    points = [[100, 100], [400, 100]]
    groups = [[10, 200, 60, 250], [300, 200, 360, 250]]
    rec = _record("u", boxes, [0.9] * 4, points, [np.nan] * 2, groups, cover=[[0.5, 0.0]])
    out = score_record(rec, 0.1, 0.0, 2)
    assert out.dense_tiles == 1
    assert out.trees == Counts(2, 2, 0) and out.dense_trees == Counts(1, 1, 0)
    assert out.groups_40 == Counts(0, 0, 2) and out.dense_groups_40 == Counts(0, 0, 1)


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
