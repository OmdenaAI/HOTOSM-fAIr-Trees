import numpy as np
import pytest

pytest.importorskip("scipy")

from experiments.recall_check import (  # noqa: E402
    at_picks,
    check_units,
    halve_unit,
    report_lines,
    scale_boxes,
    unit_breakdown,
    with_val_picks,
)
from fair_trees.evaluate import EvalUnit  # noqa: E402
from fair_trees.params import InferenceParams  # noqa: E402
from fair_trees.preprocess import array_mosaic  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402

# box ~[122.9, 122.9, 133.1, 133.1] in a 256 px mosaic, score 0.9933
CENTRE_ANCHOR_800 = (50 * 100 + 50) * 9 + 3
FAINT_ANCHOR_800 = (20 * 100 + 20) * 9 + 3  # a second box, near (52, 52), away from the first


def _unit(gt):
    mosaic = array_mosaic(np.full((256, 256, 3), 120, np.uint8), gsd_m=0.15)
    return EvalUnit("u1", mosaic, np.asarray(gt, np.float32).reshape(-1, 4), None, {"source": "a"})


def test_scale_boxes_keeps_centres():
    boxes = np.array([[0, 0, 10, 20], [5, 5, 7, 7]], np.float32)
    np.testing.assert_allclose(scale_boxes(boxes, 0.5), [[2.5, 5, 7.5, 15], [5.5, 5.5, 6.5, 6.5]])
    np.testing.assert_allclose(scale_boxes(boxes, 1.0), boxes)


def test_unit_breakdown_counts_located_misses_and_crown_sizes():
    gsd = 0.15
    gt = np.array([[0, 0, 10, 10], [50, 50, 56, 56], [100, 100, 140, 140]], np.float32)  # 1.5, 0.9, 6 m
    pred = np.array([[0, 0, 10, 10], [45, 45, 61, 61]], np.float32)  # a hit; a box too big around gt 1
    b = unit_breakdown(pred, gt, gsd)
    assert (b["tp"], b["fp"], b["fn"]) == (1, 1, 2)
    assert b["located_misses"] == 1  # gt 1's centre lies in the unmatched oversize box
    assert b["point_hits"] == 2 and b["n_gt"] == 3
    assert b["gt_by_size"] == [1, 1, 0, 0, 1] and b["tp_by_size"] == [0, 1, 0, 0, 0]


def test_unit_breakdown_counts_centre_matches_whatever_the_truth_size():
    gt = np.array([[0, 0, 10, 10], [50, 50, 56, 56], [100, 100, 140, 140]], np.float32)  # 1.5, 0.9, 6 m
    pred = np.array([[0, 0, 10, 10], [45, 45, 61, 61]], np.float32)  # the 16 px box holds gt 1's centre
    b = unit_breakdown(pred, gt, 0.15)
    assert b["centre_tp"] == 2 and b["centre_tp_by_size"] == [1, 1, 0, 0, 0]
    # side of the paired box over the truth side, per size bin: 16 / 6 for the 0.9 m crown
    assert b["centre_ratio_by_size"] == [[pytest.approx(16 / 6)], [pytest.approx(1.0)], [], [], []]


def test_shrinking_an_oversize_box_turns_a_located_miss_into_a_hit(fake_session):
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))
    units = [_unit([[125, 125, 131, 131]])]  # 6 px truth inside the 10.2 px planted box: IoU 0.35
    out = check_units(run, units, InferenceParams(), 0.2, scales=(0.8, 1.0), nms_thresholds=(), windows_m=())
    assert out["scales"]["1.0"]["tp"] == 0 and out["scales"]["1.0"]["located_misses"] == 1
    assert out["scales"]["0.8"]["tp"] == 1 and out["scales"]["0.8"]["f1"] == 1.0
    assert out["scales"]["0.8"]["units"] == [[1, 0, 0]]
    assert out["scales"]["1.0"]["point_in_box_recall"] == 1.0


def test_nms_and_window_settings_are_each_decoded_and_scored(fake_session):
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))
    units = [_unit([[123, 123, 133, 133]])]
    out = check_units(
        run,
        units,
        InferenceParams(window_m=60.0),
        0.2,
        scales=(1.0,),
        nms_thresholds=(0.05, 0.3),
        windows_m=(40.0,),
    )
    assert set(out["nms"]) == {"0.05", "0.3"} and set(out["windows_m"]) == {"40.0"}
    assert all(r["f1"] == 1.0 for r in [*out["nms"].values(), *out["windows_m"].values()])
    assert out["window_m"] == 60.0 and out["threshold"] == 0.2
    assert out["by_size"]["recall"][1] == 1.0  # the 1.5 m crown, matched at the base setting
    assert out["by_size"]["recall"][0] is None  # no crown under 1.5 m


def test_every_setting_reports_the_centre_score_and_recall_by_crown_size(fake_session):
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))
    units = [_unit([[125, 125, 131, 131]])]  # 0.9 m truth inside the 10.2 px planted box: IoU 0.35
    out = check_units(
        run,
        units,
        InferenceParams(window_m=60.0),
        0.2,
        scales=(1.0,),
        nms_thresholds=(0.3,),
        windows_m=(40.0,),
    )
    for setting in (out["scales"]["1.0"], out["nms"]["0.3"], out["windows_m"]["40.0"]):
        assert setting["f1"] == 0.0  # the IoU 0.4 score fails the oversize box
        centre = setting["centre"]
        assert (centre["tp"], centre["fp"], centre["fn"]) == (1, 0, 0) and centre["f1"] == 1.0
        assert centre["units"] == [[1, 0, 0]]
        by_size = setting["by_size"]
        assert by_size["n_gt"] == [1, 0, 0, 0, 0]
        assert by_size["recall"] == [0.0, None, None, None, None]
        assert by_size["centre_recall"] == [1.0, None, None, None, None]
        assert by_size["centre_side_ratio"][0] == pytest.approx(10.2 / 6, abs=0.05)
        assert by_size["centre_side_ratio"][1:] == [None] * 4
    assert out["by_size"] == out["scales"]["1.0"]["by_size"] | {"bins_m": [1.5, 2.5, 4.0, 6.0]}


def _swept(fake_session, planted):
    run = ort_run_fn(fake_session(planted))
    return check_units(
        run,
        [_unit([[123, 123, 133, 133]])],
        InferenceParams(window_m=60.0),
        0.2,
        scales=(1.0,),
        nms_thresholds=(),
        windows_m=(40.0,),
        pick_thresholds=(0.2, 0.3),
    )


def test_each_window_is_swept_and_picks_its_own_threshold(fake_session):
    out = _swept(fake_session, {CENTRE_ANCHOR_800: 5.0, FAINT_ANCHOR_800: -1.1})  # scores 0.99 and 0.25
    assert set(out["threshold_sweep"]) == {"60.0", "40.0"}
    for window in ("60.0", "40.0"):
        sweep = out["threshold_sweep"][window]
        assert [s["threshold"] for s in sweep] == [0.2, 0.3]
        assert [(s["tp"], s["fp"], s["fn"]) for s in sweep] == [
            (1, 1, 0),
            (1, 0, 0),
        ]  # 0.3 drops the faint box
        assert [s["centre"]["fp"] for s in sweep] == [1, 0]
        assert sweep[0]["by_size"]["recall"][1] == 1.0
        assert out["picked"][window] == {"f1": 0.3, "centre_f1": 0.3}


def test_a_tie_picks_the_lowest_threshold(fake_session):
    out = _swept(fake_session, {CENTRE_ANCHOR_800: 5.0})
    assert out["picked"]["60.0"] == {"f1": 0.2, "centre_f1": 0.2}


def test_at_picks_reads_each_window_at_thresholds_picked_elsewhere(fake_session):
    out = _swept(fake_session, {CENTRE_ANCHOR_800: 5.0, FAINT_ANCHOR_800: -1.1})
    rows = at_picks(out, {"60.0": {"f1": 0.2, "centre_f1": 0.3}, "30.0": {"f1": 0.2, "centre_f1": 0.2}})
    assert set(rows) == {"60.0"}  # 40 m has no pick and 30 m was not scored here
    assert rows["60.0"]["f1"]["threshold"] == 0.2 and rows["60.0"]["f1"]["fp"] == 1
    assert rows["60.0"]["centre_f1"]["threshold"] == 0.3 and rows["60.0"]["centre_f1"]["centre"]["fp"] == 0
    with pytest.raises(KeyError, match="0.25"):
        at_picks(out, {"60.0": {"f1": 0.25, "centre_f1": 0.3}})


def test_halve_unit_averages_two_by_two_blocks_and_keeps_labels_in_step():
    rgb = np.zeros((5, 4, 3), np.uint8)  # the odd fifth row is cropped
    rgb[:2, :2] = [[[10, 0, 0], [20, 0, 0]], [[30, 0, 0], [40, 0, 0]]]
    rgb[2:4, 2:] = 200
    valid = np.ones((5, 4), bool)
    valid[3, 3] = False
    ignore = np.zeros((5, 4), bool)
    ignore[0, 3] = True
    unit = EvalUnit(
        "u1", array_mosaic(rgb, 0.15, valid), np.array([[0, 0, 4, 2]], np.float32), ignore, {"zoom": "20"}
    )
    half = halve_unit(unit)
    assert half.mosaic.rgb.shape == (2, 2, 3) and half.mosaic.rgb.dtype == np.uint8
    assert half.mosaic.rgb[0, 0].tolist() == [25, 0, 0]  # the mean of 10, 20, 30, 40
    assert half.mosaic.valid.tolist() == [
        [True, True],
        [True, False],
    ]  # a block with an invalid pixel is invalid
    assert half.mosaic.rgb[1, 1].tolist() == [0, 0, 0]  # and blank, as invalid pixels are in a mosaic
    assert half.mosaic.gsd_m == pytest.approx(0.3)
    np.testing.assert_allclose(half.gt_boxes, [[0, 0, 2, 1]])
    assert half.ignore_mask.tolist() == [[False, True], [False, False]]  # ignored if any of its pixels was
    assert half.unit_id == "u1"
    assert unit.mosaic.rgb.shape == (5, 4, 3)  # the unit itself is untouched


def test_half_resolution_is_scored_with_its_own_sweep(fake_session):
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))
    units = [_unit([[123, 123, 133, 133]])]
    out = check_units(
        run,
        units,
        InferenceParams(window_m=60.0),
        0.2,
        scales=(1.0,),
        nms_thresholds=(),
        windows_m=(),
        pick_thresholds=(0.2, 0.3),
        half_resolution=True,
    )
    half = out["half_resolution"]
    assert half["gsd_m"] == pytest.approx(0.3)
    # the planted box sits at the same share of the window, so it still covers the halved truth box
    assert half["scales"]["1.0"]["f1"] == 1.0 and half["by_size"]["n_gt"] == [0, 1, 0, 0, 0]
    assert [row["threshold"] for row in half["threshold_sweep"]["60.0"]] == [0.2, 0.3]
    assert half["picked"]["60.0"] == {"f1": 0.2, "centre_f1": 0.2}
    assert "half_resolution" not in half
    plain = check_units(
        run,
        units,
        InferenceParams(),
        0.2,
        scales=(1.0,),
        nms_thresholds=(),
        windows_m=(),
        pick_thresholds=(0.2,),
    )
    assert "half_resolution" not in plain


def _full(fake_session, planted):
    run = ort_run_fn(fake_session(planted))
    return check_units(
        run,
        [_unit([[125, 125, 131, 131]])],  # 0.9 m truth inside the 10.2 px planted box: IoU 0.35
        InferenceParams(window_m=60.0),
        0.2,
        scales=(1.0,),
        nms_thresholds=(),
        windows_m=(40.0,),
        pick_thresholds=(0.2, 0.3),
        half_resolution=True,
    )


def test_a_held_out_part_is_read_at_the_thresholds_val_picked(fake_session):
    val = _full(fake_session, {CENTRE_ANCHOR_800: 5.0, FAINT_ANCHOR_800: -1.1})  # centre pick 0.3
    test = with_val_picks(_full(fake_session, {CENTRE_ANCHOR_800: 5.0}), val)
    assert val["picked"]["40.0"]["centre_f1"] == 0.3
    assert test["at_val_pick"]["40.0"]["centre_f1"]["threshold"] == 0.3
    assert test["at_val_pick"]["60.0"]["f1"]["threshold"] == val["picked"]["60.0"]["f1"]
    assert test["half_resolution"]["at_val_pick"]["60.0"]["centre_f1"]["threshold"] == 0.3
    assert "at_val_pick" not in val


def test_a_val_file_from_before_the_sweep_adds_nothing(fake_session):
    test = _full(fake_session, {CENTRE_ANCHOR_800: 5.0})
    old_val = {"run_id": "r", "part": "val", "threshold": 0.2}
    assert with_val_picks(dict(test), old_val) == test


def test_report_lines_show_both_scores_the_picks_and_half_resolution(fake_session):
    val = _full(fake_session, {CENTRE_ANCHOR_800: 5.0, FAINT_ANCHOR_800: -1.1})
    text = "\n".join(report_lines(with_val_picks(_full(fake_session, {CENTRE_ANCHOR_800: 5.0}), val)))
    assert "scale 1.0  F1 0.000" in text and "centre F1 1.000" in text
    assert "<1.5: IoU 0.00, centre 1.00, box/label 1.7 (n 1)" in text
    assert "window 40.0 m: own pick 0.2 (F1 0.000), own centre pick 0.2 (centre F1 1.000)" in text
    assert "window 40.0 m at val's picks: F1 0.000 @0.2, centre F1 1.000 @0.3" in text
    assert "half resolution (0.30 m/px)" in text
    assert "half resolution, window 60.0 m at val's picks" in text
