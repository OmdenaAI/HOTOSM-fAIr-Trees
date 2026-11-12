import numpy as np
import pytest

pytest.importorskip("scipy")

from experiments.recall_check import check_units, scale_boxes, unit_breakdown  # noqa: E402
from fair_trees.evaluate import EvalUnit  # noqa: E402
from fair_trees.params import InferenceParams  # noqa: E402
from fair_trees.preprocess import array_mosaic  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402

# box ~[122.9, 122.9, 133.1, 133.1] in a 256 px mosaic, score 0.9933
CENTRE_ANCHOR_800 = (50 * 100 + 50) * 9 + 3


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
