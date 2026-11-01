import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from fair_trees.evaluate import (  # noqa: E402
    MIN_PICK_GT,
    PICK_THRESHOLDS,
    EvalReport,
    EvalUnit,
    evaluate_units,
    f1_at,
    pick_threshold,
)
from fair_trees.params import InferenceParams  # noqa: E402
from fair_trees.preprocess import array_mosaic  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402

# box ~[122.9, 122.9, 133.1, 133.1] in a 256 px mosaic, score 0.9933
CENTRE_ANCHOR_800 = (50 * 100 + 50) * 9 + 3
PARAMS = InferenceParams(confidence_threshold=0.5)


def _unit(uid, gt, ignore=None, source="a"):
    mosaic = array_mosaic(np.full((256, 256, 3), 120, np.uint8), gsd_m=0.15)
    return EvalUnit(uid, mosaic, np.asarray(gt, np.float32).reshape(-1, 4), ignore, {"source": source})


@pytest.fixture
def run(fake_session):
    return ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))


def test_true_positive_unit(run):
    r = evaluate_units(run, [_unit("u1", [[123, 123, 133, 133]])], PARAMS)
    assert (r.precision, r.recall, r.f1, r.aoi_f1) == (1.0, 1.0, 1.0, 1.0)
    assert r.f1_max == 1.0 and r.conf_at_max == 0.05  # ties resolve to the lowest threshold
    assert r.ap40 == pytest.approx(1.0)
    assert r.units == [{"unit_id": "u1", "meta": {"source": "a"}, "tp": 1, "fp": 0, "fn": 0}]


def test_false_positive_and_crowd_ignore(run):
    assert evaluate_units(run, [_unit("u1", [])], PARAMS).units[0]["fp"] == 1
    ignore = np.zeros((256, 256), bool)
    ignore[120:136, 120:136] = True
    r = evaluate_units(run, [_unit("u1", [], ignore)], PARAMS)
    assert (r.units[0]["tp"], r.units[0]["fp"], r.units[0]["fn"]) == (0, 0, 0)
    r = evaluate_units(run, [_unit("u1", [], ignore)], PARAMS, ignore=False)
    assert r.units[0]["fp"] == 1


def test_threshold_above_score_gives_misses(run):
    r = evaluate_units(
        run, [_unit("u1", [[123, 123, 133, 133]])], InferenceParams(confidence_threshold=0.995)
    )
    assert (r.precision, r.recall, r.f1) == (0.0, 0.0, 0.0) and r.units[0]["fn"] == 1


def test_sweep_and_breakdown(run):
    units = [_unit("u1", [[123, 123, 133, 133]], source="a"), _unit("u2", [], source="b")]
    r = evaluate_units(run, units, PARAMS)
    assert [s["threshold"] for s in r.sweep] == pytest.approx(list(np.round(np.linspace(0.05, 0.95, 19), 2)))
    assert r.breakdown["source"]["a"]["f1"] == 1.0 and r.breakdown["source"]["b"]["precision"] == 0.0
    assert r.breakdown["source"]["b"]["n_units"] == 1
    assert [c.tp for c in r.unit_counts()] == [1, 0]


def test_report_with_no_units_is_finite(run):
    r = evaluate_units(run, [], PARAMS)
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["f1"] == 0.0 and d["ap40"] == 0.0 and d["units"] == []


def test_report_survives_non_string_meta(run):
    unit = _unit("u1", [[123, 123, 133, 133]])
    unit = EvalUnit(unit.unit_id, unit.mosaic, unit.gt_boxes, None, {"zoom": np.int64(20), 3: 4})
    d = evaluate_units(run, [unit], PARAMS).to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert d["units"][0]["meta"] == {"zoom": "20", "3": "4"}
    assert d["breakdown"]["zoom"]["20"]["n_units"] == 1


def test_decodes_below_a_low_operating_point(fake_session):
    weak = ort_run_fn(fake_session({CENTRE_ANCHOR_800: -6.0}))  # score 0.0025
    r = evaluate_units(
        weak, [_unit("u1", [[123, 123, 133, 133]])], InferenceParams(confidence_threshold=0.002)
    )
    assert r.units[0]["tp"] == 1


def test_scoring_ignores_model_thresholds(run):
    params = InferenceParams(confidence_threshold=0.5, class_thresholds=(0.999,))
    r = evaluate_units(run, [_unit("u1", [[123, 123, 133, 133]])], params)
    assert r.units[0]["tp"] == 1 and r.sweep[0]["f1"] == 1.0


def test_pick_sweep_covers_the_pick_grid(run):
    r = evaluate_units(run, [_unit("u1", [[123, 123, 133, 133]])], PARAMS)
    assert [s["threshold"] for s in r.pick_sweep] == pytest.approx([0.1 + 0.025 * i for i in range(9)])
    assert all(s["f1"] == 1.0 for s in r.pick_sweep)
    assert r.to_dict()["pick_sweep"] == r.pick_sweep


def _report(f1s, n_gt):
    sweep = [
        {"threshold": float(t), "precision": f, "recall": f, "f1": f}
        for t, f in zip(PICK_THRESHOLDS, f1s, strict=True)
    ]
    units = [{"unit_id": "u", "meta": {}, "tp": n_gt // 2, "fp": 0, "fn": n_gt - n_gt // 2}]
    return EvalReport(0.1, 0.4, 0, 0, 0, 0, 0, 0, 0, [], units, {}, pick_sweep=sweep)


def test_pick_threshold_takes_the_lowest_best_on_the_grid():
    f1s = [0.40, 0.50, 0.55, 0.57, 0.57, 0.52, 0.50, 0.45, 0.40]
    assert pick_threshold(_report(f1s, 500), prior=0.15) == pytest.approx(0.175)


def test_pick_threshold_falls_back_to_the_prior_on_small_val():
    f1s = [0.40, 0.50, 0.55, 0.57, 0.60, 0.52, 0.50, 0.45, 0.40]
    assert pick_threshold(_report(f1s, MIN_PICK_GT - 1), prior=0.15) == 0.15
    assert pick_threshold(_report(f1s, MIN_PICK_GT), prior=0.15) == pytest.approx(0.2)


def test_f1_at_reads_the_pick_sweep():
    f1s = [0.40, 0.50, 0.55, 0.57, 0.60, 0.52, 0.50, 0.45, 0.40]
    assert f1_at(_report(f1s, 500), 0.2)["f1"] == 0.60
    with pytest.raises(KeyError):
        f1_at(_report(f1s, 500), 0.33)
