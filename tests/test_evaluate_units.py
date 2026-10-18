import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from fair_trees.evaluate import EvalUnit, evaluate_units  # noqa: E402
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
