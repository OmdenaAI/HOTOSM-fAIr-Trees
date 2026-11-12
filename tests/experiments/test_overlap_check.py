import numpy as np
import pytest

pytest.importorskip("scipy")

from experiments.overlap_check import check_units, report_lines  # noqa: E402
from fair_trees.decode import make_anchors  # noqa: E402
from fair_trees.evaluate import EvalUnit, evaluate_units  # noqa: E402
from fair_trees.params import InferenceParams  # noqa: E402
from fair_trees.preprocess import array_mosaic  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402

GSD = 0.15
PARAMS = InferenceParams(window_m=38.4, group_window_m=38.4)  # 256 px windows, upscaled 3.125x to 800 px
BOX = np.array([122.88, 122.88, 133.12, 133.12], np.float32)  # the centre anchor's box in its window
# the same anchor at P3 column 5: 7.68 px from the window's left side, inside a 0.05 band (12.8 px) only
LEFT_BOX = np.array([7.68, 122.88, 17.92, 133.12], np.float32)

# P5 (25 x 25 locations after P3 and P4) at row 12, column 1: a 128 px box clipped by the window's left side,
# [0, 102.4, 30.72, 143.36] in its window
P5_LEFT = 9 * (100**2 + 50**2) + (12 * 25 + 1) * 9 + 3


def _p3(row, col):
    return (row * 100 + col) * 9 + 3  # aspect 1.0, size 32 at an 800 px input


def _unit(height, width, gt, ignore=None):
    mosaic = array_mosaic(np.full((height, width, 3), 120, np.uint8), gsd_m=GSD)
    return EvalUnit("u1", mosaic, np.asarray(gt, np.float32).reshape(-1, 4), ignore)


def _shifted(box, *offsets):
    return [box + np.array([dx, dy, dx, dy], np.float32) for dx, dy in offsets]


def _setting(out, overlap, band=0.05, rule="drop_edge"):
    [found] = [
        s for s in out["settings"] if (s["overlap"], s["edge_band"], s["rule"]) == (overlap, band, rule)
    ]
    return found


def test_each_overlap_runs_its_own_windows_and_is_scored_at_every_threshold(fake_session):
    # every window answers with one box at its centre: 4 windows without overlap, 9 at half overlap
    session = fake_session({_p3(50, 50): 5.0})
    unit = _unit(512, 512, _shifted(BOX, (0, 0), (256, 0), (0, 256), (256, 256)))
    out = check_units(
        ort_run_fn(session), [unit], PARAMS, 0, overlaps=(0.5, 0.0), bands=(0.05,), thresholds=(0.2, 0.995)
    )
    assert out["class"] == "tree" and out["window_m"] == 38.4 and out["unit_ids"] == ["u1"]
    assert out["thresholds"] == [0.2, 0.995]
    assert out["area_km2"] == pytest.approx((512 * GSD) ** 2 / 1e6)
    half, none = _setting(out, 0.5), _setting(out, 0.0)
    assert (half["passes"], none["passes"]) == (9, 4) and len(session.calls) == 13
    assert half["rows"][0]["units"] == [[4, 5, 0]] and none["rows"][0]["units"] == [[4, 0, 0]]
    assert half["rows"][0]["f1"] == pytest.approx(8 / 13) and none["rows"][0]["f1"] == 1.0
    assert none["rows"][1]["units"] == [[0, 0, 4]]  # the planted score is 0.9933
    assert half["default"] and none["default"]


def test_a_narrower_edge_band_keeps_a_box_beside_an_interior_window_side(fake_session):
    # two windows side by side; the right one's box sits 7.68 px from the side it shares with the left one
    session = fake_session({_p3(50, 5): 5.0})
    unit = _unit(256, 512, _shifted(LEFT_BOX, (0, 0), (256, 0)))
    out = check_units(
        ort_run_fn(session), [unit], PARAMS, 0, overlaps=(0.0,), bands=(0.05, 0.02), thresholds=(0.2,)
    )
    assert _setting(out, 0.0, 0.05)["rows"][0]["units"] == [[1, 0, 1]]
    assert _setting(out, 0.0, 0.02)["rows"][0]["units"] == [[2, 0, 0]]
    assert _setting(out, 0.0, 0.05)["default"] and not _setting(out, 0.0, 0.02)["default"]
    assert len(session.calls) == 2  # both bands are scored from one run of the windows


def test_groups_are_scored_under_every_merge_rule_from_one_run(fake_session):
    session = fake_session({(_p3(50, 5), 1): 5.0}, num_classes=2)
    unit = _unit(256, 512, _shifted(LEFT_BOX, (0, 0), (256, 0)))
    out = check_units(
        ort_run_fn(session), [unit], PARAMS, 1, overlaps=(0.0,), bands=(0.05,), thresholds=(0.2,)
    )
    assert out["class"] == "tree_group"
    assert {s["rule"] for s in out["settings"]} == {"drop_edge", "join", "overlap"}
    assert _setting(out, 0.0, rule="drop_edge")["rows"][0]["units"] == [[1, 0, 1]]  # the cut box is dropped
    assert _setting(out, 0.0, rule="join")["rows"][0]["units"] == [[2, 0, 0]]  # a cut box alone is kept
    assert [s["rule"] for s in out["settings"] if s["default"]] == ["drop_edge"]
    assert len(session.calls) == 2


def test_a_faint_cut_group_box_joins_only_at_the_thresholds_it_passes():
    # the left window is sure of a small group; the right window sees a faint large one across their shared
    # side (score 0.23). Joined, the pair no longer matches the truth, so serving filters before the join.
    planted = [(_p3(50, 50), 5.0), (P5_LEFT, -1.2)]  # the two windows' (anchor, logit), in run order
    calls = []

    def run(batch):
        n = sum(len(a) for a in make_anchors(batch.shape[2], batch.shape[3]))
        cls = np.full((1, n, 2), -10.0, np.float32)
        anchor, logit = planted[len(calls)]
        cls[0, anchor, 1] = logit
        calls.append(batch.shape)
        return cls, np.zeros((1, n, 4), np.float32)

    out = check_units(
        run, [_unit(256, 384, [BOX])], PARAMS, 1, overlaps=(0.5,), bands=(0.05,), thresholds=(0.2, 0.25)
    )
    assert [row["units"] for row in _setting(out, 0.5, rule="join")["rows"]] == [[[0, 1, 1]], [[1, 0, 0]]]
    dropped = _setting(out, 0.5, rule="drop_edge")["rows"]
    assert [row["units"] for row in dropped] == [[[1, 0, 0]], [[1, 0, 0]]]


def test_detections_in_the_ignore_mask_are_not_scored(fake_session):
    ignore = np.zeros((512, 512), bool)
    ignore[250:262, 250:262] = True  # the middle window's box, seen at half overlap only
    unit = _unit(512, 512, _shifted(BOX, (0, 0), (256, 0), (0, 256), (256, 256)), ignore)
    run = ort_run_fn(fake_session({_p3(50, 50): 5.0}))
    out = check_units(run, [unit], PARAMS, 0, overlaps=(0.5,), bands=(0.05,), thresholds=(0.2,))
    assert _setting(out, 0.5)["rows"][0]["units"] == [[4, 4, 0]]


@pytest.mark.parametrize("class_index", [0, 1])
def test_the_default_setting_repeats_what_evaluate_units_counts(fake_session, class_index):
    # scores 0.99, 0.73 and 0.23: the pick grid (0.1 to 0.3) drops the last one from 0.25 up
    planted = {(_p3(50, 50), class_index): 5.0, (_p3(50, 5), class_index): 1.0}
    planted[(_p3(20, 50), class_index)] = -1.2
    run = ort_run_fn(fake_session(planted, num_classes=2))
    ignore = np.zeros((512, 512), bool)
    ignore[250:262, 250:262] = True
    unit = _unit(512, 512, _shifted(BOX, (0, 0), (256, 0)) + _shifted(LEFT_BOX, (0, 256)), ignore)

    report = evaluate_units(run, [unit], PARAMS, class_index=class_index)
    out = check_units(run, [unit], PARAMS, class_index, overlaps=(0.5,), bands=(0.05,))

    [default] = [s for s in out["settings"] if s["default"]]
    assert [row["threshold"] for row in default["rows"]] == [s["threshold"] for s in report.pick_sweep]
    assert [row["units"] for row in default["rows"]] == [s["units"] for s in report.pick_sweep]
    assert default["rows"][0]["units"] != default["rows"][-1]["units"]  # the grid does change the counts
    assert [row["f1"] for row in default["rows"]] == [s["f1"] for s in report.pick_sweep]


def test_report_pairs_each_setting_with_the_first_overlap_and_names_a_better_one(fake_session):
    # each window answers with its centre box and a box beside its left side; the truth is what the two
    # windows without overlap hold. Half overlap adds a third window, whose centre box is a false positive.
    run = ort_run_fn(fake_session({_p3(50, 50): 5.0, _p3(50, 5): 5.0}))
    unit = _unit(256, 512, _shifted(BOX, (0, 0), (256, 0)) + _shifted(LEFT_BOX, (0, 0), (256, 0)))
    out = check_units(run, [unit], PARAMS, 0, overlaps=(0.5, 0.0), bands=(0.05, 0.02), thresholds=(0.2,))
    assert _setting(out, 0.5, 0.05)["rows"][0]["units"] == [[3, 1, 1]]
    assert _setting(out, 0.5, 0.02)["rows"][0]["units"] == [[4, 2, 0]]
    assert _setting(out, 0.0, 0.05)["rows"][0]["units"] == [[3, 0, 1]]
    assert _setting(out, 0.0, 0.02)["rows"][0]["units"] == [[4, 0, 0]]

    lines = report_lines({"run_id": "r", "part": "val", "served": {"tree": 0.2}, "classes": {"tree": out}})

    assert lines[0].startswith("tree: 38.4 m windows, threshold 0.2, 1 units")
    assert len(lines) == 5
    first, first_better, second, second_better = lines[1:]
    assert first.lstrip().startswith("overlap 0.5 ") and "3 passes" in first and "F1 0.750" in first
    assert "vs overlap 0.5" not in first  # the reference
    assert "band 0.02" in first_better and "F1 0.800" in first_better
    assert "vs overlap 0.5: +0.050" in first_better
    assert second.lstrip().startswith("overlap 0 ") and "2 passes" in second and "F1 0.857" in second
    assert "vs overlap 0.5: +0.107" in second
    assert "band 0.02" in second_better and "F1 1.000" in second_better
    assert "vs overlap 0.5: +0.250" in second_better


def test_report_names_no_other_setting_when_none_beats_the_default(fake_session):
    run = ort_run_fn(fake_session({_p3(50, 50): 5.0}))
    unit = _unit(512, 512, _shifted(BOX, (0, 0), (256, 0), (0, 256), (256, 256)))
    out = check_units(run, [unit], PARAMS, 0, overlaps=(0.5, 0.0), bands=(0.05, 0.02), thresholds=(0.2,))
    lines = report_lines({"run_id": "r", "part": "val", "served": {"tree": 0.2}, "classes": {"tree": out}})
    assert len(lines) == 3 and not any("band" in line for line in lines)
