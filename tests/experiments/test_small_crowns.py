import numpy as np
import pytest

pytest.importorskip("scipy")

from experiments.small_crowns import (  # noqa: E402
    TILE_PX,
    contact_sheet,
    crown_rows,
    draw_tile,
    sample_rows,
    summary,
)
from fair_trees.decode import Detections  # noqa: E402
from fair_trees.evaluate import EvalUnit  # noqa: E402
from fair_trees.preprocess import array_mosaic  # noqa: E402

GSD = 0.1
# 1.0 m, 2.0 m, 2.0 m and 5.0 m truth crowns
GT = np.array([[0, 0, 10, 10], [50, 50, 70, 70], [100, 100, 120, 120], [150, 150, 200, 200]], np.float32)


def _dets(boxes, scores):
    return Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32))


def test_crown_rows_cover_small_crowns_only_with_match_iou_and_centre_score():
    # a hit on crown 0, a faint box over crown 1's centre, a kept box too big around crown 2
    dets = _dets([[0, 0, 10, 10], [55, 55, 65, 65], [90, 90, 130, 130]], [0.9, 0.1, 0.5])
    rows = crown_rows(dets, GT, GSD, threshold=0.2)
    assert [r["index"] for r in rows] == [0, 1, 2]  # the 5 m crown gets no row
    assert [r["bin"] for r in rows] == [0, 1, 1]
    assert [r["matched"] for r in rows] == [True, False, False]
    assert rows[1]["best_iou"] == 0.0 and rows[1]["centre_score"] == pytest.approx(0.1)
    assert rows[2]["best_iou"] == pytest.approx(0.25) and rows[2]["centre_score"] == pytest.approx(0.5)


def test_crown_rows_without_detections_are_all_misses():
    rows = crown_rows(_dets([], []), GT, GSD, threshold=0.2)
    assert [(r["matched"], r["best_iou"], r["centre_score"]) for r in rows] == [(False, 0.0, 0.0)] * 3


def test_sample_rows_splits_over_bins_and_stops_at_a_small_pool():
    rows = [{"bin": 0, "side_m": 1.0 + i / 10} for i in range(10)] + [{"bin": 1, "side_m": 2.0}]
    picked = sample_rows(rows, 6, np.random.default_rng(0))
    assert [r["bin"] for r in picked] == [0, 0, 0, 1]
    assert picked == sorted(picked, key=lambda r: r["side_m"])


def test_summary_splits_misses_into_located_and_faint():
    rows = [
        {"bin": 1, "matched": True, "best_iou": 0.6, "centre_score": 0.9},
        {"bin": 1, "matched": False, "best_iou": 0.25, "centre_score": 0.5},  # located by a kept box
        {"bin": 1, "matched": False, "best_iou": 0.0, "centre_score": 0.1},  # faint
        {"bin": 1, "matched": False, "best_iou": 0.0, "centre_score": 0.0},  # nothing
    ]
    small, mid = summary(rows, threshold=0.2)
    assert small == {
        "bin_m": [0.0, 1.5],
        "n_gt": 0,
        "recall": None,
        "misses_located": None,
        "misses_faint": None,
        "misses_median_best_iou": None,
    }
    assert mid["n_gt"] == 4 and mid["recall"] == 0.25 and mid["bin_m"] == [1.5, 2.5]
    assert mid["misses_located"] == pytest.approx(1 / 3) and mid["misses_faint"] == pytest.approx(1 / 3)


def test_tiles_draw_the_crown_on_a_sheet():
    unit = EvalUnit("u1", array_mosaic(np.full((256, 256, 3), 120, np.uint8), gsd_m=GSD), GT)
    row = crown_rows(_dets([], []), GT, GSD, threshold=0.2)[1]
    tile = draw_tile(unit, _dets([], []), 0.2, row, "1  2.0 m")
    assert tile.size == (TILE_PX, TILE_PX)
    assert (255, 59, 48) in {c for _, c in tile.getcolors(TILE_PX * TILE_PX)}  # the red crown box
    sheet = contact_sheet([tile] * 7, "title", cols=6)
    assert sheet.size[1] > 2 * TILE_PX and contact_sheet([], "empty").size[0] > 6 * TILE_PX
