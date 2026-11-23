import csv

import numpy as np
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("scipy")

from affine import Affine  # noqa: E402

from experiments.gold_score import (  # noqa: E402
    _calibrated,
    global_to_mosaic,
    load_gold,
    place,
    summary,
    tree_row,
)
from fair_trees.evaluate import EvalUnit  # noqa: E402
from fair_trees.georef import tile_bounds_3857  # noqa: E402
from fair_trees.preprocess import Mosaic  # noqa: E402

X, Y, Z = 485900, 499600, 20
GSD = 0.1477


def _unit(gt=(), ignore=None, x=X):
    minx, _, maxx, maxy = tile_bounds_3857(x, Y, Z)
    res = (maxx - minx) / 256
    mosaic = Mosaic(
        np.zeros((256, 256, 3), np.uint8),
        np.ones((256, 256), bool),
        Affine(res, 0, minx, 0, -res, maxy),
        Z,
        GSD,
    )
    return EvalUnit("u", mosaic, np.asarray(gt, np.float32).reshape(-1, 4), ignore)


def _gold(gold_id, px, py, box=None, status="box", diameter=1.0):
    gx, gy = X * 256 + px, Y * 256 + py
    return {
        "gold_id": gold_id,
        "split": "validation",
        "stratum": "<1.5",
        "diameter_m": diameter,
        "status": status,
        "z": Z,
        "point": (gx, gy),
        "box": None if box is None else (gx + box[0], gy + box[1], gx + box[2], gy + box[3]),
    }


def test_global_pixels_land_on_the_mosaic_grid():
    u = _unit()
    np.testing.assert_allclose(
        global_to_mosaic([[X * 256 + 10.5, Y * 256 + 200.0]], Z, u.mosaic), [[10.5, 200.0]], atol=1e-6
    )
    # one zoom down: a global pixel spans two mosaic pixels
    np.testing.assert_allclose(global_to_mosaic([[X * 128, Y * 128]], Z - 1, u.mosaic), [[0, 0]], atol=1e-6)


def test_place_skips_points_off_the_unit_invalid_or_ignored():
    ignore = np.zeros((256, 256), bool)
    ignore[:, 200:] = True
    units = [_unit(x=X + 5), _unit(ignore=ignore)]
    gold = [_gold("a", 50, 50, box=(-10, -10, 10, 10)), _gold("b", 220, 50), _gold("c", 300, 50)]
    placed = place(gold, units)
    assert [(i, r["gold_id"]) for i, r in placed] == [(1, "a")]
    np.testing.assert_allclose(placed[0][1]["box_px"], [40, 40, 60, 60], atol=1e-6)


def test_tree_row_scores_against_drawn_box_and_tag():
    # drawn 20 px (~3 m) crown; tag box 7 px (1 m); the kept box matches the drawn crown, not the tag
    row = _gold("a", 50, 50) | {
        "point_px": np.array([50.0, 50.0]),
        "box_px": np.array([40.0, 40.0, 60.0, 60.0]),
    }
    tag = [[46.5, 46.5, 53.5, 53.5]]
    out = tree_row([[41, 41, 61, 61]], tag, row, GSD)
    assert out["gold_hit"] and not out["tag_hit"] and out["point_hit"]
    assert out["size_ratio"] == pytest.approx(1.0)
    assert not out["tag_band"] and out["calibrated_band"]  # 1 m tag -> 2.5 m: 17.5 px against 20 px
    assert out["gold_side_m"] == pytest.approx(20 * GSD)


def test_tree_row_without_a_nearby_tag_or_box():
    row = _gold("a", 50, 50) | {
        "point_px": np.array([50.0, 50.0]),
        "box_px": np.array([40.0, 40.0, 60.0, 60.0]),
    }
    out = tree_row(np.zeros((0, 4)), [[100, 100, 110, 110]], row, GSD)
    assert not out["gold_hit"] and not out["point_hit"]
    assert "tag_hit" not in out and "size_ratio" not in out
    not_tree = _gold("b", 50, 50, status="not_a_tree") | {"point_px": np.array([50.0, 50.0]), "box_px": None}
    assert tree_row([[45, 45, 55, 55]], [], not_tree, GSD) == {
        "gold_id": "b",
        "split": "validation",
        "stratum": "<1.5",
        "status": "not_a_tree",
        "point_hit": True,
    }


def test_calibration_only_resizes_the_listed_tags():
    tag = np.array([10.0, 10.0, 20.0, 20.0])
    np.testing.assert_allclose(_calibrated(tag, 2.0), [5.625, 5.625, 24.375, 24.375])
    np.testing.assert_allclose(_calibrated(tag, 2.231), tag)


def test_summary_shares_by_stratum():
    rows = [
        {
            "stratum": "<1.5",
            "status": "box",
            "point_hit": True,
            "gold_hit": True,
            "tag_hit": False,
            "size_ratio": 1.2,
        },
        {"stratum": "<1.5", "status": "box", "point_hit": False, "gold_hit": False},
        {"stratum": ">=6", "status": "not_a_tree", "point_hit": True},
    ]
    s = summary(rows)
    assert s["<1.5"]["n_box"] == 2 and s["<1.5"]["gold_recall"] == 0.5 and s["<1.5"]["tag_recall"] == 0.0
    assert s["<1.5"]["size_ratio_median"] == pytest.approx(1.2) and s["<1.5"]["n_with_tag"] == 1
    assert s[">=6"]["n_box"] == 0 and s[">=6"]["not_a_tree_detected"] == 1.0
    assert set(s) == {"all", "<1.5", ">=6"} and s["all"]["n_box"] == 2


def test_load_gold_joins_key_and_boxes_in_global_pixels(tmp_path):
    key = tmp_path / "manifest.csv"
    with key.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["gold_id", "split", "stratum", "diameter_m", "z", "crop_x0", "crop_y0", "point_x", "point_y"]
        )
        w.writerow(["g1", "test", "<1.5", "1.0", "20", "1000", "2000", "68.5", "68.5"])
        w.writerow(["g2", "train", "unsized", "", "20", "0", "0", "68.5", "68.5"])
    boxes = tmp_path / "boxes.csv"
    with boxes.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "status", "x0", "y0", "x1", "y1", "set"])
        w.writerow(["g1", "box", "60", "61", "80", "81", "items"])
        w.writerow(["g1", "box", "50", "51", "90", "91", "items_second"])  # second annotator: ignored
        w.writerow(["g2", "not_a_tree", "", "", "", "", "items"])
    g1, g2 = load_gold(key, boxes)
    assert g1["point"] == (1068.5, 2068.5) and g1["box"] == (1060.0, 2061.0, 1080.0, 2081.0)
    assert g2["box"] is None and np.isnan(g2["diameter_m"]) and g2["status"] == "not_a_tree"


def test_score_units_decodes_once_and_scores_placed_trees(fake_session):
    from experiments.gold_score import score_units
    from fair_trees.params import InferenceParams
    from fair_trees.serve import ort_run_fn

    run = ort_run_fn(fake_session({(50 * 100 + 50) * 9 + 3: 5.0}))  # box ~[122.9, 133.1], score 0.9933
    unit = _unit(gt=[[126, 126, 130, 130]])
    unit.mosaic.rgb[:] = 120
    gold = [_gold("a", 128, 128, box=(-5, -5, 5, 5)), _gold("b", 300, 50)]
    out = score_units(run, [unit], InferenceParams(), 0.2, gold)
    assert out["n_gold"] == 2 and out["n_placed"] == 1 and out["placed_by_split"] == {"validation": 1}
    (tree,) = out["trees"]
    assert tree["gold_hit"] and tree["point_hit"] and not tree["tag_hit"]
    assert out["summary"]["all"]["gold_recall"] == 1.0
