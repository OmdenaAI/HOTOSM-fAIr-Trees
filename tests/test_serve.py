import logging
import shutil

import numpy as np
import pytest
from pyproj import Transformer

from fair_trees.georef import tile_bounds_3857
from fair_trees.params import InferenceParams
from fair_trees.preprocess import mosaic_chips, window_grid
from fair_trees.serve import ort_run_fn, predict_mosaic, predict_mosaic_classes, predict_session

X0 = Y0 = 2**19
CENTRE_ANCHOR_800 = (50 * 100 + 50) * 9 + 3  # maps to the centre of a 256 px window
EMPTY = {"type": "FeatureCollection", "features": []}


def test_zero_logits_give_empty_result(write_oam_chip, fake_session, tmp_path):
    write_oam_chip()
    s = fake_session()
    assert predict_session(s, tmp_path, None) == EMPTY
    assert s.calls == [(1, 3, 800, 800)]


def test_planted_logit_lands_on_tile_centre(write_oam_chip, fake_session, tmp_path):
    write_oam_chip()
    fc = predict_session(fake_session({CENTRE_ANCHOR_800: 5.0}), tmp_path, {})
    [f] = fc["features"]
    assert f["properties"] == {"class": 1, "score": 0.9933}
    ring = np.array(f["geometry"]["coordinates"][0])
    minx, miny, maxx, maxy = tile_bounds_3857(X0, Y0, 20)
    lon, lat = Transformer.from_crs(3857, 4326, always_xy=True).transform(
        (minx + maxx) / 2, (miny + maxy) / 2
    )
    assert (ring[0, 0] + ring[2, 0]) / 2 == pytest.approx(lon, abs=1e-7)
    assert (ring[0, 1] + ring[2, 1]) / 2 == pytest.approx(lat, abs=1e-7)


def test_one_run_per_non_blank_window(write_oam_chip, fake_session):
    [m] = mosaic_chips([write_oam_chip(X0, Y0), write_oam_chip(X0 + 1, Y0), write_oam_chip(X0, Y0 + 1)])
    s = fake_session()
    predict_mosaic(ort_run_fn(s), m, InferenceParams(min_valid_fraction=0.5))
    # 9 windows of 268 px at x, y in {0, 134, 244}; (134, 244), (244, 134), (244, 244) are < 50% valid
    assert len(s.calls) == 6


def test_k2_session_emits_both_classes(write_oam_chip, fake_session, tmp_path):
    write_oam_chip()
    s = fake_session({(CENTRE_ANCHOR_800, 0): 5.0, (CENTRE_ANCHOR_800, 1): 4.0}, num_classes=2)
    fc = predict_session(s, tmp_path, {})
    assert sorted((f["properties"]["class"], f["properties"]["score"]) for f in fc["features"]) == [
        (1, 0.9933),
        (2, 0.982),
    ]
    assert len(s.calls) == 1  # the 40 m and 120 m windows both clamp to the one tile: run once


def test_k2_session_without_groups_gives_trees_only(write_oam_chip, fake_session, tmp_path):
    write_oam_chip()
    fc = predict_session(fake_session({CENTRE_ANCHOR_800: 5.0}, num_classes=2), tmp_path, {})
    assert [f["properties"]["class"] for f in fc["features"]] == [1]


def test_group_windows_run_once_per_distinct_size(write_oam_chip, fake_session):
    [m] = mosaic_chips([write_oam_chip(X0 + i, Y0 + j) for i in range(4) for j in range(4)])
    p = InferenceParams()
    trees = window_grid(*m.rgb.shape[:2], m.gsd_m, p.window_m, p.window_overlap)
    groups = window_grid(*m.rgb.shape[:2], m.gsd_m, p.group_window_m, p.window_overlap)
    assert len(groups) < len(trees)
    s = fake_session(num_classes=2)
    out = predict_mosaic_classes(ort_run_fn(s), m, p)
    assert len(out) == 2 and len(s.calls) == len(trees) + len(groups)
    s1 = fake_session(num_classes=2)
    assert len(predict_mosaic_classes(ort_run_fn(s1), m, p, (0,))) == 1 and len(s1.calls) == len(trees)
    s2 = fake_session()
    with pytest.raises(ValueError, match="class_index 1"):
        predict_mosaic_classes(ort_run_fn(s2), m, p, (1,))


def test_two_tile_groups_and_noise_files(write_oam_chip, fake_session, tmp_path, caplog):
    first = write_oam_chip(X0, Y0)
    write_oam_chip(X0 + 10, Y0)
    (tmp_path / "notes.txt").write_text("x")
    shutil.copy(first, tmp_path / "photo.tif")
    s = fake_session({CENTRE_ANCHOR_800: 5.0})
    with caplog.at_level(logging.WARNING):
        fc = predict_session(s, tmp_path, None)
    assert len(fc["features"]) == 2 and len(s.calls) == 2
    assert "photo.tif" in caplog.text


def test_mixed_zooms_warn(write_oam_chip, fake_session, tmp_path, caplog):
    write_oam_chip(X0, Y0, 20)
    write_oam_chip(X0 // 2 + 3, Y0 // 2, 19)
    s = fake_session({CENTRE_ANCHOR_800: 5.0})
    with caplog.at_level(logging.WARNING, logger="fair_trees.serve"):
        fc = predict_session(s, tmp_path, None)
    # z20: one 256 px window; z19: 40 m = 134 px windows at x, y in {0, 67, 122} -> 9 windows
    assert len(s.calls) == 10 and len(fc["features"]) == 10
    assert "zoom" in caplog.text


def test_empty_directory(fake_session, tmp_path):
    s = fake_session()
    assert predict_session(s, tmp_path, None) == EMPTY and s.calls == []


def test_all_black_chip_is_skipped(write_oam_chip, fake_session, tmp_path):
    write_oam_chip(fill=0)
    s = fake_session({CENTRE_ANCHOR_800: 5.0})
    assert predict_session(s, tmp_path, None) == EMPTY and s.calls == []


def test_bbox_and_params_are_applied(write_oam_chip, fake_session, tmp_path):
    write_oam_chip()
    assert (
        predict_session(fake_session({CENTRE_ANCHOR_800: 5.0}), tmp_path, None, bbox=(10, 10, 11, 11))
        == EMPTY
    )
    assert predict_session(fake_session({CENTRE_ANCHOR_800: 5.0}), tmp_path, {"score_thresh": 0.999}) == EMPTY


def test_single_file_input(write_oam_chip, fake_session):
    path = write_oam_chip()
    assert len(predict_session(fake_session({CENTRE_ANCHOR_800: 5.0}), path, None)["features"]) == 1


def test_request_exceeding_window_budget_is_rejected(write_oam_chip, fake_session, tmp_path):
    write_oam_chip()
    s = fake_session()
    with pytest.raises(ValueError, match=r"windows.*\d+.*\d+"):
        predict_session(s, tmp_path, {"window_m": 5, "window_overlap": 0.9})
    assert s.calls == []
    predict_session(s, tmp_path, {})
    assert len(s.calls) == 1


def test_only_bad_chips_raise_with_reason(write_oam_chip, fake_session, tmp_path):
    write_oam_chip(count=1)
    with pytest.raises(ValueError, match="3 bands"):
        predict_session(fake_session(), tmp_path, None)


def test_only_misnamed_chip_raises(write_oam_chip, fake_session, tmp_path):
    write_oam_chip(name="photo.tif")
    with pytest.raises(ValueError, match="photo.tif"):
        predict_session(fake_session(), tmp_path, None)


def test_directory_without_images_is_empty(fake_session, tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    assert predict_session(fake_session(), tmp_path, None) == EMPTY
