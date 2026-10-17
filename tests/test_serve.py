import logging
import shutil

import numpy as np
import pytest
from pyproj import Transformer

from fair_trees.georef import tile_bounds_3857
from fair_trees.params import InferenceParams
from fair_trees.preprocess import mosaic_chips
from fair_trees.serve import ort_run_fn, predict_mosaic, predict_session

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
