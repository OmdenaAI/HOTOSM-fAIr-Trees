import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from fair_trees.georef import tile_bounds_3857, tile_bounds_4326, true_gsd
from fair_trees.preprocess import (
    Window,
    array_mosaic,
    is_blank,
    mosaic_chips,
    preprocess,
    read_chip,
    window_grid,
)

X0 = Y0 = 2**19
GSD_EQ = true_gsd(20, 0.0)  # 0.149291 m; 40 m -> 268 px


def test_window_grid_overlap_and_shift_back():
    wins = window_grid(512, 512, GSD_EQ, 40.0, 0.5)
    assert sorted({w.x for w in wins}) == [0, 134, 244]
    assert sorted({w.y for w in wins}) == [0, 134, 244]
    assert len(wins) == 9 and all((w.width, w.height) == (268, 268) for w in wins)
    assert wins[:2] == [Window(0, 0, 268, 268), Window(134, 0, 268, 268)]


def test_window_larger_than_mosaic_is_clipped():
    assert window_grid(256, 256, GSD_EQ, 40.0, 0.5) == [Window(0, 0, 256, 256)]


def test_window_grid_non_square():
    wins = window_grid(256, 600, GSD_EQ, 40.0, 0.0)
    assert [w.x for w in wins] == [0, 268, 332] and {w.height for w in wins} == {256}


@pytest.mark.parametrize("h, w, overlap", [(700, 900, 0.25), (300, 1000, 0.5), (269, 269, 0.0)])
def test_window_grid_covers_every_pixel(h, w, overlap):
    covered = np.zeros((h, w), bool)
    for win in window_grid(h, w, GSD_EQ, 40.0, overlap):
        assert win.x + win.width <= w and win.y + win.height <= h
        covered[win.y : win.y + win.height, win.x : win.x + win.width] = True
    assert covered.all()


def test_read_chip(write_oam_chip):
    rgb, valid, xyz = read_chip(write_oam_chip())
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    assert valid.all() and xyz == (X0, Y0, 20)


def test_read_chip_alpha_marks_invalid(write_oam_chip):
    alpha = np.full((256, 256), 255, np.uint8)
    alpha[:, :100] = 0
    rgb, valid, _ = read_chip(write_oam_chip(count=4, alpha=alpha))
    assert not valid[:, :100].any() and valid[:, 100:].all()
    assert (rgb[:, :100] == 0).all()


def test_read_chip_black_is_invalid(write_oam_chip):
    rgb, valid, _ = read_chip(write_oam_chip(fill=0))
    assert not valid.any()


def test_read_chip_rejects(write_oam_chip):
    with pytest.raises(ValueError, match="3"):
        read_chip(write_oam_chip(count=1))
    with pytest.raises(ValueError, match="uint8"):
        read_chip(write_oam_chip(X0 + 1, dtype="uint16"))
    with pytest.raises(ValueError, match="OAM"):
        read_chip(write_oam_chip(name="photo.tif"))


def test_read_chip_sidecar_check(write_oam_chip, write_sidecar):
    path = write_oam_chip()
    w, s, e, n = tile_bounds_4326(X0, Y0, 20)
    write_sidecar(path, (w, s, e, n), 256, 256)
    read_chip(path)
    write_sidecar(path, (w + 0.001, s, e + 0.001, n), 256, 256)
    with pytest.raises(ValueError, match="sidecar"):
        read_chip(path)
    read_chip(path, use_sidecar=False)


def test_mosaic_l_shape(write_oam_chip):
    paths = [write_oam_chip(X0, Y0), write_oam_chip(X0 + 1, Y0), write_oam_chip(X0, Y0 + 1)]
    [m] = mosaic_chips(paths)
    assert m.rgb.shape == (512, 512, 3) and m.z == 20
    assert not m.valid[256:, 256:].any() and m.valid[:256, :].all() and m.valid[256:, :256].all()
    assert (m.rgb[256:, 256:] == 0).all()
    np.testing.assert_array_equal(m.rgb[:256, :256], read_chip(paths[0])[0])
    minx, _, _, maxy = tile_bounds_3857(X0, Y0, 20)
    assert m.transform_3857 @ (0, 0) == pytest.approx((minx, maxy))
    centre_lat = tile_bounds_4326(X0, Y0, 20)[1]  # the row boundary is the mosaic centre
    assert m.gsd_m == pytest.approx(true_gsd(20, centre_lat))


def test_mosaic_disjoint_and_diagonal(write_oam_chip, tmp_path):
    ms = mosaic_chips([write_oam_chip(X0 + 5, Y0), write_oam_chip(X0, Y0)])
    assert len(ms) == 2 and all(m.rgb.shape == (256, 256, 3) for m in ms)
    assert ms[0].transform_3857 @ (0, 0) == pytest.approx(tile_bounds_3857(X0, Y0, 20)[::3])
    diag = tmp_path / "diag"
    diag.mkdir()
    ms = mosaic_chips(
        [write_oam_chip(X0, Y0, directory=diag), write_oam_chip(X0 + 1, Y0 + 1, directory=diag)]
    )
    assert len(ms) == 1 and ms[0].rgb.shape == (512, 512, 3)


def test_mosaic_mixed_zooms(write_oam_chip):
    ms = mosaic_chips([write_oam_chip(X0, Y0, 20), write_oam_chip(X0 // 2, Y0 // 2, 19)])
    assert sorted(m.z for m in ms) == [19, 20]


def test_mosaic_skips_bad_chips(write_oam_chip, tmp_path, caplog):
    good = write_oam_chip()
    bad = write_oam_chip(X0 + 1, Y0, count=1)
    (tmp_path / "notes.txt").write_text("x")
    renamed = tmp_path / "photo.tif"
    shutil.copy(good, renamed)
    with caplog.at_level(logging.WARNING, logger="fair_trees.preprocess"):
        ms = mosaic_chips([good, bad, tmp_path / "notes.txt", renamed])
    assert len(ms) == 1 and ms[0].rgb.shape == (256, 256, 3) and ms[0].valid.all()
    assert "photo.tif" in caplog.text and bad.name in caplog.text
    assert "notes.txt" not in caplog.text


def test_mosaic_mixed_tile_sizes_raise(write_oam_chip):
    with pytest.raises(ValueError, match="size"):
        mosaic_chips([write_oam_chip(X0, Y0), write_oam_chip(X0 + 1, Y0, size=512)])


def test_array_mosaic_and_is_blank():
    m = array_mosaic(np.ones((10, 20, 3), np.uint8), gsd_m=0.1)
    assert m.valid.all() and m.z == -1 and m.gsd_m == 0.1
    valid = np.ones((100, 100), bool)
    valid[:, :70] = False
    m = array_mosaic(np.full((100, 100, 3), 5, np.uint8), gsd_m=0.1, valid=valid)
    assert (m.rgb[:, :70] == 0).all()
    assert is_blank(m, Window(0, 0, 100, 100), 0.5)
    assert not is_blank(m, Window(0, 0, 100, 100), 0.3)  # exactly at the threshold is not blank


def test_preprocess_single_chip_contract(write_oam_chip):
    path = write_oam_chip()
    batch, scale = preprocess(path, 256)
    assert batch.shape == (1, 3, 800, 800) and scale.resized_width == 800
    with pytest.raises(ValueError, match="256"):
        preprocess(path, 512)


def test_mosaic_duplicate_tile_keeps_first_and_warns(write_oam_chip, tmp_path, caplog):
    other = tmp_path / "other"
    other.mkdir()
    first = write_oam_chip()
    second = write_oam_chip(fill=77, directory=other)
    with caplog.at_level(logging.WARNING, logger="fair_trees.preprocess"):
        [m] = mosaic_chips([first, second])
    assert m.rgb.shape == (256, 256, 3)
    np.testing.assert_array_equal(m.rgb, read_chip(first)[0])
    assert "duplicate" in caplog.text and f"{X0}-{Y0}-20" in caplog.text


def test_malformed_sidecar_is_a_value_error_and_skipped(write_oam_chip, tmp_path, caplog):
    path = write_oam_chip()
    Path(f"{path}.aux.xml").write_text("<<not xml")
    with pytest.raises(ValueError, match="sidecar"):
        read_chip(path)
    with caplog.at_level(logging.WARNING, logger="fair_trees.preprocess"):
        assert mosaic_chips([path]) == []
    assert path.name in caplog.text


def test_non_square_chip_is_skipped_not_fatal(write_oam_chip, tmp_path, caplog):
    import rasterio

    good = write_oam_chip()
    odd = write_oam_chip(X0 + 1, Y0)
    with rasterio.open(odd) as src:
        data = src.read()[:, :, :255]
    profile = {"driver": "GTiff", "height": 256, "width": 255, "count": 3, "dtype": "uint8"}
    with rasterio.open(odd, "w", crs="EPSG:3857", transform=Affine(2, 0, 100, 0, -2, 100), **profile) as dst:
        dst.write(data)
    with caplog.at_level(logging.WARNING, logger="fair_trees.preprocess"):
        ms = mosaic_chips([good, odd])
    assert len(ms) == 1 and ms[0].rgb.shape == (256, 256, 3)
    assert odd.name in caplog.text


def test_sidecar_directory_is_not_a_sidecar(write_oam_chip):
    path = write_oam_chip()
    Path(f"{path}.aux.xml").mkdir()
    read_chip(path)


def test_mosaic_chips_collects_skip_reasons(write_oam_chip):
    skipped = []
    assert mosaic_chips([write_oam_chip(count=1)], skipped=skipped) == []
    assert [name for name, _ in skipped] == ["OAM-524288-524288-20.tif"]
    assert "3 bands" in skipped[0][1]
