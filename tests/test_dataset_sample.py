import dataclasses

import numpy as np

from fair_trees.dataset import RegionCache, SampleSpec, load_regions, load_sample, plan_epoch
from fair_trees.georef import true_gsd
from fair_trees.params import AugmentParams, TrainParams
from fair_trees.preprocess import mosaic_chips

NO_AUG = TrainParams(augment=AugmentParams(downscale=False))
SIZE_200 = 200 * true_gsd(20, 0.0)  # 200 px at z20 on the equator


def _spec(cx=128.0, cy=128.0, size_m=SIZE_200, seed=0, region="r"):
    return SampleSpec(region, 20, (cx, cy), size_m, seed)


def test_forty_percent_rule_and_ignore(write_region):
    boxes = [(10, 50, 60, 100), (0, 120, 40, 160), (100, 100, 140, 130), (150, 150, 200, 200)]
    d = write_region("r", tiles=((0, 0),), boxes_px=boxes, ignore_px=[(140, 140, 210, 210)])
    cache = RegionCache(load_regions(d.parent))
    rgb, kept = load_sample(_spec(), cache, NO_AUG)
    assert rgb.shape == (200, 200, 3) and rgb.dtype == np.uint8
    np.testing.assert_allclose(kept, [[0, 22, 32, 72], [72, 72, 112, 102]], atol=1e-3)
    assert (rgb[112:182, 112:182] == 0).all()  # the ignore polygon is blacked out


def test_crop_matches_mosaic_across_tiles(write_region):
    d = write_region("r", tiles=((0, 0), (1, 0), (0, 1), (1, 1)))
    regions = load_regions(d.parent)
    rgb, _ = load_sample(_spec(cx=256.0, cy=256.0), RegionCache(regions), NO_AUG)
    [m] = mosaic_chips(regions[0].chips[20])
    np.testing.assert_array_equal(rgb, m.rgb[156:356, 156:356])


def test_masked_pixels_stay_black_after_jitter(write_region):
    alpha = np.full((256, 256), 255, np.uint8)
    alpha[:, :100] = 0
    d = write_region("r", tiles=((0, 0),), alpha=alpha)
    cache = RegionCache(load_regions(d.parent))
    cfg = TrainParams(augment=AugmentParams(downscale=True, colour_jitter=True))
    for seed in range(5):
        rgb, _ = load_sample(_spec(seed=seed), cache, cfg)
        assert (rgb[:, :72] == 0).all() and rgb[:, 72:].any()


def test_load_sample_is_deterministic(write_region):
    d = write_region("r", tiles=((0, 0), (1, 0)), boxes_px=[(30, 30, 60, 70)])
    regions = load_regions(d.parent)
    cfg = TrainParams(augment=AugmentParams(flips=True, rot90=True, colour_jitter=True))
    spec = plan_epoch(regions, dataclasses.replace(cfg, samples_per_epoch=1), 0, 0)[0]
    a = load_sample(spec, RegionCache(regions), cfg)
    b = load_sample(spec, RegionCache(regions), cfg)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
