import dataclasses
import logging
import pickle

import numpy as np
import pytest

from fair_trees.dataset import (
    RegionCache,
    SampleSpec,
    any_label_on_chips,
    load_regions,
    plan_epoch,
    sampler_stats,
)
from fair_trees.params import TrainParams

CFG = TrainParams(samples_per_epoch=200)


def _cfg(**kw):
    return dataclasses.replace(CFG, **kw)


def test_plan_is_pure_and_epoch_dependent(write_region):
    d = write_region("a", tiles=((0, 0), (1, 0), (0, 1), (1, 1)), boxes_px=[(10, 10, 30, 30)])
    regions = load_regions(d.parent)
    a = plan_epoch(regions, CFG, 42, 0)
    assert a == plan_epoch(regions, CFG, 42, 0)
    assert a != plan_epoch(regions, CFG, 42, 1) and a != plan_epoch(regions, CFG, 43, 0)
    assert len(a) == 200 and all(isinstance(s, SampleSpec) and s.zoom == 20 for s in a)
    assert all(30.0 <= s.size_m <= 80.0 for s in a)


def test_samples_per_epoch_defaults_to_chip_count(write_region):
    d = write_region("a", tiles=((0, 0), (1, 0), (0, 1)))
    assert len(plan_epoch(load_regions(d.parent), _cfg(samples_per_epoch=None), 0, 0)) == 3


@pytest.mark.parametrize("alpha, expected", [(0.0, 0.5), (1.0, 0.25)])
def test_source_temperature(write_region, tmp_path, alpha, expected):
    root = tmp_path / "regions"
    write_region("a", tiles=((0, 0),), source="a", root=root)
    write_region("b", tiles=((4, 0), (5, 0), (6, 0)), source="b", root=root)  # 3x the valid area
    specs = plan_epoch(load_regions(root), _cfg(samples_per_epoch=4000, source_alpha=alpha), 0, 0)
    assert np.mean([s.region_id == "a" for s in specs]) == pytest.approx(expected, abs=0.03)


@pytest.mark.parametrize("alpha, expected", [(0.0, 0.5), (1.0, 0.25)])
def test_biome_temperature(write_region, tmp_path, alpha, expected):
    root = tmp_path / "regions"
    write_region("a", tiles=((0, 0),), biome="x", root=root)
    write_region("b", tiles=((4, 0), (5, 0), (6, 0)), biome="y", root=root)
    specs = plan_epoch(load_regions(root), _cfg(samples_per_epoch=4000, biome_alpha=alpha), 0, 0)
    assert np.mean([s.region_id == "a" for s in specs]) == pytest.approx(expected, abs=0.03)


def test_region_repeat_cap(write_region, tmp_path):
    root = tmp_path / "regions"
    for i, area in enumerate([1.0, 1.0, 1000.0]):
        write_region(f"r{i}", tiles=((4 * i, 0),), root=root, area=area)
    regions = load_regions(root)
    specs = plan_epoch(regions, _cfg(samples_per_epoch=6, region_repeat_cap=2), 0, 0)
    assert sorted(np.unique([s.region_id for s in specs], return_counts=True)[1]) == [2, 2, 2]
    with pytest.raises(ValueError, match="region_repeat_cap"):
        plan_epoch(regions, _cfg(samples_per_epoch=7, region_repeat_cap=2), 0, 0)


def test_crops_never_leave_present_tiles(write_region):
    d = write_region("L", tiles=((0, 0), (1, 0), (0, 1)))  # tile (1, 1) is absent
    regions = load_regions(d.parent)
    cache = RegionCache(regions)
    for s in plan_epoch(regions, _cfg(samples_per_epoch=300), 1, 0, cache=cache):
        w = cache.crop_window(s)
        g = cache.geom(s.region_id, s.zoom)
        cols = range(w.x // 256, (w.x + w.width - 1) // 256 + 1)
        rows = range(w.y // 256, (w.y + w.height - 1) // 256 + 1)
        assert {(c, r) for c in cols for r in rows} <= {(0, 0), (1, 0), (0, 1)}
        assert 0 <= w.x and w.x + w.width <= g.width and 0 <= w.y and w.y + w.height <= g.height


def test_mostly_masked_crops_are_rejected(write_region):
    d = write_region("m", tiles=((0, 0), (1, 0)), ignore_px=[(256, 0, 512, 256)])
    regions = load_regions(d.parent)
    cache = RegionCache(regions)
    specs = plan_epoch(regions, _cfg(samples_per_epoch=300), 2, 0, cache=cache)
    assert max(cache.masked_fraction(s) for s in specs) <= 0.8


def test_empty_crop_cap(write_region):
    d = write_region("e", tiles=((0, 0), (1, 0), (2, 0)), boxes_px=[(20, 100, 40, 120)])
    regions = load_regions(d.parent)
    cache = RegionCache(regions)

    def share(specs):
        return np.mean([cache.is_empty(s) for s in specs])

    assert share(plan_epoch(regions, _cfg(samples_per_epoch=400), 3, 0, cache=cache)) > 0.5
    capped = plan_epoch(regions, _cfg(samples_per_epoch=400, empty_crop_cap=0.2), 3, 0, cache=cache)
    assert share(capped) <= 0.25


def test_zoom_weights_are_renormalised(write_region):
    write_region("z", tiles=((0, 0), (1, 0), (0, 1), (1, 1)), z=20)
    d = write_region("z", tiles=((0, 0),), z=19)
    specs = plan_epoch(load_regions(d.parent), _cfg(samples_per_epoch=3000), 0, 0)
    assert np.mean([s.zoom == 20 for s in specs]) == pytest.approx(2 / 3, abs=0.03)


def test_regions_without_a_weighted_zoom_are_skipped(write_region, caplog):
    write_region("a", tiles=((0, 0),))
    d = write_region("only-z22", tiles=((0, 0),), z=22)
    regions = load_regions(d.parent)
    with caplog.at_level(logging.WARNING, logger="fair_trees.dataset"):
        specs = plan_epoch(regions, _cfg(samples_per_epoch=20), 0, 0)
    assert {s.region_id for s in specs} == {"a"} and "only-z22" in caplog.text
    with pytest.raises(ValueError, match="zoom"):
        plan_epoch([r for r in regions if r.id == "only-z22"], CFG, 0, 0)


def test_sampler_stats(write_region):
    d = write_region("s", tiles=((0, 0), (1, 0)), boxes_px=[(20, 20, 40, 40)])
    regions = load_regions(d.parent)
    cache = RegionCache(regions)
    stats = sampler_stats(plan_epoch(regions, _cfg(samples_per_epoch=50), 0, 0, cache=cache), cache)
    assert stats["sampler/draws/source/src"] == 50 and stats["sampler/draws/biome/b1"] == 50
    assert stats["sampler/draws/zoom/20"] == 50
    assert stats["sampler/region_repeat_max"] == 50 and stats["sampler/region_repeat_mean"] == 50
    assert 0 <= stats["sampler/empty_share/src"] <= 1
    assert all(isinstance(v, float) for v in stats.values())


def test_cache_pickles_without_rasters(write_region):
    d = write_region("p", tiles=((0, 0),))
    regions = load_regions(d.parent)
    cache = RegionCache(regions)
    spec = plan_epoch(regions, _cfg(samples_per_epoch=1), 0, 0, cache=cache)[0]
    cache.chip(regions[0].chips[20][0])
    clone = pickle.loads(pickle.dumps(cache))
    assert clone.crop_window(spec) == cache.crop_window(spec)


@pytest.mark.parametrize(
    "boxes_px, groups_px, classes, expected",
    [
        ([(10, 10, 30, 30)], [], ("tree",), True),
        ([(300, 10, 320, 30)], [], ("tree",), False),  # in the frame, on the absent middle tile
        ([(500, 10, 520, 30)], [], ("tree",), True),  # from the absent tile into the third one
        ([(300, 10, 320, 30)], [(10, 10, 60, 60)], ("tree",), False),  # a group is no label of a tree model
        ([(300, 10, 320, 30)], [(10, 10, 60, 60)], ("tree", "tree_group"), True),
    ],
)
def test_any_label_on_chips_counts_labels_on_present_tiles(
    write_region, boxes_px, groups_px, classes, expected
):
    d = write_region(
        "h", tiles=((0, 0), (2, 0)), boxes_px=boxes_px, groups_px=groups_px, classes=("tree", "tree_group")
    )
    regions = load_regions(d.parent)
    assert any_label_on_chips(regions, classes, RegionCache(regions)) is expected
