import dataclasses
import json

import numpy as np

from fair_trees.dataset import (
    RegionCache,
    SampleSpec,
    load_fair_dataset,
    load_regions,
    load_sample,
    plan_epoch,
)
from fair_trees.georef import true_gsd
from fair_trees.params import CLASSES, AugmentParams, TrainParams
from fair_trees.preprocess import mosaic_chips

NO_AUG = TrainParams(augment=AugmentParams(downscale=False))
SIZE_200 = 200 * true_gsd(20, 0.0)  # 200 px at z20 on the equator


def _spec(cx=128.0, cy=128.0, size_m=SIZE_200, seed=0, region="r"):
    return SampleSpec(region, 20, (cx, cy), size_m, seed)


def test_forty_percent_rule_and_ignore(write_region):
    boxes = [(10, 50, 60, 100), (0, 120, 40, 160), (100, 100, 140, 130), (150, 150, 200, 200)]
    d = write_region("r", tiles=((0, 0),), boxes_px=boxes, ignore_px=[(140, 140, 210, 210)])
    cache = RegionCache(load_regions(d.parent))
    s = load_sample(_spec(), cache, NO_AUG)
    assert s.rgb.shape == (200, 200, 3) and s.rgb.dtype == np.uint8
    np.testing.assert_allclose(s.boxes, [[0, 22, 32, 72], [72, 72, 112, 102]], atol=1e-3)
    assert (s.rgb[112:182, 112:182] == 0).all()  # the ignore polygon is blacked out


def test_masked_tree_points_are_blacked_out(write_oam_chip, tmp_path, geo):
    sized = {"natural": "tree", "diameter_crown": "6"}
    features = [
        geo.point_feature((60, 60), tags=sized, label=1),
        geo.point_feature((190, 60), tags=sized, label=1),
        geo.point_feature((128, 170), tags={"natural": "tree"}, label=1),  # no size: masked, as most have one
    ]
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    [region] = load_fair_dataset([write_oam_chip(2**19, 2**19)], labels)
    spec = SampleSpec(region.id, 20, (128.0, 128.0), 256 * true_gsd(20, 0.0), 0)  # the whole chip
    s = load_sample(spec, RegionCache([region]), NO_AUG)
    assert len(s.boxes) == 2
    assert (s.rgb[158:182, 116:140] == 0).all()  # a 4 m (26.8 px) square on the tree without a size
    assert (s.rgb[:150] > 0).all() and (s.rgb[190:] > 0).all()


def test_tree_only_region_sample_is_unchanged(write_region):
    d = write_region("r", tiles=((0, 0),), boxes_px=[(10, 50, 60, 100), (100, 100, 140, 130)])
    s = load_sample(_spec(), RegionCache(load_regions(d.parent)), NO_AUG)
    np.testing.assert_array_equal(s.labels, [0, 0])
    assert s.labels.dtype == np.int64
    assert s.unsupervised.shape == (len(CLASSES), 200, 200) and s.unsupervised.dtype == bool
    assert not s.unsupervised[0].any()


def test_unlabelled_class_is_unsupervised_everywhere(write_region):
    d = write_region("r", tiles=((0, 0),), boxes_px=[(10, 50, 60, 100)])
    s = load_sample(_spec(), RegionCache(load_regions(d.parent)), NO_AUG)
    assert s.unsupervised[1].all()


def test_group_label_gives_box_and_tree_unsupervised_inside(write_region):
    d = write_region("r", boxes_px=[(150, 60, 170, 80)], groups_px=[(40, 40, 120, 100)], classes=CLASSES)
    s = load_sample(_spec(), RegionCache(load_regions(d.parent)), NO_AUG)  # crop origin (28, 28)
    np.testing.assert_allclose(s.boxes, [[122, 32, 142, 52], [12, 12, 92, 72]], atol=1e-3)
    np.testing.assert_array_equal(s.labels, [0, 1])
    assert s.unsupervised[0][13:71, 13:91].all()
    assert not s.unsupervised[0][75:].any() and not s.unsupervised[0][:, 95:].any()
    assert not s.unsupervised[1].any()
    assert (s.rgb[13:71, 13:91] > 0).all()  # a group is not blacked out


def test_tree_stays_supervised_in_a_group_hole(write_region, geo):
    d = write_region("r", boxes_px=[(95, 95, 105, 105)], classes=CLASSES)
    outer = geo.px_to_4326([(40, 40), (40, 160), (160, 160), (160, 40), (40, 40)])
    hole = geo.px_to_4326([(80, 80), (120, 80), (120, 120), (80, 120), (80, 80)])
    labels = json.loads((d / "labels.geojson").read_text())
    group = {"type": "Polygon", "coordinates": [outer, hole]}
    labels["features"].append({"type": "Feature", "geometry": group, "properties": {"label": 2}})
    (d / "labels.geojson").write_text(json.dumps(labels))
    s = load_sample(_spec(), RegionCache(load_regions(d.parent)), NO_AUG)  # crop origin (28, 28)
    assert s.unsupervised[0][15:50, 15:50].all()  # inside the group, outside its hole
    assert not s.unsupervised[0][55:89, 55:89].any()  # the hole, where the tree is


def test_group_cut_by_crop_is_unsupervised(write_region):
    d = write_region("r", groups_px=[(20, 20, 620, 620)], classes=CLASSES)  # 3x the crop's side
    s = load_sample(_spec(), RegionCache(load_regions(d.parent)), NO_AUG)
    assert len(s.boxes) == 0
    assert s.unsupervised[1].all() and s.unsupervised[0].all()


def test_flip_and_rot_move_unsupervised_with_image(write_region):
    d = write_region("r", groups_px=[(40, 40, 120, 100)], classes=CLASSES)
    cache = RegionCache(load_regions(d.parent))
    cfg = TrainParams(augment=AugmentParams(downscale=False, flips=True, rot90=True))
    for seed in range(8):
        s = load_sample(_spec(seed=seed), cache, cfg)
        [(x1, y1, x2, y2)] = s.boxes.astype(int)
        assert s.unsupervised[0][y1 + 1 : y2 - 1, x1 + 1 : x2 - 1].all()
        assert s.unsupervised[0].sum() == (x2 - x1) * (y2 - y1)


def test_crop_matches_mosaic_across_tiles(write_region):
    d = write_region("r", tiles=((0, 0), (1, 0), (0, 1), (1, 1)))
    regions = load_regions(d.parent)
    s = load_sample(_spec(cx=256.0, cy=256.0), RegionCache(regions), NO_AUG)
    [m] = mosaic_chips(regions[0].chips[20])
    np.testing.assert_array_equal(s.rgb, m.rgb[156:356, 156:356])


def test_masked_pixels_stay_black_after_jitter(write_region):
    alpha = np.full((256, 256), 255, np.uint8)
    alpha[:, :100] = 0
    d = write_region("r", tiles=((0, 0),), alpha=alpha)
    cache = RegionCache(load_regions(d.parent))
    cfg = TrainParams(augment=AugmentParams(downscale=True, colour_jitter=True))
    for seed in range(5):
        rgb = load_sample(_spec(seed=seed), cache, cfg).rgb
        assert (rgb[:, :72] == 0).all() and rgb[:, 72:].any()


def test_load_sample_is_deterministic(write_region):
    d = write_region("r", tiles=((0, 0), (1, 0)), boxes_px=[(30, 30, 60, 70)])
    regions = load_regions(d.parent)
    cfg = TrainParams(augment=AugmentParams(flips=True, rot90=True, colour_jitter=True))
    spec = plan_epoch(regions, dataclasses.replace(cfg, samples_per_epoch=1), 0, 0)[0]
    a = load_sample(spec, RegionCache(regions), cfg)
    b = load_sample(spec, RegionCache(regions), cfg)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
