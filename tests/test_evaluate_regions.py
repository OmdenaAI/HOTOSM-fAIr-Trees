import json

import numpy as np
import pytest

pytest.importorskip("scipy")

from fair_trees.dataset import load_fair_dataset, load_regions  # noqa: E402
from fair_trees.evaluate import evaluate_regions, region_units  # noqa: E402
from fair_trees.params import CLASSES, InferenceParams  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402

CENTRE_ANCHOR_800 = (50 * 100 + 50) * 9 + 3  # box ~[122.9, 122.9, 133.1, 133.1] in a 256 px mosaic
PARAMS = InferenceParams(confidence_threshold=0.5)


def test_region_units_pick_zoom_and_clip_labels(write_region):
    boxes = [(120, 120, 136, 136), (250, 10, 300, 30)]
    write_region("r", tiles=((0, 0),), z=19, boxes_px=boxes)
    d = write_region("r", tiles=((0, 0),), z=20, boxes_px=boxes)
    [u] = region_units(load_regions(d.parent))
    assert u.unit_id == "r" and u.meta == {"source": "src", "biome": "b1", "zoom": "20"}
    np.testing.assert_allclose(u.gt_boxes, [[120, 120, 136, 136]], atol=1e-3)  # the second box is 12 % inside
    [u19] = region_units(load_regions(d.parent), zooms=(19,))
    assert u19.meta["zoom"] == "19"


def test_falls_back_to_the_regions_lowest_zoom(write_region):
    d = write_region("r", tiles=((0, 0),), z=22)
    [u] = region_units(load_regions(d.parent))
    assert u.meta["zoom"] == "22"


def test_disconnected_chips_become_separate_units(write_region):
    d = write_region("r", tiles=((0, 0), (5, 0)))
    assert [u.unit_id for u in region_units(load_regions(d.parent))] == ["r#0", "r#1"]


def test_evaluate_regions_scores_planted_box(write_region, fake_session):
    d = write_region("r", tiles=((0, 0),), boxes_px=[(123, 123, 133, 133)])
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))
    r = evaluate_regions(run, load_regions(d.parent), PARAMS)
    assert (r.precision, r.recall, r.f1) == (1.0, 1.0, 1.0)
    assert r.breakdown["zoom"]["20"]["n_units"] == 1


def test_group_units_score_group_boxes(write_region, fake_session):
    d = write_region("r", boxes_px=[(10, 10, 20, 20)], groups_px=[(123, 123, 133, 133)], classes=CLASSES)
    write_region("s", boxes_px=[(10, 10, 20, 20)])  # labels no groups: not a group unit
    regions = load_regions(d.parent)
    assert [u.unit_id for u in region_units(regions, label=2)] == ["r"]
    run = ort_run_fn(fake_session({(CENTRE_ANCHOR_800, 1): 5.0}, num_classes=2))
    groups = evaluate_regions(run, regions, PARAMS, label=2)
    assert (groups.precision, groups.recall, groups.f1) == (1.0, 1.0, 1.0)
    trees = evaluate_regions(run, regions, PARAMS)
    assert trees.recall == 0.0 and len(trees.units) == 2


def test_group_units_ignore_dropped_groups(write_region, fake_session):
    d = write_region("r", groups_px=[(100, 100, 500, 160)], classes=CLASSES)  # 39 % inside the unit
    [u] = region_units(load_regions(d.parent), label=2)
    assert len(u.gt_boxes) == 0
    run = ort_run_fn(fake_session({(CENTRE_ANCHOR_800, 1): 5.0}, num_classes=2))
    r = evaluate_regions(run, load_regions(d.parent), PARAMS, label=2)
    assert (r.units[0]["fp"], r.units[0]["fn"]) == (0, 0)


def test_tree_units_ignore_group_polygons(write_region, fake_session):
    d = write_region("r", groups_px=[(110, 110, 150, 150)], classes=CLASSES)
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))
    assert evaluate_regions(run, load_regions(d.parent), PARAMS).units[0]["fp"] == 0


def test_ignore_polygons_drop_predictions(write_region, fake_session):
    d = write_region("r", tiles=((0, 0),), ignore_px=[(110, 110, 150, 150)])
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))
    assert evaluate_regions(run, load_regions(d.parent), PARAMS).units[0]["fp"] == 0
    assert evaluate_regions(run, load_regions(d.parent), PARAMS, ignore=False).units[0]["fp"] == 1


def test_masked_tree_points_drop_predictions(write_oam_chip, tmp_path, geo, fake_session):
    sized = {"natural": "tree", "diameter_crown": "6"}
    features = [
        geo.point_feature((50, 50), tags=sized, label=1),
        geo.point_feature((200, 50), tags=sized, label=1),
        geo.point_feature((128, 128), tags={"natural": "tree"}, label=1),  # no size: masked, as most have one
    ]
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    regions = load_fair_dataset([write_oam_chip(2**19, 2**19)], labels, split="val")
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))  # one detection, on the tree without a size
    [unit] = evaluate_regions(run, regions, PARAMS).units
    assert (unit["tp"], unit["fp"], unit["fn"]) == (0, 0, 2)
