import numpy as np
import pytest

pytest.importorskip("scipy")

from fair_trees.dataset import load_regions  # noqa: E402
from fair_trees.evaluate import evaluate_regions, region_units  # noqa: E402
from fair_trees.params import InferenceParams  # noqa: E402
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


def test_ignore_polygons_drop_predictions(write_region, fake_session):
    d = write_region("r", tiles=((0, 0),), ignore_px=[(110, 110, 150, 150)])
    run = ort_run_fn(fake_session({CENTRE_ANCHOR_800: 5.0}))
    assert evaluate_regions(run, load_regions(d.parent), PARAMS).units[0]["fp"] == 0
    assert evaluate_regions(run, load_regions(d.parent), PARAMS, ignore=False).units[0]["fp"] == 1
