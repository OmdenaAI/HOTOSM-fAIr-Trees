import json

import numpy as np
import pytest

from experiments import teacher
from fair_trees.dataset import load_regions
from fair_trees.decode import Detections
from fair_trees.params import InferenceParams
from fair_trees.preprocess import mosaic_chips

PARAMS = InferenceParams(window_m=60.0, confidence_threshold=0.3)


def test_label_free_tiles_marks_present_tiles_without_labels():
    valid = np.ones((8, 8), bool)
    valid[4:, 4:] = False
    free = teacher.label_free_tiles(np.array([[1, 1, 3, 3]], float), valid, 4)
    assert free.tolist() == [[False, True], [True, False]]


def test_label_free_tiles_edge_touch_is_not_overlap():
    free = teacher.label_free_tiles(np.array([[1, 1, 4, 3]], float), np.ones((4, 8), bool), 4)
    assert free.tolist() == [[False, True]]


@pytest.mark.parametrize("label", [(-50, 10, -10, 40), (600, 10, 640, 40), (512, 10, 540, 40)])
def test_label_free_tiles_ignores_labels_outside_the_mosaic(label):
    free = teacher.label_free_tiles(np.array([label], float), np.ones((256, 512), bool), 256)
    assert free.tolist() == [[True, True]]


def test_select_teacher_drops_boxes_on_labels():
    labels = np.array([[10, 10, 20, 20]], float)
    boxes = np.array([[15, 15, 25, 25], [20, 0, 30, 10]], float)
    free = np.ones((1, 1), bool)
    for scope in teacher.SCOPES:
        assert teacher.select_teacher(boxes, labels, free, 64, scope).tolist() == [False, True]


def test_select_teacher_label_free_uses_centre_tile():
    free = np.array([[False, True]])
    boxes = np.array([[2, 0, 7, 2], [1, 0, 6, 2]], float)  # centres in tile 1 and tile 0
    keep = teacher.select_teacher(boxes, np.zeros((0, 4)), free, 4, "label_free")
    assert keep.tolist() == [True, False]
    assert teacher.select_teacher(boxes, np.zeros((0, 4)), free, 4, "all").tolist() == [True, True]


def test_select_teacher_unknown_scope_raises():
    with pytest.raises(ValueError, match="scope"):
        teacher.select_teacher(np.zeros((0, 4)), np.zeros((0, 4)), np.ones((1, 1), bool), 4, "some")


def _stub_predictions(monkeypatch, boxes, scores):
    det = Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32))
    monkeypatch.setattr(teacher, "predict_mosaic", lambda run_fn, mosaic, params: det)


def _features(path):
    return json.loads(path.read_text())["features"]


def region_gsd(region):
    return mosaic_chips(region.chips[20], use_sidecar=False)[0].gsd_m


def _tree_bytes(root):
    return {p.relative_to(root): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_write_teacher_sets_adds_features_to_train_only(monkeypatch, write_region, tmp_path):
    src = tmp_path / "hot"
    write_region(
        "t",
        tiles=((0, 0), (1, 0)),
        root=src,
        source="hot",
        boxes_px=[(10, 10, 40, 40)],
        ignore_px=[(200, 0, 220, 20)],
    )
    write_region("v", tiles=((4, 0),), root=src, source="hot", split="val", boxes_px=[(1030, 10, 1060, 40)])
    before = _tree_bytes(src)
    # on the label, in the label-free tile 1, in the labelled tile 0 off the label
    _stub_predictions(
        monkeypatch, [(12, 12, 30, 30), (300, 50, 330, 80), (100, 100, 130, 130)], [0.5, 0.6, 0.7]
    )

    stats = teacher.write_teacher_sets(src, tmp_path / "out", run_fn=None, params=PARAMS)

    assert _tree_bytes(src) == before
    original = _features(src / "t" / "ignore.geojson")
    for name, n_teacher in (("hot-teacher-all", 2), ("hot-teacher-empty", 1)):
        root = tmp_path / "out" / name
        feats = _features(root / "t" / "ignore.geojson")
        assert feats[: len(original)] == original
        added = feats[len(original) :]
        assert len(added) == n_teacher
        assert all(f["properties"]["reason"] == "teacher" and "score" in f["properties"] for f in added)
        assert _tree_bytes(root / "v") == {
            k.relative_to("v"): b for k, b in before.items() if k.parts[0] == "v"
        }
        assert json.loads((root / "teacher.json").read_text())["kept"] == n_teacher
        assert [r.id for r in load_regions(root)] == ["t", "v"]
    assert [f["properties"]["score"] for f in added] == [0.6]
    assert stats["regions"] == 1 and stats["teacher"] == 3 and stats["on_labels"] == 1
    assert stats["sets"]["hot-teacher-all"]["kept"] == 2
    assert (
        stats["sets"]["hot-teacher-all"]["kept_area_share"]
        > stats["sets"]["hot-teacher-empty"]["kept_area_share"]
    )


def test_write_teacher_sets_creates_missing_ignore(monkeypatch, write_region, tmp_path):
    src = tmp_path / "hot"
    write_region("t", tiles=((0, 0),), root=src, source="hot")
    _stub_predictions(monkeypatch, [(100, 100, 130, 130)], [0.9])

    teacher.write_teacher_sets(src, tmp_path / "out", run_fn=None, params=PARAMS)

    for name in teacher.SCOPES.values():
        (region,) = load_regions(tmp_path / "out" / name)
        assert region.ignore_path is not None
        assert len(_features(region.ignore_path)) == 1


def test_teacher_features_skip_boxes_already_masked(monkeypatch, write_region, tmp_path):
    write_region(
        "t",
        tiles=((0, 0), (1, 0)),
        root=tmp_path / "hot",
        boxes_px=[(10, 10, 40, 40)],
        ignore_px=[(290, 40, 340, 90)],  # e.g. a tree group
    )
    # centred inside the existing ignore square; off every label in the labelled tile 0
    _stub_predictions(monkeypatch, [(300, 50, 330, 80), (100, 100, 130, 130)], [0.5, 0.6])
    (region,) = load_regions(tmp_path / "hot")

    features, counts = teacher.teacher_features(region, None, PARAMS)

    assert [f["properties"]["score"] for f in features["all"]] == [0.6]
    assert features["label_free"] == []
    assert counts["already_masked"] == 1 and counts["all"] == 1 and counts["label_free"] == 0


def test_teacher_features_count_tiles_over_the_mask_limit(monkeypatch, write_region, tmp_path):
    write_region("t", tiles=((0, 0), (1, 0)), root=tmp_path / "hot", boxes_px=[(10, 10, 40, 40)])
    _stub_predictions(monkeypatch, [(256, 0, 506, 250)], [0.9])  # 95% of the label-free tile 1
    (region,) = load_regions(tmp_path / "hot")

    _, counts = teacher.teacher_features(region, None, PARAMS)

    assert counts["tiles"] == 2
    assert counts["tiles_over_max_masked"] == {"existing": 0, "all": 1, "label_free": 1}


def test_teacher_features_cover_every_mosaic_of_a_region(monkeypatch, write_region, tmp_path):
    write_region("t", tiles=((0, 0), (3, 0)), root=tmp_path / "hot", source="hot")  # two separate mosaics
    _stub_predictions(monkeypatch, [(10, 10, 40, 40)], [0.8])
    (region,) = load_regions(tmp_path / "hot")

    features, counts = teacher.teacher_features(region, None, PARAMS)

    rings = [f["geometry"]["coordinates"][0][0] for f in features["all"]]
    assert len(rings) == 2 and rings[0] != rings[1]
    assert {k: counts[k] for k in ("teacher", "on_labels", "all", "label_free")} == {
        "teacher": 2,
        "on_labels": 0,
        "all": 2,
        "label_free": 2,
    }
    assert counts["area_m2"]["all"] == pytest.approx(2 * (30 * region_gsd(region)) ** 2, rel=1e-6)
