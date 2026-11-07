import json

import numpy as np
import pytest

from fair_trees.postprocess import to_feature_collection


def ring(x0, y0, size=1.0):
    return [[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]


def test_properties_exactly_class_and_score():
    fc = to_feature_collection([ring(0, 0)], np.array([0.123456], np.float32))
    assert fc["type"] == "FeatureCollection"
    [f] = fc["features"]
    assert f["type"] == "Feature"
    assert f["properties"] == {"class": 1, "score": 0.1235}
    assert type(f["properties"]["score"]) is float
    assert f["geometry"] == {"type": "Polygon", "coordinates": [ring(0, 0)]}


def test_label_when_enabled():
    [f] = to_feature_collection([ring(0, 0)], np.array([0.5]), include_label=True)["features"]
    assert f["properties"] == {"class": 1, "score": 0.5, "label": "tree"}


def test_classes_per_feature():
    fc = to_feature_collection(
        [ring(0, 0), ring(5, 5)], np.array([0.5, 0.6]), classes=np.array([1, 2]), include_label=True
    )
    assert [f["properties"] for f in fc["features"]] == [
        {"class": 1, "score": 0.5, "label": "tree"},
        {"class": 2, "score": 0.6, "label": "tree_group"},
    ]


def test_in_group_flag_only_where_given():
    fc = to_feature_collection(
        [ring(0, 0), ring(5, 5), ring(9, 9)],
        np.array([0.5, 0.6, 0.7]),
        classes=np.array([1, 1, 2]),
        in_group=[True, False, None],
    )
    assert [f["properties"].get("in_group") for f in fc["features"]] == [True, False, None]
    assert "in_group" not in fc["features"][2]["properties"]


def test_empty():
    assert to_feature_collection([], np.zeros(0)) == {"type": "FeatureCollection", "features": []}


def test_bbox_keeps_by_centre_inclusive():
    fc = to_feature_collection(
        [ring(0, 0), ring(10, 10), ring(1.5, 1.5)], np.array([0.9, 0.8, 0.7]), bbox=(0, 0, 2, 2)
    )
    assert [f["properties"]["score"] for f in fc["features"]] == [0.9, 0.7]


def test_json_serialisable_without_nan():
    json.dumps(to_feature_collection([ring(0, 0)], np.array([0.5], np.float32)), allow_nan=False)


def test_length_mismatch():
    with pytest.raises(ValueError):
        to_feature_collection([ring(0, 0)], np.array([0.5, 0.6]))
