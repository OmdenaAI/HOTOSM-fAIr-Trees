"""Turn EPSG:4326 crown rings and scores into a GeoJSON FeatureCollection."""

from collections.abc import Sequence

import numpy as np

from fair_trees.params import CLASSES


def to_feature_collection(
    rings: Sequence[Sequence[Sequence[float]]],
    scores: np.ndarray,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    include_label: bool = False,
    classes: np.ndarray | None = None,
    in_group: Sequence[bool | None] | None = None,
) -> dict:
    """Build a FeatureCollection of polygons, one per ring, in input order.

    ``bbox`` is ``(west, south, east, north)`` in EPSG:4326. When given, a ring is
    kept only if its centre lies inside the box (boundaries inclusive). ``classes`` holds
    each ring's class number (1 tree, 2 tree_group); default all 1. ``in_group`` holds an
    ``in_group`` property per ring (None: no property).
    """
    if len(rings) != len(scores):
        raise ValueError(f"got {len(rings)} rings but {len(scores)} scores")
    if classes is None:
        classes = np.ones(len(scores), np.int64)
    if in_group is None:
        in_group = [None] * len(scores)

    features = []
    for ring, score, cls, flag in zip(rings, scores, classes, in_group, strict=True):
        if bbox is not None:
            west, south, east, north = bbox
            cx = (ring[0][0] + ring[2][0]) / 2
            cy = (ring[0][1] + ring[2][1]) / 2
            if not (west <= cx <= east and south <= cy <= north):
                continue
        properties: dict = {"class": int(cls), "score": round(float(score), 4)}
        if include_label:
            properties["label"] = CLASSES[int(cls) - 1]
        if flag is not None:
            properties["in_group"] = bool(flag)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}
