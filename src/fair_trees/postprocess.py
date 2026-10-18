"""Turn EPSG:4326 crown rings and scores into a GeoJSON FeatureCollection."""

from collections.abc import Sequence

import numpy as np


def to_feature_collection(
    rings: Sequence[Sequence[Sequence[float]]],
    scores: np.ndarray,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    include_label: bool = False,
) -> dict:
    """Build a FeatureCollection of polygons, one per ring, in input order.

    ``bbox`` is ``(west, south, east, north)`` in EPSG:4326. When given, a ring is
    kept only if its centre lies inside the box (boundaries inclusive).
    """
    if len(rings) != len(scores):
        raise ValueError(f"got {len(rings)} rings but {len(scores)} scores")

    features = []
    for ring, score in zip(rings, scores, strict=True):
        if bbox is not None:
            west, south, east, north = bbox
            cx = (ring[0][0] + ring[2][0]) / 2
            cy = (ring[0][1] + ring[2][1]) / 2
            if not (west <= cx <= east and south <= cy <= north):
                continue
        properties: dict = {"class": 1, "score": round(float(score), 4)}
        if include_label:
            properties["label"] = "tree"
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}
