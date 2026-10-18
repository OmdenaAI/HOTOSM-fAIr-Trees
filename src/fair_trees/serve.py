"""Serving entry points: sliding-window prediction over a mosaic and the fAIr chip-directory API."""

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from fair_trees.decode import (
    Detections,
    concat_detections,
    decode_retinanet_output,
    drop_edge_boxes,
    interior_sides,
    merge_window_detections,
    window_to_mosaic,
)
from fair_trees.georef import mosaic_px_to_4326_rings
from fair_trees.params import OUTPUT_NAMES, InferenceParams, parse_inference_params
from fair_trees.postprocess import to_feature_collection
from fair_trees.preprocess import Mosaic, is_blank, mosaic_chips, preprocess_window, window_grid

logger = logging.getLogger(__name__)

MAX_WINDOW_FACTOR = 16  # a request may cost at most this many times the default window count

RunFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def ort_run_fn(session) -> RunFn:
    """Wrap an onnxruntime InferenceSession as a ``batch -> (cls_logits, bbox_regression)`` function."""
    name = session.get_inputs()[0].name

    def run(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cls, reg = session.run(list(OUTPUT_NAMES), {name: batch})[:2]
        return cls, reg

    return run


def predict_mosaic(run_fn: RunFn, mosaic: Mosaic, params: InferenceParams) -> Detections:
    """Detect crowns in a mosaic with overlapping windows; boxes are in mosaic pixels."""
    height, width = mosaic.rgb.shape[:2]
    windows = window_grid(height, width, mosaic.gsd_m, params.window_m, params.window_overlap)
    xs = sorted({w.x for w in windows})
    ys = sorted({w.y for w in windows})
    active = [w for w in windows if not is_blank(mosaic, w, params.min_valid_fraction)]
    active_xy = {(w.x, w.y) for w in active}

    parts = []
    for w in active:
        batch, scale = preprocess_window(mosaic.rgb[w.y : w.y + w.height, w.x : w.x + w.width])
        cls, reg = run_fn(batch)
        d = decode_retinanet_output(
            cls,
            reg,
            batch.shape[2:],
            (scale.resized_height, scale.resized_width),
            params.confidence_threshold,
            params.nms_threshold,
        )
        d = window_to_mosaic(d, w, scale)
        parts.append(drop_edge_boxes(d, w, interior_sides(w, active_xy, xs, ys), params.edge_band))
    return merge_window_detections(concat_detections(parts), params.iou_threshold)


def _window_count(mosaics: list[Mosaic], params: InferenceParams) -> int:
    return sum(
        len(window_grid(*m.rgb.shape[:2], m.gsd_m, params.window_m, params.window_overlap)) for m in mosaics
    )


def predict_session(
    session,
    input_images: Path,
    params: Mapping | None,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """Predict crowns for a chip file or directory of chips and return a GeoJSON FeatureCollection."""
    p = parse_inference_params(params)
    input_images = Path(input_images)
    paths = [input_images] if input_images.is_file() else sorted(input_images.iterdir())
    skipped: list[tuple[str, str]] = []
    mosaics = mosaic_chips(paths, skipped=skipped)
    if not mosaics and skipped:
        reasons = "; ".join(f"{name}: {why}" for name, why in skipped[:3])
        raise ValueError(f"no usable chips ({len(skipped)} skipped): {reasons}")
    requested, baseline = _window_count(mosaics, p), _window_count(mosaics, InferenceParams())
    if requested > MAX_WINDOW_FACTOR * baseline:
        raise ValueError(
            f"window_m={p.window_m:g} and window_overlap={p.window_overlap:g} need {requested} windows, "
            f"more than {MAX_WINDOW_FACTOR}x the {baseline} of the default settings"
        )
    if len({m.z for m in mosaics}) > 1:
        logger.warning(
            "Chips span several zoom levels %s; each tile group is predicted separately",
            sorted({m.z for m in mosaics}),
        )

    run = ort_run_fn(session)
    rings: list = []
    scores: list = []
    for m in mosaics:
        d = predict_mosaic(run, m, p)
        rings.extend(mosaic_px_to_4326_rings(d.boxes, m))
        scores.extend(d.scores)
    return to_feature_collection(rings, np.asarray(scores, np.float32), bbox=bbox)
