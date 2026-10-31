"""Serving entry points: sliding-window prediction over a mosaic and the fAIr chip-directory API."""

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from fair_trees.decode import (
    Detections,
    concat_detections,
    cut_boxes,
    decode_retinanet_output,
    drop_edge_boxes,
    interior_sides,
    merge_group_detections,
    merge_window_detections,
    window_to_mosaic,
)
from fair_trees.georef import mosaic_px_to_4326_rings
from fair_trees.params import CLASSES, OUTPUT_NAMES, InferenceParams, parse_inference_params
from fair_trees.postprocess import to_feature_collection
from fair_trees.preprocess import (
    Mosaic,
    Window,
    WindowScale,
    is_blank,
    mosaic_chips,
    preprocess_window,
    window_grid,
)

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


class _WindowRun(NamedTuple):
    """One window's raw network outputs and where they sit in the mosaic."""

    window: Window
    interior: set[str]
    cls: np.ndarray
    reg: np.ndarray
    input_hw: tuple[int, int]
    scale: WindowScale


def _run_windows(run_fn: RunFn, mosaic: Mosaic, params: InferenceParams, window_m: float) -> list[_WindowRun]:
    """Run the network once on every non-blank window of `window_m` metres."""
    height, width = mosaic.rgb.shape[:2]
    windows = window_grid(height, width, mosaic.gsd_m, window_m, params.window_overlap)
    xs = sorted({w.x for w in windows})
    ys = sorted({w.y for w in windows})
    active = [w for w in windows if not is_blank(mosaic, w, params.min_valid_fraction)]
    active_xy = {(w.x, w.y) for w in active}
    runs = []
    for w in active:
        batch, scale = preprocess_window(mosaic.rgb[w.y : w.y + w.height, w.x : w.x + w.width])
        cls, reg = run_fn(batch)
        runs.append(_WindowRun(w, interior_sides(w, active_xy, xs, ys), cls, reg, batch.shape[2:], scale))
    return runs


def _decode(run: _WindowRun, params: InferenceParams, class_index: int) -> Detections:
    d = decode_retinanet_output(
        run.cls,
        run.reg,
        run.input_hw,
        (run.scale.resized_height, run.scale.resized_width),
        params.confidence_threshold,
        params.nms_threshold,
        class_index,
    )
    return window_to_mosaic(d, run.window, run.scale)


def _trees(runs: Sequence[_WindowRun], params: InferenceParams) -> Detections:
    parts = [drop_edge_boxes(_decode(r, params, 0), r.window, r.interior, params.edge_band) for r in runs]
    return merge_window_detections(concat_detections(parts), params.iou_threshold)


def _groups(runs: Sequence[_WindowRun], params: InferenceParams) -> Detections:
    parts = []
    for r in runs:
        d = _decode(r, params, 1)
        parts.append((d, cut_boxes(d, r.window, r.interior, params.edge_band)))
    return merge_group_detections(parts, params.iou_threshold)


def predict_mosaic_classes(
    run_fn: RunFn, mosaic: Mosaic, params: InferenceParams, class_indices: Sequence[int] | None = None
) -> list[Detections]:
    """Detections per class (in `class_indices` order; None = every class the model outputs), in mosaic
    pixels.

    Trees come from `window_m` windows with edge boxes dropped; tree_groups from `group_window_m` windows
    with boxes cut by a window edge joined across windows. The network runs once per window per distinct
    window grid.
    """
    runs: dict[tuple, list[_WindowRun]] = {}

    def runs_for(window_m: float) -> list[_WindowRun]:
        height, width = mosaic.rgb.shape[:2]
        key = tuple(window_grid(height, width, mosaic.gsd_m, window_m, params.window_overlap))
        if key not in runs:
            runs[key] = _run_windows(run_fn, mosaic, params, window_m)
        return runs[key]

    if class_indices is None:
        first = runs_for(params.window_m)
        class_indices = range(first[0].cls.shape[-1] if first else 1)
    out = []
    for c in class_indices:
        if c == 0:
            out.append(_trees(runs_for(params.window_m), params))
        elif c == 1:
            out.append(_groups(runs_for(params.group_window_m), params))
        else:
            raise ValueError(f"class_index {c} out of range: the classes are {CLASSES}")
    return out


def predict_mosaic(run_fn: RunFn, mosaic: Mosaic, params: InferenceParams) -> Detections:
    """Detect crowns in a mosaic with overlapping windows; boxes are in mosaic pixels."""
    return predict_mosaic_classes(run_fn, mosaic, params, (0,))[0]


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
    """Predict crowns (class 1) and, for a two-class model, tree groups (class 2) for a chip file or
    directory of chips and return a GeoJSON FeatureCollection."""
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
    classes: list = []
    for m in mosaics:
        for c, d in enumerate(predict_mosaic_classes(run, m, p)):
            rings.extend(mosaic_px_to_4326_rings(d.boxes, m))
            scores.extend(d.scores)
            classes.extend([c + 1] * len(d.scores))
    return to_feature_collection(
        rings, np.asarray(scores, np.float32), bbox=bbox, classes=np.asarray(classes, np.int64)
    )
