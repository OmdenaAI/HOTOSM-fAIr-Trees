"""Serving entry points: sliding-window prediction over a mosaic and the fAIr chip-directory API."""

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np

from fair_trees.decode import (
    Detections,
    centres_inside,
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
from fair_trees.params import (
    CLASSES,
    OUTPUT_NAMES,
    THRESHOLDS_KEY,
    WINDOWS_KEY,
    InferenceParams,
    caller_sets_threshold,
    parse_class_thresholds,
    parse_inference_params,
    parse_model_windows,
)
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


def _grid(mosaic: Mosaic, params: InferenceParams, window_m: float) -> list[Window]:
    return window_grid(*mosaic.rgb.shape[:2], mosaic.gsd_m, window_m, params.window_overlap)


def _run_windows(
    run_fn: RunFn, mosaic: Mosaic, params: InferenceParams, window_m: float
) -> Iterator[_WindowRun]:
    """Run the network on every non-blank window of `window_m` metres, one window at a time."""
    windows = _grid(mosaic, params, window_m)
    xs = sorted({w.x for w in windows})
    ys = sorted({w.y for w in windows})
    active = [w for w in windows if not is_blank(mosaic, w, params.min_valid_fraction)]
    active_xy = {(w.x, w.y) for w in active}
    for w in active:
        batch, scale = preprocess_window(mosaic.rgb[w.y : w.y + w.height, w.x : w.x + w.width])
        cls, reg = run_fn(batch)
        yield _WindowRun(w, interior_sides(w, active_xy, xs, ys), cls, reg, batch.shape[2:], scale)


def _decode(run: _WindowRun, params: InferenceParams, class_index: int) -> Detections:
    d = decode_retinanet_output(
        run.cls,
        run.reg,
        run.input_hw,
        (run.scale.resized_height, run.scale.resized_width),
        params.threshold(class_index),
        params.nms_threshold,
        class_index,
    )
    return window_to_mosaic(d, run.window, run.scale)


def predict_mosaic_classes(
    run_fn: RunFn, mosaic: Mosaic, params: InferenceParams, class_indices: Sequence[int] | None = None
) -> list[Detections]:
    """Detections per class (in `class_indices` order; None = every class the model outputs), in mosaic
    pixels.

    Trees come from `window_m` windows with edge boxes dropped; tree_groups from `group_window_m` windows
    with boxes cut by a window edge joined across windows. Each window is decoded as soon as it has run,
    and a window grid both classes share runs once.
    """
    if class_indices is not None and not set(class_indices) <= {0, 1}:
        raise ValueError(f"class_index out of range in {list(class_indices)}: the classes are {CLASSES}")
    trees: list[Detections] = []
    groups: list[tuple[Detections, np.ndarray]] = []
    seen_k = 1

    def sweep(window_m: float, decode_classes) -> None:
        nonlocal seen_k
        for run in _run_windows(run_fn, mosaic, params, window_m):
            seen_k = run.cls.shape[-1]
            for c in decode_classes(seen_k):
                d = _decode(run, params, c)
                if c == 0:
                    trees.append(drop_edge_boxes(d, run.window, run.interior, params.edge_band))
                else:
                    groups.append((d, cut_boxes(d, run.window, run.interior, params.edge_band)))

    def wanted(k: int) -> list[int]:
        return list(range(min(k, len(CLASSES)))) if class_indices is None else list(class_indices)

    shared = _grid(mosaic, params, params.window_m) == _grid(mosaic, params, params.group_window_m)
    tree_pass = class_indices is None or 0 in class_indices
    if tree_pass:
        sweep(params.window_m, lambda k: [c for c in wanted(k) if c == 0 or shared])
    if 1 in wanted(seen_k) and not (tree_pass and shared):
        sweep(params.group_window_m, lambda k: [1])
    return [
        merge_window_detections(concat_detections(trees), params.iou_threshold)
        if c == 0
        else merge_group_detections(groups, params.iou_threshold, params.group_merge)
        for c in wanted(seen_k)
    ]


def predict_mosaic(run_fn: RunFn, mosaic: Mosaic, params: InferenceParams) -> Detections:
    """Detect crowns in a mosaic with overlapping windows; boxes are in mosaic pixels."""
    return predict_mosaic_classes(run_fn, mosaic, params, (0,))[0]


def predict_group_parts(
    run_fn: RunFn, mosaic: Mosaic, params: InferenceParams
) -> list[tuple[Detections, np.ndarray]]:
    """Each `group_window_m` window's tree_group boxes (mosaic pixels) and whether each is cut, before the
    cross-window join: what `merge_group_detections` takes, so a caller can filter scores first."""
    parts = []
    for run in _run_windows(run_fn, mosaic, params, params.group_window_m):
        d = _decode(run, params, 1)
        parts.append((d, cut_boxes(d, run.window, run.interior, params.edge_band)))
    return parts


def _window_count(mosaics: list[Mosaic], params: InferenceParams) -> int:
    """Windows the tree and group grids would run, counted as if both run (the model's classes are not
    known before the first window)."""
    return sum(
        len(_grid(m, params, params.window_m)) + len(_grid(m, params, params.group_window_m)) for m in mosaics
    )


def _metadata(session) -> Mapping[str, str]:
    """The ONNX model's custom metadata; empty for a session that exposes none."""
    get = getattr(session, "get_modelmeta", None)
    return dict(get().custom_metadata_map) if get is not None else {}


def predict_session(
    session,
    input_images: Path,
    params: Mapping | None,
    bbox: tuple[float, float, float, float] | None = None,
) -> dict:
    """Predict crowns (class 1) and, for a two-class model, tree groups (class 2) for a chip file or
    directory of chips and return a GeoJSON FeatureCollection.

    Unless the caller names a confidence threshold, each class is served at the threshold the model's
    fine-tune picked on val (ONNX metadata ``THRESHOLDS_KEY``), else at the default. Likewise a tree or
    group window the caller leaves unset is the one those thresholds were picked at (``WINDOWS_KEY``)."""
    p = parse_inference_params(params)
    meta = _metadata(session)
    if not caller_sets_threshold(params):
        p = replace(p, class_thresholds=parse_class_thresholds(meta.get(THRESHOLDS_KEY)))
    windows = parse_model_windows(meta.get(WINDOWS_KEY))
    p = replace(p, **{k: v for k, v in windows.items() if (params or {}).get(k) is None})
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
    in_group: list = []  # a two-class model's trees: whether a served group box covers the centre
    for m in mosaics:
        per_class = predict_mosaic_classes(run, m, p)
        for c, d in enumerate(per_class):
            rings.extend(mosaic_px_to_4326_rings(d.boxes, m))
            scores.extend(d.scores)
            classes.extend([c + 1] * len(d.scores))
            if c == 0 and len(per_class) > 1:
                in_group.extend(centres_inside(d.boxes, per_class[1].boxes).tolist())
            else:
                in_group.extend([None] * len(d.scores))
    return to_feature_collection(
        rings,
        np.asarray(scores, np.float32),
        bbox=bbox,
        classes=np.asarray(classes, np.int64),
        in_group=in_group,
    )
