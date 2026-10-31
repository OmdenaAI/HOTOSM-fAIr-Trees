"""Numpy RetinaNet decoding: anchors, box decode, deterministic NMS.

Reproduces torchvision's AnchorGenerator and RetinaNet.postprocess_detections
one class at a time, working in model-input pixels (no CRS).
"""

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from fair_trees.preprocess import Window, WindowScale

TOPK_CANDIDATES = 1000
DETECTIONS_PER_IMG = 300
ANCHOR_SIZES = tuple((s, int(s * 2 ** (1 / 3)), int(s * 2 ** (2 / 3))) for s in (32, 64, 128, 256, 512))
ASPECT_RATIOS = (0.5, 1.0, 2.0)
BBOX_XFORM_CLIP = math.log(1000 / 16)

_FIRST_LEVEL_HALVINGS = 3  # P3 is the input halved three times


@dataclass(frozen=True, eq=False)
class Detections:
    """XYXY float32 boxes (N, 4) with float32 scores (N,)."""

    boxes: np.ndarray
    scores: np.ndarray

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32))


def _grid_size(n: int, halvings: int) -> int:
    for _ in range(halvings):
        n = (n - 1) // 2 + 1
    return n


def _cell_anchors(sizes: tuple[int, ...]) -> np.ndarray:
    scales = np.array(sizes, np.float32)
    h_ratios = np.sqrt(np.array(ASPECT_RATIOS, np.float32))
    w_ratios = np.float32(1.0) / h_ratios
    ws = (w_ratios[:, None] * scales[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * scales[None, :]).reshape(-1)
    return np.round(np.stack([-ws, -hs, ws, hs], axis=1) / np.float32(2.0)).astype(np.float32)


def make_anchors(input_h: int, input_w: int) -> list[np.ndarray]:
    """Anchors (L, 4) float32 per level P3-P7, location-major and anchor-minor."""
    levels = []
    for i, sizes in enumerate(ANCHOR_SIZES):
        gh = _grid_size(input_h, _FIRST_LEVEL_HALVINGS + i)
        gw = _grid_size(input_w, _FIRST_LEVEL_HALVINGS + i)
        sx = np.arange(gw, dtype=np.float32) * np.float32(input_w // gw)
        sy = np.arange(gh, dtype=np.float32) * np.float32(input_h // gh)
        shift_y, shift_x = np.meshgrid(sy, sx, indexing="ij")
        shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)
        shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=1)
        levels.append((shifts[:, None, :] + _cell_anchors(sizes)[None]).reshape(-1, 4).astype(np.float32))
    return levels


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS; ties break by ascending index. Suppresses IoU strictly above the threshold."""
    n = len(scores)
    if n == 0:
        return np.zeros(0, np.int64)
    order = np.lexsort((np.arange(n), -scores))
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    suppressed = np.zeros(n, bool)
    keep = []
    for pos, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(i)
        rest = order[pos + 1 :]
        iw = np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0])
        ih = np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1])
        inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        suppressed[rest[iou > iou_threshold]] = True
    return np.array(keep, np.int64)


def _decode_boxes(reg: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    cx = anchors[:, 0] + np.float32(0.5) * w
    cy = anchors[:, 1] + np.float32(0.5) * h
    dx, dy = reg[:, 0], reg[:, 1]
    dw = np.minimum(reg[:, 2], np.float32(BBOX_XFORM_CLIP))
    dh = np.minimum(reg[:, 3], np.float32(BBOX_XFORM_CLIP))
    pcx = dx * w + cx
    pcy = dy * h + cy
    pw = np.exp(dw) * w
    ph = np.exp(dh) * h
    half = np.float32(0.5)
    return np.stack([pcx - half * pw, pcy - half * ph, pcx + half * pw, pcy + half * ph], axis=1).astype(
        np.float32
    )


def _squeeze_batch(a: np.ndarray, last: int | None, name: str) -> np.ndarray:
    """(A, last) from (1, A, last) or (A, last); `last` None accepts any width."""
    a = np.asarray(a, np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or (last is not None and a.shape[1] != last):
        want = "K" if last is None else last
        raise ValueError(f"{name} must have shape (1, A, {want}) or (A, {want}), got {a.shape}")
    return a


def decode_retinanet_output(
    cls_logits: np.ndarray,
    bbox_regression: np.ndarray,
    input_hw: tuple[int, int],
    image_hw: tuple[int, int],
    confidence_threshold: float,
    nms_threshold: float,
    class_index: int = 0,
) -> Detections:
    """Decode one class of raw RetinaNet outputs into clipped, NMS-filtered detections.

    `cls_logits` holds one column per class; `input_hw` is the padded network input; `image_hw` the
    resized, unpadded size boxes are clipped to.
    """
    all_logits = _squeeze_batch(cls_logits, None, "cls_logits")
    if not 0 <= class_index < all_logits.shape[1]:
        raise ValueError(f"class_index {class_index} out of range for {all_logits.shape[1]} classes")
    logits = all_logits[:, class_index]
    reg = _squeeze_batch(bbox_regression, 4, "bbox_regression")
    levels = make_anchors(*input_hw)
    total = sum(len(a) for a in levels)
    if len(logits) != total or len(reg) != total:
        raise ValueError(
            f"expected {total} anchors for input {tuple(input_hw)}, got {len(logits)} and {len(reg)}"
        )

    boxes_out, scores_out = [], []
    start = 0
    for anchors in levels:
        stop = start + len(anchors)
        with np.errstate(over="ignore"):
            scores = (np.float32(1.0) / (np.float32(1.0) + np.exp(-logits[start:stop]))).astype(np.float32)
        idx = np.flatnonzero(scores > confidence_threshold)
        idx = idx[np.lexsort((idx, -scores[idx]))][:TOPK_CANDIDATES]
        boxes_out.append(_decode_boxes(reg[start:stop][idx], anchors[idx]))
        scores_out.append(scores[idx])
        start = stop

    boxes = np.concatenate(boxes_out)
    scores = np.concatenate(scores_out)
    if len(scores) == 0:
        return Detections.empty()
    image_h, image_w = image_hw
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_h)
    finite = np.isfinite(boxes).all(axis=1)
    boxes, scores = boxes[finite], scores[finite]
    keep = nms_xyxy(boxes, scores, nms_threshold)[:DETECTIONS_PER_IMG]
    return Detections(boxes[keep].astype(np.float32), scores[keep].astype(np.float32))


def window_to_mosaic(dets: Detections, window: Window, scale: WindowScale) -> Detections:
    """Map model-input pixel boxes back to mosaic pixels."""
    shift = np.array([window.x, window.y, window.x, window.y], np.float32)
    div = np.array([scale.scale_x, scale.scale_y, scale.scale_x, scale.scale_y], np.float32)
    return Detections((dets.boxes / div + shift).astype(np.float32), dets.scores)


def interior_sides(
    window: Window, active: Collection[tuple[int, int]], xs: Sequence[int], ys: Sequence[int]
) -> set[str]:
    """Sides of the window that border another non-blank window."""
    i, j = xs.index(window.x), ys.index(window.y)
    sides = set()
    if i > 0 and (xs[i - 1], window.y) in active:
        sides.add("left")
    if i < len(xs) - 1 and (xs[i + 1], window.y) in active:
        sides.add("right")
    if j > 0 and (window.x, ys[j - 1]) in active:
        sides.add("top")
    if j < len(ys) - 1 and (window.x, ys[j + 1]) in active:
        sides.add("bottom")
    return sides


def cut_boxes(dets: Detections, window: Window, interior: set[str], band: float) -> np.ndarray:
    """Boxes (bool per box) reaching into the edge band of an interior window side."""
    bx, by = band * window.width, band * window.height
    x1, y1, x2, y2 = dets.boxes.T
    cut = np.zeros(len(dets.scores), bool)
    if "left" in interior:
        cut |= x1 < window.x + bx
    if "right" in interior:
        cut |= x2 > window.x + window.width - bx
    if "top" in interior:
        cut |= y1 < window.y + by
    if "bottom" in interior:
        cut |= y2 > window.y + window.height - by
    return cut


def drop_edge_boxes(dets: Detections, window: Window, interior: set[str], band: float) -> Detections:
    """Drop boxes reaching into the edge band of an interior window side."""
    keep = ~cut_boxes(dets, window, interior, band)
    return Detections(dets.boxes[keep], dets.scores[keep])


def merge_group_detections(
    parts: Sequence[tuple[Detections, np.ndarray]], iou_threshold: float
) -> Detections:
    """Join tree_group boxes cut by window edges, then NMS.

    ``parts`` holds each window's boxes (mosaic pixels) and whether each is cut (``cut_boxes``). A cut box
    joins every box of another window it overlaps, transitively; a joined set becomes its union extent with
    the highest score.
    """
    dets = concat_detections([d for d, _ in parts])
    if not len(dets.scores):
        return Detections.empty()
    cut = np.concatenate([np.asarray(c, bool) for _, c in parts])
    window = np.repeat(np.arange(len(parts)), [len(d.scores) for d, _ in parts])
    b = dets.boxes.astype(np.float64)
    parent = np.arange(len(b))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in np.flatnonzero(cut):  # one row at a time: memory stays linear in the box count
        iw = np.minimum(b[i, 2], b[:, 2]) - np.maximum(b[i, 0], b[:, 0])
        ih = np.minimum(b[i, 3], b[:, 3]) - np.maximum(b[i, 1], b[:, 1])
        for j in np.flatnonzero((iw > 0) & (ih > 0) & (window != window[i])):
            parent[find(j)] = find(i)
    roots = np.array([find(i) for i in range(len(b))], np.int64)
    _, member_of = np.unique(roots, return_inverse=True)
    n = int(member_of.max()) + 1
    lo, hi = np.full((n, 2), np.inf), np.full((n, 2), -np.inf)
    np.minimum.at(lo, member_of, b[:, :2])
    np.maximum.at(hi, member_of, b[:, 2:])
    scores = np.full(n, -np.inf, np.float32)
    np.maximum.at(scores, member_of, dets.scores)
    joined = Detections(np.hstack([lo, hi]).astype(np.float32), scores)
    return merge_window_detections(joined, iou_threshold)


def filter_group_parts(
    parts: Sequence[tuple[Detections, np.ndarray]], threshold: float
) -> list[tuple[Detections, np.ndarray]]:
    """Each window's group boxes scoring strictly above the threshold, with their cut flags."""
    return [(filter_score(d, threshold), np.asarray(c, bool)[d.scores > threshold]) for d, c in parts]


def merge_window_detections(dets: Detections, iou_threshold: float) -> Detections:
    keep = nms_xyxy(dets.boxes, dets.scores, iou_threshold)
    return Detections(dets.boxes[keep], dets.scores[keep])


def filter_score(dets: Detections, threshold: float) -> Detections:
    """Keep scores strictly greater than the threshold."""
    keep = dets.scores > threshold
    return Detections(dets.boxes[keep], dets.scores[keep])


def concat_detections(parts: Sequence[Detections]) -> Detections:
    if not parts:
        return Detections.empty()
    return Detections(np.concatenate([p.boxes for p in parts]), np.concatenate([p.scores for p in parts]))
