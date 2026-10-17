"""Numpy RetinaNet decoding: anchors, box decode, deterministic NMS.

Reproduces torchvision's AnchorGenerator and RetinaNet.postprocess_detections
for a single-class head, working in model-input pixels (no CRS).
"""

import math
from dataclasses import dataclass

import numpy as np

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


def _squeeze_batch(a: np.ndarray, last: int, name: str) -> np.ndarray:
    a = np.asarray(a, np.float32)
    if a.ndim == 3 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 2 or a.shape[1] != last:
        raise ValueError(f"{name} must have shape (1, A, {last}) or (A, {last}), got {a.shape}")
    return a


def decode_retinanet_output(
    cls_logits: np.ndarray,
    bbox_regression: np.ndarray,
    input_hw: tuple[int, int],
    image_hw: tuple[int, int],
    confidence_threshold: float,
    nms_threshold: float,
) -> Detections:
    """Decode raw single-class RetinaNet outputs into clipped, NMS-filtered detections.

    `input_hw` is the padded network input; `image_hw` the resized, unpadded size boxes are clipped to.
    """
    logits = _squeeze_batch(cls_logits, 1, "cls_logits")[:, 0]
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
    keep = nms_xyxy(boxes, scores, nms_threshold)[:DETECTIONS_PER_IMG]
    return Detections(boxes[keep].astype(np.float32), scores[keep].astype(np.float32))
