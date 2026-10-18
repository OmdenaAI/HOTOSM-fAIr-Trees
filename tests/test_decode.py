import numpy as np
import pytest

from fair_trees.decode import (
    ANCHOR_SIZES,
    DETECTIONS_PER_IMG,
    decode_retinanet_output,
    make_anchors,
    nms_xyxy,
)

# P3 row 50, col 50, aspect 1.0, size 32 at an 800 x 800 input -> box [384, 384, 416, 416]
CENTRE_ANCHOR_800 = (50 * 100 + 50) * 9 + 3


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _outputs(input_hw, planted=None, reg=None):
    n = sum(len(a) for a in make_anchors(*input_hw))
    cls = np.full((1, n, 1), -10.0, np.float32)
    box = np.zeros((1, n, 4), np.float32)
    for i, v in (planted or {}).items():
        cls[0, i, 0] = v
    for i, v in (reg or {}).items():
        box[0, i] = v
    return cls, box


def test_anchor_count_and_levels():
    levels = make_anchors(800, 800)
    assert [len(a) for a in levels] == [90000, 22500, 5625, 1521, 441]
    assert sum(len(a) for a in levels) == 120087
    assert all(a.dtype == np.float32 for a in levels)


def test_anchor_sizes_truncated():
    assert ANCHOR_SIZES == ((32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812))


def test_cell_anchor_layout():
    np.testing.assert_array_equal(
        make_anchors(800, 800)[0][:9],
        [
            [-23, -11, 23, 11],
            [-28, -14, 28, 14],
            [-35, -18, 35, 18],
            [-16, -16, 16, 16],
            [-20, -20, 20, 20],
            [-25, -25, 25, 25],
            [-11, -23, 11, 23],
            [-14, -28, 14, 28],
            [-18, -35, 18, 35],
        ],
    )


def test_p6_stride_61():
    p6 = make_anchors(800, 800)[3]
    assert p6[9, 0] - p6[0, 0] == 61  # next location along x
    assert p6[13 * 9, 1] - p6[0, 1] == 61  # next row (13 columns)


def test_planted_logit_decodes_to_its_anchor():
    cls, box = _outputs((800, 800), {CENTRE_ANCHOR_800: 5.0})
    d = decode_retinanet_output(cls, box, (800, 800), (800, 800), 0.1, 0.05)
    np.testing.assert_allclose(d.boxes, [[384, 384, 416, 416]], atol=1e-4)
    np.testing.assert_allclose(d.scores, [_sigmoid(5.0)], atol=1e-6)
    assert d.boxes.dtype == np.float32 and d.scores.dtype == np.float32


def test_dw_is_clamped_before_exp():
    idx = (50 * 200 + 50) * 9 + 3  # P3 row 50, col 50, aspect 1, size 32 at 800 x 1600
    cls, box = _outputs((800, 1600), {idx: 5.0}, {idx: [0.0, 0.0, 5.0, 0.0]})
    d = decode_retinanet_output(cls, box, (800, 1600), (800, 1600), 0.1, 0.05)
    # clamp: w = 32 * 1000/16 = 2000 -> [-600, 1400] -> clipped x1 = 0; unclamped exp(5) would reach 1600
    np.testing.assert_allclose(d.boxes, [[0, 384, 1400, 416]], atol=1e-3)


def test_clip_to_resized_size_not_padded_size():
    idx = (50 * 120 + 119) * 9 + 3  # P3 row 50, last column at 800 x 960
    cls, box = _outputs((800, 960), {idx: 5.0})
    d = decode_retinanet_output(cls, box, (800, 960), (800, 937), 0.1, 0.05)
    np.testing.assert_allclose(d.boxes, [[936, 384, 937, 416]], atol=1e-4)


def test_threshold_is_strict():
    cls, box = _outputs((800, 800), {CENTRE_ANCHOR_800: 0.0})  # score exactly 0.5
    assert len(decode_retinanet_output(cls, box, (800, 800), (800, 800), 0.5, 0.05).scores) == 0
    assert len(decode_retinanet_output(cls, box, (800, 800), (800, 800), 0.49, 0.05).scores) == 1


def test_empty_output_shapes():
    cls, box = _outputs((800, 800))
    d = decode_retinanet_output(cls, box, (800, 800), (800, 800), 0.1, 0.05)
    assert d.boxes.shape == (0, 4) and d.scores.shape == (0,)


def test_anchor_count_mismatch_raises():
    cls, box = _outputs((800, 800))
    with pytest.raises(ValueError):
        decode_retinanet_output(cls, box, (800, 832), (800, 800), 0.1, 0.05)


def test_cap_300_keeps_highest():
    locs = [(r, c) for r in range(0, 100, 5) for c in range(0, 100, 5)]  # 400 non-overlapping P3 boxes
    planted = {(r * 100 + c) * 9 + 3: 2.0 + 0.001 * i for i, (r, c) in enumerate(locs)}
    cls, box = _outputs((800, 800), planted)
    d = decode_retinanet_output(cls, box, (800, 800), (800, 800), 0.1, 0.05)
    assert len(d.scores) == DETECTIONS_PER_IMG
    kth = np.sort(_sigmoid(np.array(list(planted.values()), np.float32)))[-DETECTIONS_PER_IMG]
    assert d.scores.min() >= kth - 1e-6
    assert np.all(np.diff(d.scores) <= 0)


def test_nms_ties_and_strict_threshold():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    assert nms_xyxy(boxes, np.array([0.9, 0.9, 0.8], np.float32), 0.5).tolist() == [0, 2]
    b = np.array([[0, 0, 10, 10], [0, 0, 10, 5]], np.float32)  # IoU exactly 0.5
    s = np.array([0.9, 0.8], np.float32)
    assert nms_xyxy(b, s, 0.5).tolist() == [0, 1]
    assert nms_xyxy(b, s, 0.49).tolist() == [0]
    assert nms_xyxy(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), 0.5).tolist() == []


def test_nan_regression_row_is_dropped():
    good, bad = CENTRE_ANCHOR_800, CENTRE_ANCHOR_800 + 9 * 100 * 10 * 3
    cls, box = _outputs((800, 800), {good: 5.0, bad: 5.0}, {bad: [np.nan, 0.0, 0.0, 0.0]})
    d = decode_retinanet_output(cls, box, (800, 800), (800, 800), 0.1, 0.05)
    assert np.isfinite(d.boxes).all()
    np.testing.assert_allclose(d.boxes, [[384, 384, 416, 416]], atol=1e-4)
