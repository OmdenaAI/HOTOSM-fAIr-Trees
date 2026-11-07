import numpy as np

from fair_trees.decode import (
    Detections,
    concat_detections,
    cut_boxes,
    decode_retinanet_output,
    drop_edge_boxes,
    filter_score,
    interior_sides,
    make_anchors,
    merge_group_detections,
    merge_window_detections,
    window_to_mosaic,
)
from fair_trees.preprocess import Window, WindowScale


def _dets(boxes, scores):
    return Detections(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores, np.float32))


def _as_set(d):
    return {(*np.round(b, 3).tolist(), round(float(s), 6)) for b, s in zip(d.boxes, d.scores, strict=True)}


def test_window_to_mosaic():
    d = window_to_mosaic(
        _dets([[384, 384, 416, 416]], [0.9]), Window(10, 20, 256, 256), WindowScale(800, 800, 3.125, 3.125)
    )
    np.testing.assert_allclose(d.boxes, [[132.88, 142.88, 143.12, 153.12]], atol=1e-4)
    np.testing.assert_allclose(d.scores, [0.9])


def test_interior_sides_respect_blank_neighbours():
    xs = ys = [0, 134, 244]
    active = {(x, y) for x in xs for y in ys} - {(244, 134)}
    assert interior_sides(Window(134, 134, 268, 268), active, xs, ys) == {"left", "top", "bottom"}
    assert interior_sides(Window(0, 0, 268, 268), active, xs, ys) == {"right", "bottom"}
    assert interior_sides(Window(0, 0, 256, 256), {(0, 0)}, [0], [0]) == set()


def test_drop_edge_boxes():
    win = Window(100, 100, 200, 200)  # band 0.05 -> 10 px
    d = _dets(
        [[105, 150, 140, 180], [150, 150, 180, 180], [260, 150, 295, 180], [150, 105, 180, 140]],
        [0.9, 0.8, 0.7, 0.6],
    )
    np.testing.assert_allclose(drop_edge_boxes(d, win, {"left"}, 0.05).scores, [0.8, 0.7, 0.6])
    np.testing.assert_allclose(drop_edge_boxes(d, win, {"left", "right"}, 0.05).scores, [0.8, 0.6])
    np.testing.assert_allclose(drop_edge_boxes(d, win, {"top"}, 0.05).scores, [0.9, 0.8, 0.7])
    np.testing.assert_allclose(drop_edge_boxes(d, win, set(), 0.05).scores, [0.9, 0.8, 0.7, 0.6])
    np.testing.assert_allclose(
        drop_edge_boxes(d, win, {"left", "right", "top"}, 0.0).scores, [0.9, 0.8, 0.7, 0.6]
    )


def test_cut_boxes_are_the_ones_drop_edge_boxes_drops():
    win = Window(100, 100, 200, 200)
    d = _dets([[105, 150, 140, 180], [150, 150, 180, 180], [260, 150, 295, 180]], [0.9, 0.8, 0.7])
    assert cut_boxes(d, win, {"left", "right"}, 0.05).tolist() == [True, False, True]


def test_group_split_across_windows_is_joined():
    a = (_dets([[80, 10, 100, 30]], [0.5]), np.array([True]))
    b = (_dets([[95, 12, 130, 32]], [0.7]), np.array([True]))
    m = merge_group_detections([a, b], 0.15, rule="join")
    np.testing.assert_allclose(m.boxes, [[80, 10, 130, 32]])
    np.testing.assert_allclose(m.scores, [0.7])


def test_cut_group_joins_full_box_of_other_window():
    a = (_dets([[80, 10, 100, 30]], [0.5]), np.array([True]))
    b = (_dets([[60, 10, 130, 30], [300, 300, 340, 340]], [0.9, 0.4]), np.array([False, False]))
    m = merge_group_detections([a, b], 0.15, rule="join")
    assert _as_set(m) == {(60.0, 10.0, 130.0, 30.0, 0.9), (300.0, 300.0, 340.0, 340.0, 0.4)}


def test_separate_groups_stay_separate():
    a = (_dets([[0, 0, 10, 10], [12, 0, 20, 10]], [0.9, 0.8]), np.array([True, False]))
    b = (_dets([[50, 50, 60, 60]], [0.5]), np.array([True]))
    m = merge_group_detections([a, b], 0.15, rule="join")
    assert len(m.scores) == 3  # a cut box only joins boxes of another window it overlaps


def test_merge_keeps_highest_and_filter_is_strict():
    d = concat_detections(
        [_dets([[0, 0, 10, 10]], [0.6]), _dets([[0, 0, 10, 9]], [0.9]), _dets([[50, 50, 60, 60]], [0.5])]
    )
    np.testing.assert_allclose(merge_window_detections(d, 0.15).scores, [0.9, 0.5])
    np.testing.assert_allclose(
        filter_score(_dets([[0, 0, 1, 1], [2, 2, 3, 3]], [0.5, 0.6]), 0.5).scores, [0.6]
    )


def test_concat_empty():
    d = concat_detections([])
    assert d.boxes.shape == (0, 4) and d.scores.shape == (0,)


def _outputs(input_hw, planted):
    n = sum(len(a) for a in make_anchors(*input_hw))
    cls = np.full((1, n, 1), -10.0, np.float32)
    for i, v in planted.items():
        cls[0, i, 0] = v
    return cls, np.zeros((1, n, 4), np.float32)


def test_sweep_equivalence_at_cap_boundary_with_ties():
    locs = [(r, c) for r in range(0, 100, 5) for c in range(0, 100, 5)]  # 400 non-overlapping boxes
    logits = {(r * 100 + c) * 9 + 3: 4.0 - 0.01 * i for i, (r, c) in enumerate(locs)}
    for i in range(295, 306):  # equal scores straddling the 300 cap
        r, c = locs[i]
        logits[(r * 100 + c) * 9 + 3] = 1.0
    cls, reg = _outputs((800, 800), logits)
    low = decode_retinanet_output(cls, reg, (800, 800), (800, 800), 0.01, 0.05)
    assert len(low.scores) == 300
    for t in (0.05, float(1 / (1 + np.exp(-0.99))), 0.9):
        high = decode_retinanet_output(cls, reg, (800, 800), (800, 800), t, 0.05)
        assert _as_set(filter_score(low, t)) == _as_set(high)


def test_sweep_equivalence_on_merged_output():
    rng = np.random.default_rng(0)
    hw, n = (800, 800), 120087
    windows = [Window(0, 0, 256, 256), Window(134, 0, 256, 256)]
    xs, ys, active = [0, 134], [0], {(0, 0), (134, 0)}
    scale = WindowScale(800, 800, 3.125, 3.125)
    outs = [
        (
            rng.normal(-3.0, 1.5, (1, n, 1)).astype(np.float32),
            rng.normal(0, 0.2, (1, n, 4)).astype(np.float32),
        )
        for _ in windows
    ]

    def pipeline(t):
        parts = []
        for w, (cls, reg) in zip(windows, outs, strict=True):
            d = window_to_mosaic(decode_retinanet_output(cls, reg, hw, hw, t, 0.05), w, scale)
            parts.append(drop_edge_boxes(d, w, interior_sides(w, active, xs, ys), 0.05))
        return merge_window_detections(concat_detections(parts), 0.15)

    low = pipeline(0.01)
    for t in (0.05, 0.2, 0.5):
        assert _as_set(filter_score(low, t)) == _as_set(pipeline(t))


def test_overlap_rule_joins_only_substantial_overlaps():
    a = (_dets([[0, 0, 60, 40], [200, 0, 260, 40]], [0.5, 0.6]), np.array([True, True]))
    b = (_dets([[25, 0, 100, 40], [255, 0, 330, 40]], [0.7, 0.4]), np.array([True, True]))
    m = merge_group_detections([a, b], 0.15, rule="overlap")
    # 35/60 of the smaller box shared: joined; 5/60: kept apart
    assert _as_set(m) == {
        (0.0, 0.0, 100.0, 40.0, 0.7),
        (200.0, 0.0, 260.0, 40.0, 0.6),
        (255.0, 0.0, 330.0, 40.0, 0.4),
    }


def test_drop_edge_rule_keeps_only_uncut_boxes():
    a = (_dets([[80, 10, 100, 30], [0, 0, 20, 20]], [0.5, 0.3]), np.array([True, False]))
    b = (_dets([[60, 10, 130, 30]], [0.9]), np.array([False]))
    m = merge_group_detections([a, b], 0.15, rule="drop_edge")
    assert _as_set(m) == {(60.0, 10.0, 130.0, 30.0, 0.9), (0.0, 0.0, 20.0, 20.0, 0.3)}


def test_drop_edge_is_the_default_rule():
    a = (_dets([[0, 0, 60, 40], [200, 0, 220, 20]], [0.5, 0.4]), np.array([True, False]))
    b = (_dets([[55, 0, 100, 40]], [0.7]), np.array([True]))
    assert _as_set(merge_group_detections([a, b], 0.15)) == {(200.0, 0.0, 220.0, 20.0, 0.4)}


def test_unknown_rule_rejected():
    import pytest

    with pytest.raises(ValueError, match="group_merge"):
        merge_group_detections([], 0.15, rule="glue")
