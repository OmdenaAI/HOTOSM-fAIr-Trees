import numpy as np
import pytest

pytest.importorskip("scipy")

from fair_trees.decode import make_anchors  # noqa: E402
from fair_trees.evaluate import EvalUnit, evaluate_units  # noqa: E402
from fair_trees.params import InferenceParams  # noqa: E402
from fair_trees.preprocess import mosaic_chips, window_grid  # noqa: E402
from fair_trees.serve import predict_mosaic_classes  # noqa: E402

X0 = Y0 = 2**19
# P7 at an 800 px input: 7 x 7 locations after P3-P6 (100, 50, 25, 13 per side); location (6, 6),
# aspect 1.0, size 812 -> a box from 278 px to the bottom-right corner, cut by an interior window edge
P7_BIG = 9 * (100**2 + 50**2 + 25**2 + 13**2) + (6 * 7 + 6) * 9 + 5


def _one_confident_window(n_windows):
    """Group logits: the first window of each pass scores the big box 0.95, every other window 0.047."""
    calls = []

    def run(batch):
        n = sum(len(a) for a in make_anchors(batch.shape[2], batch.shape[3]))
        cls = np.full((1, n, 2), -10.0, np.float32)
        cls[0, P7_BIG, 1] = 3.0 if len(calls) % n_windows == 0 else -3.0
        calls.append(batch.shape)
        return cls, np.zeros((1, n, 4), np.float32)

    return run


def test_group_eval_matches_serving_at_its_threshold(write_oam_chip):
    [m] = mosaic_chips([write_oam_chip(X0 + i, Y0 + j) for i in range(8) for j in range(4)])
    params = InferenceParams(confidence_threshold=0.5, group_merge="join")
    n_windows = len(window_grid(*m.rgb.shape[:2], m.gsd_m, params.group_window_m, params.window_overlap))
    run = _one_confident_window(n_windows)
    [served] = predict_mosaic_classes(run, m, params, (1,))
    assert len(served.scores) == 1  # the low boxes never reach the merge in serving
    unit = EvalUnit("u", m, gt_boxes=served.boxes)
    report = evaluate_units(run, [unit], params, class_index=1)
    assert (report.units[0]["tp"], report.units[0]["fp"], report.units[0]["fn"]) == (1, 0, 0)
