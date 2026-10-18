"""Opt-in parity: our pipeline vs DeepForest predict_tile on OSBS_029, DeepForest-compatible mode."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from experiments.neon_eval import DF_COMPAT  # noqa: E402
from fair_trees.evaluate import match_hungarian  # noqa: E402
from fair_trees.preprocess import array_mosaic  # noqa: E402
from fair_trees.serve import predict_mosaic  # noqa: E402
from fair_trees.train import torch_run_fn  # noqa: E402

DF_REVISION = "cc21436bc5d572dde8ff5f93c1e71a32f563cace"  # weecology/deepforest-tree commit


def test_end_to_end_matches_deepforest(neon_model, sample_rgb, sample_path):
    pytest.importorskip("deepforest")
    from deepforest import main

    df_model = main.deepforest()
    df_model.load_model("weecology/deepforest-tree", revision=DF_REVISION)
    df = df_model.predict_tile(path=str(sample_path), patch_size=400, patch_overlap=0.05, iou_threshold=0.15)
    theirs = df[["xmin", "ymin", "xmax", "ymax"]].to_numpy(np.float32)
    their_scores = df["score"].to_numpy(np.float32)

    ours = predict_mosaic(
        torch_run_fn(neon_model, torch.device("cpu")), array_mosaic(sample_rgb, 0.1), DF_COMPAT
    )

    # DeepForest truncates box coordinates to whole pixels; apply the same to ours before comparing.
    ours_boxes = np.floor(ours.boxes)
    # Criteria [P]: counts within max(2, 5%), >= 95% matched at IoU > 0.9, median score gap < 0.02.
    assert abs(len(ours.scores) - len(theirs)) <= max(2, round(0.05 * len(theirs)))
    pairs = match_hungarian(ours_boxes, theirs, iou_threshold=0.9)
    assert len(pairs) >= 0.95 * min(len(ours.scores), len(theirs))
    gaps = [abs(float(ours.scores[i]) - float(their_scores[j])) for i, j in pairs]
    assert np.median(gaps) < 0.02
