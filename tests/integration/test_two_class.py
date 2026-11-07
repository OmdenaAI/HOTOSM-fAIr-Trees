from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")

from fair_trees.preprocess import preprocess_window  # noqa: E402
from fair_trees.train import build_model, torch_run_fn  # noqa: E402

WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "NEON.pt"  # as conftest's neon_model loads


def test_release_weights_grow_a_group_class_with_tree_outputs_unchanged(neon_model, sample_rgb):
    batch, _ = preprocess_window(sample_rgb)
    two = build_model(WEIGHTS, trainable_layers=0, num_classes=2)
    assert two.head.classification_head.num_classes == 2
    cls1, reg1 = torch_run_fn(neon_model, torch.device("cpu"))(batch)
    cls2, reg2 = torch_run_fn(two, torch.device("cpu"))(batch)
    np.testing.assert_allclose(cls2[..., :1], cls1, atol=1e-5)
    np.testing.assert_allclose(reg2, reg1, atol=1e-5)
    assert (1 / (1 + np.exp(-cls2[..., 1]))).max() < 0.1  # the new group class starts near its 0.01 prior
