"""Opt-in parity: ONNX vs torch raw outputs on NEON.pt, atol 2e-3."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from fair_trees.export import export_onnx_bytes  # noqa: E402
from fair_trees.preprocess import preprocess_window  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402
from fair_trees.train import torch_run_fn  # noqa: E402


@pytest.fixture(scope="module")
def session(neon_model):
    return ort.InferenceSession(export_onnx_bytes(neon_model), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("hw", [(400, 400), (256, 256), (300, 420), (800, 800), (800, 1088)])
def test_onnx_equals_torch(neon_model, session, sample_rgb, hw):
    reps = (-(-hw[0] // 400), -(-hw[1] // 400), 1)
    rgb = np.ascontiguousarray(np.tile(sample_rgb, reps)[: hw[0], : hw[1]])
    batch, _ = preprocess_window(rgb)
    o_cls, o_reg = ort_run_fn(session)(batch)
    t_cls, t_reg = torch_run_fn(neon_model, torch.device("cpu"))(batch)
    np.testing.assert_allclose(o_cls, t_cls, atol=2e-3)
    np.testing.assert_allclose(o_reg, t_reg, atol=2e-3)
