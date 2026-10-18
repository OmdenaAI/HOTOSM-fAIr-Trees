import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from fair_trees.export import OPSET, export_onnx_bytes  # noqa: E402
from fair_trees.params import INPUT_NAME, OUTPUT_NAMES  # noqa: E402
from fair_trees.preprocess import preprocess_window  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402
from fair_trees.train import build_model, torch_run_fn  # noqa: E402


@pytest.fixture(scope="module")
def exported():
    torch.manual_seed(0)
    model = build_model(None, trainable_layers=0).eval()
    return model, export_onnx_bytes(model)


def test_graph_contract(exported):
    _, data = exported
    assert isinstance(data, bytes)
    m = onnx.load_from_string(data)
    onnx.checker.check_model(m)
    assert [i.name for i in m.graph.input] == [INPUT_NAME]
    assert [o.name for o in m.graph.output] == list(OUTPUT_NAMES)
    assert next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")) == OPSET == 17
    dims = m.graph.input[0].type.tensor_type.shape.dim
    assert dims[0].dim_value == 1 and dims[1].dim_value == 3
    assert dims[2].dim_param and dims[3].dim_param


@pytest.mark.parametrize("hw", [(256, 256), (300, 420)])
def test_ort_matches_torch(exported, hw):
    model, data = exported
    rgb = np.random.default_rng(1).integers(0, 256, (*hw, 3), dtype=np.uint8)
    batch, _ = preprocess_window(rgb)
    o_cls, o_reg = ort_run_fn(ort.InferenceSession(data, providers=["CPUExecutionProvider"]))(batch)
    t_cls, t_reg = torch_run_fn(model, torch.device("cpu"))(batch)
    np.testing.assert_allclose(o_cls, t_cls, atol=2e-3)
    np.testing.assert_allclose(o_reg, t_reg, atol=2e-3)


def test_cli_writes_onnx(tmp_path):
    torch.manual_seed(0)
    ckpt = tmp_path / "w.pt"
    torch.save(build_model(None, trainable_layers=0).state_dict(), ckpt)
    out = tmp_path / "m.onnx"
    subprocess.run([sys.executable, "-m", "fair_trees.export", str(ckpt), str(out)], check=True)
    onnx.checker.check_model(onnx.load(str(out)))
