"""End-to-end parity: the served GeoJSON of a two-class model is the same through ONNX Runtime and PyTorch.

Unlike ``test_export``, which compares raw network outputs, this runs the whole serving path (sliding windows,
decoding, dropping boxes cut by a window edge, cross-window merging, georeferencing) through both engines.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from fair_trees.decode import merge_group_detections  # noqa: E402
from fair_trees.export import export_onnx_bytes  # noqa: E402
from fair_trees.params import OUTPUT_NAMES, THRESHOLDS_KEY, WINDOWS_KEY, InferenceParams  # noqa: E402
from fair_trees.preprocess import mosaic_chips  # noqa: E402
from fair_trees.serve import ort_run_fn, predict_group_parts, predict_session  # noqa: E402
from fair_trees.train import build_model, torch_run_fn  # noqa: E402

# One z20 chip (256 px, about 38 m) with 20 m windows and no overlap: a 2x2 grid with interior edges that the
# tree and group classes share, so each engine runs 4 windows.
THRESHOLDS = {"tree": 0.5, "tree_group": 0.5}
WINDOWS = {"window_m": 20.0, "group_window_m": 20.0}
OVERLAP = 0.0


class TorchSession:
    """The PyTorch model behind the onnxruntime session interface ``predict_session`` uses."""

    def __init__(self, model, metadata):
        self._run = torch_run_fn(model, torch.device("cpu"))
        self._metadata = metadata

    def get_inputs(self):
        return [SimpleNamespace(name="images")]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=self._metadata)

    def run(self, output_names, feeds):
        assert list(output_names) == list(OUTPUT_NAMES)
        return list(self._run(next(iter(feeds.values()))))


@pytest.fixture(scope="module")
def sessions():
    """A seeded two-class model whose class head gives a few hundred boxes over 0.5, as ONNX and torch."""
    torch.manual_seed(0)
    model = build_model(None, trainable_layers=0, num_classes=2).eval()
    with torch.no_grad():  # the default head prior (0.01) puts every score far below any threshold
        model.head.classification_head.cls_logits.weight.normal_(0, 0.05)
        model.head.classification_head.cls_logits.bias.fill_(-1.5)
    onnx_session = ort.InferenceSession(
        export_onnx_bytes(model, thresholds=THRESHOLDS, windows=WINDOWS), providers=["CPUExecutionProvider"]
    )
    metadata = dict(onnx_session.get_modelmeta().custom_metadata_map)
    return onnx_session, TorchSession(model, metadata)


@pytest.fixture
def chip(write_oam_chip):
    return write_oam_chip()


def _key(feature):
    ring = np.asarray(feature["geometry"]["coordinates"][0])
    return (feature["properties"]["class"], *ring.min(axis=0).round(7), *ring.max(axis=0).round(7))


def test_onnx_geojson_equals_torch(sessions, chip, tmp_path):
    onnx_session, torch_session = sessions
    assert json.loads(torch_session.get_modelmeta().custom_metadata_map[THRESHOLDS_KEY]) == THRESHOLDS
    assert json.loads(torch_session.get_modelmeta().custom_metadata_map[WINDOWS_KEY]) == WINDOWS

    params = {"window_overlap": OVERLAP}  # windows and thresholds come from the model's metadata
    got = sorted(predict_session(onnx_session, tmp_path, params)["features"], key=_key)
    want = sorted(predict_session(torch_session, tmp_path, params)["features"], key=_key)

    classes = [f["properties"]["class"] for f in want]
    assert classes.count(1) > 10 and classes.count(2) > 10  # both classes are exercised
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g["properties"]["class"] == w["properties"]["class"]
        assert g["properties"].get("in_group") == w["properties"].get("in_group")
        assert g["properties"]["score"] == pytest.approx(w["properties"]["score"], abs=2e-4)
        # 1e-8 degrees is about 1 mm on the ground
        np.testing.assert_allclose(g["geometry"]["coordinates"], w["geometry"]["coordinates"], atol=1e-8)


def test_groups_cut_by_a_window_edge_are_dropped_the_same_in_both_engines(sessions, chip):
    onnx_session, torch_session = sessions
    [mosaic] = mosaic_chips([chip])
    params = InferenceParams(class_thresholds=tuple(THRESHOLDS.values()), window_overlap=OVERLAP, **WINDOWS)
    assert params.group_merge == "drop_edge"

    parts = {}
    for name, session in (("onnx", onnx_session), ("torch", torch_session)):
        parts[name] = predict_group_parts(ort_run_fn(session), mosaic, params)
        groups = merge_group_detections(parts[name], params.iou_threshold, params.group_merge)
        assert len(groups.scores) > 10, name
        uncut = np.concatenate([d.boxes[~cut] for d, cut in parts[name]])
        # every served group is a whole (uncut) window box: no cut box survives drop_edge
        assert all(np.abs(uncut - box).max(axis=1).min() < 1e-3 for box in groups.boxes), name

    assert sum(int(cut.sum()) for _, cut in parts["torch"]) > 10  # the edge rule is exercised
    for (d_onnx, cut_onnx), (d_torch, cut_torch) in zip(parts["onnx"], parts["torch"], strict=True):
        np.testing.assert_array_equal(cut_onnx, cut_torch)  # the same boxes are cut in each window
        np.testing.assert_allclose(d_onnx.boxes, d_torch.boxes, atol=1e-3)
