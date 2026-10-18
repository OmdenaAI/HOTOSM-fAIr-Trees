"""Opt-in parity with torchvision on NEON.pt: strict load, anchors, preprocessing and decoding."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")

from torchvision.models.detection.image_list import ImageList  # noqa: E402

from fair_trees.decode import decode_retinanet_output, make_anchors  # noqa: E402
from fair_trees.preprocess import preprocess_window  # noqa: E402


def test_strict_load_301_keys(neon_model):
    assert len(neon_model.state_dict()) == 301


@pytest.mark.parametrize("hw", [(800, 800), (800, 1088), (800, 960), (256, 256)])
def test_anchors_equal_torchvision(neon_model, hw):
    images = torch.zeros(1, 3, *hw)
    with torch.no_grad():
        feats = list(neon_model.backbone(images).values())
    tv = neon_model.anchor_generator(ImageList(images, [hw]), feats)[0].numpy()
    np.testing.assert_array_equal(np.concatenate(make_anchors(*hw)), tv)


@pytest.mark.parametrize("hw", [(400, 400), (256, 256), (300, 420)])
def test_preprocess_equals_transform_on_upscale(neon_model, sample_rgb, hw):
    rgb = np.ascontiguousarray(
        np.pad(sample_rgb, ((0, 0), (0, 20), (0, 0)), mode="reflect")[: hw[0], : hw[1]]
    )
    ours, scale = preprocess_window(rgb)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)
    image_list, _ = neon_model.transform([x])
    assert tuple(image_list.image_sizes[0]) == (scale.resized_height, scale.resized_width)
    assert tuple(image_list.tensors.shape) == ours.shape
    assert np.abs(image_list.tensors.numpy() - ours).max() < 0.05


@pytest.mark.parametrize("conf", [0.1, 0.3])
def test_numpy_decode_equals_postprocess_detections(neon_model, sample_rgb, conf, monkeypatch):
    batch, scale = preprocess_window(sample_rgb)
    image_hw = (scale.resized_height, scale.resized_width)
    x = torch.from_numpy(batch)
    monkeypatch.setattr(neon_model, "score_thresh", conf)
    monkeypatch.setattr(neon_model, "nms_thresh", 0.05)
    with torch.no_grad():
        feats = list(neon_model.backbone(x).values())
        head = neon_model.head(feats)
        anchors = neon_model.anchor_generator(ImageList(x, [image_hw]), feats)
        per_level = [f.size(2) * f.size(3) for f in feats]
        a = head["cls_logits"].size(1) // sum(per_level)
        split = [n * a for n in per_level]
        split_head = {k: list(v.split(split, dim=1)) for k, v in head.items()}
        split_anchors = [list(anc.split(split)) for anc in anchors]
        tv = neon_model.postprocess_detections(split_head, split_anchors, [image_hw])[0]
    ours = decode_retinanet_output(
        head["cls_logits"].numpy(), head["bbox_regression"].numpy(), batch.shape[2:], image_hw, conf, 0.05
    )
    assert len(ours.scores) == len(tv["scores"]) > 0
    order = tv["scores"].argsort(descending=True)
    np.testing.assert_allclose(ours.scores, tv["scores"][order].numpy(), atol=2e-3)
    np.testing.assert_allclose(ours.boxes, tv["boxes"][order].numpy(), atol=0.5)
