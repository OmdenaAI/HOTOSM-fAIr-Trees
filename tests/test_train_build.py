import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torchvision.ops.misc import FrozenBatchNorm2d  # noqa: E402

from fair_trees.decode import make_anchors  # noqa: E402
from fair_trees.train import build_model, checkpoint_num_classes, torch_run_fn  # noqa: E402


@pytest.fixture(scope="module")
def model3():
    torch.manual_seed(0)
    return build_model(None, trainable_layers=3)


def test_frozen_bn_layout_301_keys(model3):
    assert len(model3.state_dict()) == 301
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model3.modules())
    assert sum(isinstance(m, FrozenBatchNorm2d) for m in model3.backbone.body.modules()) == 53


@pytest.mark.parametrize(
    "layers, trainable, frozen",
    [
        (
            3,
            ("backbone.body.layer2.", "backbone.body.layer3.", "backbone.body.layer4."),
            ("backbone.body.conv1.", "backbone.body.layer1."),
        ),
        (0, (), ("backbone.body.",)),
    ],
)
def test_requires_grad_follows_trainable_layers(layers, trainable, frozen):
    model = build_model(None, trainable_layers=layers)
    for name, p in model.named_parameters():
        if name.startswith(("backbone.fpn.", "head.")):
            assert p.requires_grad, name
        elif name.startswith(frozen):
            assert not p.requires_grad, name
        elif name.startswith(trainable):
            assert p.requires_grad, name


def test_invalid_trainable_layers():
    with pytest.raises(ValueError):
        build_model(None, trainable_layers=6)


def test_round_trip_and_strict_load(model3, tmp_path):
    path = tmp_path / "w.pt"
    torch.save(model3.state_dict(), path)
    loaded = build_model(path, trainable_layers=0)
    for (k, a), (_, b) in zip(model3.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert torch.equal(a, b), k


def test_strict_load_rejects_trainable_bn_layout(tmp_path):
    from torchvision.models.detection import retinanet_resnet50_fpn

    sd = retinanet_resnet50_fpn(weights=None, weights_backbone=None, num_classes=1).state_dict()
    assert len(sd) == 354
    torch.save(sd, tmp_path / "bn.pt")
    with pytest.raises(RuntimeError):
        build_model(tmp_path / "bn.pt", trainable_layers=3)


def test_torch_run_fn_shapes_match_numpy_anchors(model3):
    run = torch_run_fn(model3, torch.device("cpu"))
    batch = np.zeros((1, 3, 256, 320), np.float32)
    cls, reg = run(batch)
    a = sum(len(x) for x in make_anchors(256, 320))
    assert cls.shape == (1, a, 1) and reg.shape == (1, a, 4)
    assert cls.dtype == np.float32 and not model3.training


def _cls_weight(state):
    return state["head.classification_head.cls_logits.weight"]


def test_build_reads_k_from_checkpoint(tmp_path):
    path = tmp_path / "k2.pt"
    torch.save(build_model(None, trainable_layers=0, num_classes=2).state_dict(), path)
    model = build_model(path, trainable_layers=0)
    assert model.head.classification_head.num_classes == 2
    assert checkpoint_num_classes(torch.load(path, weights_only=True)) == 2
    assert build_model(None, trainable_layers=0).head.classification_head.num_classes == 1


def test_expand_k1_to_k2_keeps_tree_rows(model3, tmp_path):
    path = tmp_path / "k1.pt"
    torch.save(model3.state_dict(), path)
    a = build_model(path, trainable_layers=0, num_classes=2)
    b = build_model(path, trainable_layers=0, num_classes=2)
    old, new = model3.state_dict(), a.state_dict()
    bias_key = "head.classification_head.cls_logits.bias"
    assert torch.equal(_cls_weight(new)[0::2], _cls_weight(old))
    assert torch.equal(new[bias_key][0::2], old[bias_key])
    assert torch.allclose(new[bias_key][1::2], torch.full((9,), -math.log(99)))
    assert torch.equal(_cls_weight(new)[1::2], _cls_weight(b.state_dict())[1::2])
    assert 0.005 < float(_cls_weight(new)[1::2].std()) < 0.015
    batch = np.random.default_rng(0).normal(size=(1, 3, 128, 160)).astype(np.float32)
    cls1, _ = torch_run_fn(model3, torch.device("cpu"))(batch)
    cls2, _ = torch_run_fn(a, torch.device("cpu"))(batch)
    np.testing.assert_allclose(cls2[..., 0], cls1[..., 0], atol=1e-5)


def test_expansion_keeps_state_metadata(model3):
    from fair_trees.train import _add_group_rows

    state = model3.state_dict()
    assert _add_group_rows(state)._metadata is state._metadata


def test_k2_into_k1_refused(tmp_path):
    path = tmp_path / "k2.pt"
    torch.save(build_model(None, trainable_layers=0, num_classes=2).state_dict(), path)
    with pytest.raises(ValueError, match="2 classes.*1"):
        build_model(path, trainable_layers=0, num_classes=1)
