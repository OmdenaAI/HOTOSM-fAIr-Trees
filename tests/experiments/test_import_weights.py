import pytest
import torch

from experiments.import_weights import import_retinanet
from fair_trees.train import build_model


def _plain_bn_state():
    """A torchvision RetinaNet state dict as saved with trainable BatchNorm: ours plus the BN counters."""
    state = build_model(None, trainable_layers=0).state_dict()
    for name in [k for k in state if k.endswith(".running_mean")]:
        state[name.replace("running_mean", "num_batches_tracked")] = torch.tensor(7)
    return state


def test_import_drops_bn_counters_and_loads_strictly(tmp_path):
    src, dst = tmp_path / "plain.pt", tmp_path / "out" / "frozen.pt"
    state = _plain_bn_state()
    torch.save(state, src)
    with pytest.raises(RuntimeError, match="trainable batch-norm"):
        build_model(src, trainable_layers=0)

    assert import_retinanet(src, dst) == dst

    saved = torch.load(dst, weights_only=True)
    assert not any(k.endswith("num_batches_tracked") for k in saved)
    model = build_model(dst, trainable_layers=0)
    name = "head.classification_head.cls_logits.weight"
    assert torch.equal(model.state_dict()[name], state[name])


def test_import_refuses_a_state_dict_that_is_not_our_retinanet(tmp_path):
    src = tmp_path / "plain.pt"
    state = _plain_bn_state()
    state.pop("head.regression_head.bbox_reg.bias")
    torch.save(state, src)
    with pytest.raises(RuntimeError, match="bbox_reg.bias"):
        import_retinanet(src, tmp_path / "frozen.pt")
    assert not (tmp_path / "frozen.pt").exists()
