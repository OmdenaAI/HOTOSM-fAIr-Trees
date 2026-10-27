import re

import pytest

torch = pytest.importorskip("torch")

from experiments.interpolate import blend, interpolate  # noqa: E402
from fair_trees.train import build_model  # noqa: E402


def test_floats_are_blended_and_other_tensors_kept():
    base = {"w": torch.zeros(2), "n": torch.tensor([3])}
    state = blend(base, {"w": torch.ones(2), "n": torch.tensor([3])}, 0.3)
    assert torch.allclose(state["w"], torch.full((2,), 0.3))
    assert state["n"].dtype == torch.int64 and state["n"].item() == 3


def test_alpha_0_and_1_return_the_two_inputs():
    base, tuned = {"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([5.0, -2.0])}
    assert blend(base, tuned, 0.0)["w"].tolist() == [1.0, 2.0]
    assert blend(base, tuned, 1.0)["w"].tolist() == [5.0, -2.0]


def test_refuses_mismatched_checkpoints():
    base = {"w": torch.zeros(2), "n": torch.tensor([3])}
    other_keys = {"w": torch.zeros(2)}
    other_shape = {"w": torch.zeros(3), "n": torch.tensor([3])}
    other_int = {"w": torch.zeros(2), "n": torch.tensor([4])}
    for tuned, message in ((other_keys, "keys"), (other_shape, "shape"), (other_int, "differs")):
        with pytest.raises(ValueError, match=message):
            blend(base, tuned, 0.5)


def test_alpha_outside_0_1_is_refused():
    with pytest.raises(ValueError, match="alpha"):
        blend({"w": torch.zeros(2)}, {"w": torch.zeros(2)}, 1.5)


def test_blend_of_two_models_loads(tmp_path):
    for name in ("base.pt", "tuned.pt"):
        torch.save(build_model(None, trainable_layers=0).state_dict(), tmp_path / name)
    out = interpolate(tmp_path / "base.pt", tmp_path / "tuned.pt", 0.5, tmp_path / "blend" / "w.pt")
    build_model(out, trainable_layers=0)


def test_legacy_fpn_keys_are_read_through_the_model(tmp_path):
    # the release checkpoint stores FPN convs as `inner_blocks.0.weight`; current torchvision saves
    # `inner_blocks.0.0.weight` and renames the old keys only when loading into a model
    base_state = build_model(None, trainable_layers=0).state_dict()
    legacy = {re.sub(r"(fpn\.(?:inner|layer)_blocks\.\d+)\.0\.", r"\1.", k): v for k, v in base_state.items()}
    assert "backbone.fpn.inner_blocks.0.weight" in legacy
    torch.save(legacy, tmp_path / "legacy.pt")
    tuned_state = build_model(None, trainable_layers=0).state_dict()
    torch.save(tuned_state, tmp_path / "tuned.pt")
    out = interpolate(tmp_path / "legacy.pt", tmp_path / "tuned.pt", 1.0, tmp_path / "blend.pt")
    state = torch.load(out, weights_only=True)
    assert state.keys() == tuned_state.keys()
    key = "backbone.fpn.inner_blocks.0.0.weight"
    assert torch.equal(state[key], tuned_state[key])
