import pytest

torch = pytest.importorskip("torch")

from experiments.interpolate import interpolate  # noqa: E402
from fair_trees.train import build_model  # noqa: E402


def _save(tmp_path, name, state):
    path = tmp_path / name
    torch.save(state, path)
    return path


def test_floats_are_blended_and_other_tensors_kept(tmp_path):
    base = _save(tmp_path, "base.pt", {"w": torch.zeros(2), "n": torch.tensor([3])})
    tuned = _save(tmp_path, "tuned.pt", {"w": torch.ones(2), "n": torch.tensor([3])})
    out = interpolate(base, tuned, 0.3, tmp_path / "out" / "w.pt")
    state = torch.load(out, weights_only=True)
    assert torch.allclose(state["w"], torch.full((2,), 0.3))
    assert state["n"].dtype == torch.int64 and state["n"].item() == 3


def test_alpha_0_and_1_return_the_two_inputs(tmp_path):
    base = _save(tmp_path, "base.pt", {"w": torch.tensor([1.0, 2.0])})
    tuned = _save(tmp_path, "tuned.pt", {"w": torch.tensor([5.0, -2.0])})
    for alpha, expected in ((0.0, [1.0, 2.0]), (1.0, [5.0, -2.0])):
        state = torch.load(interpolate(base, tuned, alpha, tmp_path / f"{alpha}.pt"), weights_only=True)
        assert state["w"].tolist() == expected


def test_refuses_mismatched_checkpoints(tmp_path):
    base = _save(tmp_path, "base.pt", {"w": torch.zeros(2), "n": torch.tensor([3])})
    other_keys = _save(tmp_path, "a.pt", {"w": torch.zeros(2)})
    other_shape = _save(tmp_path, "b.pt", {"w": torch.zeros(3), "n": torch.tensor([3])})
    other_int = _save(tmp_path, "c.pt", {"w": torch.zeros(2), "n": torch.tensor([4])})
    for tuned, message in ((other_keys, "keys"), (other_shape, "shape"), (other_int, "differs")):
        with pytest.raises(ValueError, match=message):
            interpolate(base, tuned, 0.5, tmp_path / "o.pt")


def test_alpha_outside_0_1_is_refused(tmp_path):
    base = _save(tmp_path, "base.pt", {"w": torch.zeros(2)})
    with pytest.raises(ValueError, match="alpha"):
        interpolate(base, base, 1.5, tmp_path / "o.pt")


def test_blend_of_two_models_loads(tmp_path):
    base = _save(tmp_path, "base.pt", build_model(None, trainable_layers=0).state_dict())
    tuned = _save(tmp_path, "tuned.pt", build_model(None, trainable_layers=0).state_dict())
    build_model(interpolate(base, tuned, 0.5, tmp_path / "blend.pt"), trainable_layers=0)
