"""Blend two checkpoints of the same model: ``(1 - alpha) * base + alpha * tuned`` (weight-space ensembling).

    python -m experiments.interpolate base.pt tuned.pt 0.5 out.pt

Used to check whether a point between the release weights and a fine-tune keeps the fine-tune's gain
without its forgetting. Float tensors are blended; any other tensor must be equal in both files.
"""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path


def blend(base: Mapping, tuned: Mapping, alpha: float) -> dict:
    """``(1 - alpha) * base + alpha * tuned`` per float tensor; alpha 0 is ``base``, 1 is ``tuned``."""
    import torch

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if base.keys() != tuned.keys():
        raise ValueError(f"checkpoint keys differ: {sorted(base.keys() ^ tuned.keys())[:5]}")
    blended = {}
    for key, x in base.items():
        y = tuned[key]
        if x.shape != y.shape:
            raise ValueError(f"{key}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
        if x.is_floating_point():
            blended[key] = torch.lerp(x, y.to(x.dtype), alpha)
        elif torch.equal(x, y):
            blended[key] = x
        else:
            raise ValueError(f"{key}: non-float tensor differs between the checkpoints")
    return blended


def interpolate(base: Path, tuned: Path, alpha: float, out: Path) -> Path:
    """Blend two model checkpoints into ``out`` and return it.

    Both are read through the model, so a checkpoint saved by an older torchvision (the release weights
    keep the old FPN key names) is renamed to the current layout before blending.
    """
    import torch

    from fair_trees.train import build_model

    states = [build_model(p, trainable_layers=0).state_dict() for p in (base, tuned)]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blend(*states, alpha), out)
    return out


def cli(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("base", type=Path)
    p.add_argument("tuned", type=Path)
    p.add_argument("alpha", type=float)
    p.add_argument("out", type=Path)
    args = p.parse_args(argv)
    print(interpolate(args.base, args.tuned, args.alpha, args.out))


if __name__ == "__main__":
    cli()
