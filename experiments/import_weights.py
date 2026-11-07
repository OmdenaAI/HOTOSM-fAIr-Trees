"""Bring a torchvision RetinaNet state dict saved with trainable BatchNorm into our frozen-BN layout.

    python -m experiments.import_weights NEON_OAM.pt NEON_OAM-frozen-bn.pt

Used to score outside weights (e.g. another team's DeepForest fine-tune) with our harness. Our models use
FrozenBatchNorm2d, which has no ``num_batches_tracked``; ``build_model`` refuses a checkpoint carrying those
counters so a trainable-BN layout is never loaded by accident. Here they are dropped on purpose: the running
statistics, weights and biases are kept, every other key must match our model exactly.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path


def import_retinanet(src: Path, dst: Path) -> Path:
    """Write ``src`` without its BatchNorm counters to ``dst``, loaded strictly through our model first."""
    import torch

    from fair_trees.train import build_model, checkpoint_num_classes

    state = torch.load(src, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.endswith("num_batches_tracked")}
    model = build_model(None, trainable_layers=0, num_classes=checkpoint_num_classes(state))
    model.load_state_dict(state, strict=True)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), dst)
    return dst


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args(argv)
    print(import_retinanet(args.src, args.dst))


if __name__ == "__main__":
    main()
