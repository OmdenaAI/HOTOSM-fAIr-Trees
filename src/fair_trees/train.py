"""Torch side: RetinaNet builder with frozen batch norm, and a numpy-facing run function."""

from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet50
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

from fair_trees.serve import RunFn


def build_model(weights: Path | None, *, trainable_layers: int) -> RetinaNet:
    """Single-class RetinaNet (ResNet-50 FPN, frozen BN); loads `weights` strictly when given."""
    if not 0 <= trainable_layers <= 5:
        raise ValueError(f"trainable_layers must be in 0..5, got {trainable_layers}")
    body = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(
        body, trainable_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    model = RetinaNet(backbone, num_classes=1)
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # FrozenBatchNorm2d silently drops `num_batches_tracked`, so strict=True alone would
        # accept a trainable-BN checkpoint; reject that layout explicitly.
        if any(k.endswith("num_batches_tracked") for k in state):
            raise RuntimeError(f"{weights}: trainable batch-norm layout, expected frozen-BN state_dict")
        model.load_state_dict(state, strict=True)
    return model


def torch_run_fn(model: RetinaNet, device: torch.device) -> RunFn:
    """Raw network forward (backbone, FPN, head) on numpy batches, returning numpy outputs."""
    model.eval()
    model.to(device)

    def run(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(device)
            feats = list(model.backbone(x).values())
            out = model.head(feats)
            return (
                out["cls_logits"].float().cpu().numpy(),
                out["bbox_regression"].float().cpu().numpy(),
            )

    return run
