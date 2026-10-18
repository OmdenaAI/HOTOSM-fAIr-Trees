"""Torch side: RetinaNet builder, numpy-facing run function, and the training-step pieces."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.image_list import ImageList
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

from fair_trees.dataset import Region, RegionCache, SampleSpec, load_sample
from fair_trees.params import TrainParams
from fair_trees.preprocess import preprocess_window
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


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Detached clones of the trainable parameters: the L2-SP reference point."""
    return {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}


def l2sp_penalty(model: nn.Module, ref: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Sum of squared distances to `ref` over the parameters named in it (constant alpha, not alpha/2)."""
    terms = [
        ((p.float() - ref[name].to(p.device).float()) ** 2).sum()
        for name, p in model.named_parameters()
        if name in ref
    ]
    if not terms:
        return torch.zeros((), dtype=torch.float32)
    return torch.stack(terms).sum()


class Batch(NamedTuple):
    indices: list[int]
    images: torch.Tensor  # (B, 3, H, W) float32, zero-padded bottom and right
    image_sizes: list[tuple[int, int]]  # resized (h, w) before padding
    targets: list[dict[str, torch.Tensor]]


def collate(items: Sequence[tuple[int, np.ndarray, tuple[int, int], np.ndarray]]) -> Batch:
    """Stack `(index, image (3, Hp, Wp), resized (h, w), boxes (K, 4))` items into one padded batch."""
    height = max(item[1].shape[1] for item in items)
    width = max(item[1].shape[2] for item in items)
    images = torch.zeros((len(items), 3, height, width), dtype=torch.float32)
    for row, (_, image, _, _) in enumerate(items):
        images[row, :, : image.shape[1], : image.shape[2]] = torch.from_numpy(np.ascontiguousarray(image))
    targets = []
    for _, _, _, boxes in items:
        boxes_t = torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)).reshape(-1, 4)
        targets.append({"boxes": boxes_t, "labels": torch.zeros(len(boxes_t), dtype=torch.int64)})
    return Batch([item[0] for item in items], images, [tuple(item[2]) for item in items], targets)


class SampleDataset(Dataset):
    """Planned samples from `start` on, preprocessed exactly as serving does; boxes in resized pixels."""

    def __init__(
        self, specs: Sequence[SampleSpec], regions: Sequence[Region], cfg: TrainParams, *, start: int = 0
    ):
        self.specs = specs
        self.regions = regions
        self.cfg = cfg
        self.start = start
        self._cache: RegionCache | None = None

    def __getstate__(self) -> dict:
        # the cache holds open rasters; each DataLoader worker builds its own
        return {**self.__dict__, "_cache": None}

    def __len__(self) -> int:
        return len(self.specs) - self.start

    def __getitem__(self, i: int) -> tuple[int, np.ndarray, tuple[int, int], np.ndarray]:
        if self._cache is None:
            self._cache = RegionCache(self.regions)
        rgb, boxes = load_sample(self.specs[self.start + i], self._cache, self.cfg)
        batch, scale = preprocess_window(rgb)
        boxes = boxes * np.array([scale.scale_x, scale.scale_y, scale.scale_x, scale.scale_y], np.float32)
        # a degenerate box makes the box coder's log(w) infinite
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
        return (
            self.start + i,
            batch[0],
            (scale.resized_height, scale.resized_width),
            np.ascontiguousarray(boxes[keep], dtype=np.float32),
        )


def make_loader(
    specs: Sequence[SampleSpec],
    regions: Sequence[Region],
    cfg: TrainParams,
    *,
    start: int = 0,
    num_workers: int = 2,
) -> DataLoader:
    """Sequential loader over `specs[start:]`; DataLoader keeps worker order, so a resume is exact."""
    return DataLoader(
        SampleDataset(specs, regions, cfg, start=start),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )


def forward_loss(
    model: RetinaNet,
    images: torch.Tensor,
    image_sizes: list[tuple[int, int]],
    targets: list[dict],
) -> dict[str, torch.Tensor]:
    """Training losses on preprocessed batches; GeneralizedRCNNTransform never runs."""
    features = list(model.backbone(images).values())
    head_outputs = model.head(features)
    anchors = model.anchor_generator(ImageList(images, image_sizes), features)
    return model.compute_loss(targets, head_outputs, anchors)
