"""Torch side: RetinaNet builder, numpy-facing run function, and the training-step pieces."""

import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
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

logger = logging.getLogger(__name__)

CKPT_PATTERN = re.compile(r"^ckpt-e(\d{3})-s(\d{6})\.pt$")


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


def checkpoint_name(epoch: int, step: int) -> str:
    return f"ckpt-e{epoch:03d}-s{step:06d}.pt"


def config_hash(cfg: TrainParams) -> str:
    """sha256 of the canonical JSON of the config (zoom_weights keys as strings)."""
    data = dataclasses.asdict(cfg)
    data["zoom_weights"] = {str(k): v for k, v in data["zoom_weights"].items()}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def state_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    """sha256 over the sorted tensor names and their raw bytes."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        h.update(name.encode())
        h.update(tensors[name].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _checkpoints(out_dir: Path) -> list[tuple[tuple[int, int], Path]]:
    """Checkpoint files in `out_dir`, oldest first by (epoch, step)."""
    found = []
    for path in out_dir.iterdir() if out_dir.is_dir() else ():
        m = CKPT_PATTERN.match(path.name)
        if m:
            found.append(((int(m.group(1)), int(m.group(2))), path))
    return sorted(found)


def save_file(obj, out_dir: Path, name: str, *, tmp_dir: Path, keep_last: int | None = None) -> Path:
    """Save locally, copy to `<name>.part`, check the size, rename; rotate only after the rename."""
    if keep_last is not None and keep_last < 1:
        raise ValueError(f"keep_last must be None or >= 1, got {keep_last}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local = tmp_dir / name
    part = out_dir / f"{name}.part"
    try:
        torch.save(obj, local)
        size = local.stat().st_size
        try:
            shutil.copyfile(local, part)
            if part.stat().st_size != size:
                raise OSError(f"size mismatch copying {name}")
            os.replace(part, out_dir / name)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    finally:
        local.unlink(missing_ok=True)
    if keep_last is not None:
        existing = _checkpoints(out_dir)
        for _, old in existing[: max(len(existing) - keep_last, 0)]:
            old.unlink()
    return out_dir / name


def latest_checkpoint(out_dir: Path) -> tuple[Path, dict] | None:
    """Newest checkpoint that loads; unreadable ones are logged and skipped."""
    for _, path in reversed(_checkpoints(out_dir)):
        try:
            return path, torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            logger.warning(f"{path.name}: unreadable checkpoint ({error}); trying the previous one")
    return None


def check_resume(
    state: Mapping, *, config_hash: str, w0_hash: str, manifest_hash: str, repo_sha: str, force: bool
) -> None:
    """Refuse to resume against changed config, start weights or data unless `force`."""
    if not force:
        if state["config_hash"] != config_hash:
            raise ValueError("config hash changed since the checkpoint; pass force=True to resume anyway")
        if state["w0_hash"] != w0_hash:
            raise ValueError("w0 hash changed: the start weights differ")
        if state["manifest_hash"] != manifest_hash:
            raise ValueError("manifest hash changed: the training data differ")
    if state["repo_sha"] != repo_sha:
        logger.warning(f"repo SHA changed since the checkpoint: {state['repo_sha']} -> {repo_sha}")
