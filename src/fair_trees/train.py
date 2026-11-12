"""Torch side: RetinaNet builder, numpy-facing run function, the training loop and `fine_tune`."""

import csv
import dataclasses
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops
from torchvision.ops import sigmoid_focal_loss
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

from fair_trees.dataset import (
    Region,
    RegionCache,
    SampleSpec,
    any_label_on_chips,
    load_fair_dataset,
    load_sample,
    plan_epoch,
    sampler_stats,
    total_chips,
)
from fair_trees.evaluate import (
    PICK_THRESHOLDS,
    PRIOR_THRESHOLD,
    EvalUnit,
    evaluate_units,
    pick_threshold,
    region_units,
)
from fair_trees.params import (
    CLASSES,
    WINDOW_FIELDS,
    InferenceParams,
    TrainParams,
    parse_class_thresholds,
    parse_model_windows,
    parse_train_params,
)
from fair_trees.preprocess import preprocess_window
from fair_trees.serve import RunFn

logger = logging.getLogger(__name__)

CKPT_PATTERN = re.compile(r"^ckpt-e(\d{3})-s(\d{6})\.pt$")


CLS_WEIGHT = "head.classification_head.cls_logits.weight"
CLS_BIAS = "head.classification_head.cls_logits.bias"
NUM_ANCHORS = 9  # 3 sizes x 3 aspect ratios per location


def checkpoint_num_classes(state: Mapping[str, torch.Tensor]) -> int:
    """Classes of a RetinaNet state dict: classification-head output rows per anchor."""
    return state[CLS_WEIGHT].shape[0] // NUM_ANCHORS


def weights_num_classes(weights: Path) -> int:
    """Classes of the RetinaNet state dict saved at ``weights``."""
    return checkpoint_num_classes(torch.load(weights, map_location="cpu", weights_only=True))


def _add_group_rows(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Grow a tree-only state to tree + tree_group in place: row ``a*2`` keeps tree row ``a``; the group
    rows start as torchvision initialises a head (N(0, 0.01) weights, prior-0.01 bias) from a fixed seed.

    In place, so the state keeps its ``_metadata`` (module versions torchvision needs to rename the old
    FPN keys of the release weights).
    """
    weight, bias = state[CLS_WEIGHT], state[CLS_BIAS]
    generator = torch.Generator().manual_seed(0)
    new_weight = torch.empty((2 * NUM_ANCHORS, *weight.shape[1:]), dtype=weight.dtype)
    new_weight[0::2] = weight
    new_weight[1::2] = torch.normal(0.0, 0.01, size=weight.shape, generator=generator).to(weight.dtype)
    new_bias = torch.empty(2 * NUM_ANCHORS, dtype=bias.dtype)
    new_bias[0::2] = bias
    new_bias[1::2] = -math.log(99)
    state[CLS_WEIGHT], state[CLS_BIAS] = new_weight, new_bias
    return state


def build_model(weights: Path | None, *, trainable_layers: int, num_classes: int | None = None) -> RetinaNet:
    """RetinaNet (ResNet-50 FPN, frozen BN) with 1 or 2 classes; loads `weights` strictly when given.

    `num_classes` None takes the checkpoint's (1 without weights). A tree-only checkpoint loads into a
    two-class model with its tree rows kept and fresh tree_group rows; the reverse is refused.
    """
    if not 0 <= trainable_layers <= 5:
        raise ValueError(f"trainable_layers must be in 0..5, got {trainable_layers}")
    state = None
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # FrozenBatchNorm2d silently drops `num_batches_tracked`, so strict=True alone would
        # accept a trainable-BN checkpoint; reject that layout explicitly.
        if any(k.endswith("num_batches_tracked") for k in state):
            raise RuntimeError(f"{weights}: trainable batch-norm layout, expected frozen-BN state_dict")
        found = checkpoint_num_classes(state)
        num_classes = num_classes or found
        if found > num_classes:
            raise ValueError(f"{weights} has {found} classes; it cannot load into a model of {num_classes}")
        if found < num_classes:
            state = _add_group_rows(state)
    body = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(
        body, trainable_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )
    model = RetinaNet(backbone, num_classes=num_classes or 1)
    if state is not None:
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


Item = tuple[int, np.ndarray, tuple[int, int], np.ndarray, np.ndarray, np.ndarray]


def collate(items: Sequence[Item]) -> Batch:
    """Stack `(index, image (3, Hp, Wp), resized (h, w), boxes (K, 4), labels (K,), unsupervised (C, h, w))`
    items into one padded batch; the unsupervised rasters are padded with False."""
    height = max(item[1].shape[1] for item in items)
    width = max(item[1].shape[2] for item in items)
    images = torch.zeros((len(items), 3, height, width), dtype=torch.float32)
    targets = []
    for row, (_, image, _, boxes, labels, unsupervised) in enumerate(items):
        images[row, :, : image.shape[1], : image.shape[2]] = torch.from_numpy(np.ascontiguousarray(image))
        padded = torch.zeros((len(unsupervised), height, width), dtype=torch.bool)
        padded[:, : unsupervised.shape[1], : unsupervised.shape[2]] = torch.from_numpy(unsupervised)
        targets.append(
            {
                "boxes": torch.from_numpy(np.ascontiguousarray(boxes, dtype=np.float32)).reshape(-1, 4),
                "labels": torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64)),
                "unsupervised": padded,
            }
        )
    return Batch([item[0] for item in items], images, [tuple(item[2]) for item in items], targets)


def _resize_masks(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour resize of (C, h, w) bool masks to (C, height, width)."""
    out = [
        np.asarray(Image.fromarray(m.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST))
        for m in masks
    ]
    return np.stack(out).astype(bool).reshape(len(masks), height, width)


class SampleDataset(Dataset):
    """Planned samples from `start` on, preprocessed exactly as serving does; boxes in resized pixels.

    Only the model's first `num_classes` classes are kept: their boxes and unsupervised rasters.
    """

    def __init__(
        self,
        specs: Sequence[SampleSpec],
        regions: Sequence[Region],
        cfg: TrainParams,
        *,
        start: int = 0,
        num_classes: int = 1,
    ):
        self.specs = specs
        self.regions = regions
        self.cfg = cfg
        self.start = start
        self.num_classes = num_classes
        self._cache: RegionCache | None = None

    def __getstate__(self) -> dict:
        # the cache holds open rasters; each DataLoader worker builds its own
        return {**self.__dict__, "_cache": None}

    def __len__(self) -> int:
        return len(self.specs) - self.start

    def __getitem__(self, i: int) -> Item:
        if self._cache is None:
            self._cache = RegionCache(self.regions)
        sample = load_sample(self.specs[self.start + i], self._cache, self.cfg)
        batch, scale = preprocess_window(sample.rgb)
        ours = sample.labels < self.num_classes
        boxes = sample.boxes[ours] * np.array(
            [scale.scale_x, scale.scale_y, scale.scale_x, scale.scale_y], np.float32
        )
        # a degenerate box makes the box coder's log(w) infinite
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1.0) & ((boxes[:, 3] - boxes[:, 1]) >= 1.0)
        size = (scale.resized_height, scale.resized_width)
        return (
            self.start + i,
            batch[0],
            size,
            np.ascontiguousarray(boxes[keep], dtype=np.float32),
            sample.labels[ours][keep],
            _resize_masks(sample.unsupervised[: self.num_classes], *size),
        )


def make_loader(
    specs: Sequence[SampleSpec],
    regions: Sequence[Region],
    cfg: TrainParams,
    *,
    start: int = 0,
    num_workers: int = 2,
    num_classes: int = 1,
) -> DataLoader:
    """Sequential loader over `specs[start:]`; DataLoader keeps worker order, so a resume is exact."""
    return DataLoader(
        SampleDataset(specs, regions, cfg, start=start, num_classes=num_classes),
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
    """Training losses on preprocessed batches; GeneralizedRCNNTransform never runs.

    torchvision's own loss runs unless some target has unsupervised pixels.
    """
    features = list(model.backbone(images).values())
    head_outputs = model.head(features)
    anchors = model.anchor_generator(ImageList(images, image_sizes), features)
    if any(t.get("unsupervised") is not None and t["unsupervised"].any() for t in targets):
        return masked_compute_loss(model, targets, head_outputs, anchors)
    return model.compute_loss(targets, head_outputs, anchors)


def masked_compute_loss(
    model: RetinaNet, targets: list[dict], head_outputs: dict[str, torch.Tensor], anchors: list[torch.Tensor]
) -> dict[str, torch.Tensor]:
    """torchvision's RetinaNet loss, except that the negative classification terms of an anchor whose centre
    lies on an unsupervised pixel of a class (``targets[i]["unsupervised"]``, (C, H, W) bool) are dropped.

    Matching, the focal loss, its normalisation by the foreground count and the box regression are
    torchvision's.
    """
    head = model.head.classification_head
    matched_idxs = []
    for anchors_per_image, t in zip(anchors, targets, strict=True):
        if t["boxes"].numel() == 0:
            matched_idxs.append(
                torch.full((len(anchors_per_image),), -1, dtype=torch.int64, device=anchors_per_image.device)
            )
        else:
            matched_idxs.append(model.proposal_matcher(box_ops.box_iou(t["boxes"], anchors_per_image)))

    losses = []
    for t, logits, anchors_per_image, matched in zip(
        targets, head_outputs["cls_logits"], anchors, matched_idxs, strict=True
    ):
        foreground = matched >= 0
        target = torch.zeros_like(logits)
        target[foreground, t["labels"][matched[foreground]]] = 1.0
        weight = torch.ones_like(logits)
        unsupervised = t.get("unsupervised")
        if unsupervised is not None:
            _, h, w = unsupervised.shape
            centres = (anchors_per_image[:, :2] + anchors_per_image[:, 2:]) / 2
            cx = centres[:, 0].floor().long().clamp(0, w - 1)
            cy = centres[:, 1].floor().long().clamp(0, h - 1)
            off = unsupervised[:, cy, cx].T  # (anchors, C)
            weight = torch.where(off & (target == 0), 0.0, 1.0).to(logits.dtype)
        valid = matched != head.BETWEEN_THRESHOLDS
        terms = sigmoid_focal_loss(logits[valid], target[valid], reduction="none") * weight[valid]
        losses.append(terms.sum() / max(1, int(foreground.sum())))
    return {
        "classification": torch.stack(losses).sum() / len(targets),
        "bbox_regression": model.head.regression_head.compute_loss(
            targets, head_outputs, anchors, matched_idxs
        ),
    }


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


@dataclasses.dataclass
class TrainResult:
    best_checkpoint: Path
    last_checkpoint: Path
    best_metrics: dict
    history: list[dict]
    # the start weights' own thresholds for the model's classes (``thresholds.json`` beside them)
    start_thresholds: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def thresholds(self) -> dict[str, float]:
        """The operating threshold per class (what ``thresholds.json`` holds): the best epoch's val pick,
        else the start weights' own threshold for a class val could not score."""
        return _serving_thresholds(self.best_metrics, self.start_thresholds)


def _thresholds(metrics: Mapping) -> dict[str, float]:
    return {c: float(metrics[f"val/threshold_{c}"]) for c in CLASSES if f"val/threshold_{c}" in metrics}


def _serving_thresholds(metrics: Mapping, start: Mapping[str, float]) -> dict[str, float]:
    picked = _thresholds(metrics)
    return {c: picked.get(c, start.get(c)) for c in CLASSES if c in picked or c in start}


def _window_field(name: str) -> str:
    """The ``InferenceParams`` window a class is predicted with, and its threshold picked at."""
    return WINDOW_FIELDS[CLASSES.index(name)]


def _beside(weights: Path, name: str) -> str | None:
    """The text of the file ``name`` beside the start weights (a run writes its own beside ``best.pt``), or
    None."""
    path = Path(weights).with_name(name)
    return path.read_text() if path.is_file() else None


def _unlabelled_classes(
    classes: Sequence[str], start_classes: Sequence[str], train_regions: Sequence[Region]
) -> list[str]:
    """The model's classes that no train region labels: they get no loss term, so the model keeps the
    start weights' detection of them.

    Refused: such a class that the start weights lack (its new rows would ship untrained), and data that
    labels none of the model's classes (nothing to learn).
    """
    labelled = {name for r in train_regions for name in r.classes}
    unlabelled = [name for name in classes if name not in labelled]
    new = [name for name in unlabelled if name not in start_classes]
    if new:
        raise ValueError(
            f"the start weights have no {new[0]} class and no train region labels {new[0]}: its new rows "
            f"would ship untrained. Use data with {new[0]} labels, or leave num_classes unset."
        )
    if len(unlabelled) == len(classes):
        raise ValueError(
            f"no train region labels a class of the model: it has {list(classes)}, "
            f"the regions label {sorted(labelled)}"
        )
    return unlabelled


def _fresh_sums() -> dict:
    return {"cls": 0.0, "reg": 0.0, "l2sp": 0.0, "batches": 0}


@dataclasses.dataclass
class _LoopState:
    """Everything a resume restores besides the torch state dicts; `epoch` and `step` are the next to run.

    Only ints, floats, strings, bools, lists and dicts, so the checkpoint loads with `weights_only=True`.
    """

    epoch: int = 0
    step: int = 0
    best_f1: float = -math.inf
    best_ap: float = -math.inf
    patience_left: int = 0
    epoch_seen: list[int] = dataclasses.field(default_factory=list)
    sums: dict = dataclasses.field(default_factory=_fresh_sums)
    history: list[dict] = dataclasses.field(default_factory=list)
    best_metrics: dict = dataclasses.field(default_factory=dict)
    done: bool = False


def amp_dtype(device: torch.device, amp: bool) -> torch.dtype | None:
    """None without AMP or off CUDA; bf16 on A100/H100, fp16 elsewhere (T4, L4)."""
    if not amp or device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    return torch.bfloat16 if "A100" in name or "H100" in name else torch.float16


def _validate(
    model: RetinaNet,
    device: torch.device,
    units: Mapping[str, Sequence[EvalUnit]],
    params: InferenceParams,
    epoch: int,
) -> dict:
    """Val metrics through the serving decode path; puts the model back in train mode.

    ``units`` maps each class of the model to its units (none for a class val does not label). Each
    (source, class) is scored on its own and the scores are the equal-weight means over them (with
    ``f1_max_<source>`` / ``ap40_<source>`` for trees and ``f1_max_<source>_tree_group`` /
    ``ap40_<source>_tree_group`` for groups alongside when there are several), so neither a large source
    nor one class drowns out the rest in early stopping.
    ``threshold_<class>`` is the operating threshold picked on val (``pick_threshold``; the mean over
    sources, snapped to ``PICK_THRESHOLDS``), with the prior for a model of that many classes.
    """
    t0 = time.monotonic()
    run_fn = torch_run_fn(model, device)
    prior = PRIOR_THRESHOLD[len(units)]
    reports = {}
    picks: dict[str, list[float]] = {}
    for name, class_units in units.items():
        by_source: dict[str, list[EvalUnit]] = {}
        for u in class_units:
            by_source.setdefault(u.meta["source"], []).append(u)
        for s, us in sorted(by_source.items()):
            key = s if name == "tree" else f"{s}_{name}"
            reports[key] = evaluate_units(run_fn, us, params, class_index=CLASSES.index(name))
            picks.setdefault(name, []).append(pick_threshold(reports[key], prior))
    model.train()
    keys = ("f1_max", "conf_at_max", "ap40", "aoi_f1")
    metrics = {k: float(np.mean([getattr(r, k) for r in reports.values()])) for k in keys}
    grid = np.asarray(PICK_THRESHOLDS)  # a mean over sources snaps back to the grid every report uses
    metrics |= {
        f"threshold_{name}": float(grid[np.abs(grid - np.mean(ts)).argmin()]) for name, ts in picks.items()
    }
    if len(reports) > 1:
        for key, r in reports.items():
            # not "f1_max/<source>": MLflow's file store cannot hold "f1_max" as both a file and a folder
            metrics |= {f"f1_max_{key}": r.f1_max, f"ap40_{key}": r.ap40}
    return metrics | {"seconds": time.monotonic() - t0}


def _load_scaler_state(scaler: torch.amp.GradScaler, state: Mapping) -> None:
    """Load a GradScaler state only when both sides use it.

    A disabled scaler (CPU, bf16 or amp off) saves an empty dict, and the device is not in the config
    hash, so a run may resume under a different AMP mode; the scale then restarts from its default.
    """
    if state and scaler.is_enabled():
        scaler.load_state_dict(state)


def _supersede(out_dir: Path) -> Path:
    """Move an earlier run's checkpoints, best.pt, metrics.csv, thresholds.json and windows.json into
    `superseded-<n>`."""
    n = 0
    while (out_dir / f"superseded-{n}").exists():
        n += 1
    dest = out_dir / f"superseded-{n}"
    dest.mkdir()
    old = [path for _, path in _checkpoints(out_dir)]
    names = ("best.pt", "metrics.csv", "thresholds.json", "windows.json")
    old += [out_dir / name for name in names if (out_dir / name).exists()]
    for path in old:
        os.replace(path, dest / path.name)
    logger.warning(f"force=True with resume=False: moved the earlier run's {len(old)} files to {dest}")
    return dest


def _plan_sha256(specs: Sequence[SampleSpec]) -> str:
    lines = (
        f"{s.region_id}|{s.zoom}|{s.centre_px[0]:.6f}|{s.centre_px[1]:.6f}|{s.size_m:.6f}|{s.aug_seed}"
        for s in specs
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _write_metrics_csv(path: Path, history: Sequence[Mapping]) -> None:
    keys = sorted({k for row in history for k in row} - {"epoch", "improved"})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "improved", *keys], restval="")
        writer.writeheader()
        writer.writerows(history)


def train(
    cfg: TrainParams,
    train_regions: Sequence[Region],
    val_regions: Sequence[Region],
    *,
    weights: Path,
    out_dir: Path,
    on_epoch_end: Callable[[dict, int], None] | None = None,
    resume: bool = True,
    force: bool = False,
    val_zooms: Sequence[int] = (20, 19),
    val_params: InferenceParams | None = None,
    num_workers: int = 2,
    device: str | None = None,
    tmp_dir: Path | None = None,
    repo_sha: str = "",
    manifest_hash: str = "",
) -> TrainResult:
    """Train from `weights` with early stopping on val f1_max (the mean over sources when val mixes
    sources); resumes from the newest checkpoint.

    The epoch plan is regenerated from (seed, epoch) on resume, so an interrupted run trains on the same
    samples as an uninterrupted one. A finished run returns its result without training.

    A class of the start weights that no train region labels is kept as it is: it gets no loss term, val
    does not score it, and it is served at the threshold and window in the ``thresholds.json`` and
    ``windows.json`` beside `weights`. So a tree and tree_group model fine-tunes on tree-only data.

    Refused before the first epoch: data with no label of the model's classes on any train chip
    (``any_label_on_chips``), such as a labels file for another area; empty val chips are allowed.
    """
    if not train_regions:
        raise ValueError("no train regions")
    if val_params is None:
        val_params = InferenceParams()
    out_dir = Path(out_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(weights, trainable_layers=cfg.trainable_layers, num_classes=cfg.num_classes)
    model = model.to(device).train()
    num_classes = model.head.classification_head.num_classes
    classes = CLASSES[:num_classes]
    for name in _unlabelled_classes(classes, CLASSES[: weights_num_classes(weights)], train_regions):
        logger.warning(
            f"no train region labels {name}: the model keeps the start weights' {name} detection "
            "(no loss term here) and threshold"
        )
    cache = RegionCache(train_regions)
    if not any_label_on_chips(train_regions, classes, cache):
        raise ValueError(
            "no label falls on any train chip: there is nothing to learn. Check that the labels file covers "
            "the same area as the chips (GeoJSON in longitude and latitude)."
        )
    start_thresholds = dict(
        zip(classes, parse_class_thresholds(_beside(weights, "thresholds.json")), strict=False)
    )
    start_windows = parse_model_windows(_beside(weights, "windows.json"))
    ref = snapshot(model)
    w0 = state_hash(ref)
    chash = config_hash(cfg)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    n = cfg.samples_per_epoch or total_chips(train_regions)
    steps = math.ceil(n / cfg.batch_size)
    total = cfg.epochs * steps

    def lr_factor(t: int) -> float:
        if cfg.schedule == "cosine":
            return 0.5 * (1 + math.cos(math.pi * min(t, total) / total))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    dtype = amp_dtype(device, cfg.amp)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype is torch.float16)
    tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.gettempdir()) / "fair_trees_ckpt"

    s = _LoopState(patience_left=cfg.patience)
    last_path = None
    found = latest_checkpoint(out_dir) if resume else None
    if resume and found is None and _checkpoints(out_dir):
        # a fresh start here would be rotated away by the old, higher-numbered checkpoints
        raise RuntimeError(
            f"{out_dir}: no readable checkpoint; move the unreadable ckpt-*.pt aside or use a new out_dir"
        )
    if found is not None:
        last_path, state = found
        check_resume(
            state, config_hash=chash, w0_hash=w0, manifest_hash=manifest_hash, repo_sha=repo_sha, force=force
        )
        s = _LoopState(**{f.name: state[f.name] for f in dataclasses.fields(_LoopState)})
        if s.done:
            return TrainResult(out_dir / "best.pt", last_path, s.best_metrics, s.history, start_thresholds)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        _load_scaler_state(scaler, state["scaler"])
        logger.info(f"resuming from {last_path.name}: epoch {s.epoch}, step {s.step}")
    elif not resume and _checkpoints(out_dir):
        if not force:
            raise ValueError("out_dir already holds checkpoints: use resume=True or a new out_dir")
        _supersede(out_dir)

    # after the resume check, so a finished run returns without reading the val set
    val_units = {
        name: region_units(val_regions, zooms=val_zooms, label=k + 1) for k, name in enumerate(classes)
    }
    if not any(val_units.values()):
        val_units = {}
        logger.warning(
            "no val region labels a class of the model: every epoch counts as an improvement and best.pt "
            "is the last epoch"
        )

    def save_state(name: str) -> Path:
        state = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            **dataclasses.asdict(s),
            "w0_hash": w0,
            "config_hash": chash,
            "manifest_hash": manifest_hash,
            "repo_sha": repo_sha,
        }
        return save_file(state, out_dir, name, tmp_dir=tmp_dir, keep_last=cfg.keep_last)

    out_dir.mkdir(parents=True, exist_ok=True)
    last_save = time.monotonic()
    for epoch in range(s.epoch, cfg.epochs):
        s.epoch = epoch
        specs = plan_epoch(train_regions, cfg, cfg.seed, epoch, cache=cache)
        stats = sampler_stats(specs, cache)
        t_epoch = time.monotonic()
        trained = 0  # samples trained in this session; a resumed epoch counts only its remainder
        loader = make_loader(
            specs,
            train_regions,
            cfg,
            start=s.step * cfg.batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
        )
        for batch in loader:
            images = batch.images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in batch.targets]
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                losses = forward_loss(model, images, batch.image_sizes, targets)
            l2 = l2sp_penalty(model, ref) if cfg.l2sp_alpha > 0 else torch.zeros((), device=device)
            loss = losses["classification"].float() + losses["bbox_regression"].float() + cfg.l2sp_alpha * l2
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {s.step}")
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.max_grad_norm is not None:
                scaler.unscale_(opt)  # clip the true gradient, not the loss-scaled one
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], cfg.max_grad_norm
                )
            scaler.step(opt)
            scaler.update()
            sched.step()

            s.step += 1
            s.epoch_seen.extend(batch.indices)
            s.sums["cls"] += losses["classification"].detach().item()
            s.sums["reg"] += losses["bbox_regression"].detach().item()
            s.sums["l2sp"] += l2.detach().item()
            s.sums["batches"] += 1
            trained += len(batch.indices)
            if time.monotonic() - last_save >= cfg.ckpt_every_min * 60 and s.step < steps:
                last_path = save_state(checkpoint_name(epoch, s.step))
                last_save = time.monotonic()
        train_s = time.monotonic() - t_epoch

        batches = max(s.sums["batches"], 1)
        metrics = {
            "train/loss_cls": s.sums["cls"] / batches,
            "train/loss_reg": s.sums["reg"] / batches,
            "train/l2sp": s.sums["l2sp"] / batches,
            "train/lr": float(sched.get_last_lr()[0]),
            "train/samples_per_s": trained / train_s if train_s > 0 else 0.0,
        }
        if val_units:
            v = _validate(model, device, val_units, val_params, epoch)
            metrics |= {f"val/{k}": float(value) for k, value in v.items()}
            if v["seconds"] > 0.15 * (train_s + v["seconds"]):
                logger.warning(
                    f"validation took {v['seconds']:.1f} s, over 15% of the epoch's wall time: "
                    "cap the val set"
                )
            f1, ap = v["f1_max"], v["ap40"]
            improved = f1 > s.best_f1 + cfg.min_delta or (
                abs(f1 - s.best_f1) <= cfg.min_delta and ap > s.best_ap
            )
        else:
            improved = True
        if improved:
            if val_units:
                s.best_f1, s.best_ap = max(s.best_f1, f1), ap
            s.patience_left = cfg.patience
            s.best_metrics = dict(metrics)
            save_file(model.state_dict(), out_dir, "best.pt", tmp_dir=tmp_dir, keep_last=None)
            if thresholds := _serving_thresholds(s.best_metrics, start_thresholds):
                windows = {name: getattr(val_params, name) for name in WINDOW_FIELDS}
                for name in thresholds.keys() - _thresholds(s.best_metrics).keys():
                    # not picked on this val set: the threshold holds at the start weights' window
                    field = _window_field(name)
                    windows[field] = start_windows.get(field, windows[field])
                for name, values in (("windows.json", windows), ("thresholds.json", thresholds)):
                    part = out_dir / f"{name}.part"
                    part.write_text(json.dumps(values, indent=2))
                    os.replace(part, out_dir / name)
        else:
            s.patience_left -= 1

        s.history.append(
            {
                "epoch": epoch,
                **metrics,
                **stats,
                "sampler/plan_sha256": _plan_sha256(specs),
                "sampler/sample_ids_sha256": hashlib.sha256(
                    "\n".join(map(str, s.epoch_seen)).encode()
                ).hexdigest(),
                "improved": improved,
            }
        )
        _write_metrics_csv(out_dir / "metrics.csv", s.history)

        s.epoch, s.step, s.epoch_seen, s.sums = epoch + 1, 0, [], _fresh_sums()
        s.done = s.patience_left == 0 or epoch + 1 == cfg.epochs
        last_path = save_state(checkpoint_name(epoch + 1, 0))
        last_save = time.monotonic()
        if on_epoch_end is not None:
            on_epoch_end({k: float(v) for k, v in {**metrics, **stats}.items()}, epoch)
        if s.patience_left == 0:
            break
    return TrainResult(out_dir / "best.pt", last_path, s.best_metrics, s.history, start_thresholds)


def fair_data_hash(train_chips: Sequence[Path], val_chips: Sequence[Path], labels_path: Path) -> str:
    """sha256 over each chip's split, name and bytes (in name order) and the labels file's bytes."""
    h = hashlib.sha256()
    for split, chips in (("train", train_chips), ("val", val_chips)):
        for path in sorted(map(Path, chips), key=lambda p: p.name):
            h.update(f"{split}/{path.name}\n".encode())
            h.update(hashlib.sha256(path.read_bytes()).digest())
    h.update(Path(labels_path).read_bytes())
    return h.hexdigest()


def fine_tune(
    train_chips: Sequence[Path],
    val_chips: Sequence[Path],
    labels_path: Path,
    hyperparameters: Mapping | None,
    *,
    weights: Path,
    out_dir: Path,
    on_epoch_end=None,
    resume: bool = True,
    **train_kwargs,
) -> TrainResult:
    """Fine-tune on a mapper's chips, already split into train and val by fAIr; no internal re-split.

    Without explicit `zoom_weights`, every zoom present in the train chips gets weight 1, so a z18 or z22
    dataset is never filtered out by the base-training defaults. Unless given, the manifest hash covers
    the chip files, their split and the labels, so a resume on changed data is refused. Labels that fall
    on no train chip are refused before training (see `train`).
    """
    train_kwargs.setdefault("manifest_hash", fair_data_hash(train_chips, val_chips, labels_path))
    hp = hyperparameters or {}
    cfg = parse_train_params(hyperparameters)
    train_regions = load_fair_dataset(train_chips, labels_path, split="train")
    val_regions = load_fair_dataset(val_chips, labels_path, split="val")
    if hp.get("zoom_weights") is None:
        zooms = sorted({z for r in train_regions for z in r.chips})
        cfg = dataclasses.replace(cfg, zoom_weights={z: 1.0 for z in zooms})
    return train(
        cfg,
        train_regions,
        val_regions,
        weights=weights,
        out_dir=out_dir,
        on_epoch_end=on_epoch_end,
        resume=resume,
        **train_kwargs,
    )
