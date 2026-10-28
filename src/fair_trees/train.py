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
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.image_list import ImageList
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops.misc import FrozenBatchNorm2d

from fair_trees.dataset import (
    Region,
    RegionCache,
    SampleSpec,
    load_fair_dataset,
    load_sample,
    plan_epoch,
    sampler_stats,
    total_chips,
)
from fair_trees.evaluate import EvalUnit, evaluate_units, region_units
from fair_trees.params import InferenceParams, TrainParams, parse_train_params
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


@dataclasses.dataclass
class TrainResult:
    best_checkpoint: Path
    last_checkpoint: Path
    best_metrics: dict
    history: list[dict]


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
    model: RetinaNet, device: torch.device, units: Sequence[EvalUnit], params: InferenceParams, epoch: int
) -> dict:
    """Val metrics through the serving decode path; puts the model back in train mode.

    When the units come from several sources, each source is scored on its own and the scores are the
    equal-weight means over sources (with ``f1_max/<source>`` and ``ap40/<source>`` alongside), so a large
    source cannot drown out a small one in early stopping.
    """
    t0 = time.monotonic()
    run_fn = torch_run_fn(model, device)
    by_source: dict[str, list[EvalUnit]] = {}
    for u in units:
        by_source.setdefault(u.meta["source"], []).append(u)
    reports = {s: evaluate_units(run_fn, us, params) for s, us in sorted(by_source.items())}
    model.train()
    keys = ("f1_max", "conf_at_max", "ap40", "aoi_f1")
    metrics = {k: float(np.mean([getattr(r, k) for r in reports.values()])) for k in keys}
    if len(reports) > 1:
        for s, r in reports.items():
            metrics |= {f"f1_max/{s}": r.f1_max, f"ap40/{s}": r.ap40}
    return metrics | {"seconds": time.monotonic() - t0}


def _load_scaler_state(scaler: torch.amp.GradScaler, state: Mapping) -> None:
    """Load a GradScaler state only when both sides use it.

    A disabled scaler (CPU, bf16 or amp off) saves an empty dict, and the device is not in the config
    hash, so a run may resume under a different AMP mode; the scale then restarts from its default.
    """
    if state and scaler.is_enabled():
        scaler.load_state_dict(state)


def _supersede(out_dir: Path) -> Path:
    """Move an earlier run's checkpoints, best.pt and metrics.csv into `superseded-<n>`; delete nothing."""
    n = 0
    while (out_dir / f"superseded-{n}").exists():
        n += 1
    dest = out_dir / f"superseded-{n}"
    dest.mkdir()
    old = [path for _, path in _checkpoints(out_dir)]
    old += [out_dir / name for name in ("best.pt", "metrics.csv") if (out_dir / name).exists()]
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
    """
    if not train_regions:
        raise ValueError("no train regions")
    if val_params is None:
        val_params = InferenceParams()
    out_dir = Path(out_dir)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(weights, trainable_layers=cfg.trainable_layers).to(device).train()
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
            return TrainResult(out_dir / "best.pt", last_path, s.best_metrics, s.history)
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
    val_units = region_units(val_regions, zooms=val_zooms) if val_regions else []
    if not val_regions:
        logger.warning("no val regions: every epoch counts as an improvement and best.pt is the last epoch")

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
    cache = RegionCache(train_regions)
    last_save = time.monotonic()
    for epoch in range(s.epoch, cfg.epochs):
        s.epoch = epoch
        specs = plan_epoch(train_regions, cfg, cfg.seed, epoch, cache=cache)
        stats = sampler_stats(specs, cache)
        t_epoch = time.monotonic()
        trained = 0  # samples trained in this session; a resumed epoch counts only its remainder
        loader = make_loader(
            specs, train_regions, cfg, start=s.step * cfg.batch_size, num_workers=num_workers
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
    return TrainResult(out_dir / "best.pt", last_path, s.best_metrics, s.history)


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
    the chip files, their split and the labels, so a resume on changed data is refused.
    """
    train_kwargs.setdefault("manifest_hash", fair_data_hash(train_chips, val_chips, labels_path))
    cfg = parse_train_params(hyperparameters)
    train_regions = load_fair_dataset(train_chips, labels_path, split="train")
    val_regions = load_fair_dataset(val_chips, labels_path, split="val")
    if (hyperparameters or {}).get("zoom_weights") is None:
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
