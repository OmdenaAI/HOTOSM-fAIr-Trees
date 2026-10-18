"""Loader vs GPU-step throughput: is the data loader fast enough to keep the GPU busy?

    python -m experiments.throughput experiments/configs/neon_platform.yaml neon-platform-1

Writes ``<drive root>/runs/<run id>/throughput.json``; the loader passes when its best worker count
delivers at least 1.5x the samples/s the GPU step consumes.
"""

import argparse
import dataclasses
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from fair_trees.dataset import plan_epoch
from fair_trees.params import TrainParams, parse_train_params
from fair_trees.train import amp_dtype, build_model, collate, forward_loss, make_loader

logger = logging.getLogger(__name__)

PASS_RATIO = 1.5
GPU_IMAGE_SIZE = 800
GPU_BOXES_PER_IMAGE = 10
DESC = __doc__.splitlines()[0]


def measure_loader(
    regions: Sequence,
    cfg: TrainParams,
    *,
    workers: Sequence[int] = (0, 2, 4),
    n_batches: int = 30,
    seed: int = 0,
) -> dict[int, float]:
    """Samples/s of the training loader per worker count.

    The first ``max(1, 2 * workers)`` batches are warm-up (each worker prefetches two) and not timed.
    """
    rates = {}
    for w in workers:
        warm = max(1, w * 2)
        plan_cfg = dataclasses.replace(cfg, samples_per_epoch=(warm + n_batches) * cfg.batch_size)
        specs = plan_epoch(regions, plan_cfg, seed, 0)
        samples, timed, start = 0, 0, None
        for i, batch in enumerate(make_loader(specs, regions, cfg, num_workers=w)):
            if i < warm:
                start = time.perf_counter()
                continue
            samples += len(batch.indices)
            timed += 1
        if timed < 2:
            raise ValueError(
                f"workers={w}: only {timed} batch(es) timed after {warm} warm-up; need at least 2"
            )
        rates[w] = samples / (time.perf_counter() - start)
        logger.info(f"loader workers={w}: {rates[w]:.1f} samples/s")
    return rates


def _fixed_batch(cfg: TrainParams, device: torch.device):
    gen = torch.Generator().manual_seed(0)
    items = []
    for i in range(cfg.batch_size):
        image = torch.rand(3, GPU_IMAGE_SIZE, GPU_IMAGE_SIZE, generator=gen).numpy()
        xy = torch.rand(GPU_BOXES_PER_IMAGE, 2, generator=gen) * (GPU_IMAGE_SIZE - 100)
        wh = torch.rand(GPU_BOXES_PER_IMAGE, 2, generator=gen) * 60 + 20
        boxes = torch.cat([xy, xy + wh], dim=1).numpy()
        items.append((i, image, (GPU_IMAGE_SIZE, GPU_IMAGE_SIZE), boxes))
    batch = collate(items)
    targets = [{k: v.to(device) for k, v in t.items()} for t in batch.targets]
    return batch.images.to(device), batch.image_sizes, targets


def measure_gpu_step(
    cfg: TrainParams, *, weights: Path | None, device: str, n_steps: int = 30, warmup: int = 3
) -> float:
    """Samples/s of forward, backward and optimizer step on one fixed batch, under the training AMP rule."""
    dev = torch.device(device)
    model = build_model(weights, trainable_layers=cfg.trainable_layers).to(dev).train()
    images, sizes, targets = _fixed_batch(cfg, dev)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    dtype = amp_dtype(dev, cfg.amp)
    scaler = torch.amp.GradScaler(dev.type, enabled=dtype is torch.float16)

    def step() -> None:
        with torch.autocast(dev.type, dtype=dtype, enabled=dtype is not None):
            losses = forward_loss(model, images, sizes, targets)
        loss = losses["classification"].float() + losses["bbox_regression"].float()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    def sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        step()
    sync()
    start = time.perf_counter()
    for _ in range(n_steps):
        step()
    sync()
    return n_steps * cfg.batch_size / (time.perf_counter() - start)


def main(config: Path, run_id: str, *, drive_root: Path, data_root: Path) -> dict:
    from experiments import run

    conf = run.load_config(config)
    cfg = parse_train_params(conf["hyperparameters"])
    train, _ = run._regions(conf, data_root)
    loader = measure_loader(train, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_step = measure_gpu_step(cfg, weights=Path(drive_root) / conf["weights"], device=device)
    ratio = max(loader.values()) / gpu_step
    dtype = amp_dtype(torch.device(device), cfg.amp)
    result = {
        "loader": loader,
        "gpu_step": gpu_step,
        "ratio": ratio,
        "pass": ratio >= PASS_RATIO,
        "device": device,
        "amp_dtype": None if dtype is None else str(dtype).removeprefix("torch."),
    }
    out = Path(drive_root) / "runs" / run_id / "throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


def cli(argv: Sequence[str] | None = None) -> None:
    from experiments import run

    parser = argparse.ArgumentParser(
        prog="python -m experiments.throughput", description=__doc__.splitlines()[0]
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("run_id")
    parser.add_argument("--drive-root", type=Path, default=run.DRIVE_ROOT)
    parser.add_argument("--data-root", type=Path, default=run.DATA_ROOT)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print(
        json.dumps(
            main(args.config, args.run_id, drive_root=args.drive_root, data_root=args.data_root), indent=2
        )
    )


if __name__ == "__main__":
    cli()
