"""Colab entry points: train a run, evaluate it, export it to ONNX and compare two runs.

    python -m experiments.run train experiments/configs/neon_platform.yaml neon-platform-1
    python -m experiments.run evaluate neon-platform-1
    python -m experiments.run export neon-platform-1
    python -m experiments.run compare neon-resume-a neon-resume-b

Run outputs live in ``<drive root>/runs/<run id>``; data is staged under the data root. The library is
imported lazily inside the functions, so the config tests need no torch.
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import logging
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fair_trees.serve import RunFn
    from fair_trees.train import TrainResult

logger = logging.getLogger(__name__)

DRIVE_ROOT = Path("/content/drive/MyDrive/fair_trees")
DATA_ROOT = Path("/content/data")
TMP_ROOT = Path("/content/tmp")
BENCHMARK = Path(__file__).parent / "benchmark.json"

CONFIG_KEYS = ("weights", "sources", "hyperparameters", "val", "mlflow")
REQUIRED_KEYS = ("weights", "sources")
DEFAULT_VAL = {"zooms": [20, 19], "max_regions": None}
MLFLOW_EXPERIMENT = "fair_trees"
PARITY_UNITS = 3
PARITY_ATOL = 2e-3


def load_config(path: Path) -> dict:
    """Read a run config; unknown or missing top-level keys raise, the optional ones get defaults."""
    import yaml

    conf = yaml.safe_load(Path(path).read_text())
    if not isinstance(conf, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    unknown = sorted(set(conf) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown config keys {unknown}")
    for key in REQUIRED_KEYS:
        if key not in conf:
            raise ValueError(f"{path}: config lacks {key}")
    conf["hyperparameters"] = conf.get("hyperparameters") or {}
    conf["val"] = {**DEFAULT_VAL, **(conf.get("val") or {})}
    conf.setdefault("mlflow", True)
    return conf


def repo_sha() -> str:
    """HEAD of the repo containing ``experiments/``, or "" when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def _regions(conf: Mapping, data_root: Path) -> tuple[list, list]:
    """(train, val) regions of the configured sources; val sorted by id and cut to ``max_regions``."""
    from fair_trees.dataset import load_regions

    regions = [r for s in conf["sources"] for r in load_regions(Path(data_root) / s)]
    train = [r for r in regions if r.split == "train"]
    val = sorted((r for r in regions if r.split == "val"), key=lambda r: r.id)
    max_regions = conf["val"]["max_regions"]
    if max_regions is not None:
        val = val[:max_regions]
    return train, val


def _manifests(conf: Mapping, data_root: Path) -> dict[str, str]:
    """sha256 of each configured source's staged ``manifest.json``."""
    from experiments.shards import manifest_hash

    hashes = {}
    for s in conf["sources"]:
        path = Path(data_root) / s / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"source {s}: no manifest at {path}")
        hashes[s] = manifest_hash(path)
    return hashes


def _manifest_hash(conf: Mapping, data_root: Path) -> str:
    """One hash over all configured sources' manifests, independent of the source order."""
    lines = sorted(f"{s}:{h}" for s, h in _manifests(conf, data_root).items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main(
    config: Path,
    run_id: str,
    *,
    drive_root: Path = DRIVE_ROOT,
    data_root: Path = DATA_ROOT,
    tmp_dir: Path = TMP_ROOT,
    num_workers: int = 2,
    force: bool = False,
) -> "TrainResult":
    """Train (or resume) run ``run_id`` from ``config``; writes ``config.json`` and ``env.json`` first."""
    import torch
    import torchvision

    import fair_trees.train
    from experiments.mlflow_cb import make_mlflow_callback
    from fair_trees.params import parse_train_params

    conf = load_config(config)
    cfg = parse_train_params(conf["hyperparameters"])
    out_dir = Path(drive_root) / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # json writes the int keys of `zoom_weights` as strings
    _write_json(out_dir / "config.json", {"config": conf, "parsed": dataclasses.asdict(cfg)})

    env_path = out_dir / "env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    sha = repo_sha()
    _write_json(
        env_path,
        {
            "mlflow_run_id": None,
            "units_used": None,
            **env,
            "repo_sha": sha,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "manifests": _manifests(conf, data_root),
        },
    )

    on_epoch_end = None
    if conf["mlflow"]:
        on_epoch_end = make_mlflow_callback(
            Path(drive_root) / "mlruns", MLFLOW_EXPERIMENT, env_path, {"run_id": run_id, **conf}
        )
    train_regions, val_regions = _regions(conf, data_root)
    logger.info(f"{run_id}: {len(train_regions)} train and {len(val_regions)} val regions")
    return fair_trees.train.train(
        cfg,
        train_regions,
        val_regions,
        weights=Path(drive_root) / conf["weights"],
        out_dir=out_dir,
        on_epoch_end=on_epoch_end,
        val_zooms=tuple(conf["val"]["zooms"]),
        num_workers=num_workers,
        tmp_dir=tmp_dir,
        repo_sha=sha,
        manifest_hash=_manifest_hash(conf, data_root),
        force=force,
    )


def _run_fn(weights: Path, device: str | None) -> "RunFn":
    """Torch run function for a frozen-BN checkpoint; the patch point the tests replace."""
    import torch

    from fair_trees.train import build_model, torch_run_fn

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return torch_run_fn(build_model(weights, trainable_layers=0), dev)


def _benchmark_parts(benchmark: Path, data_root: Path) -> Iterator[tuple[str, dict, list]]:
    """(name, part, units) for each NEON eval-plot part whose data is staged; units must match the freeze."""
    from experiments.neon_eval import load_eval_plots

    for name, part in json.loads(Path(benchmark).read_text())["parts"].items():
        if part["kind"] != "neon_eval_plots":
            continue
        root = Path(data_root) / part["data_dir"]
        if not root.exists():
            logger.warning(f"benchmark part {name}: {root} not staged, skipped")
            continue
        units = load_eval_plots(root)
        if [u.unit_id for u in units] != part["unit_ids"]:
            raise ValueError(f"benchmark part {name}: units differ from the frozen list")
        yield name, part, units


def _with_ci(report) -> dict:
    from fair_trees.evaluate import bootstrap_ci

    return report.to_dict() | {"f1_ci95": list(bootstrap_ci(report.unit_counts()))}


def evaluate(
    run_id: str,
    benchmark: Path,
    *,
    drive_root: Path = DRIVE_ROOT,
    data_root: Path = DATA_ROOT,
    config: Path | None = None,
    weights: Path | None = None,
    device: str | None = None,
) -> dict:
    """Score a run on its val regions and on the frozen NEON benchmark parts; writes ``eval.json``."""
    from experiments.neon_eval import DF_COMPAT
    from fair_trees.evaluate import evaluate_regions, evaluate_units
    from fair_trees.params import InferenceParams

    out_dir = Path(drive_root) / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        conf = load_config(config)
    else:
        conf = json.loads((out_dir / "config.json").read_text())["config"]
    weights = Path(weights) if weights is not None else out_dir / "best.pt"
    run_fn = _run_fn(weights, device)

    _, val = _regions(conf, data_root)
    parts = {"val": _with_ci(evaluate_regions(run_fn, val, InferenceParams(), zooms=conf["val"]["zooms"]))}
    for name, part, units in _benchmark_parts(benchmark, data_root):
        report = evaluate_units(run_fn, units, DF_COMPAT, ignore=False)
        lower = float(part["release_f1_ci95"][0])
        parts[name] = _with_ci(report) | {
            "forgetting_lower_bound": lower,
            "forgetting_pass": report.f1 >= lower,
        }

    result = {"run_id": run_id, "weights": str(weights), "parts": parts}
    _write_json(out_dir / "eval.json", result)
    return result


def export(
    run_id: str, benchmark: Path, *, drive_root: Path = DRIVE_ROOT, data_root: Path = DATA_ROOT
) -> dict:
    """Export ``best.pt`` to ``model.onnx`` and check ONNX against torch on the first NEON benchmark units."""
    import numpy as np
    import onnxruntime
    import torch

    from fair_trees.export import export_onnx_bytes
    from fair_trees.preprocess import preprocess_window
    from fair_trees.serve import ort_run_fn
    from fair_trees.train import build_model, torch_run_fn

    out_dir = Path(drive_root) / "runs" / run_id
    units = [u for _, _, us in _benchmark_parts(benchmark, data_root) for u in us][:PARITY_UNITS]
    if not units:
        raise ValueError(
            f"{benchmark}: no staged NEON benchmark units under {data_root} for the parity check"
        )

    model = build_model(None, trainable_layers=0)
    state = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    data = export_onnx_bytes(model)
    (out_dir / "model.onnx").write_bytes(data)

    torch_run = torch_run_fn(model, torch.device("cpu"))
    ort_run = ort_run_fn(onnxruntime.InferenceSession(data, providers=["CPUExecutionProvider"]))
    max_abs_diff = 0.0
    for unit in units:
        batch, _ = preprocess_window(unit.mosaic.rgb)
        for t, o in zip(torch_run(batch), ort_run(batch), strict=True):
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(t - o))))

    result = {"max_abs_diff": max_abs_diff, "pass": max_abs_diff <= PARITY_ATOL, "windows": len(units)}
    _write_json(out_dir / "export.json", result)
    return result


def _sampler_digests(run_dir: Path) -> dict[str, tuple[str, str]] | None:
    """Per-epoch (plan, sample ids) digests from ``metrics.csv``, or None without one."""
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        return {
            row["epoch"]: (row["sampler/plan_sha256"], row["sampler/sample_ids_sha256"])
            for row in csv.DictReader(f)
        }


def _val_part(run_dir: Path) -> dict:
    return json.loads((run_dir / "eval.json").read_text())["parts"]["val"]


def _unit_counts(part: Mapping) -> dict:
    from fair_trees.evaluate import Counts

    return {u["unit_id"]: Counts(u["tp"], u["fp"], u["fn"]) for u in part["units"]}


def _equal_over(a: Mapping | None, b: Mapping | None, epochs: Sequence[str], column: int) -> bool | None:
    if a is None or b is None:
        return None
    return all(a[e][column] == b[e][column] for e in epochs)


def compare_runs(a: str, b: str, *, drive_root: Path = DRIVE_ROOT) -> dict:
    """Sampler digests over the common epochs and the val F1 of run ``b`` against run ``a``."""
    from fair_trees.evaluate import paired_bootstrap_diff

    dir_a, dir_b = Path(drive_root) / "runs" / a, Path(drive_root) / "runs" / b
    dig_a, dig_b = _sampler_digests(dir_a), _sampler_digests(dir_b)
    epochs = sorted(set(dig_a) & set(dig_b), key=int) if dig_a is not None and dig_b is not None else []

    val_a, val_b = _val_part(dir_a), _val_part(dir_b)
    counts_a, counts_b = _unit_counts(val_a), _unit_counts(val_b)
    if set(counts_a) != set(counts_b):
        raise ValueError(f"runs {a} and {b} were evaluated on different val units")
    ids = list(counts_a)
    a_ci = val_a["f1_ci95"]
    f1_b = val_b["f1"]
    return {
        "a": a,
        "b": b,
        "common_epochs": len(epochs),
        "plan_equal": _equal_over(dig_a, dig_b, epochs, 0),
        "sample_ids_equal": _equal_over(dig_a, dig_b, epochs, 1),
        "f1_a": val_a["f1"],
        "f1_b": f1_b,
        "a_f1_ci95": a_ci,
        "b_f1_in_a_ci": a_ci[0] <= f1_b <= a_ci[1],
        "paired_diff": list(paired_bootstrap_diff([counts_b[i] for i in ids], [counts_a[i] for i in ids])),
    }


def cli(argv: Sequence[str] | None = None) -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--drive-root", type=Path, default=DRIVE_ROOT)
    common.add_argument("--data-root", type=Path, default=DATA_ROOT)
    bench = argparse.ArgumentParser(add_help=False)
    bench.add_argument("--benchmark", type=Path, default=BENCHMARK)

    parser = argparse.ArgumentParser(prog="python -m experiments.run", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", parents=[common], help="train or resume a run")
    p.add_argument("config", type=Path)
    p.add_argument("run_id")
    p = sub.add_parser("evaluate", parents=[common, bench], help="write eval.json for a run")
    p.add_argument("run_id")
    p.add_argument("--config", type=Path)
    p.add_argument("--weights", type=Path)
    p = sub.add_parser("export", parents=[common, bench], help="write model.onnx and export.json")
    p.add_argument("run_id")
    p = sub.add_parser("compare", parents=[common], help="compare two runs' sampler digests and val F1")
    p.add_argument("a")
    p.add_argument("b")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.command == "train":
        result = main(args.config, args.run_id, drive_root=args.drive_root, data_root=args.data_root)
        print(f"best {result.best_checkpoint}: {result.best_metrics}")
    elif args.command == "evaluate":
        out = evaluate(
            args.run_id,
            args.benchmark,
            drive_root=args.drive_root,
            data_root=args.data_root,
            config=args.config,
            weights=args.weights,
        )
        summary = {
            name: {k: part[k] for k in ("f1", "f1_ci95", "forgetting_pass") if k in part}
            for name, part in out["parts"].items()
        }
        print(json.dumps(summary, indent=2))
    elif args.command == "export":
        parity = export(args.run_id, args.benchmark, drive_root=args.drive_root, data_root=args.data_root)
        print(json.dumps(parity))
    else:
        print(json.dumps(compare_runs(args.a, args.b, drive_root=args.drive_root), indent=2))


if __name__ == "__main__":
    cli()
