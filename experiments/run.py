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
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fair_trees.params import InferenceParams
    from fair_trees.serve import RunFn
    from fair_trees.train import TrainResult

logger = logging.getLogger(__name__)

DRIVE_ROOT = Path("/content/drive/MyDrive/fair_trees")
DATA_ROOT = Path("/content/data")
TMP_ROOT = Path("/content/tmp")
BENCHMARK = Path(__file__).parent / "benchmark.json"

CONFIG_KEYS = ("weights", "sources", "hyperparameters", "val", "mlflow")
REQUIRED_KEYS = ("weights", "sources")
DEFAULT_VAL = {"zooms": [20, 19], "max_regions": None, "window_m": 40.0, "group_window_m": 120.0}
# the label study's crown-merge setting, the group baseline for a tree-only model
MERGED_GROUPS = {"score": 0.2, "gap_m": 2.0, "min_crowns": 8}
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
    _check_max_regions(path, conf["val"]["max_regions"])
    conf.setdefault("mlflow", True)
    return conf


def _check_max_regions(path: Path, value) -> None:
    """``val.max_regions`` is None, an int >= 1, or a mapping of source -> int >= 1."""

    def positive_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool) and v >= 1

    if value is None or positive_int(value):
        return
    if isinstance(value, dict) and all(isinstance(k, str) and positive_int(v) for k, v in value.items()):
        return
    raise ValueError(
        f"{path}: val max_regions must be an int >= 1 or a mapping source -> int >= 1, got {value!r}"
    )


def _stored_config(path: Path) -> dict:
    """The ``config`` of a run's config.json, with val defaults added that it predates (e.g. ``window_m``)."""
    conf = json.loads(Path(path).read_text())["config"]
    return {**conf, "val": {**DEFAULT_VAL, **conf.get("val", {})}}


def val_params(conf: Mapping) -> "InferenceParams":
    """Inference params to score val regions with the configured tree and group windows."""
    from fair_trees.params import parse_inference_params

    val = conf["val"]
    return parse_inference_params(
        {
            "window_m": val["window_m"],
            "group_window_m": val.get("group_window_m", DEFAULT_VAL["group_window_m"]),
        }
    )


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
    """(train, val) regions of the configured sources; val sorted by id and cut to ``max_regions``.

    An int ``max_regions`` cuts the pooled list; a mapping ``{source: n}`` keeps the first ``n`` val regions
    (by id) of each named source and all of the others.
    """
    from fair_trees.dataset import load_regions

    regions = [r for s in conf["sources"] for r in load_regions(Path(data_root) / s)]
    train = [r for r in regions if r.split == "train"]
    val = sorted((r for r in regions if r.split == "val"), key=lambda r: r.id)
    max_regions = conf["val"]["max_regions"]
    if isinstance(max_regions, Mapping):
        kept: Counter[str] = Counter()
        capped = []
        for r in val:
            if r.source not in max_regions or kept[r.source] < max_regions[r.source]:
                capped.append(r)
                kept[r.source] += 1
        val = capped
    elif max_regions is not None:
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


def _run_changes(
    config_path: Path, env: Mapping, conf: Mapping, parsed: Mapping, manifests: Mapping
) -> list[str]:
    """What differs from the existing run in ``config_path.parent``: config, parsed params or data."""
    changed = []
    if config_path.exists():
        stored = json.loads(config_path.read_text())
        # round-tripped through json like the stored copy
        if _stored_config(config_path) != json.loads(json.dumps(conf)):
            changed.append("the config")
        if stored["parsed"] != parsed:
            changed.append("the parsed hyperparameters")
    if "manifests" in env and env["manifests"] != manifests:
        changed.append("the data manifests")
    return changed


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
    """Train (or resume) run ``run_id`` from ``config``; writes ``config.json`` and ``env.json`` first.

    An existing run whose config, parsed hyperparameters or data manifests differ is refused before any
    file is written. With ``force`` it is started over instead: train moves the old checkpoints into
    ``superseded-<n>/`` and the new run logs to a new MLflow run.
    """
    import torch
    import torchvision

    import fair_trees.train
    from experiments.mlflow_cb import make_mlflow_callback
    from fair_trees.params import parse_train_params

    conf = load_config(config)
    cfg = parse_train_params(conf["hyperparameters"])
    # round-tripped through json (int `zoom_weights` keys become strings) to compare with config.json
    parsed = json.loads(json.dumps(dataclasses.asdict(cfg)))
    manifests = _manifests(conf, data_root)
    out_dir = Path(drive_root) / "runs" / run_id
    config_path, env_path = out_dir / "config.json", out_dir / "env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    changed = _run_changes(config_path, env, conf, parsed, manifests)
    if changed and not force:
        raise ValueError(
            f"{out_dir}: {' and '.join(changed)} differ from the existing run; "
            "pass force=True to start it over"
        )
    if changed:
        env = {k: v for k, v in env.items() if k != "mlflow_run_id"}

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(config_path, {"config": conf, "parsed": parsed})
    sha = repo_sha()
    _write_json(
        env_path,
        {
            "mlflow_run_id": None,
            "units_used": None,
            **env,
            "num_workers": num_workers,  # of the latest session; samples/s depends on it
            "repo_sha": sha,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
            "manifests": manifests,
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
        val_params=val_params(conf),
        num_workers=num_workers,
        tmp_dir=tmp_dir,
        repo_sha=sha,
        manifest_hash=_manifest_hash(conf, data_root),
        resume=not changed,
        force=force,
    )


def _run_fn(weights: Path, device: str | None) -> "RunFn":
    """Torch run function for a frozen-BN checkpoint; the patch point the tests replace."""
    import torch

    from fair_trees.train import build_model, torch_run_fn

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return torch_run_fn(build_model(weights, trainable_layers=0), dev)


def _benchmark_parts(benchmark: Path, data_root: Path) -> Iterator[tuple[str, dict, list]]:
    """(name, part, units) for each NEON eval-plot or region part whose data is staged; units must match
    the freeze."""
    from experiments.neon_eval import load_eval_plots
    from fair_trees.dataset import load_regions
    from fair_trees.evaluate import region_units

    for name, part in json.loads(Path(benchmark).read_text())["parts"].items():
        if part["kind"] not in ("neon_eval_plots", "regions"):
            continue
        root = Path(data_root) / part["data_dir"]
        if not root.exists():
            logger.warning(f"benchmark part {name}: {root} not staged, skipped")
            continue
        if part["kind"] == "regions":
            regions = [r for r in load_regions(root) if r.split == part["split"]]
            units = region_units(regions, zooms=part["zooms"])
        else:
            units = load_eval_plots(root)
        if [u.unit_id for u in units] != part["unit_ids"]:
            raise ValueError(f"benchmark part {name}: units differ from the frozen list")
        yield name, part, units


def _with_ci(report) -> dict:
    from fair_trees.evaluate import bootstrap_ci

    return report.to_dict() | {"f1_ci95": list(bootstrap_ci(report.unit_counts()))}


def _num_classes(run_fn: "RunFn") -> int:
    """Classes the model outputs, from one forward pass on an empty window."""
    import numpy as np

    return run_fn(np.zeros((1, 3, 800, 800), np.float32))[0].shape[-1]


def _merged_groups(
    run_fn: "RunFn", regions: Sequence, params: "InferenceParams", zooms: Sequence[int]
) -> dict:
    """Group boxes built by merging a tree-only model's crowns (``MERGED_GROUPS``), scored like a group
    head."""
    from experiments.label_study.merge import merge_crowns
    from fair_trees.decode import filter_score
    from fair_trees.evaluate import (
        Counts,
        _drop_ignored,
        aoi_f1,
        bootstrap_ci,
        count_matches,
        prf,
        region_units,
    )
    from fair_trees.serve import predict_mosaic

    rows, counts = [], []
    for u in region_units(regions, zooms=zooms, label=2):
        crowns = filter_score(predict_mosaic(run_fn, u.mosaic, params), MERGED_GROUPS["score"])
        groups = merge_crowns(
            crowns, MERGED_GROUPS["gap_m"] / u.mosaic.gsd_m, MERGED_GROUPS["min_crowns"]
        ).groups
        if u.ignore_mask is not None:
            groups = _drop_ignored(groups, u.ignore_mask)
        c = count_matches(groups.boxes, u.gt_boxes)
        counts.append(c)
        rows.append({"unit_id": u.unit_id, "meta": dict(u.meta), "tp": c.tp, "fp": c.fp, "fn": c.fn})
    p, r, f1 = prf(sum(counts, Counts(0, 0, 0)))
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "aoi_f1": aoi_f1(counts),
        "f1_ci95": list(bootstrap_ci(counts)),
        "units": rows,
        "setting": dict(MERGED_GROUPS),
    }


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
    """Score a run on its val regions and on the frozen benchmark parts that are staged; writes ``eval.json``.

    NEON eval-plot parts carry the forgetting bound; region parts are report-only.
    """
    from experiments.neon_eval import DF_COMPAT
    from fair_trees.evaluate import evaluate_regions, evaluate_units
    from fair_trees.params import InferenceParams

    out_dir = Path(drive_root) / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if config is not None:
        conf = load_config(config)
    else:
        conf = _stored_config(out_dir / "config.json")
    weights = Path(weights) if weights is not None else out_dir / "best.pt"
    run_fn = _run_fn(weights, device)

    _, val = _regions(conf, data_root)
    params = val_params(conf)
    zooms = conf["val"]["zooms"]
    report = evaluate_regions(run_fn, val, params, zooms=zooms)
    parts = {"val": _with_ci(report) | {"window_m": params.window_m}}
    if any("tree_group" in r.classes for r in val):
        if _num_classes(run_fn) == 2:
            report = evaluate_regions(run_fn, val, params, zooms=zooms, label=2)
            parts["val_group"] = _with_ci(report) | {"window_m": params.group_window_m}
        else:
            parts["val_group_merged"] = _merged_groups(run_fn, val, params, zooms)
    for name, part, units in _benchmark_parts(benchmark, data_root):
        if part["kind"] == "regions":  # report-only: no forgetting bound
            report = evaluate_units(run_fn, units, InferenceParams(window_m=part["window_m"]))
            parts[name] = _with_ci(report) | {"window_m": part["window_m"]}
            continue
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
    neon = (us for _, part, us in _benchmark_parts(benchmark, data_root) if part["kind"] == "neon_eval_plots")
    units = [u for us in neon for u in us][:PARITY_UNITS]
    if not units:
        raise ValueError(
            f"{benchmark}: no staged NEON benchmark units under {data_root} for the parity check"
        )

    model = build_model(out_dir / "best.pt", trainable_layers=0)  # one or two classes, as trained
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


def _eval_part(run_dir: Path, part: str = "val") -> dict:
    parts = json.loads((run_dir / "eval.json").read_text())["parts"]
    if part not in parts:
        raise ValueError(f"{run_dir.name}: eval.json has no part {part!r}, only {sorted(parts)}")
    return parts[part]


def _unit_counts(part: Mapping) -> dict:
    from fair_trees.evaluate import Counts

    return {u["unit_id"]: Counts(u["tp"], u["fp"], u["fn"]) for u in part["units"]}


def _equal_over(a: Mapping | None, b: Mapping | None, epochs: Sequence[str], column: int) -> bool | None:
    """Digest equality over ``epochs``; None without a metrics.csv on either side or a common epoch."""
    if a is None or b is None or not epochs:
        return None
    return all(a[e][column] == b[e][column] for e in epochs)


def compare_runs(
    a: str, b: str, *, drive_root: Path = DRIVE_ROOT, part_a: str = "val", part_b: str | None = None
) -> dict:
    """Sampler digests over the common epochs and the F1 of part ``part_b`` of run ``b`` against part
    ``part_a`` of run ``a`` (default: both ``val``), paired over their units.

    Two different parts (e.g. ``val_group`` against ``val_group_merged``) may use different windows; the
    same part must use the same one.
    """
    from fair_trees.evaluate import paired_bootstrap_diff

    part_b = part_b or part_a
    dir_a, dir_b = Path(drive_root) / "runs" / a, Path(drive_root) / "runs" / b
    dig_a, dig_b = _sampler_digests(dir_a), _sampler_digests(dir_b)
    epochs = sorted(set(dig_a) & set(dig_b), key=int) if dig_a is not None and dig_b is not None else []

    val_a, val_b = _eval_part(dir_a, part_a), _eval_part(dir_b, part_b)
    counts_a, counts_b = _unit_counts(val_a), _unit_counts(val_b)
    if set(counts_a) != set(counts_b):
        raise ValueError(f"runs {a} and {b} were evaluated on different val units")
    window_a, window_b = (v.get("window_m", DEFAULT_VAL["window_m"]) for v in (val_a, val_b))
    if part_a == part_b and window_a != window_b:  # eval.json before window_m: the 40 m default
        raise ValueError(
            f"runs {a} and {b} were scored with different val windows: {window_a} m and {window_b} m"
        )
    ids = list(counts_a)
    a_ci = val_a["f1_ci95"]
    f1_b = val_b["f1"]
    return {
        "a": a,
        "b": b,
        "part_a": part_a,
        "part_b": part_b,
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
