"""Reproduce DeepForest's published NEON F1 with our pipeline.

    python -m experiments.neon_repro --weights weights/NEON.pt \
        --data data_local/NeonTreeEvaluation --out results/neon.json
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from experiments.neon_eval import DF_COMPAT, load_eval_plots
from fair_trees.evaluate import EvalUnit, bootstrap_ci, evaluate_units

PUBLISHED_F1 = 0.719
PASS_F1 = 0.72
PASS_TOLERANCE = 0.03


def run_neon_repro(
    units: Sequence[EvalUnit], run_fn, out_path: Path, *, n_boot: int = 1000, seed: int = 0
) -> dict:
    report = evaluate_units(run_fn, units, DF_COMPAT, ignore=False)
    lo, hi = bootstrap_ci(report.unit_counts(), n=n_boot, seed=seed)
    payload = report.to_dict() | {
        "f1_ci95": [lo, hi],
        "published_f1": PUBLISHED_F1,
        "pass": abs(report.f1 - PASS_F1) <= PASS_TOLERANCE,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, allow_nan=False))
    return payload


def main(argv=None) -> None:
    import torch

    from fair_trees.train import build_model, torch_run_fn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    model = build_model(Path(args.weights), trainable_layers=0)
    units = load_eval_plots(Path(args.data))
    result = run_neon_repro(units, torch_run_fn(model, torch.device(args.device)), Path(args.out))
    lo, hi = result["f1_ci95"]
    print(
        f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f} "
        f"CI95=[{lo:.3f}, {hi:.3f}] plots={len(units)} {'PASS' if result['pass'] else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
