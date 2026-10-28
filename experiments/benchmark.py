"""The benchmark freeze rule.

A part of ``benchmark.json`` is written once, before any training run reads that source's shards, and
never changes. ``freeze_part`` refuses to alter a written part and ``load_benchmark`` refuses a part whose
unit list no longer matches its hash.

    python -m experiments.benchmark freeze-neon --data-root D --data-dir NeonTreeEvaluation \\
        --archive-sha256 S --release-eval experiments/results/neon-release/eval.json --frozen-on 2026-10-26
"""

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

BENCHMARK = Path(__file__).parent / "benchmark.json"


def ids_sha256(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def load_benchmark(path: Path) -> dict:
    """The benchmark file, or ``{"parts": {}}`` when absent; a part whose ids do not match its hash raises."""
    path = Path(path)
    if not path.exists():
        return {"parts": {}}
    data = json.loads(path.read_text())
    for name, part in data["parts"].items():
        if part["unit_ids_sha256"] != ids_sha256(part["unit_ids"]):
            raise ValueError(
                f"benchmark part {name}: unit_ids_sha256 does not match its unit_ids (hand-edited?)"
            )
    return data


def freeze_part(path: Path, name: str, part: dict) -> dict:
    """Add ``part`` under ``name``; an identical re-freeze returns it, a different one raises."""
    frozen = part | {"unit_ids_sha256": ids_sha256(part["unit_ids"])}
    data = load_benchmark(path)
    if name in data["parts"]:
        if data["parts"][name] == frozen:
            return frozen
        raise ValueError(f"benchmark part {name} is frozen and cannot change")
    data["parts"][name] = frozen
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return frozen


def neon_part(
    data_root: Path, data_dir: str, *, archive_sha256: str, release_eval: Path, frozen_on: str
) -> dict:
    """The NEON eval-plot part: the plot ids in the staged archive plus the release score to beat."""
    from experiments.neon_eval import load_eval_plots

    ids = [u.unit_id for u in load_eval_plots(Path(data_root) / data_dir)]
    release = json.loads(Path(release_eval).read_text())
    release = release.get("parts", {}).get("neon", release)
    return {
        "kind": "neon_eval_plots",
        "data_dir": data_dir,
        "unit_ids": ids,
        "archive_sha256": archive_sha256,
        "mode": "deepforest_compatible",
        "release_f1": release["f1"],
        "release_f1_ci95": list(release["f1_ci95"]),
        "frozen_on": frozen_on,
    }


def regions_part(
    data_root: Path, data_dir: str, *, split: str, zooms: Sequence[int], window_m: float, frozen_on: str
) -> dict:
    """A report-only part over the staged regions of one split, one unit per region mosaic."""
    from fair_trees.dataset import load_regions
    from fair_trees.evaluate import region_units

    regions = [r for r in load_regions(Path(data_root) / data_dir) if r.split == split]
    return {
        "kind": "regions",
        "data_dir": data_dir,
        "split": split,
        "zooms": list(zooms),
        "window_m": float(window_m),
        "unit_ids": [u.unit_id for u in region_units(regions, zooms=zooms)],
        "frozen_on": frozen_on,
    }


def cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.benchmark", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("freeze-neon", help="freeze the NEON eval-plot part")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--release-eval", type=Path, required=True)
    p.add_argument("--frozen-on", required=True)
    p.add_argument("--benchmark", type=Path, default=BENCHMARK)
    p = sub.add_parser("freeze-regions", help="freeze a report-only part over staged regions of one split")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-dir", required=True, help="the staged source folder; also the part's name")
    p.add_argument("--split", required=True)
    p.add_argument("--zooms", type=int, nargs="+", default=[20, 19])
    p.add_argument("--window-m", type=float, default=40.0)
    p.add_argument("--frozen-on", required=True)
    p.add_argument("--benchmark", type=Path, default=BENCHMARK)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if args.command == "freeze-regions":
        part = regions_part(
            args.data_root,
            args.data_dir,
            split=args.split,
            zooms=args.zooms,
            window_m=args.window_m,
            frozen_on=args.frozen_on,
        )
        frozen = freeze_part(args.benchmark, args.data_dir, part)
        print(
            f"{args.data_dir}: {len(frozen['unit_ids'])} units, unit_ids_sha256 {frozen['unit_ids_sha256']}"
        )
        return
    part = neon_part(
        args.data_root,
        args.data_dir,
        archive_sha256=args.archive_sha256,
        release_eval=args.release_eval,
        frozen_on=args.frozen_on,
    )
    frozen = freeze_part(args.benchmark, "neon", part)
    print(f"neon: {len(frozen['unit_ids'])} plots, unit_ids_sha256 {frozen['unit_ids_sha256']}")


if __name__ == "__main__":
    cli()
