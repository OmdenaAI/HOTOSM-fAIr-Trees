"""What a cheaper serving setting costs in F1, scored without training.

    python -m experiments.overlap_check RUN_ID [--part val]

Serving time is the number of windows the network runs, and ``window_overlap`` sets it. Each overlap's
windows are run once per class; their boxes are then scored at every pick threshold under each edge band
and, for tree groups, each merge rule, since a narrower band or ``join`` may win back what a smaller overlap
loses beside window edges. Every setting keeps its per-unit counts, so any two can be paired and a val pick
confirmed once on held-out test. Writes ``runs/<RUN_ID>/overlap_check_<part>.json``.
"""

import argparse
import dataclasses
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

OVERLAPS = (0.5, 0.25, 0.125, 0.0)  # the first is the reference the report pairs the others against
# share of the window side: a box reaching into it beside another window is cut. The run's own band (the
# first) must stay in, since it marks the default settings.
EDGE_BANDS = (0.05, 0.02, 0.0)


def window_parts(run_fn, mosaic, params, class_index: int) -> list[tuple]:
    """Each non-blank window's boxes of one class (mosaic pixels, before any edge rule), with the window and
    the sides it shares with another window."""
    from fair_trees.serve import _decode, _run_windows

    window_m = params.group_window_m if class_index else params.window_m
    return [
        (_decode(run, params, class_index), run.window, run.interior)
        for run in _run_windows(run_fn, mosaic, params, window_m)
    ]


def flag_cut(parts: Sequence[tuple], band: float) -> list[tuple]:
    """Each window's boxes with whether each reaches into ``band`` beside another window."""
    from fair_trees.decode import cut_boxes

    return [(d, cut_boxes(d, window, sides, band)) for d, window, sides in parts]


def detections_at(
    flagged: Sequence[tuple], rule: str, iou_threshold: float, thresholds: Sequence[float]
) -> list:
    """Per threshold, a unit's boxes once ``rule`` has dropped or joined the cut ones
    (``merge_group_detections``; trees are served with ``drop_edge``)."""
    from fair_trees.decode import filter_group_parts, filter_score, merge_group_detections

    if rule == "drop_edge":  # greedy NMS keeps the same boxes whether scores are filtered before or after
        merged = merge_group_detections(flagged, iou_threshold, rule)
        return [filter_score(merged, t) for t in thresholds]
    return [merge_group_detections(filter_group_parts(flagged, t), iou_threshold, rule) for t in thresholds]


def unit_counts(dets, unit) -> list[int]:
    """One unit's [tp, fp, fn], boxes centred in its ignore mask left unscored."""
    from fair_trees.evaluate import _drop_ignored, count_matches

    if unit.ignore_mask is not None:
        dets = _drop_ignored(dets, np.asarray(unit.ignore_mask, bool))
    c = count_matches(dets.boxes, unit.gt_boxes)
    return [c.tp, c.fp, c.fn]


def _row(threshold: float, units: list[list[int]]) -> dict:
    """Pooled P/R/F1 of one setting at one threshold, with its per-unit [tp, fp, fn]."""
    from fair_trees.evaluate import Counts, prf

    p, r, f1 = prf(Counts(*(sum(u[k] for u in units) for k in range(3))))
    return {"threshold": threshold, "precision": p, "recall": r, "f1": f1, "units": units}


def check_units(
    run_fn,
    units: Sequence,
    params,
    class_index: int,
    *,
    overlaps: Sequence[float] = OVERLAPS,
    bands: Sequence[float] = EDGE_BANDS,
    thresholds: Sequence[float] | None = None,
) -> dict:
    """Score class ``class_index`` of ``units`` under each overlap, edge band and (tree groups) merge rule,
    at each threshold (default: the pick grid). ``params`` is the run's val setting; its band and rule mark
    the ``default`` settings, and ``passes`` counts the windows an overlap ran."""
    from fair_trees.evaluate import PICK_THRESHOLDS
    from fair_trees.params import CLASSES, GROUP_MERGE_RULES

    thresholds = [float(t) for t in (PICK_THRESHOLDS if thresholds is None else thresholds)]
    rules = GROUP_MERGE_RULES if class_index else ("drop_edge",)
    default = (params.edge_band, params.group_merge if class_index else "drop_edge")
    settings = []
    for overlap in overlaps:
        low = dataclasses.replace(
            params, window_overlap=overlap, confidence_threshold=min(thresholds), class_thresholds=()
        )
        counts = {(band, rule): [[] for _ in thresholds] for band in bands for rule in rules}
        passes, started = 0, time.perf_counter()
        for u in units:
            parts = window_parts(run_fn, u.mosaic, low, class_index)
            passes += len(parts)
            flagged = {band: flag_cut(parts, band) for band in bands}
            for (band, rule), rows in counts.items():
                kept = detections_at(flagged[band], rule, params.iou_threshold, thresholds)
                for row, dets in zip(rows, kept, strict=True):
                    row.append(unit_counts(dets, u))
        logger.info(
            f"{CLASSES[class_index]}, overlap {overlap:g}: {passes} passes over {len(units)} units "
            f"in {time.perf_counter() - started:.0f} s"
        )
        settings += [
            {
                "overlap": float(overlap),
                "edge_band": float(band),
                "rule": rule,
                "default": (band, rule) == default,
                "passes": passes,
                "rows": [_row(t, row) for t, row in zip(thresholds, rows, strict=True)],
            }
            for (band, rule), rows in counts.items()
        ]
    return {
        "class": CLASSES[class_index],
        "window_m": float(params.group_window_m if class_index else params.window_m),
        "thresholds": thresholds,
        "unit_ids": [u.unit_id for u in units],
        "area_km2": float(sum(u.mosaic.valid.sum() * u.mosaic.gsd_m**2 for u in units) / 1e6),
        "settings": settings,
    }


def run_overlap_check(
    run_id: str,
    *,
    part: str = "val",
    drive_root=None,
    data_root=None,
    benchmark=None,
    overlaps: Sequence[float] = OVERLAPS,
) -> dict:
    """``check_units`` for each class of a run's ``best.pt`` at its val windows, on its val regions or a
    frozen benchmark region part; ``served`` holds the run's picked thresholds (else the prior)."""
    from experiments import run
    from fair_trees.evaluate import PRIOR_THRESHOLD, region_units
    from fair_trees.export import checkpoint_thresholds
    from fair_trees.params import CLASSES

    drive_root = Path(drive_root or run.DRIVE_ROOT)
    data_root = Path(data_root or run.DATA_ROOT)
    out_dir = drive_root / "runs" / run_id
    conf = run._stored_config(out_dir / "config.json")
    params = run.val_params(conf)
    if part == "val":
        val = run._regions(conf, data_root)[1]

        def units_of(class_index: int) -> list:  # one class at a time: each holds its own mosaics
            return region_units(val, zooms=conf["val"]["zooms"], label=class_index + 1)
    else:
        benchmark = Path(benchmark or run.BENCHMARK)
        found = [(u, g) for name, _, u, g in run._benchmark_parts(benchmark, data_root) if name == part]
        if not found:
            raise ValueError(f"{part}: not a staged part of {benchmark}")

        def units_of(class_index: int) -> list:
            return found[0][class_index]

    weights = out_dir / "best.pt"
    run_fn = run._run_fn(weights, None)
    names = CLASSES[: run._num_classes(run_fn)]
    picked = checkpoint_thresholds(weights) or {}
    served = {name: float(picked.get(name, PRIOR_THRESHOLD[len(names)])) for name in names}
    classes = {}
    for class_index, name in enumerate(names):
        units = units_of(class_index)
        if units:
            classes[name] = check_units(run_fn, units, params, class_index, overlaps=overlaps)
    result = {"run_id": run_id, "part": part, "served": served, "classes": classes}
    (out_dir / f"overlap_check_{part}.json").write_text(json.dumps(result, allow_nan=False) + "\n")
    return result


def _scores(row: dict) -> str:
    return f"F1 {row['f1']:.3f}  P {row['precision']:.3f}  R {row['recall']:.3f}"


def _versus(setting: dict, reference: dict, i: int) -> str:
    """Report text: a setting's paired F1 difference against the reference at threshold index ``i``."""
    from fair_trees.evaluate import Counts, paired_bootstrap_diff

    a, b = ([Counts(*u) for u in s["rows"][i]["units"]] for s in (setting, reference))
    diff, lo, hi = paired_bootstrap_diff(a, b)
    return f"  vs overlap {reference['overlap']:g}: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _class_lines(part: dict, served: float) -> list[str]:
    thresholds = part["thresholds"]
    i = min(range(len(thresholds)), key=lambda k: abs(thresholds[k] - served))
    defaults = [s for s in part["settings"] if s["default"]]
    several_rules = len({s["rule"] for s in part["settings"]}) > 1
    lines = [
        f"{part['class']}: {part['window_m']:g} m windows, threshold {thresholds[i]:g}, "
        f"{len(part['unit_ids'])} units, {part['area_km2']:.2f} km2"
    ]
    for s in defaults:
        versus = "" if s is defaults[0] else _versus(s, defaults[0], i)
        lines.append(f"  overlap {s['overlap']:<6g}{s['passes']:>7} passes  {_scores(s['rows'][i])}{versus}")
        f1 = s["rows"][i]["f1"]
        better = [o for o in part["settings"] if o["overlap"] == s["overlap"] and o["rows"][i]["f1"] > f1]
        if better:
            best = max(better, key=lambda o: o["rows"][i]["f1"])
            label = f"band {best['edge_band']:g}" + (f", {best['rule']}" if several_rules else "")
            lines.append(f"      {label:<25}  {_scores(best['rows'][i])}{_versus(best, defaults[0], i)}")
    return lines


def report_lines(result: dict) -> list[str]:
    """A short printable view. Per class and overlap: the default setting at the served threshold and, when
    one beats it, the best other band or rule; each paired against the first overlap's default."""
    return [
        line
        for name, part in result["classes"].items()
        for line in _class_lines(part, result["served"][name])
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--part", default="val")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print("\n".join(report_lines(run_overlap_check(args.run_id, part=args.part))))


if __name__ == "__main__":
    main()
