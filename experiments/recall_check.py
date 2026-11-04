"""Where a run's missing tree recall goes, scored without training.

    python -m experiments.recall_check RUN_ID [--part val]

Each setting is decoded once at a low threshold and scored at the run's picked tree threshold:

- box scale: every box shrunk or grown about its centre (mapper crowns are often smaller than our boxes);
  ``located_misses`` counts truth boxes whose centre lies in a prediction that failed the IoU match;
- crown size: recall per truth size bin (a crown under ~1.7 m is below the smallest anchor at 60 m);
- NMS: the per-window suppression threshold (neighbouring crowns can suppress each other);
- window: the tree window (smaller windows enlarge small crowns).

Every setting keeps its per-unit counts, so a val winner can be paired once on held-out test. Writes
``runs/<RUN_ID>/recall_check_<part>.json``.
"""

import argparse
import dataclasses
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCALES = (0.8, 0.85, 0.9, 0.95, 1.0)
NMS_THRESHOLDS = (0.05, 0.1, 0.15, 0.2, 0.3)
WINDOWS_M = (40.0,)  # besides the run's own val window
SIZE_BINS_M = (1.5, 2.5, 4.0, 6.0)  # truth box side in metres: <1.5, 1.5-2.5, 2.5-4, 4-6, >=6


def scale_boxes(boxes: np.ndarray, factor: float) -> np.ndarray:
    """Each xyxy box scaled by ``factor`` about its centre."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    centre = (boxes[:, :2] + boxes[:, 2:]) / 2
    half = (boxes[:, 2:] - boxes[:, :2]) * (factor / 2)
    return np.concatenate([centre - half, centre + half], axis=1)


def _inside(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(points, boxes) bool: point i inside box j, edges inclusive."""
    x, y = points[:, None, 0], points[:, None, 1]
    return (
        (x >= boxes[None, :, 0])
        & (x <= boxes[None, :, 2])
        & (y >= boxes[None, :, 1])
        & (y <= boxes[None, :, 3])
    )


def unit_breakdown(pred: np.ndarray, gt: np.ndarray, gsd_m: float, iou_threshold: float = 0.4) -> dict:
    """One unit's matches: counts, truth located by an unmatched box, centres in any box, size bins."""
    from fair_trees.evaluate import match_hungarian

    pred = np.asarray(pred, np.float32).reshape(-1, 4)
    gt = np.asarray(gt, np.float32).reshape(-1, 4)
    pairs = match_hungarian(pred, gt, iou_threshold)
    hit_pred = np.zeros(len(pred), bool)
    hit_gt = np.zeros(len(gt), bool)
    for p, g in pairs:
        hit_pred[p], hit_gt[g] = True, True
    centres = (gt[:, :2] + gt[:, 2:]) / 2
    inside = _inside(centres, pred) if len(pred) and len(gt) else np.zeros((len(gt), len(pred)), bool)
    located = (~hit_gt) & inside[:, ~hit_pred].any(axis=1) if len(gt) else np.zeros(0, bool)
    side_m = ((gt[:, 2] - gt[:, 0]) + (gt[:, 3] - gt[:, 1])) / 2 * gsd_m
    bins = np.digitize(side_m, SIZE_BINS_M)
    n_bins = len(SIZE_BINS_M) + 1
    return {
        "tp": len(pairs),
        "fp": int(len(pred) - len(pairs)),
        "fn": int(len(gt) - len(pairs)),
        "located_misses": int(located.sum()),
        "point_hits": int(inside.any(axis=1).sum()) if len(gt) else 0,
        "n_gt": len(gt),
        "gt_by_size": np.bincount(bins, minlength=n_bins).tolist(),
        "tp_by_size": np.bincount(bins[hit_gt], minlength=n_bins).tolist(),
    }


def _summary(rows: Sequence[dict]) -> dict:
    """Pooled P/R/F1 and recall diagnostics of one setting, with its per-unit [tp, fp, fn]."""
    from fair_trees.evaluate import Counts, prf

    total = Counts(sum(r["tp"] for r in rows), sum(r["fp"] for r in rows), sum(r["fn"] for r in rows))
    p, r, f1 = prf(total)
    n_gt = sum(row["n_gt"] for row in rows)
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": total.tp,
        "fp": total.fp,
        "fn": total.fn,
        "located_misses": sum(row["located_misses"] for row in rows),
        "point_in_box_recall": sum(row["point_hits"] for row in rows) / n_gt if n_gt else 0.0,
        "units": [[row["tp"], row["fp"], row["fn"]] for row in rows],
    }


def check_units(
    run_fn,
    units: Sequence,
    params,
    threshold: float,
    *,
    scales: Sequence[float] = SCALES,
    nms_thresholds: Sequence[float] = NMS_THRESHOLDS,
    windows_m: Sequence[float] = WINDOWS_M,
) -> dict:
    """Score tree class ``units`` at ``threshold`` under each box scale (base decode), NMS threshold and
    tree window; ``params`` is the run's val setting."""
    from fair_trees.decode import filter_score
    from fair_trees.evaluate import DECODE_THRESHOLD, _drop_ignored
    from fair_trees.serve import predict_mosaic_classes

    def decode(p) -> list[np.ndarray]:
        low = dataclasses.replace(p, confidence_threshold=DECODE_THRESHOLD, class_thresholds=())
        out = []
        for u in units:
            dets = predict_mosaic_classes(run_fn, u.mosaic, low, (0,))[0]
            if u.ignore_mask is not None:
                dets = _drop_ignored(dets, np.asarray(u.ignore_mask, bool))
            out.append(filter_score(dets, threshold).boxes)
        return out

    def rows(boxes: Sequence[np.ndarray], factor: float = 1.0) -> list[dict]:
        return [
            unit_breakdown(scale_boxes(b, factor), u.gt_boxes, u.mosaic.gsd_m)
            for b, u in zip(boxes, units, strict=True)
        ]

    base = decode(params)
    base_rows = rows(base)
    gt_by_size = np.sum([r["gt_by_size"] for r in base_rows], axis=0) if base_rows else []
    tp_by_size = np.sum([r["tp_by_size"] for r in base_rows], axis=0) if base_rows else []
    by_size = {
        "bins_m": list(SIZE_BINS_M),
        "n_gt": [int(n) for n in gt_by_size],
        "recall": [float(t / n) if n else None for t, n in zip(tp_by_size, gt_by_size, strict=True)],
    }
    nms = {
        str(t): _summary(
            base_rows
            if t == params.nms_threshold
            else rows(decode(dataclasses.replace(params, nms_threshold=t)))
        )
        for t in nms_thresholds
    }
    windows = {
        str(w): _summary(
            base_rows if w == params.window_m else rows(decode(dataclasses.replace(params, window_m=w)))
        )
        for w in windows_m
    }
    return {
        "threshold": float(threshold),
        "window_m": float(params.window_m),
        "nms_threshold": float(params.nms_threshold),
        "scales": {str(s): _summary(rows(base, s)) for s in scales},
        "by_size": by_size,
        "nms": nms,
        "windows_m": windows,
    }


def run_recall_check(
    run_id: str, *, part: str = "val", drive_root=None, data_root=None, benchmark=None
) -> dict:
    """``check_units`` on a run's ``best.pt``: its val regions, or a frozen benchmark region part."""
    from experiments import run
    from fair_trees.evaluate import PRIOR_THRESHOLD, region_units
    from fair_trees.export import checkpoint_thresholds

    drive_root = Path(drive_root or run.DRIVE_ROOT)
    data_root = Path(data_root or run.DATA_ROOT)
    out_dir = drive_root / "runs" / run_id
    conf = run._stored_config(out_dir / "config.json")
    params = run.val_params(conf)
    if part == "val":
        units = region_units(run._regions(conf, data_root)[1], zooms=conf["val"]["zooms"])
    else:
        benchmark = Path(benchmark or Path(__file__).parent / "benchmark.json")
        found = [u for name, _, u, _ in run._benchmark_parts(benchmark, data_root) if name == part]
        if not found:
            raise ValueError(f"{part}: not a staged part of {benchmark}")
        units = found[0]
    weights = out_dir / "best.pt"
    threshold = (checkpoint_thresholds(weights) or {}).get("tree", PRIOR_THRESHOLD[1])
    result = {"run_id": run_id, "part": part} | check_units(
        run._run_fn(weights, None), units, params, threshold
    )
    path = out_dir / f"recall_check_{part}.json"
    path.write_text(json.dumps(result, indent=2))
    return result


def report_lines(result: dict) -> list[str]:
    """A short printable view of ``check_units`` output."""

    def line(name: str, s: dict) -> str:
        return (
            f"{name:>14}  F1 {s['f1']:.3f}  P {s['precision']:.3f}  R {s['recall']:.3f}  "
            f"located misses {s['located_misses']}  point-in-box R {s['point_in_box_recall']:.3f}"
        )

    out = [f"threshold {result['threshold']}, window {result['window_m']:g} m, NMS {result['nms_threshold']}"]
    out += [line(f"scale {k}", s) for k, s in result["scales"].items()]
    out += [line(f"NMS {k}", s) for k, s in result["nms"].items()]
    out += [line(f"window {k} m", s) for k, s in result["windows_m"].items()]
    bins = ["<1.5", "1.5-2.5", "2.5-4", "4-6", ">=6"]
    recall = result["by_size"]["recall"]
    out.append(
        "recall by crown size (m): "
        + ", ".join(
            f"{b}: {'-' if r is None else f'{r:.2f}'} (n {n})"
            for b, r, n in zip(bins, recall, result["by_size"]["n_gt"], strict=True)
        )
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--part", default="val")
    args = parser.parse_args(argv)
    print("\n".join(report_lines(run_recall_check(args.run_id, part=args.part))))


if __name__ == "__main__":
    main()
