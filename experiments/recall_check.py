"""Where a run's missing tree recall goes, scored without training.

    python -m experiments.recall_check RUN_ID [--part val]

Each setting is decoded once at a low threshold and scored at the run's picked tree threshold, with two
scores. The IoU score is the run's own (one-to-one, IoU > 0.4). The centre score (``match_centres``) pairs a
truth crown with a box that holds its centre, one-to-one, whatever the mapper's crown-size tag says; the
gap between the two is what box size costs at the IoU gate. Recall per truth size bin is reported for both
in every setting, with the median side of a centre-matched box over its truth side.

- box scale: every box shrunk or grown about its centre (mapper crowns are often smaller than our boxes);
  ``located_misses`` counts truth boxes whose centre lies in a prediction that failed the IoU match;
- NMS: the per-window suppression threshold (neighbouring crowns can suppress each other);
- window: the tree window (smaller windows enlarge small crowns). The run's window and each other one are
  swept over the pick thresholds, so a window is read at its own pick: ``picked`` per part, and for a part
  other than val ``at_val_pick`` (run val first), the rows at the thresholds val picked;
- half resolution: the units with each 2 x 2 block of pixels averaged, zoom 19 emulated from zoom 20, with
  a sweep and picks of its own.

Every setting keeps its per-unit counts, so a val winner can be paired once on held-out test. Writes
``runs/<RUN_ID>/recall_check_<part>.json``.
"""

import argparse
import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
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
    """One unit's matches: counts, truth located by an unmatched box, centres in any box, size bins, and
    the centre match (``match_centres``) with the side of each paired box over its truth side."""
    from fair_trees.evaluate import match_centres, match_hungarian

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
    side = ((gt[:, 2] - gt[:, 0]) + (gt[:, 3] - gt[:, 1])) / 2
    bins = np.digitize(side * gsd_m, SIZE_BINS_M)
    n_bins = len(SIZE_BINS_M) + 1
    centre_pairs = match_centres(pred, gt)
    centre_gt = np.array([g for _, g in centre_pairs], int)
    pred_side = ((pred[:, 2] - pred[:, 0]) + (pred[:, 3] - pred[:, 1])) / 2
    ratios: list[list[float]] = [[] for _ in range(n_bins)]
    for p, g in centre_pairs:
        if side[g] > 0:
            ratios[bins[g]].append(float(pred_side[p] / side[g]))
    return {
        "tp": len(pairs),
        "fp": int(len(pred) - len(pairs)),
        "fn": int(len(gt) - len(pairs)),
        "located_misses": int(located.sum()),
        "point_hits": int(inside.any(axis=1).sum()) if len(gt) else 0,
        "n_gt": len(gt),
        "n_pred": len(pred),
        "gt_by_size": np.bincount(bins, minlength=n_bins).tolist(),
        "tp_by_size": np.bincount(bins[hit_gt], minlength=n_bins).tolist(),
        "centre_tp": len(centre_pairs),
        "centre_tp_by_size": np.bincount(bins[centre_gt], minlength=n_bins).tolist(),
        "centre_ratio_by_size": ratios,
    }


def _by_size(rows: Sequence[dict]) -> dict:
    """Per truth size bin over ``rows``: truth count, recall at the IoU match, recall at the centre match,
    and the median side of a centre-matched box over its truth side (None for a bin without any)."""
    n_bins = len(SIZE_BINS_M) + 1

    def total(key: str) -> list[int]:
        return [sum(r[key][b] for r in rows) for b in range(n_bins)]

    n_gt, tp, centre_tp = total("gt_by_size"), total("tp_by_size"), total("centre_tp_by_size")
    ratios = [[x for r in rows for x in r["centre_ratio_by_size"][b]] for b in range(n_bins)]
    return {
        "n_gt": n_gt,
        "recall": [t / n if n else None for t, n in zip(tp, n_gt, strict=True)],
        "centre_recall": [t / n if n else None for t, n in zip(centre_tp, n_gt, strict=True)],
        "centre_side_ratio": [float(np.median(x)) if x else None for x in ratios],
    }


def _summary(rows: Sequence[dict]) -> dict:
    """Pooled P/R/F1 and recall diagnostics of one setting with its per-unit [tp, fp, fn], the same for
    the centre match under ``centre``, and both recalls per crown size under ``by_size``."""
    from fair_trees.evaluate import Counts, prf

    def scored(units: list[list[int]]) -> dict:
        total = Counts(*(sum(u[k] for u in units) for k in range(3)))
        p, r, f1 = prf(total)
        return {"precision": p, "recall": r, "f1": f1, "tp": total.tp, "fp": total.fp, "fn": total.fn}

    units = [[row["tp"], row["fp"], row["fn"]] for row in rows]
    centre_units = [
        [row["centre_tp"], row["n_pred"] - row["centre_tp"], row["n_gt"] - row["centre_tp"]] for row in rows
    ]
    n_gt = sum(row["n_gt"] for row in rows)
    return scored(units) | {
        "located_misses": sum(row["located_misses"] for row in rows),
        "point_in_box_recall": sum(row["point_hits"] for row in rows) / n_gt if n_gt else 0.0,
        "units": units,
        "centre": scored(centre_units) | {"units": centre_units},
        "by_size": _by_size(rows),
    }


def decode_trees(run_fn, units: Sequence, params) -> list:
    """Each unit's tree detections at ``DECODE_THRESHOLD``, those centred in its ignore mask dropped."""
    from fair_trees.evaluate import DECODE_THRESHOLD, _drop_ignored
    from fair_trees.serve import predict_mosaic_classes

    low = dataclasses.replace(params, confidence_threshold=DECODE_THRESHOLD, class_thresholds=())
    out = []
    for u in units:
        dets = predict_mosaic_classes(run_fn, u.mosaic, low, (0,))[0]
        if u.ignore_mask is not None:
            dets = _drop_ignored(dets, np.asarray(u.ignore_mask, bool))
        out.append(dets)
    return out


def halve_unit(unit):
    """``unit`` at half its resolution, as one zoom level coarser imagery of the same ground: each 2 x 2
    block of pixels becomes its mean (an odd last row or column is cropped), boxes follow, and a block is
    invalid if any of its pixels was and ignored if any of its pixels was."""
    from affine import Affine

    from fair_trees.preprocess import Mosaic

    m = unit.mosaic
    h, w = m.rgb.shape[0] // 2, m.rgb.shape[1] // 2

    def blocks(a: np.ndarray) -> np.ndarray:
        return a[: 2 * h, : 2 * w].reshape(h, 2, w, 2, *a.shape[2:])

    valid = blocks(m.valid).all(axis=(1, 3))
    rgb = np.rint(blocks(m.rgb).mean(axis=(1, 3))).astype(np.uint8)
    rgb[~valid] = 0
    mosaic = Mosaic(
        rgb, valid, m.transform_3857 @ Affine.scale(2.0), m.z - 1 if m.z > 0 else m.z, m.gsd_m * 2
    )
    boxes = np.asarray(unit.gt_boxes, np.float32).reshape(-1, 4) / 2
    boxes = np.clip(boxes, 0, np.array([w, h, w, h], np.float32))
    ignore = None if unit.ignore_mask is None else blocks(np.asarray(unit.ignore_mask, bool)).any(axis=(1, 3))
    return dataclasses.replace(unit, mosaic=mosaic, gt_boxes=boxes, ignore_mask=ignore)


def _pick(sweep: Sequence[dict], score) -> float:
    """The lowest threshold of ``sweep`` at which ``score`` peaks, as a fine-tune picks its own."""
    best = max(score(row) for row in sweep)
    return min(row["threshold"] for row in sweep if score(row) == best)


def at_picks(result: Mapping, picks: Mapping) -> dict:
    """``result``'s sweep rows at thresholds picked elsewhere (on val, for a held-out part): per window of
    ``picks`` that ``result`` swept, the row at its IoU pick (``f1``) and at its centre pick
    (``centre_f1``)."""
    out = {}
    for window, pick in picks.items():
        if window not in result["threshold_sweep"]:
            continue
        rows = {row["threshold"]: row for row in result["threshold_sweep"][window]}
        missing = [t for t in pick.values() if t not in rows]
        if missing:
            raise KeyError(
                f"window {window} m: threshold {missing[0]} is not one of the swept {sorted(rows)}"
            )
        out[window] = {name: rows[t] for name, t in pick.items()}
    return out


def check_units(
    run_fn,
    units: Sequence,
    params,
    threshold: float,
    *,
    scales: Sequence[float] = SCALES,
    nms_thresholds: Sequence[float] = NMS_THRESHOLDS,
    windows_m: Sequence[float] = WINDOWS_M,
    pick_thresholds: Sequence[float] | None = None,
    half_resolution: bool = False,
) -> dict:
    """Score tree class ``units`` at ``threshold`` under each box scale (base decode), NMS threshold and
    tree window; ``params`` is the run's val setting.

    The run's window and each of ``windows_m`` are also scored at every one of ``pick_thresholds`` (default:
    where a fine-tune picks), so a window is judged at a threshold of its own: ``picked`` holds, per window,
    the threshold with the best IoU F1 and the one with the best centre F1 on these units.

    ``half_resolution`` adds the same units at half their resolution (``halve_unit``: zoom 19 emulated from
    zoom 20), scored at the run's window with a sweep and picks of their own.
    """
    from fair_trees.decode import filter_score
    from fair_trees.evaluate import PICK_THRESHOLDS

    if pick_thresholds is None:
        pick_thresholds = PICK_THRESHOLDS

    def decode(p) -> list:
        return decode_trees(run_fn, units, p)

    def rows(dets: Sequence, at: float = threshold, factor: float = 1.0) -> list[dict]:
        return [
            unit_breakdown(scale_boxes(filter_score(d, at).boxes, factor), u.gt_boxes, u.mosaic.gsd_m)
            for d, u in zip(dets, units, strict=True)
        ]

    by_window = {float(params.window_m): decode(params)}
    for w in windows_m:
        if float(w) not in by_window:
            by_window[float(w)] = decode(dataclasses.replace(params, window_m=w))
    base = by_window[float(params.window_m)]
    base_rows = rows(base)
    nms = {
        str(t): _summary(
            base_rows
            if t == params.nms_threshold
            else rows(decode(dataclasses.replace(params, nms_threshold=t)))
        )
        for t in nms_thresholds
    }
    sweep = {
        str(w): [{"threshold": float(t)} | _summary(rows(dets, float(t))) for t in pick_thresholds]
        for w, dets in by_window.items()
    }
    half = {}
    if half_resolution:
        halved = [halve_unit(u) for u in units]
        half["half_resolution"] = {"gsd_m": float(np.median([u.mosaic.gsd_m for u in halved]))} | check_units(
            run_fn,
            halved,
            params,
            threshold,
            scales=(1.0,),
            nms_thresholds=(),
            windows_m=(),
            pick_thresholds=pick_thresholds,
        )
    return half | {
        "threshold": float(threshold),
        "window_m": float(params.window_m),
        "nms_threshold": float(params.nms_threshold),
        "scales": {str(s): _summary(rows(base, factor=s)) for s in scales},
        "by_size": {"bins_m": list(SIZE_BINS_M)} | _summary(base_rows)["by_size"],
        "nms": nms,
        "windows_m": {str(float(w)): _summary(rows(by_window[float(w)])) for w in windows_m},
        "threshold_sweep": sweep,
        "picked": {
            w: {
                "f1": _pick(s, lambda row: row["f1"]),
                "centre_f1": _pick(s, lambda row: row["centre"]["f1"]),
            }
            for w, s in sweep.items()
        },
    }


def run_units(run_id: str, *, part: str = "val", drive_root=None, data_root=None, benchmark=None):
    """(run folder, val params, tree units of ``part``, run function of ``best.pt``, picked tree threshold):
    ``part`` is the run's val regions or a frozen benchmark region part."""
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
    return out_dir, params, units, run._run_fn(weights, None), threshold


def with_val_picks(result: Mapping, val: Mapping) -> dict:
    """``result`` (a held-out part) with ``at_val_pick``: its sweep rows at the thresholds ``val`` (the same
    run's val result) picked, for the full and the half resolution. A val result without picks adds
    nothing."""
    out = dict(result)
    if "picked" in val:
        out["at_val_pick"] = at_picks(result, val["picked"])
    half, val_half = result.get("half_resolution"), val.get("half_resolution")
    if half and val_half and "picked" in val_half:
        out["half_resolution"] = half | {"at_val_pick": at_picks(half, val_half["picked"])}
    return out


def run_recall_check(
    run_id: str, *, part: str = "val", drive_root=None, data_root=None, benchmark=None
) -> dict:
    """``check_units`` on a run's ``best.pt``: its val regions, or a frozen benchmark region part. A part
    other than val is also read at the thresholds picked on val, when the run's val result is there."""
    out_dir, params, units, run_fn, threshold = run_units(
        run_id, part=part, drive_root=drive_root, data_root=data_root, benchmark=benchmark
    )
    result = {"run_id": run_id, "part": part} | check_units(
        run_fn, units, params, threshold, half_resolution=True
    )
    val_path = out_dir / "recall_check_val.json"
    if part != "val" and val_path.is_file():
        result = with_val_picks(result, json.loads(val_path.read_text()))
    path = out_dir / f"recall_check_{part}.json"
    path.write_text(json.dumps(result, indent=2))
    return result


SIZE_BIN_NAMES = ("<1.5", "1.5-2.5", "2.5-4", "4-6", ">=6")


def _setting_line(name: str, s: Mapping) -> str:
    c = s["centre"]
    return (
        f"{name:>14}  F1 {s['f1']:.3f}  P {s['precision']:.3f}  R {s['recall']:.3f}  "
        f"centre F1 {c['f1']:.3f}  P {c['precision']:.3f}  R {c['recall']:.3f}  "
        f"located misses {s['located_misses']}  point-in-box R {s['point_in_box_recall']:.3f}"
    )


def _size_line(name: str, by_size: Mapping) -> str:
    cells = []
    for i, b in enumerate(SIZE_BIN_NAMES):
        n = by_size["n_gt"][i]
        if not n:
            cells.append(f"{b}: - (n 0)")
            continue
        ratio = by_size["centre_side_ratio"][i]
        cells.append(
            f"{b}: IoU {by_size['recall'][i]:.2f}, centre {by_size['centre_recall'][i]:.2f}, "
            f"box/label {'-' if ratio is None else f'{ratio:.1f}'} (n {n})"
        )
    return f"recall by crown size (m), {name}: " + "; ".join(cells)


def _pick_lines(result: Mapping, prefix: str = "") -> list[str]:
    out = []
    for window, pick in result["picked"].items():
        rows = {row["threshold"]: row for row in result["threshold_sweep"][window]}
        out.append(
            f"{prefix}window {window} m: own pick {pick['f1']:g} (F1 {rows[pick['f1']]['f1']:.3f}), "
            f"own centre pick {pick['centre_f1']:g} (centre F1 {rows[pick['centre_f1']]['centre']['f1']:.3f})"
        )
        at_val = result.get("at_val_pick", {}).get(window)
        if at_val:
            out.append(
                f"{prefix}window {window} m at val's picks: F1 {at_val['f1']['f1']:.3f} "
                f"@{at_val['f1']['threshold']:g}, centre F1 {at_val['centre_f1']['centre']['f1']:.3f} "
                f"@{at_val['centre_f1']['threshold']:g}"
            )
    return out


def report_lines(result: dict) -> list[str]:
    """A short printable view of ``check_units`` output: each setting at the run's threshold with the IoU
    and the centre score, recall by crown size, each window's own picks, and the half resolution."""
    out = [f"threshold {result['threshold']}, window {result['window_m']:g} m, NMS {result['nms_threshold']}"]
    out += [_setting_line(f"scale {k}", s) for k, s in result["scales"].items()]
    out += [_setting_line(f"NMS {k}", s) for k, s in result["nms"].items()]
    out += [_setting_line(f"window {k} m", s) for k, s in result["windows_m"].items()]
    out.append(_size_line(f"window {result['window_m']:g} m", result["by_size"]))
    out += [_size_line(f"window {k} m", s["by_size"]) for k, s in result["windows_m"].items()]
    out += _pick_lines(result)
    half = result.get("half_resolution")
    if half:
        name = f"half resolution ({half['gsd_m']:.2f} m/px)"
        out.append(_setting_line(name, half["scales"]["1.0"]))
        out.append(_size_line(name, half["by_size"]))
        out += _pick_lines(half, "half resolution, ")
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--part", default="val")
    args = parser.parse_args(argv)
    print("\n".join(report_lines(run_recall_check(args.run_id, part=args.part))))


if __name__ == "__main__":
    main()
