"""A run's tree boxes scored against hand-drawn crowns instead of the mapper tags, without training.

    python -m experiments.gold_score RUN_ID GOLD_MANIFEST GOLD_BOXES [--part val]

``GOLD_MANIFEST`` is the key written by ``experiments.gold_crowns`` and ``GOLD_BOXES`` the annotator page's
CSV export. Each gold tree whose point lies on a valid, unmasked pixel of a ``part`` unit is placed on that
unit's mosaic. The run's tree class is decoded once per unit and kept at its picked tree threshold; per gold
tree drawn as a box:

- ``gold_hit``: some kept box has IoU > 0.4 with the drawn box;
- ``tag_hit``: some kept box has IoU > 0.4 with the unit's label box at that point (the frozen score);
- ``point_hit``: some kept box covers the point; ``size_ratio``: side of the covering box closest in IoU to
  the drawn box, over the drawn side;
- ``tag_band`` / ``calibrated_band``: the label box itself, and the label box resized by ``CALIBRATION_M``,
  have IoU > 0.4 with the drawn box (what a corrected scoreboard would do).

Trees marked not a tree only get ``point_hit``. Matches are per tree, not one-to-one: gold trees are sparse,
so this is a recall-style check of box size, not a replacement for the benchmark F1. Writes
``runs/<RUN_ID>/gold_score_<part>.json``.
"""

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from experiments.gold_crowns import STRATA, UNSIZED
from fair_trees.georef import ORIGIN_SHIFT

IOU = 0.4
CALIBRATION_M = {1.0: 2.5, 2.0: 3.75}  # train gold medians of the 1 m and 2 m tags
CALIBRATION_TOL_M = 1e-6
TAG_MATCH_M = 0.5  # a label box counts as the tree's own when its centre is this close to the point
GROUPS = [name for name, *_ in STRATA] + [UNSIZED]


def load_gold(manifest: Path, boxes: Path) -> list[dict]:
    """Per gold tree: gold_id, split, stratum, diameter_m, status, z, the point and (for a box) the drawn
    box in global pixels at zoom z. Only the main set; the last save of an id wins."""
    with Path(manifest).open() as f:
        key = {r["gold_id"]: r for r in csv.DictReader(f)}
    drawn: dict[str, dict] = {}
    with Path(boxes).open() as f:
        for r in csv.DictReader(f):
            if r["set"] == "items" and r["id"] in key:
                drawn[r["id"]] = r
    out = []
    for gold_id, r in sorted(drawn.items()):
        k = key[gold_id]
        x0, y0 = float(k["crop_x0"]), float(k["crop_y0"])
        row = {
            "gold_id": gold_id,
            "split": k["split"],
            "stratum": k["stratum"],
            "diameter_m": float(k["diameter_m"]) if k["diameter_m"] else float("nan"),
            "status": r["status"],
            "z": int(k["z"]),
            "point": (x0 + float(k["point_x"]), y0 + float(k["point_y"])),
            "box": None,
        }
        if r["status"] == "box":
            row["box"] = (x0 + float(r["x0"]), y0 + float(r["y0"]), x0 + float(r["x1"]), y0 + float(r["y1"]))
        out.append(row)
    return out


def global_to_mosaic(xy: np.ndarray, z: int, mosaic) -> np.ndarray:
    """(N, 2) global pixels at zoom ``z`` as (N, 2) pixels of ``mosaic``."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    res = 2 * ORIGIN_SHIFT / (256 * 2**z)
    mx, my = -ORIGIN_SHIFT + xy[:, 0] * res, ORIGIN_SHIFT - xy[:, 1] * res
    inv = ~mosaic.transform_3857
    return np.stack([inv.a * mx + inv.b * my + inv.c, inv.d * mx + inv.e * my + inv.f], axis=1)


def place(gold: Sequence[dict], units: Sequence) -> list[tuple[int, dict]]:
    """(unit index, row with ``point_px`` and ``box_px``) for each gold tree on a valid, unmasked pixel of a
    unit; the first unit wins."""
    placed = []
    for row in gold:
        for i, u in enumerate(units):
            ((px, py),) = global_to_mosaic([row["point"]], row["z"], u.mosaic)
            h, w = u.mosaic.valid.shape
            c, r = int(np.floor(px)), int(np.floor(py))
            if not (0 <= c < w and 0 <= r < h) or not u.mosaic.valid[r, c]:
                continue
            if u.ignore_mask is not None and u.ignore_mask[r, c]:
                continue
            box = None
            if row["box"] is not None:
                corners = global_to_mosaic(np.reshape(row["box"], (2, 2)), row["z"], u.mosaic)
                box = np.concatenate([corners.min(axis=0), corners.max(axis=0)])
            placed.append((i, row | {"point_px": np.array([px, py]), "box_px": box}))
            break
    return placed


def _calibrated(tag: np.ndarray, diameter_m: float) -> np.ndarray:
    """The tag box scaled about its centre when its tag is in ``CALIBRATION_M``, else unchanged."""
    for tag_m, size_m in CALIBRATION_M.items():
        if abs(diameter_m - tag_m) <= CALIBRATION_TOL_M:
            centre, half = (tag[:2] + tag[2:]) / 2, (tag[2:] - tag[:2]) * (size_m / tag_m) / 2
            return np.concatenate([centre - half, centre + half])
    return tag


def tree_row(kept: np.ndarray, gt: np.ndarray, row: Mapping, gsd_m: float) -> dict:
    """The scores of one placed gold tree against a unit's kept boxes and labels (all mosaic pixels)."""
    from experiments.recall_check import _inside
    from fair_trees.evaluate import iou_matrix

    kept = np.asarray(kept, np.float64).reshape(-1, 4)
    gt = np.asarray(gt, np.float64).reshape(-1, 4)
    point = np.asarray(row["point_px"], np.float64)
    covering = _inside(point[None], kept)[0] if len(kept) else np.zeros(0, bool)
    out = {
        "gold_id": row["gold_id"],
        "split": row["split"],
        "stratum": row["stratum"],
        "status": row["status"],
        "point_hit": bool(covering.any()),
    }
    if row["box_px"] is None:
        return out
    box = np.asarray(row["box_px"], np.float64)
    gold_iou = iou_matrix(box[None], kept)[0] if len(kept) else np.zeros(0)
    out["gold_side_m"] = float((box[2] - box[0] + box[3] - box[1]) / 2 * gsd_m)
    out["gold_hit"] = bool((gold_iou > IOU).any())
    if covering.any():
        best = kept[covering][np.argmax(gold_iou[covering])]
        out["size_ratio"] = float(
            (best[2] - best[0] + best[3] - best[1]) / (box[2] - box[0] + box[3] - box[1])
        )
    dist = np.hypot(*((gt[:, :2] + gt[:, 2:]) / 2 - point).T) * gsd_m if len(gt) else np.zeros(0)
    if len(dist) and dist.min() <= TAG_MATCH_M:
        tag = gt[np.argmin(dist)]
        out["tag_hit"] = bool(len(kept) and (iou_matrix(tag[None], kept)[0] > IOU).any())
        out["tag_band"] = bool(iou_matrix(tag[None], box[None])[0, 0] > IOU)
        cal = _calibrated(tag, row["diameter_m"])
        out["calibrated_band"] = bool(iou_matrix(cal[None], box[None])[0, 0] > IOU)
    return out


def summary(rows: Sequence[dict]) -> dict:
    """Per stratum and over all: counts and the shares of ``tree_row`` flags, median size ratio."""

    def share(rs: Sequence[dict], key: str) -> float | None:
        vals = [r[key] for r in rs if key in r]
        return float(np.mean(vals)) if vals else None

    def one(rs: Sequence[dict]) -> dict:
        boxes = [r for r in rs if r["status"] == "box"]
        not_trees = [r for r in rs if r["status"] == "not_a_tree"]
        ratios = [r["size_ratio"] for r in boxes if "size_ratio" in r]
        return {
            "n_box": len(boxes),
            "n_with_tag": sum("tag_hit" in r for r in boxes),
            "gold_recall": share(boxes, "gold_hit"),
            "tag_recall": share(boxes, "tag_hit"),
            "point_recall": share(boxes, "point_hit"),
            "size_ratio_median": float(np.median(ratios)) if ratios else None,
            "tag_band": share(boxes, "tag_band"),
            "calibrated_band": share(boxes, "calibrated_band"),
            "n_not_a_tree": len(not_trees),
            "not_a_tree_detected": share(not_trees, "point_hit"),
        }

    groups = {g: one([r for r in rows if r["stratum"] == g]) for g in GROUPS}
    return {"all": one(rows)} | {g: s for g, s in groups.items() if s["n_box"] or s["n_not_a_tree"]}


def score_units(run_fn, units: Sequence, params, threshold: float, gold: Sequence[dict]) -> dict:
    """``tree_row`` for every gold tree placed on ``units``, decoded once and kept at ``threshold``."""
    from experiments.recall_check import decode_trees
    from fair_trees.decode import filter_score

    placed = place(gold, units)
    needed = sorted({i for i, _ in placed})
    dets = dict(zip(needed, decode_trees(run_fn, [units[i] for i in needed], params), strict=True))
    rows = [
        tree_row(filter_score(dets[i], threshold).boxes, units[i].gt_boxes, row, units[i].mosaic.gsd_m)
        for i, row in placed
    ]
    by_split: dict[str, int] = {}
    for _, row in placed:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
    return {
        "threshold": float(threshold),
        "n_gold": len(gold),
        "n_placed": len(placed),
        "placed_by_split": by_split,
        "summary": summary(rows),
        "trees": rows,
    }


def run_gold_score(
    run_id: str,
    manifest: Path,
    boxes: Path,
    *,
    part: str = "val",
    threshold: float | None = None,
    drive_root=None,
    data_root=None,
    benchmark=None,
) -> dict:
    """``score_units`` on a run's ``best.pt``: its val regions, or a frozen benchmark region part.

    ``threshold`` replaces the run's picked tree threshold (to compare runs at one threshold); the result
    then goes to ``gold_score_<part>_t<threshold>.json`` so the picked one is kept."""
    from experiments.recall_check import run_units

    out_dir, params, units, run_fn, picked = run_units(
        run_id, part=part, drive_root=drive_root, data_root=data_root, benchmark=benchmark
    )
    if not units:
        raise ValueError(
            f"{run_id} {part}: no units; stage the run's sources (and hot-test for the test part)"
        )
    used = picked if threshold is None else threshold
    result = {"run_id": run_id, "part": part, "picked_threshold": float(picked)} | score_units(
        run_fn, units, params, used, load_gold(manifest, boxes)
    )
    name = f"gold_score_{part}.json" if threshold is None else f"gold_score_{part}_t{threshold:g}.json"
    (out_dir / name).write_text(json.dumps(result, indent=2))
    return result


def report_lines(result: dict) -> list[str]:
    """A short printable view of ``score_units`` output."""

    def pct(v: float | None) -> str:
        return "  -  " if v is None else f"{v:5.0%}"

    out = [
        f"threshold {result['threshold']}, gold trees placed {result['n_placed']} of {result['n_gold']} "
        f"{result['placed_by_split']}",
        f"{'stratum':>15} {'boxes':>5} {'gold R':>6} {'tag R':>6} {'point R':>7} {'box/gold':>8} "
        f"{'tag band':>8} {'calib band':>10} {'not-tree det':>12}",
    ]
    for name, s in result["summary"].items():
        ratio = "   -" if s["size_ratio_median"] is None else f"{s['size_ratio_median']:.2f}"
        out.append(
            f"{name:>15} {s['n_box']:>5} {pct(s['gold_recall']):>6} {pct(s['tag_recall']):>6} "
            f"{pct(s['point_recall']):>7} {ratio:>8} {pct(s['tag_band']):>8} {pct(s['calibrated_band']):>10} "
            f"{pct(s['not_a_tree_detected']):>12}"
        )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("boxes", type=Path)
    parser.add_argument("--part", default="val")
    parser.add_argument("--threshold", type=float, default=None, help="instead of the run's picked one")
    args = parser.parse_args(argv)
    result = run_gold_score(args.run_id, args.manifest, args.boxes, part=args.part, threshold=args.threshold)
    print("\n".join(report_lines(result)))


if __name__ == "__main__":
    main()
