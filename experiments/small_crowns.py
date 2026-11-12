"""Small truth crowns drawn on the imagery, to check by eye whether the missed ones are trees.

    python -m experiments.small_crowns RUN_ID [--part val] [--n 30]

The run's tree class is decoded once per unit at a low threshold. Every truth crown under the largest of
``BINS_M`` gets a row: its size, whether it matched at the picked threshold, its best IoU with a kept box
and the top score of any decoded box covering its centre. A seeded sample of misses, split evenly over the
size bins, and a smaller sample of hits for contrast are drawn on crops of the mosaic. Writes
``runs/<RUN_ID>/small_crowns_<part>_misses.png``, ``..._hits.png`` and ``small_crowns_<part>.json``
(pooled shares per bin, and each drawn crown with its tile number and lon/lat).
"""

import argparse
import json
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from experiments.recall_check import _inside, decode_trees, run_units

BINS_M = (1.5, 2.5)  # truth box side: under 1.5 m, 1.5-2.5 m; larger crowns get no row
CROP_M = 12.0  # ground side of each drawn crop
TILE_PX = 256
FAINT_SCORE = 0.05  # a box at or above this, below the picked threshold, is drawn as a faint detection
COLOURS = {"crown": (255, 59, 48), "truth": (255, 214, 10), "kept": (0, 220, 255), "faint": (255, 0, 200)}
LEGEND = "red: this label  yellow: other labels  cyan: kept boxes  magenta: boxes below the threshold"


def crown_rows(
    dets, gt: np.ndarray, gsd_m: float, threshold: float, iou_threshold: float = 0.4
) -> list[dict]:
    """One row per truth box under ``BINS_M[-1]``: index, side_m, bin, matched, best_iou, centre_score."""
    from fair_trees.decode import filter_score
    from fair_trees.evaluate import iou_matrix, match_hungarian

    gt = np.asarray(gt, np.float32).reshape(-1, 4)
    kept = filter_score(dets, threshold).boxes
    matched = np.zeros(len(gt), bool)
    for _, g in match_hungarian(kept, gt, iou_threshold):
        matched[g] = True
    best_iou = iou_matrix(gt, kept).max(axis=1) if len(kept) else np.zeros(len(gt))
    inside = _inside((gt[:, :2] + gt[:, 2:]) / 2, dets.boxes)
    centre_score = (
        np.where(inside, dets.scores[None, :], 0.0).max(axis=1) if len(dets.boxes) else np.zeros(len(gt))
    )
    side_m = ((gt[:, 2] - gt[:, 0]) + (gt[:, 3] - gt[:, 1])) / 2 * gsd_m
    return [
        {
            "index": int(i),
            "side_m": float(side_m[i]),
            "bin": int(np.digitize(side_m[i], BINS_M)),
            "matched": bool(matched[i]),
            "best_iou": float(best_iou[i]),
            "centre_score": float(centre_score[i]),
        }
        for i in np.flatnonzero(side_m < BINS_M[-1])
    ]


def sample_rows(rows: Sequence[dict], n: int, rng: np.random.Generator) -> list[dict]:
    """Up to ``n`` rows without replacement, split evenly over the bins, smallest first."""
    out = []
    for b in range(len(BINS_M)):
        pool = [r for r in rows if r["bin"] == b]
        k = min(len(pool), n // len(BINS_M) + (b < n % len(BINS_M)))
        out += [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
    return sorted(out, key=lambda r: r["side_m"])


def summary(rows: Sequence[dict], threshold: float) -> list[dict]:
    """Per bin: truth count, recall, and of the misses the share located by a kept box (centre inside one),
    the share seen only faintly and the median best IoU."""
    out = []
    for b in range(len(BINS_M)):
        in_bin = [r for r in rows if r["bin"] == b]
        misses = [r for r in in_bin if not r["matched"]]
        scores = np.array([r["centre_score"] for r in misses])
        out.append(
            {
                "bin_m": [0.0, *BINS_M][b : b + 2],
                "n_gt": len(in_bin),
                "recall": 1 - len(misses) / len(in_bin) if in_bin else None,
                "misses_located": float((scores > threshold).mean()) if misses else None,
                "misses_faint": float(((scores >= FAINT_SCORE) & (scores <= threshold)).mean())
                if misses
                else None,
                "misses_median_best_iou": float(np.median([r["best_iou"] for r in misses]))
                if misses
                else None,
            }
        )
    return out


def _near(boxes: np.ndarray, window: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = window
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return boxes[(boxes[:, 2] > x0) & (boxes[:, 0] < x1) & (boxes[:, 3] > y0) & (boxes[:, 1] < y1)]


def draw_tile(unit, dets, threshold: float, row: dict, caption: str) -> Image.Image:
    """A ``CROP_M`` crop centred on ``row``'s truth box, scaled to ``TILE_PX``, with boxes and a caption."""
    crown = unit.gt_boxes[row["index"]]
    cx, cy = (crown[0] + crown[2]) / 2, (crown[1] + crown[3]) / 2
    half = CROP_M / 2 / unit.mosaic.gsd_m
    window = (cx - half, cy - half, cx + half, cy + half)
    tile = Image.fromarray(unit.mosaic.rgb).crop(tuple(round(v) for v in window))
    tile = tile.resize((TILE_PX, TILE_PX), Image.BILINEAR)
    scale = TILE_PX / (2 * half)
    draw = ImageDraw.Draw(tile)

    def boxes(b: np.ndarray, colour: str, width: int) -> None:
        for x0, y0, x1, y1 in _near(b, window):
            xy = [
                (x0 - window[0]) * scale,
                (y0 - window[1]) * scale,
                (x1 - window[0]) * scale,
                (y1 - window[1]) * scale,
            ]
            draw.rectangle(xy, outline=COLOURS[colour], width=width)

    faint = (dets.scores >= FAINT_SCORE) & (dets.scores <= threshold)
    boxes(dets.boxes[faint], "faint", 1)
    boxes(dets.boxes[dets.scores > threshold], "kept", 2)
    boxes(np.delete(unit.gt_boxes, row["index"], axis=0), "truth", 1)
    boxes(crown[None], "crown", 3)
    draw.rectangle([0, TILE_PX - 18, TILE_PX, TILE_PX], fill=(0, 0, 0))
    draw.text((4, TILE_PX - 16), caption, fill=(255, 255, 255))
    return tile


def contact_sheet(tiles: Sequence[Image.Image], title: str, cols: int = 6, gap: int = 4) -> Image.Image:
    """Tiles in a grid under a title and the colour legend."""
    rows = max(1, -(-len(tiles) // cols))
    head = 40
    sheet = Image.new("RGB", (cols * (TILE_PX + gap) + gap, head + rows * (TILE_PX + gap) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 4), title, fill=(0, 0, 0))
    draw.text((gap, 22), LEGEND, fill=(80, 80, 80))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, (gap + (k % cols) * (TILE_PX + gap), head + gap + (k // cols) * (TILE_PX + gap)))
    return sheet


def _lonlat(unit, row: dict) -> list[float]:
    from fair_trees.georef import mosaic_px_to_4326_rings

    ring = np.asarray(mosaic_px_to_4326_rings(unit.gt_boxes[row["index"]][None], unit.mosaic)[0])
    return [round(float(v), 7) for v in (ring[:4, 0].mean(), ring[:4, 1].mean())]


def run_small_crowns(
    run_id: str,
    *,
    part: str = "val",
    n: int = 30,
    n_hits: int = 12,
    seed: int = 0,
    drive_root=None,
    data_root=None,
    benchmark=None,
) -> dict:
    """Rows, samples and contact sheets for a run's ``best.pt`` on ``part`` (see the module docstring)."""
    out_dir, params, units, run_fn, threshold = run_units(
        run_id, part=part, drive_root=drive_root, data_root=data_root, benchmark=benchmark
    )
    dets = decode_trees(run_fn, units, params)
    rows = [
        {"unit": k, "unit_id": u.unit_id, **r}
        for k, (u, d) in enumerate(zip(units, dets, strict=True))
        for r in crown_rows(d, u.gt_boxes, u.mosaic.gsd_m, threshold)
    ]
    rng = np.random.default_rng(seed)
    result = {"run_id": run_id, "part": part, "threshold": float(threshold), "crop_m": CROP_M}
    result["by_size"] = summary(rows, threshold)
    for name, matched, k in (("misses", False, n), ("hits", True, n_hits)):
        drawn, tiles = [], []
        for tile_no, r in enumerate(sample_rows([r for r in rows if r["matched"] == matched], k, rng), 1):
            u, d = units[r["unit"]], dets[r["unit"]]
            caption = f"{tile_no}  {r['side_m']:.1f} m  IoU {r['best_iou']:.2f}  s {r['centre_score']:.2f}"
            tiles.append(draw_tile(u, d, threshold, r, caption))
            drawn.append({"tile": tile_no, **r, "lonlat": _lonlat(u, r)})
        title = f"{run_id} {part}: {name} under {BINS_M[-1]} m at threshold {threshold:g}"
        contact_sheet(tiles, title).save(out_dir / f"small_crowns_{part}_{name}.png")
        result[name] = drawn
    (out_dir / f"small_crowns_{part}.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--part", default="val")
    parser.add_argument("--n", type=int, default=30)
    args = parser.parse_args(argv)
    result = run_small_crowns(args.run_id, part=args.part, n=args.n)
    print(json.dumps(result["by_size"], indent=2))


if __name__ == "__main__":
    main()
