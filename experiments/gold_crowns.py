"""A blind, hand-measured set of HOT crowns: a stratified sample of tree points, one crop per tree.

Trees outside ``tree_group`` polygons are sampled per HOT split and per stratum of their mapper-tagged
``diameter_crown`` (``STRATA``, the 1.5-2.5 m band split between the copied 2.231 / 2.412 m values and the
rest), plus a pooled sample of trees with no usable size. Each tree gets a ``CROP_M`` square crop centred on
its point, stitched from the neighbouring tiles of any split (grey where no tile exists).

``write_gold`` writes two folders:

- ``annotate/``: ``crops/<gold_id>.png``, ``items.json`` (every tree, shuffled) and ``items_second.json``
  (a seeded subset for a second annotator, in its own order), and the annotator page. Nothing in it carries
  the tag, the stratum or the split, so the folder can be handed to annotators as it is.
- ``key/manifest.csv``: per ``gold_id`` the osm id, split, stratum, tag, lon/lat and the crop geometry.
"""

import csv
import io
import json
import math
import shutil
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from experiments.label_study.hot import TILE_PX, HotData
from fair_trees.dataset import MAX_DIAMETER_M
from fair_trees.georef import ORIGIN_SHIFT

CROP_M = 20.0  # ground side of each crop
PRESETS_M = (2.231, 2.412)  # diameter values repeated thousands of times: copied, not measured
PRESET_TOL_M = 5e-4
# (name, low, high, trees per HOT split); the presets are carved out of the 1.5-2.5 m band
STRATA = (
    ("<1.5", 0.0, 1.5, 40),
    ("1.5-2.5", 1.5, 2.5, 20),
    ("1.5-2.5 preset", 1.5, 2.5, 20),
    ("2.5-4", 2.5, 4.0, 40),
    ("4-6", 4.0, 6.0, 40),
    (">=6", 6.0, math.inf, 40),
)
UNSIZED = "unsized"
N_UNSIZED = 40  # pooled over the splits
N_SECOND = 100
MISSING_GREY = 128
ANNOTATOR_HTML = Path(__file__).with_name("gold_annotator.html")


def stratum(diameter_m: float) -> str:
    """The stratum of a tagged diameter; ``UNSIZED`` for NaN or one outside (0, MAX_DIAMETER_M]."""
    d = float(diameter_m)
    if not 0 < d <= MAX_DIAMETER_M:  # also NaN
        return UNSIZED
    if 1.5 <= d < 2.5 and any(abs(d - p) <= PRESET_TOL_M for p in PRESETS_M):
        return "1.5-2.5 preset"
    for name, low, high, _ in STRATA:
        if low <= d < high and not name.endswith("preset"):
            return name
    raise AssertionError(d)


def sample_gold(data: HotData, seed: int = 0) -> pd.DataFrame:
    """The seeded sample: osm_id, split, stratum, diameter_m, lon, lat, one row per tree.

    A stratum with fewer trees than asked gives all it has."""
    trees = data.trees
    if len(data.groups):
        inside = data.groups.sindex.query(trees.geometry, predicate="within")[0]
        trees = trees.drop(trees.index[np.unique(inside)])
    trees = trees.assign(
        stratum=[stratum(d) for d in trees["diameter_m"]], lon=trees.geometry.x, lat=trees.geometry.y
    ).sort_values("osm_id")
    rng = np.random.default_rng(seed)

    def pick(pool: pd.DataFrame, n: int) -> pd.DataFrame:
        return pool.iloc[np.sort(rng.choice(len(pool), size=min(n, len(pool)), replace=False))]

    parts = [
        pick(trees[(trees["split"] == split) & (trees["stratum"] == name)], n)
        for split in sorted(trees["split"].unique())
        for name, _, _, n in STRATA
    ]
    parts.append(pick(trees[trees["stratum"] == UNSIZED], N_UNSIZED))
    cols = ["osm_id", "split", "stratum", "diameter_m", "lon", "lat"]
    return pd.concat(parts)[cols].reset_index(drop=True)


@dataclass(frozen=True)
class CropWindow:
    x0: int  # global pixel column of the crop's left edge at zoom ``z``
    y0: int  # global pixel row of the crop's top edge
    size: int  # crop side in pixels
    point: tuple[float, float]  # the tree point in crop pixels (x, y)
    m_per_px: float  # ground metres per crop pixel at the point's latitude
    z: int


def crop_window(lon: float, lat: float, z: int, crop_m: float = CROP_M) -> CropWindow:
    """A square of ``crop_m`` on the ground, centred on (lon, lat), on the zoom-``z`` tile pixel grid."""
    world_px = TILE_PX * 2**z
    gx = (lon + 180.0) / 360.0 * world_px
    gy = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * world_px
    m_per_px = 2 * ORIGIN_SHIFT / world_px * math.cos(math.radians(lat))
    half = round(crop_m / 2 / m_per_px)
    x0, y0 = math.floor(gx) - half, math.floor(gy) - half
    return CropWindow(x0, y0, 2 * half, (gx - x0, gy - y0), m_per_px, z)


def window_tiles(w: CropWindow) -> list[tuple[int, int]]:
    """The (x, y) tiles the window overlaps."""
    xs = range(w.x0 // TILE_PX, (w.x0 + w.size - 1) // TILE_PX + 1)
    ys = range(w.y0 // TILE_PX, (w.y0 + w.size - 1) // TILE_PX + 1)
    return [(x, y) for y in ys for x in xs]


def assemble(w: CropWindow, tile_rgb: Callable[[int, int], np.ndarray | None]) -> tuple[np.ndarray, float]:
    """The window's RGB pixels from ``tile_rgb(x, y)`` (None: no tile), and the share left grey."""
    out = np.full((w.size, w.size, 3), MISSING_GREY, np.uint8)
    covered = np.zeros((w.size, w.size), bool)
    for tx, ty in window_tiles(w):
        rgb = tile_rgb(tx, ty)
        if rgb is None:
            continue
        # overlap of the tile and the window, in global pixels
        gx0, gy0 = max(w.x0, tx * TILE_PX), max(w.y0, ty * TILE_PX)
        gx1, gy1 = min(w.x0 + w.size, (tx + 1) * TILE_PX), min(w.y0 + w.size, (ty + 1) * TILE_PX)
        win = np.s_[gy0 - w.y0 : gy1 - w.y0, gx0 - w.x0 : gx1 - w.x0]
        out[win] = rgb[gy0 - ty * TILE_PX : gy1 - ty * TILE_PX, gx0 - tx * TILE_PX : gx1 - tx * TILE_PX]
        covered[win] = True
    return out, float(1 - covered.mean())


def _decode(image: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(image)).convert("RGB"))


def write_gold(
    sample: pd.DataFrame,
    data: HotData,
    images: Iterable[tuple[str, bytes]],
    out_dir: Path,
    *,
    n_second: int = N_SECOND,
    seed: int = 0,
) -> dict:
    """Crops, item lists and the key for ``sample`` (see the module docstring); ``images`` yields
    ``(tile_id, image_bytes)`` and only the tiles under a crop are kept. ``out_dir`` must not exist."""
    out_dir = Path(out_dir)
    annotate, key = out_dir / "annotate", out_dir / "key"
    (annotate / "crops").mkdir(parents=True)
    key.mkdir()
    z = int(data.tiles["z"].iloc[0])
    tile_ids: Mapping[tuple[int, int], str] = {
        (int(x), int(y)): t
        for t, x, y in zip(data.tiles["tile_id"], data.tiles["x"], data.tiles["y"], strict=True)
    }
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sample))
    rows = sample.iloc[order].reset_index(drop=True)
    rows["gold_id"] = [f"g{k + 1:04d}" for k in range(len(rows))]
    windows = [crop_window(lon, lat, z) for lon, lat in zip(rows["lon"], rows["lat"], strict=True)]

    wanted = {tile_ids[t] for w in windows for t in window_tiles(w) if t in tile_ids}
    found = {tile_id: image for tile_id, image in images if tile_id in wanted}
    by_xy = {xy: found[t] for xy, t in tile_ids.items() if t in found}
    cache: dict[tuple[int, int], np.ndarray] = {}

    def tile_rgb(x: int, y: int) -> np.ndarray | None:
        if (x, y) not in by_xy:
            return None
        if (x, y) not in cache:
            cache[(x, y)] = _decode(by_xy[(x, y)])
        return cache[(x, y)]

    items, key_rows = [], []
    for row, w in zip(rows.itertuples(), windows, strict=True):
        rgb, grey = assemble(w, tile_rgb)
        file = f"crops/{row.gold_id}.png"
        Image.fromarray(rgb).save(annotate / file)
        point = [round(v, 2) for v in w.point]
        items.append(
            {"id": row.gold_id, "file": file, "size_px": w.size, "m_per_px": w.m_per_px, "point": point}
        )
        key_rows.append(
            {
                "gold_id": row.gold_id,
                "osm_id": row.osm_id,
                "split": row.split,
                "stratum": row.stratum,
                "diameter_m": row.diameter_m,
                "lon": row.lon,
                "lat": row.lat,
                "z": w.z,
                "crop_x0": w.x0,
                "crop_y0": w.y0,
                "size_px": w.size,
                "m_per_px": w.m_per_px,
                "point_x": point[0],
                "point_y": point[1],
                "grey_share": round(grey, 4),
                "second": False,
            }
        )
    second = rng.choice(len(items), size=min(n_second, len(items)), replace=False)
    for k in second:
        key_rows[k]["second"] = True

    def item_list(name: str, chosen: list[dict]) -> None:
        body = {"set": name, "crop_m": CROP_M, "items": chosen}
        (annotate / f"{name}.json").write_text(json.dumps(body, indent=1))

    item_list("items", items)
    item_list("items_second", [items[k] for k in second])  # rng order, not the main order
    shutil.copyfile(ANNOTATOR_HTML, annotate / ANNOTATOR_HTML.name)
    with (key / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    counts = sample.groupby(["split", "stratum"]).size()
    return {
        "trees": len(items),
        "second": len(second),
        "tiles_missing": len(wanted - set(found)),
        "crops_with_grey": sum(r["grey_share"] > 0 for r in key_rows),
        "by_split_stratum": {f"{a}/{b}": int(n) for (a, b), n in counts.items()},
    }
