# Data sources

Local, git-ignored downloads used by the opt-in integration tests (`uv run pytest -m integration`).

| Item | URL | sha256 | Retrieved |
|---|---|---|---|
| `weights/NEON.pt` (DeepForest tree-crown RetinaNet) | https://huggingface.co/weecology/deepforest-tree/resolve/main/NEON.pt | 6d2194194daa694ce99c3daff8d53a95722d5856337466263e9a1bd23f8d282a | 2026-10-18 |
| `tests/data_local/OSBS_029.tif` (sample tile) | https://raw.githubusercontent.com/weecology/DeepForest/main/src/deepforest/data/OSBS_029.tif | 8521dbb5e8063d37d61895867dfcca7476491d1345103dc1b3fd96f71b186353 | 2026-10-18 |
| `data_local/NeonTreeEvaluation/annotations.zip` (NeonTreeEvaluation v0.2.2, Zenodo 10.5281/zenodo.5914554) | https://zenodo.org/api/records/5914554/files/annotations.zip/content | 2b765abc14c3492479ba929933a86aa2b5c0a8ce6142dc5d458be606eb3c20ff (md5 f577f0948a259d474ee0199fb8d76524 matches Zenodo) | 2026-10-18 |
| `data_local/NeonTreeEvaluation/evaluation/RGB/*.tif` (194 annotated plots with an RGB image; 226 annotation XMLs, 32 have no image in this release) | https://zenodo.org/api/records/5914554/files/evaluation.zip/content (3,853,532,101 bytes, md5 f0c444fc59cf0115ce8c981a576fab77 per Zenodo) | extracted by HTTP range requests, whole-zip hash not computed; sha256 of the sorted `sha256sum` listing of annotations/*.xml + evaluation/RGB/*.tif: f32048b7ccf4e77da242fb1b052378ace70f91f65fb655c6a663c6cb80c67a30 | 2026-10-18 |

```bash
mkdir -p weights tests/data_local
curl -L -o weights/NEON.pt https://huggingface.co/weecology/deepforest-tree/resolve/main/NEON.pt
curl -L -o tests/data_local/OSBS_029.tif https://raw.githubusercontent.com/weecology/DeepForest/main/src/deepforest/data/OSBS_029.tif
sha256sum weights/NEON.pt tests/data_local/OSBS_029.tif
```

NeonTreeEvaluation layout matches the loader defaults (`evaluation/RGB/<PLOT>.tif`, `annotations/<PLOT>.xml`, Pascal VOC pixel boxes); no path constants changed. Only annotated plots' RGB tifs are extracted; the zip also holds LiDAR, hyperspectral and CHM data that this project does not use.

The 32 annotation XMLs without an image are absent from the Zenodo `evaluation.zip` itself (central directory read over HTTP range requests: 2,439 RGB tifs, none with those stems), not merely unextracted; the evaluation therefore runs on 194 plots.

## NEON training data on Drive (Colab conversion)

Drive layout under `MyDrive/fair_trees/`, written by `experiments/notebooks/colab_convert.ipynb`.

| Item | URL | sha256 | Retrieved |
|---|---|---|---|
| `raw/neon/annotations.zip` (NeonTreeEvaluation v0.2.2, Zenodo 10.5281/zenodo.5914554, CC-BY-4.0) | https://zenodo.org/api/records/5914554/files/annotations.zip/content | 2b765abc14c3492479ba929933a86aa2b5c0a8ce6142dc5d458be606eb3c20ff | 2026-10-19 |
| `raw/neon/evaluation.zip` | https://zenodo.org/api/records/5914554/files/evaluation.zip/content | 97e06a47cd3c1d80859aa9a3e9424edac405775dbd42d3b3ce0af07d6cf60406 | 2026-10-19 |
| `raw/neon/training.zip` (20 RGB rasters, with CHM, LiDAR and hyperspectral; folders at the zip root) | https://zenodo.org/api/records/5914554/files/training.zip/content | 0cdba2c3794f74a9e7bb4d3eaa9247d010944d0cb76ca4f0b2a0cad0abeffa8b | 2026-10-19 |
| `raw/resolve/Ecoregions2017.zip` (RESOLVE Ecoregions 2017, 149,248,653 bytes, Last-Modified 2019-05-07) | https://storage.googleapis.com/teow2016/Ecoregions2017.zip | be36d6209e443038d02e309f0447c6e7f2a62f5fe60c605ffe90d064952f2a60 (same URL hashed locally on the same day; the Drive copy was not hashed) | 2026-10-19 |
| `weights/NEON.pt` | as in the table above | not hashed on Drive; the release-baseline run checks it against the reproduction F1 | 2026-10-19 |

### NEON checks

1. **Eval plots have a CRS: yes.** All 194 annotated plots do (`eval_plot_footprints` returns 194). The archive also holds
   unannotated plot images, some without a CRS (e.g. `JERC_047_2019.tif`); they are never scored, so the 1 km exclusion
   ignores them.
2. **Training rasters have a CRS: 17 of 20.** `2019_DSNY_5_452000_3113000_image_crop`,
   `2019_YELL_2_528000_4978000_image_crop2` and `2019_YELL_2_541000_4977000_image_crop` have none and are skipped: the name
   gives the 1 km tile corner, not the crop's offset. All 17 are 0.10 m in UTM. The three hold 1,761 boxes (DSNY 888,
   YELL 574 and 299; median box side 38-47 px, in line with 0.10 m). If each crop lies inside its named 1 km tile
   (unchecked; testable on the 17 georeferenced rasters), it can be placed to within 1 km, which is enough for the
   1 km exclusion (the nearest DSNY eval plot is 8.3 km from the DSNY tile; YELL has no eval plots) and for the
   split. Placing them from the name is left for the base-training converters: the NEON fine-tune tests the
   training stack, and 8% more boxes would not change any of its checks.
3. **Fully annotated: partly checked.** On the contact sheet the boxes sit on the crowns in every chip with trees. The
   two-rasters-per-site overlay was not done, so coverage of whole rasters is unverified.
4. **Site latitudes and zooms:** see `theory.md` section 6. Only TOOL (68.65 N) is above the z20 limit and has z19 only.
5. **Licence:** CC-BY-4.0 on the Zenodo record.

Black collar: `2019_SJER_4_251000_4103000_image` stores a black collar as (0, 0, 0) pixels with no nodata
value; before the tiler treated pure black as nodata in such sources, 491 of its 1,403 chips were mostly black yet valid.

### Split

The rasters (0.1-0.6 km) are smaller than the 1 km train/val buffer. A per-site block split could not reach a 10-20% val
share (simulated on the tiled chips, z20 counts):

| Block size (z19 tiles) | Train | Val | Dropped by buffer | Val share |
|---|---|---|---|---|
| 2 | 2 | 628 | 3,112 | 1.00 |
| 4 | 27 | 730 | 2,985 | 0.96 |
| 8 | 75 | 729 | 2,938 | 0.91 |
| 16 | 1,726 | 782 | 1,234 | 0.31 |
| 32 | 2,004 | 811 | 927 | 0.29 |

The converter uses a group split instead: rasters closer than 1 km form one group, and val is the set of whole groups whose
ground area is closest to 15%.

### Conversion result (converter 90875d94c6af571e8aa5e3834b1a3f0e51a9e1ba)

| Zoom | Train chips | Val chips |
|---|---|---|
| z19 | 1,932 | 171 |
| z20 | 2,463 | 586 |

- `val_share` (valid ground area) 0.125, `dropped_by_buffer` 0, `excluded_near_eval_plots` 2,408, `masked_pct` 5.6, 21,087
  boxes.
- Val scenes: BART, HARV, JERC, MLBS (two rasters), NIWO, DELA, LENO and ONAQ. Train: the OSBS, SJER, TEAK and TOOL scenes.
- Shard manifest sha256 `5bbeca8e879491ed5835c137dca01dcc52870d0d6623d2152217342772d9b428`.
- Drive-to-`/content` copy of `z20-0000.tar`: 97.6 MB/s (an earlier conversion measured 307.5 MB/s); restaging all
  shards takes well under 10 minutes.

## HOT Freetown trees on Drive (Colab conversion)

Written by `experiments/notebooks/colab_convert_hot.ipynb` to `MyDrive/fair_trees/shards/hot/`.

| Item | URL | Revision | Licence |
|---|---|---|---|
| `hotosm/fAIr-guru-td-trees` (labels, tile table and image parquets, read through `huggingface_hub`) | https://huggingface.co/datasets/hotosm/fAIr-guru-td-trees | `v0.1.0` | imagery CC-BY-4.0, labels ODbL-1.0 (dataset card) |

Files used: `stac/project-50661/labels.parquet`, `stac/project-50661/tiles.parquet`, `data/train-00000-of-00002.parquet`,
`data/train-00001-of-00002.parquet` and `data/validation-00000-of-00001.parquet`. The biome comes from the RESOLVE
shapefile already on Drive (`raw/resolve/Ecoregions2017.zip`).

### What is converted

- **Splits.** Only HOT `train` (as `train`) and `validation` (as `val`). HOT `test` is never converted; the label study
  reads it straight from Hugging Face.
- **Regions.** One region per connected run of tiles in a 16 x 16-tile block, since a whole split is larger than the
  mosaic limit. Each region keeps its chips (PNG or JPEG as stored) and its own `labels.geojson` and `ignore.geojson`;
  every feature that touches the region's tiles is written, so a crown cut by a block edge is labelled on both sides.
- **Labels.** A tree outside every `tree_group` with a known diameter (0 to 30 m) becomes a square of that side
  centred on the point (built in UTM 28N, stored in EPSG:4326).
- **Ignore.** Every `tree_group` polygon, plus a 4 m square around each tree with an unknown diameter that is not in a
  group. Trees inside a group get nothing, as the group already masks them. A crown from the other split that reaches
  into a region is masked there, never labelled.
- **No split buffer.** HOT's split leaves no gap between train and val, so crowns and imagery can sit next to each other
  across the boundary; the other-split masking above is the only guard.

### Conversion run (2026-10-28, converter dd883b872b4bc333637a4e63c49eae2698f55d88)

| Split | Regions | Chips | Label-free tiles |
| --- | --- | --- | --- |
| train | 224 | 38,705 | 36.9% |
| val | 30 | 4,158 | 20.6% |

- 81,458 crown boxes; 3,940 unknown-diameter trees and 9,873 groups masked; 62 crowns from the other split masked; 254
  trees inside groups left out. `val_share` (valid ground area) 0.097.
- All 254 regions fall in one biome (Tropical & Subtropical Moist Broadleaf Forests), so biome balancing has no effect.
- Shard manifest sha256 `24d84f0f98aac87f212992290ab31dd36d75f5c1cd047a9b0b55db9b23e853e2`.
- Drive-to-`/content` copy of `z20-0000.tar`: 117.2 MB/s.
- The earlier conversion (2026-10-26, manifest `aff1f4b5034fbdfe4e8cd148a89dc94c8ea2bda3efda035f70a1d49c47a3ba6d`, kept
  on Drive as `shards/hot-aff1f4b5/`) looked biomes up at region centroids. Coastal centroids fall off the RESOLVE
  polygons, so about 19.5% of the committed `hot-platform` run's draws went to `unknown`-biome regions, oversampled
  about 3.5x. Every other count above matches it, and region ids come from the tile blocks, so val scores compare
  across the two.

### Teacher-masked HOT sets (2026-10-28, code 4d069e25acd19575c35ebe0373fe02ff3cd97766)

Written by `experiments/notebooks/colab_teacher_hot.ipynb` from the `hot` shards above to
`MyDrive/fair_trees/shards/hot-teacher-all/` and `shards/hot-teacher-empty/`. The release weights ran over the 224 train
regions at a 60 m window; detections scoring at least 0.3 that overlap no HOT label and are not centred in an existing
mask became `reason: "teacher"` ignore squares. Val regions are unchanged.

| | Count |
| --- | --- |
| Teacher detections | 128,541 |
| On a HOT label (dropped) | 43,465 |
| Centred in an existing mask, e.g. a tree group (dropped) | 35,450 |
| Kept, `hot-teacher-all` (everywhere) | 49,626, 1.30% of the valid train area |
| Kept, `hot-teacher-empty` (centre on a tile no label touches) | 21,066, 0.56% of the valid train area |
| Train tiles over the sampler's 80% mask limit, before / after either set | 1,155 / 1,168 of 38,705 |

- Manifest sha256: `hot-teacher-all` `09f5d813bae7c81943605c4a3d8701fbf39f92877e1759e3e56e3008b48dd339`,
  `hot-teacher-empty` `8376daa9727746b22a33d885184363103ca10825fcaadf4eb8c55465ebd7096c`.
- Contact sheet (`shards/hot-teacher-all/qa-teacher-contact-sheet.png`): in canopy the squares sit on crowns; in built-up
  tiles some sit on roofs, mostly blue ones and tarps, so the sets also stop training from correcting those false
  positives.
