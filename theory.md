# Theory: fair_trees (DeepForest RetinaNet for fAIr)

Tags: [V] verified (source code, checkpoint, dataset card, paper, or arithmetic here), [P] proposed, [?] unknown.

## 1. Model
- torchvision RetinaNet, ResNet-50 FPN (v1), 1 class, from `weecology/deepforest-tree` `NEON.pt` (MIT) [V].
- `NEON.pt` is a 301-key state dict in the FrozenBatchNorm2d layout: body 53 convs + 53 FrozenBN x 4 = 265,
  FPN 16, heads 20. A trainable-BatchNorm2d build has 354 keys [V: arithmetic].
- `retinanet_resnet50_fpn(weights=None)` ignores `trainable_backbone_layers` and builds BatchNorm2d, so
  `build_model` assembles the network by hand with FrozenBatchNorm2d and loads with `strict=True` [V].

## 2. Preprocessing
- Order: normalise, then resize (short side 800, long side <= 1333, PIL bilinear, size = floor(h * scale)),
  then zero-pad to a multiple of 32. Padding after normalising gives 0 = the mean colour, as torchvision does [V].
- One function (`preprocess_window`) serves training and inference, so both resize identically [P].

## 3. Anchors and decoding
- Sizes (32, 40, 50), (64, 80, 101), (128, 161, 203), (256, 322, 406), (512, 645, 812): int-truncated [V].
- 120,087 anchors at 800 x 800 (13,343 locations x 9); P6 stride at 800 is 61 px [V].
- Decode mirrors `postprocess_detections`: sigmoid, score > threshold, top 1,000 per level,
  dw/dh clamped at log(1000/16), clip to the resized image, NMS, 300 cap [V].
- Confidence-sweep equivalence: decoding at a low threshold then filtering by score equals decoding at the
  higher threshold, because every stage is greedy by score or score-independent [V: reasoning, tested].

## 4. Windows
- A window is downscaled only when wider than 800 x GSD: z20 ~103-119 m, z21 ~52-60 m, z22 ~26-30 m [V].
- The largest crown kept whole in one window is window - stride (20 m at 0.5 overlap) [V: arithmetic].

## 5. Evaluation
- Hungarian matching, IoU > 0.4; AP@0.4 over 101 confidence thresholds in [0.01, 1] [P].
- Forgetting tolerance = release model's 95% CI on the NEON eval plots (recorded after the NEON reproduction run) [P].

## 6. Weights and data provenance
- `NEON.pt`: https://huggingface.co/weecology/deepforest-tree/resolve/main/NEON.pt, sha256 `6d2194194daa694ce99c3daff8d53a95722d5856337466263e9a1bd23f8d282a`, retrieved 2026-10-18 [V].
- `OSBS_029.tif`: https://raw.githubusercontent.com/weecology/DeepForest/main/src/deepforest/data/OSBS_029.tif, sha256 `8521dbb5e8063d37d61895867dfcca7476491d1345103dc1b3fd96f71b186353`, retrieved 2026-10-18 [V].
- The strict-load check (301 keys, frozen-BN layout) and the torchvision parity checks (anchors, preprocessing, decode plus NMS against `postprocess_detections`) in `tests/integration/test_parity.py` pass with torch 2.11.0 / torchvision 0.26.0 on 2026-10-18 [V].
- `FrozenBatchNorm2d` silently drops `num_batches_tracked` on load, so `load_state_dict(strict=True)` accepts a 354-key trainable-BN checkpoint; `build_model` rejects checkpoints containing `num_batches_tracked` explicitly [V]. `NEON.pt` stores the FPN blocks under the older key names (`inner_blocks.N.weight`), which torchvision remaps on load [V].
- Real-sidecar check: a real fAIr chip's `.aux.xml` bounds against the bounds derived from its OAM file name is pending a real fAIr chip [?].
- NeonTreeEvaluation v0.2.2 archives on Drive, retrieved 2026-10-19: `annotations.zip` sha256
  `2b765abc14c3492479ba929933a86aa2b5c0a8ce6142dc5d458be606eb3c20ff`, `evaluation.zip`
  `97e06a47cd3c1d80859aa9a3e9424edac405775dbd42d3b3ce0af07d6cf60406`, `training.zip`
  `0cdba2c3794f74a9e7bb4d3eaa9247d010944d0cb76ca4f0b2a0cad0abeffa8b` [V]. The frozen NEON benchmark part names
  `evaluation.zip` [V: experiments/benchmark.json].
- NEON shard manifest sha256 `5bbeca8e879491ed5835c137dca01dcc52870d0d6623d2152217342772d9b428`, converter
  90875d94c6af571e8aa5e3834b1a3f0e51a9e1ba [V].
- NEON training rasters by site, all 0.10 m; z20 needs a latitude below ~47.9 degrees [V: Colab, 2026-10-19]:

  | Site | Rasters | Latitude | Zooms |
  |---|---|---|---|
  | BART | 1 | 44.07 | z19, z20 |
  | DELA | 1 | 32.55 | z19, z20 |
  | HARV | 1 | 42.41 | z19, z20 |
  | JERC | 1 | 31.17 | z19, z20 |
  | LENO | 1 | 31.84 | z19, z20 |
  | MLBS | 2 | 37.41 | z19, z20 |
  | NIWO | 1 | 39.99 | z19, z20 |
  | ONAQ | 1 | 40.19 | z19, z20 |
  | OSBS | 3 | 29.71 | z19, z20 |
  | SJER | 3 | 37.04-37.11 | z19, z20 |
  | TEAK | 1 | 36.98 | z19, z20 |
  | TOOL | 1 | 68.65 | z19 |
  | DSNY | 1 | no CRS | skipped |
  | YELL | 2 | no CRS | skipped |

## 7. Environment
- Colab GPU runtime, 2026-10-18: Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0+cu130 [V].
  Raw output of `import sys, torch, torchvision; print(sys.version); print(torch.__version__, torchvision.__version__)`:

  ```text
  3.13.15 (main, Aug  6 2026, 11:06:22) [GCC 13.3.0]
  2.11.0+cu130 0.26.0+cu130
  ```
- Pinned in `pyproject.toml` as `torch==2.11.0` and `torchvision==0.26.0` (CPU wheels locally and in CI) [V].

## 8. Training
- One `train()` serves both regimes: base training (trainable_layers 3, mixed sources) and the platform
  fine-tune (trainable_layers 0, the mapper's chips) [P].
- The forward pass calls backbone, head, anchor generator and `compute_loss` directly on batches from
  `preprocess_window`; `GeneralizedRCNNTransform` never runs. The opt-in loss-parity test checks this path
  against `model(images, targets)` on NEON.pt [V: tests/integration/test_train_parity.py].
- Loss = classification + bbox_regression + alpha * sum ||w - w0||^2 over trainable parameters, with w0
  snapshotted after loading the start weights. The constant is alpha, not alpha/2 as in Li et al. (2018).
  When alpha > 0, weight decay is forced to 0 [P].
- Boxes narrower or shorter than 1 px after resizing are dropped, because log(w) in the box coder would be
  infinite [V: torchvision BoxCoder].
- AMP: fp16 with GradScaler on T4/L4, bf16 on A100/H100, none on CPU [P].
- Early stopping on val f1_max (biased: its threshold is chosen on val) with AP@0.4 breaking ties within
  min_delta; f1_max is never reported as a result [P].
- Checkpoints: saved locally, copied to `<name>.part`, size-checked, renamed; the oldest beyond keep_last is
  deleted only after that. A resume regenerates the epoch plan from (seed, epoch) and continues at the saved
  step; no RNG state is stored. The history carries a sha256 of the plan and of the sample indices actually
  trained, so an interrupted run can be checked against an uninterrupted one [P].
- NEON acceptance, decided 2026-10-20: the forgetting check passes if a fine-tuned model's F1 on the 194 NEON eval
  plots, DeepForest-compatible mode, is >= 0.638, the release model's lower 95% bound in
  `experiments/benchmark.json`. NEON val F1 is not compared with that bound (other images, zooms and inference
  settings); it is reported as a paired difference against the release model [P].

## 9. Sampling and crops
- Source, then biome, then region, then zoom; source and biome probabilities follow n^alpha of valid area
  (alpha 0 equal, 1 proportional) [P].
- Crops of 30-80 m at native GSD, clipped to small regions; a draw is rejected if it touches an absent tile
  or is more than 80% masked on a low-resolution mask, with a whole-tile fallback after 20 tries [P].
- A box is kept if at least 40% of its area is inside the crop and unmasked (pixel-centre count) [P].
- Masked and nodata pixels are set to 0 after colour jitter, so they match partial-nodata windows at serving [P].
- Downscale range (0.5, 1.0) with p 0.5, and hue jitter in degrees, are placeholders pending DeepForest's
  Downscale config and the unit (degrees or a 0-255 hue channel) of the planned +/-10 hue jitter [?].

## 10. Data conversion
- Chips are cut from each scene per zoom only when the zoom's true GSD at the scene centre is at least the
  source GSD, so nothing is upsampled. 10 cm sources give no z21 anywhere and no z20 above ~47.9 degrees
  (0.149 * cos(lat) < 0.10) [V: arithmetic].
- Chips are 256 px JPEG GeoTIFFs in EPSG:3857 aligned to the XYZ grid, named OAM-x-y-z.tif. JPEG cannot
  hold exact zeros, so nodata travels in an internal GDAL mask that read_chip honours [P].
- Chips more than 80% nodata are dropped [P].
- A source with no nodata value, mask or alpha treats pure-black (0, 0, 0) pixels as nodata [P]: NEON's
  `2019_SJER_4_251000_4103000_image` has a black collar stored as pixels, which made 491 of its 1,403
  chips mostly black yet valid [V: Colab check, 2026-10-19].
- NEON: group split with a 1 km buffer. Training rasters closer than 1 km form one group, and val is the
  set of whole groups whose z19 ground area is closest to 15%, so train and val lie at least 1 km apart
  and the buffer drops nothing [P]. Every chip within 1 km of an annotated NeonTreeEvaluation eval plot is
  excluded before splitting [P]. A block split cannot work here: the rasters (0.1-0.6 km) are smaller
  than the buffer, and block sizes 2-32 left 29-100% of z20 chips in val while dropping 927-3,112
  train chips [V: Colab sweep, 2026-10-19].
