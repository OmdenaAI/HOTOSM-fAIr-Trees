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
