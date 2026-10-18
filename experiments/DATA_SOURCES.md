# Data sources

Local, git-ignored downloads used by the opt-in integration tests (`uv run pytest -m integration`).

| Item | URL | sha256 | Retrieved |
|---|---|---|---|
| `weights/NEON.pt` (DeepForest tree-crown RetinaNet) | https://huggingface.co/weecology/deepforest-tree/resolve/main/NEON.pt | 6d2194194daa694ce99c3daff8d53a95722d5856337466263e9a1bd23f8d282a | 2026-10-18 |
| `tests/data_local/OSBS_029.tif` (sample tile) | https://raw.githubusercontent.com/weecology/DeepForest/main/src/deepforest/data/OSBS_029.tif | 8521dbb5e8063d37d61895867dfcca7476491d1345103dc1b3fd96f71b186353 | 2026-10-18 |

```bash
mkdir -p weights tests/data_local
curl -L -o weights/NEON.pt https://huggingface.co/weecology/deepforest-tree/resolve/main/NEON.pt
curl -L -o tests/data_local/OSBS_029.tif https://raw.githubusercontent.com/weecology/DeepForest/main/src/deepforest/data/OSBS_029.tif
sha256sum weights/NEON.pt tests/data_local/OSBS_029.tif
```
