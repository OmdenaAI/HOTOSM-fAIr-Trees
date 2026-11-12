# fair_trees

Tree and tree-group detection in RGB aerial imagery for HOT's [fAIr](https://github.com/hotosm/fAIr) platform. The
model is a torchvision RetinaNet (ResNet-50 FPN) that starts from DeepForest's tree-crown model
(`weecology/deepforest-tree`). This library holds what a fAIr model pack calls: chip preprocessing, sliding-window
prediction with ONNX Runtime (no torch), fine-tuning, evaluation and ONNX export. Predictions are a GeoJSON
FeatureCollection of box polygons, class 1 tree and class 2 tree_group.

## Install

The library is installed from GitHub, not PyPI; pin a tag or a commit.

```bash
pip install "fair-trees @ git+https://github.com/OmdenaAI/HOTOSM-fAIr-Trees@main"         # serving only, no torch
pip install "fair-trees[train] @ git+https://github.com/OmdenaAI/HOTOSM-fAIr-Trees@main"  # adds training and export
```

The `train` extra pins `torch==2.11.0` and `torchvision==0.26.0`. In a clone:

```bash
uv sync                          # serving core only
uv sync --extra train --extra test   # everything, CPU torch wheels
```

## Weights

The trained two-class model is on Hugging Face:
[`OmarHelmy/retinanet-tree-groups`](https://huggingface.co/OmarHelmy/retinanet-tree-groups), tag `v0.1.0`, with
`best.pt`, `model.onnx`, `thresholds.json` and `windows.json`. `experiments/DATA_SOURCES.md` lists their sha256 values.

## Licence

- Code: MIT (see `LICENSE`).
- Weights: CC-BY-NC-4.0, non-commercial use only, because part of their training data (the `restor/tcd-nc` tiles of
  OAM-TCD) is CC-BY-NC-4.0. The HOT labels they were fine-tuned on are (c) OpenStreetMap contributors, ODbL-1.0.

## Tests

```bash
uv run pytest -q                 # offline, CPU, no network
uv run pytest -m integration     # opt-in parity; needs weights/NEON.pt and tests/data_local/OSBS_029.tif
```

## Fine-tuning (what the fAIr pack calls)

```python
from pathlib import Path
from fair_trees.dataset import spatial_split
from fair_trees.train import fine_tune

chips = sorted(Path("chips").glob("OAM-*.tif"))
train_chips, val_chips = spatial_split(chips, val_ratio=0.2, seed=42)
result = fine_tune(
    train_chips,
    val_chips,
    Path("labels.geojson"),
    {"epochs": 10},
    weights=Path("base.pt"),
    out_dir=Path("run"),
)
print(result.best_checkpoint, result.best_metrics)
```

Calling `fine_tune` again on the same `out_dir` resumes an interrupted run, or returns the finished result.

## Integration tests

The opt-in parity tests need two local downloads (`weights/NEON.pt` and `tests/data_local/OSBS_029.tif`).
`experiments/DATA_SOURCES.md` lists the URLs, sha256 values and the download commands.

The DeepForest parity test needs `deepforest==2.1.0` (see `experiments/requirements.txt`) in its own
environment, `.venv-df`, and is run from there:
`.venv-df/bin/python -m pytest -m integration tests/integration/test_deepforest_parity.py`
(`.venv-df/Scripts/python.exe` on Windows).

See theory.md for the model facts and CLAUDE.md for the repository rules.
