# fair_trees

DeepForest RetinaNet (`weecology/deepforest-tree`) tree-crown detection for HOT's fAIr platform:
preprocessing, decoding, serving (ONNX Runtime, no torch), training, evaluation and ONNX export.

## Install

```bash
uv sync                          # serving core only
uv sync --extra train --extra test   # everything, CPU torch wheels
```

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
environment, `.venv-df`, and is run from there (Windows):
`.venv-df/Scripts/python.exe -m pytest -m integration tests/integration/test_deepforest_parity.py`

See theory.md for the model facts and CLAUDE.md for the repository rules.
