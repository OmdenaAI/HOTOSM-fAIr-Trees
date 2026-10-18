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

## Integration tests

The opt-in parity tests need two local downloads (`weights/NEON.pt` and `tests/data_local/OSBS_029.tif`).
`experiments/DATA_SOURCES.md` lists the URLs, sha256 values and the download commands.

The DeepForest parity test needs `deepforest==2.1.0` (see `experiments/requirements.txt`) in its own
environment, `.venv-df`, and is run from there (Windows):
`.venv-df/Scripts/python.exe -m pytest -m integration tests/integration/test_deepforest_parity.py`

See theory.md for the model facts and CLAUDE.md for the repository rules.
