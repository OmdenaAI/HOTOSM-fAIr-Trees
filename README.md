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

See theory.md for the model facts and CLAUDE.md for the repository rules.
