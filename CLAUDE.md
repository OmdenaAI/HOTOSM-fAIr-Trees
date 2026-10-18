# CLAUDE.md

Rules for agents working in this repo.

- The serving path (`fair_trees.{params,georef,preprocess,decode,postprocess,serve}`) imports numpy,
  Pillow, rasterio, pyproj and the stdlib only. `tests/test_imports.py` enforces it.
- Model space never uses CRS: `decode.py` works in pixels; `georef.py` knows nothing about tensors.
- Preprocessing order is normalise -> resize (PIL bilinear, short side 800, max 1333) -> pad to 32 with 0.
  `preprocess_window` is the only implementation; training and serving both call it.
- Matching is Hungarian with IoU > 0.4 (strict).
- torch/torchvision are pinned with `==`; rerun `pytest -m integration` on any bump.
- Anything a mapper's fine-tune or serving needs lives in `src/fair_trees/`; one-off base-training code
  lives in `experiments/` (not installed).
- Tag claims in theory.md and docs as [V] verified, [P] proposed, [?] unknown.
- `docs/` is local only: never `git add` it. Commit messages carry no Co-Authored-By/Claude-Session lines.
- td-hot (no LICENSE) is a reference for structure only; never copy its code.

Commands: `uv sync --extra train --extra test`, `uv run pytest -q`, `uv run pytest -m integration`,
`uv run ruff check . && uv run ruff format --check .`.
