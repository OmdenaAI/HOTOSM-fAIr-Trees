import json
from pathlib import Path

import pytest

NB = Path(__file__).resolve().parents[2] / "experiments" / "notebooks"

REQUIRED = {
    "colab_convert.ipynb": [
        "SOURCE",
        "REPO_SHA",
        "drive.mount",
        "git clone",
        "GITHUB_TOKEN",
        "sha256",
        "experiments.converters.neon",
        "resolve_biome_fn",
        "write_shards",
        "manifest_hash",
        "contact",
        "MB/s",
        "flush_and_unmount",
    ],
    "colab_train.ipynb": [
        "RUN_ID",
        "CONFIG",
        "REPO_SHA",
        "drive.mount",
        "git clone",
        "GITHUB_TOKEN",
        "[train,test]",
        "torch.__version__",
        "stage_shards",
        "experiments.throughput",
        "run.main",
        "run.evaluate",
        "run.export",
        "run.compare_runs",
        "flush_and_unmount",
    ],
}


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_notebook_structure(name):
    nb = json.loads((NB / name).read_text())
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    missing = [s for s in REQUIRED[name] if s not in code]
    assert not missing, missing
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")  # no outputs committed
