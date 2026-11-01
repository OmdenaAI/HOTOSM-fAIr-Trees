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
    "colab_convert_hot.ipynb": [
        "REPO_SHA",
        "drive.mount",
        "git clone",
        "GITHUB_TOKEN",
        "experiments.converters.hot",
        "REVISION",
        "iter_batches",
        "resolve_biome_fn",
        "write_shards",
        "manifest_hash",
        "contact",
        "MB/s",
        "flush_and_unmount",
    ],
    "colab_train.ipynb": [
        "RUN_ID",
        "BASELINE_ID",
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
        "picked=True",
        "run.compare_group_merges",
        "flush_and_unmount",
    ],
    "colab_convert_oamtcd.ipynb": [
        "REPO_SHA",
        "drive.mount",
        "git clone",
        "GITHUB_TOKEN",
        "restor/tcd",
        "restor/tcd-nc",
        "REVISION",
        "iter_batches",
        "category_ids",
        "read_hot",
        "resolve_biome_fn",
        "experiments.converters.oamtcd",
        "write_shards",
        "manifest_hash",
        "regions_part",
        "contact",
        "MB/s",
        "flush_and_unmount",
    ],
    "colab_teacher_hot.ipynb": [
        "REPO_SHA",
        "drive.mount",
        "git clone",
        "GITHUB_TOKEN",
        "stage_shards",
        "write_teacher_sets",
        "write_shards",
        "manifest_hash",
        "contact",
        "flush_and_unmount",
    ],
    "label_study.ipynb": [
        "REPO_SHA",
        "drive.mount",
        "git clone",
        "GITHUB_TOKEN",
        "[train,test]",
        "NEON.pt",
        "read_hot",
        "split_units",
        "run_units",
        "sweep",
        "choose",
        "score_test",
        "groups_40_all",
        "DRIVE_RESULTS",
        "masking_cost",
        "audit_table",
        "flush_and_unmount",
        "MODEL",
        "WINDOWS_M",
        "window_m=",
        "pick_window",
        "80.0",
        "window_check.json",
        "weights.sha256",
        "cache-release-40m",
        "RUN_AUDIT",
        'MODEL != "release"',
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


@pytest.mark.parametrize("name", sorted(REQUIRED))
def test_token_clone_never_raises_with_its_url(name):
    # check=True would raise CalledProcessError, whose message holds the URL and so the token
    nb = json.loads((NB / name).read_text())
    lines = [line for c in nb["cells"] for line in c["source"] if '"git", "clone"' in line]
    assert lines and all("url" in line and "check=True" not in line for line in lines)
