import json

import pytest

from experiments.benchmark import cli, freeze_part, ids_sha256, load_benchmark, regions_part

PART = {
    "kind": "neon_eval_plots",
    "data_dir": "NeonTreeEvaluation",
    "unit_ids": ["A_001", "B_002"],
    "release_f1_ci95": [0.69, 0.75],
    "frozen_on": "2026-10-26",
}


def test_freeze_writes_and_is_idempotent(tmp_path):
    path = tmp_path / "benchmark.json"
    assert load_benchmark(path) == {"parts": {}}
    frozen = freeze_part(path, "neon", PART)
    assert frozen["unit_ids_sha256"] == ids_sha256(["A_001", "B_002"])
    assert load_benchmark(path)["parts"]["neon"] == frozen
    assert freeze_part(path, "neon", PART) == frozen  # identical re-freeze is a no-op


def test_frozen_parts_never_change(tmp_path):
    path = tmp_path / "benchmark.json"
    freeze_part(path, "neon", PART)
    with pytest.raises(ValueError, match="frozen"):
        freeze_part(path, "neon", PART | {"unit_ids": ["A_001"]})
    freeze_part(path, "other", PART | {"data_dir": "x"})  # a new part is fine
    assert set(load_benchmark(path)["parts"]) == {"neon", "other"}


def test_hand_edits_are_detected(tmp_path):
    path = tmp_path / "benchmark.json"
    freeze_part(path, "neon", PART)
    data = json.loads(path.read_text())
    data["parts"]["neon"]["unit_ids"].append("C_003")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unit_ids_sha256"):
        load_benchmark(path)


@pytest.fixture
def oamtcd_root(write_region, tmp_path):
    data = tmp_path / "data"
    for rid, split, dx in (("t2", "test", 0), ("t1", "test", 2), ("a", "train", 4)):
        write_region(rid, tiles=((dx, 0),), root=data / "oamtcd", source="oamtcd", split=split)
    return data


def test_regions_part_lists_split_unit_ids(oamtcd_root):
    part = regions_part(
        oamtcd_root, "oamtcd", split="test", zooms=[20, 19], window_m=40.0, frozen_on="2026-10-29"
    )
    assert part == {
        "kind": "regions",
        "data_dir": "oamtcd",
        "split": "test",
        "zooms": [20, 19],
        "window_m": 40.0,
        "unit_ids": ["t1", "t2"],
        "frozen_on": "2026-10-29",
    }


def test_cli_freezes_a_regions_part_once(oamtcd_root, tmp_path):
    bench = tmp_path / "benchmark.json"
    argv = [
        "freeze-regions",
        *("--data-root", str(oamtcd_root), "--data-dir", "oamtcd", "--split", "test"),
        *("--zooms", "20", "19", "--window-m", "40", "--frozen-on", "2026-10-29", "--benchmark", str(bench)),
    ]
    cli(argv)
    cli(argv)  # an identical re-freeze is a no-op
    part = load_benchmark(bench)["parts"]["oamtcd"]
    assert part["unit_ids"] == ["t1", "t2"] and part["kind"] == "regions"
