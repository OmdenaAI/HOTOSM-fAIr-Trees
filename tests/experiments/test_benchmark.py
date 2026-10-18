import json

import pytest

from experiments.benchmark import freeze_part, ids_sha256, load_benchmark

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
