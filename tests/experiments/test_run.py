import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

pytest.importorskip("yaml")
torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from experiments import run  # noqa: E402
from fair_trees.params import parse_train_params  # noqa: E402
from fair_trees.serve import ort_run_fn  # noqa: E402

# the fake eval plots are plain pixel rasters, as in test_neon_eval
pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")

CONFIG = """\
weights: weights/NEON.pt
sources: [neon]
hyperparameters: {epochs: 2, lr: 0.002}
val: {zooms: [20, 19], max_regions: null}
mlflow: false
"""


def _plot(root, name, boxes):
    (root / "evaluation" / "RGB").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    tif = root / "evaluation" / "RGB" / f"{name}.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=400, height=400, count=3, dtype="uint8") as dst:
        dst.write(np.full((3, 400, 400), 120, np.uint8))
    objs = "".join(
        f"<object><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for a, b, c, d in boxes
    )
    (root / "annotations" / f"{name}.xml").write_text(f"<annotation>{objs}</annotation>")


@pytest.fixture
def setup(write_region, tmp_path, monkeypatch):
    data, drive = tmp_path / "data", tmp_path / "drive"
    write_region("a", tiles=((0, 0),), root=data / "neon")
    write_region("v", tiles=((4, 0),), root=data / "neon", split="val", boxes_px=[(30, 30, 60, 60)])
    (data / "neon" / "manifest.json").write_text('{"source": "neon"}\n')
    for name in ("SITE_001", "SITE_002"):
        _plot(data / "NeonTreeEvaluation", name, [(10, 10, 60, 60)])
    (drive / "weights").mkdir(parents=True)
    (drive / "weights" / "NEON.pt").write_bytes(b"x")
    config = tmp_path / "c.yaml"
    config.write_text(CONFIG)
    bench = tmp_path / "benchmark.json"
    part = {
        "kind": "neon_eval_plots",
        "data_dir": "NeonTreeEvaluation",
        "unit_ids": ["SITE_001", "SITE_002"],
        "release_f1_ci95": [0.69, 0.75],
    }
    bench.write_text(json.dumps({"parts": {"neon": part}}))
    monkeypatch.setattr("fair_trees.train.train", lambda *a, **k: None)
    run.main(config, "r1", drive_root=drive, data_root=data, tmp_dir=tmp_path / "tmp")
    (drive / "runs" / "r1" / "best.pt").write_bytes(b"x")
    return data, drive, config, bench


def test_load_config_validates(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(CONFIG + "extra: 1\n")
    with pytest.raises(ValueError, match="extra"):
        run.load_config(p)
    p.write_text("sources: [neon]\n")
    with pytest.raises(ValueError, match="weights"):
        run.load_config(p)


CONFIGS = Path(run.__file__).parent / "configs"


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_shipped_configs_parse(name):
    conf = run.load_config(CONFIGS / name)
    parse_train_params(conf["hyperparameters"])


def test_main_writes_run_files_and_calls_train(monkeypatch, setup, tmp_path):
    data, drive, config, _ = setup
    calls = {}

    def fake_train(cfg, train_regions, val_regions, **kw):
        calls.update(cfg=cfg, train=[r.id for r in train_regions], val=[r.id for r in val_regions], **kw)
        return "result"

    monkeypatch.setattr("fair_trees.train.train", fake_train)
    assert run.main(config, "r2", drive_root=drive, data_root=data, tmp_dir=tmp_path / "tmp") == "result"
    assert calls["train"] == ["a"] and calls["val"] == ["v"] and calls["cfg"].learning_rate == 0.002
    assert calls["weights"] == drive / "weights" / "NEON.pt" and calls["out_dir"] == drive / "runs" / "r2"
    assert calls["manifest_hash"] and calls["on_epoch_end"] is None
    env = json.loads((drive / "runs" / "r2" / "env.json").read_text())
    assert {
        "repo_sha",
        "torch",
        "torchvision",
        "cuda",
        "gpu",
        "manifests",
        "mlflow_run_id",
        "units_used",
    } <= set(env)
    assert env["num_workers"] == 2
    assert json.loads((drive / "runs" / "r2" / "config.json").read_text())["parsed"]["learning_rate"] == 0.002


def test_main_passes_the_val_window_to_train(monkeypatch, setup, tmp_path):
    data, drive, config, _ = setup
    kw = dict(drive_root=drive, data_root=data, tmp_dir=tmp_path / "tmp")
    assert _rerun(monkeypatch, config, "r1", kw)["val_params"].window_m == 40.0
    config.write_text(CONFIG.replace("max_regions: null", "max_regions: null, window_m: 60"))
    calls = _rerun(monkeypatch, config, "r2", kw)
    assert calls["val_params"].window_m == 60.0


def test_stored_config_without_the_val_window_is_not_a_change(monkeypatch, setup, tmp_path):
    data, drive, config, _ = setup
    kw = dict(drive_root=drive, data_root=data, tmp_dir=tmp_path / "tmp")
    path = drive / "runs" / "r1" / "config.json"
    stored = json.loads(path.read_text())
    del stored["config"]["val"]["window_m"]
    path.write_text(json.dumps(stored))
    assert _rerun(monkeypatch, config, "r1", kw)["resume"] is True  # old run, same config: no refusal
    stored["config"]["val"]["window_m"] = 60.0
    path.write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="config.*force=True"):
        run.main(config, "r1", **kw)


def test_load_config_validates_the_val_window(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(CONFIG.replace("max_regions: null", "max_regions: null, window_m: 1000"))
    conf = run.load_config(p)
    with pytest.raises(ValueError, match="window_m"):
        run.val_params(conf)


def test_evaluate_scores_val_with_the_configured_window(monkeypatch, setup, fake_session):
    data, drive, config, bench = setup
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(fake_session()))
    import fair_trees.evaluate as ev

    windows = []
    real = ev.evaluate_regions
    monkeypatch.setattr(
        ev,
        "evaluate_regions",
        lambda fn, val, params, **k: windows.append(params.window_m) or real(fn, val, params, **k),
    )
    run.evaluate("r1", bench, drive_root=drive, data_root=data)  # stored config: no window_m
    config.write_text(CONFIG.replace("max_regions: null", "max_regions: null, window_m: 60"))
    run.evaluate("r1", bench, drive_root=drive, data_root=data, config=config)
    assert windows == [40.0, 60.0]
    assert json.loads((drive / "runs" / "r1" / "eval.json").read_text())["parts"]["val"]["window_m"] == 60.0


def _rerun(monkeypatch, config, run_id, kw, **main_kw):
    """Run `main` again and return the keyword arguments it passed to train."""
    calls = {}
    monkeypatch.setattr("fair_trees.train.train", lambda *a, **k: calls.update(k))
    run.main(config, run_id, **kw, **main_kw)
    return calls


def _set_mlflow_run_id(out, value):
    env = json.loads((out / "env.json").read_text())
    (out / "env.json").write_text(json.dumps({**env, "mlflow_run_id": value}))


def _refused_then_forced(monkeypatch, config, run_id, kw, match):
    """Rerun `run_id` after a change: refused with no file touched, then started over with force=True."""
    out = kw["drive_root"] / "runs" / run_id
    _set_mlflow_run_id(out, "old")
    before = {name: (out / name).read_bytes() for name in ("config.json", "env.json")}
    with pytest.raises(ValueError, match=f"{match}.*force=True"):
        run.main(config, run_id, **kw)
    assert {name: (out / name).read_bytes() for name in before} == before
    calls = _rerun(monkeypatch, config, run_id, kw, force=True)
    assert calls["resume"] is False and calls["force"] is True
    env = json.loads((out / "env.json").read_text())
    assert env["mlflow_run_id"] is None  # the new run logs to a new mlflow run
    return json.loads((out / "config.json").read_text()), env


def test_main_refuses_changed_hyperparameters_unless_forced(monkeypatch, setup, tmp_path):
    data, drive, config, _ = setup
    kw = dict(drive_root=drive, data_root=data, tmp_dir=tmp_path / "tmp")
    _set_mlflow_run_id(drive / "runs" / "r1", "old")
    assert _rerun(monkeypatch, config, "r1", kw)["resume"] is True  # an unchanged rerun resumes
    assert json.loads((drive / "runs" / "r1" / "env.json").read_text())["mlflow_run_id"] == "old"
    config.write_text(CONFIG.replace("lr: 0.002", "lr: 0.003"))
    conf, _ = _refused_then_forced(monkeypatch, config, "r1", kw, "hyperparameters")
    assert conf["parsed"]["learning_rate"] == 0.003


def test_main_refuses_changed_manifests_unless_forced(monkeypatch, setup, tmp_path):
    data, drive, config, _ = setup
    kw = dict(drive_root=drive, data_root=data, tmp_dir=tmp_path / "tmp")
    old = json.loads((drive / "runs" / "r1" / "env.json").read_text())["manifests"]
    (data / "neon" / "manifest.json").write_text('{"source": "neon", "restaged": true}\n')
    _, env = _refused_then_forced(monkeypatch, config, "r1", kw, "manifests")
    assert env["manifests"] != old


def test_main_refuses_changed_weights_unless_forced(monkeypatch, setup, tmp_path):
    data, drive, config, _ = setup
    kw = dict(drive_root=drive, data_root=data, tmp_dir=tmp_path / "tmp")
    config.write_text(CONFIG.replace("weights/NEON.pt", "weights/other.pt"))
    conf, _ = _refused_then_forced(monkeypatch, config, "r1", kw, "config")
    assert conf["config"]["weights"] == "weights/other.pt"


def test_evaluate_writes_parts(monkeypatch, setup, fake_session):
    data, drive, _, bench = setup
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(fake_session()))
    out = run.evaluate("r1", bench, drive_root=drive, data_root=data)
    assert json.loads((drive / "runs" / "r1" / "eval.json").read_text()) == out
    assert set(out["parts"]) == {"val", "neon"}
    neon = out["parts"]["neon"]
    assert neon["forgetting_lower_bound"] == 0.69 and neon["forgetting_pass"] is False  # no detections
    assert "f1_ci95" in neon and "f1_ci95" in out["parts"]["val"]


P3_800 = 100  # P3 locations per row at an 800 px input; a P3 anchor's box is 32 input px


def _p3(row, col):
    return (row * P3_800 + col) * 9 + 3  # aspect 1.0, size 32


GROUP_PX = (
    2048 + 123,
    123,
    2048 + 154,
    154,
)  # on tile (8, 0); 3 x 3 touching crowns at P3 (50 + 4i, 50 + 4j)


def _group_region(write_region, data):
    from fair_trees.params import CLASSES

    write_region("g", tiles=((8, 0),), root=data / "neon", split="val", groups_px=[GROUP_PX], classes=CLASSES)


def test_evaluate_adds_val_group_for_k2(monkeypatch, setup, write_region, fake_session):
    data, drive, _, bench = setup
    _group_region(write_region, data)
    session = fake_session({(_p3(50, 50), 1): 5.0}, num_classes=2)
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(session))
    out = run.evaluate("r1", bench, drive_root=drive, data_root=data)
    assert out["parts"]["val_group"]["window_m"] == 120.0
    assert [u["unit_id"] for u in out["parts"]["val_group"]["units"]] == ["g"]
    assert "val_group_merged" not in out["parts"]


def _k2_tree_and_group(fake_session):
    return fake_session({(_p3(50, 50), 0): 5.0, (_p3(50, 50), 1): 5.0}, num_classes=2)


def test_evaluate_picks_thresholds_on_val(monkeypatch, setup, write_region, fake_session):
    data, drive, _, bench = setup
    _group_region(write_region, data)
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(_k2_tree_and_group(fake_session)))
    out = run.evaluate("r1", bench, drive_root=drive, data_root=data)
    parts = out["parts"]
    # a handful of val boxes: the two-class prior, for every part of each class
    assert out["thresholds"] == {"tree": 0.2, "tree_group": 0.2} and out["threshold_source"] == "val"
    for name in ("val", "val_group", "neon"):
        assert parts[name]["picked_threshold"] == 0.2
        assert {"f1_picked", "precision_picked", "recall_picked"} <= set(parts[name])
    assert json.loads((drive / "runs" / "r1" / "thresholds.json").read_text()) == out["thresholds"]

    (drive / "runs" / "r1" / "thresholds.json").write_text(json.dumps({"tree": 0.125, "tree_group": 0.3}))
    out = run.evaluate("r1", bench, drive_root=drive, data_root=data)
    assert out["thresholds"] == {"tree": 0.125, "tree_group": 0.3} and out["threshold_source"] == "training"
    assert out["parts"]["val"]["picked_threshold"] == 0.125


def test_evaluate_reports_trees_inside_groups(monkeypatch, setup, write_region, fake_session):
    data, drive, _, bench = setup
    _group_region(write_region, data)
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(_k2_tree_and_group(fake_session)))
    report = run.evaluate("r1", bench, drive_root=drive, data_root=data)["parts"]["val_in_group"]
    assert report["tree_boxes"] == 2  # one per val tile
    assert report["share_in_served_groups"] == 1.0 and report["share_in_gt_groups"] == 0.5
    assert set(report) >= {"f1", "f1_dropping_in_group", "tp_lost", "paired_diff", "thresholds"}


def test_compare_group_merges(monkeypatch, setup, write_region, fake_session):
    data, drive, _, bench = setup
    _group_region(write_region, data)
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(_k2_tree_and_group(fake_session)))
    out = run.compare_group_merges("r1", drive_root=drive, data_root=data)
    assert set(out["rules"]) == {"join", "overlap", "drop_edge"}
    assert out["rules"]["join"]["paired_diff_vs_join"] == [0.0, 0.0, 0.0]
    assert {"f1", "f1_max", "ap40", "threshold"} <= set(out["rules"]["drop_edge"])
    assert json.loads((drive / "runs" / "r1" / "group_merge.json").read_text()) == out


def test_evaluate_adds_merged_baseline_for_k1(monkeypatch, setup, write_region, fake_session):
    data, drive, _, bench = setup
    _group_region(write_region, data)
    crowns = {_p3(50 + 4 * i, 50 + 4 * j): 5.0 for i in range(3) for j in range(3)}
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(fake_session(crowns)))
    out = run.evaluate("r1", bench, drive_root=drive, data_root=data)
    merged = out["parts"]["val_group_merged"]
    assert (merged["precision"], merged["recall"], merged["f1"]) == (1.0, 1.0, 1.0)
    assert merged["setting"] == {"score": 0.2, "gap_m": 2.0, "min_crowns": 8}
    assert [u["unit_id"] for u in merged["units"]] == ["g"] and "f1_ci95" in merged
    assert "val_group" not in out["parts"]


def test_export_k2_checkpoint(setup):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from fair_trees.train import build_model

    data, drive, _, bench = setup
    torch.manual_seed(0)
    model = build_model(None, trainable_layers=0, num_classes=2)
    torch.save(model.state_dict(), drive / "runs" / "r1" / "best.pt")
    assert run.export("r1", bench, drive_root=drive, data_root=data)["pass"] is True


def _val_conf(max_regions):
    return {"sources": ["neon", "oamtcd"], "val": {**run.DEFAULT_VAL, "max_regions": max_regions}}


def test_val_cap_per_source(write_region, tmp_path):
    data = tmp_path / "data"
    write_region("n1", tiles=((0, 0),), root=data / "neon", source="neon", split="val")
    write_region("o1", tiles=((2, 0),), root=data / "oamtcd", source="oamtcd", split="val")
    write_region("o2", tiles=((4, 0),), root=data / "oamtcd", source="oamtcd", split="val")

    def val_ids(max_regions):
        return [r.id for r in run._regions(_val_conf(max_regions), data)[1]]

    assert val_ids({"oamtcd": 1}) == ["n1", "o1"]
    assert val_ids({"other": 1}) == ["n1", "o1", "o2"]
    assert val_ids(2) == ["n1", "o1"]
    assert val_ids(None) == ["n1", "o1", "o2"]


@pytest.mark.parametrize("bad", ["{oamtcd: 0}", "{oamtcd: x}", "0", "[1]"])
def test_load_config_validates_max_regions(tmp_path, bad):
    p = tmp_path / "c.yaml"
    p.write_text(CONFIG.replace("max_regions: null", f"max_regions: {bad}"))
    with pytest.raises(ValueError, match="max_regions"):
        run.load_config(p)


def _add_regions_part(bench, **changes):
    data = json.loads(bench.read_text())
    part = {
        "kind": "regions",
        "data_dir": "oamtcd",
        "split": "test",
        "zooms": [20, 19],
        "window_m": 40.0,
        "unit_ids": ["t1"],
        "frozen_on": "2026-10-29",
    }
    data["parts"]["oamtcd"] = part | changes
    bench.write_text(json.dumps(data))


def test_evaluate_scores_a_regions_part(monkeypatch, setup, write_region, fake_session):
    data, drive, _, bench = setup
    oamtcd = data / "oamtcd"
    write_region(
        "t1", tiles=((8, 0),), root=oamtcd, source="oamtcd", split="test", boxes_px=[(2100, 30, 2130, 60)]
    )
    _add_regions_part(bench)
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(fake_session()))

    out = run.evaluate("r1", bench, drive_root=drive, data_root=data)

    part = out["parts"]["oamtcd"]
    assert "f1" in part and "f1_ci95" in part and part["window_m"] == 40.0
    assert "forgetting_pass" not in part
    assert [u["unit_id"] for u in part["units"]] == ["t1"]


def test_evaluate_scores_groups_of_a_regions_part_for_k2(monkeypatch, setup, write_region, fake_session):
    from fair_trees.params import CLASSES

    data, drive, _, bench = setup
    write_region(
        "t1", tiles=((8, 0),), root=data / "oamtcd", source="oamtcd", split="test",
        groups_px=[GROUP_PX], classes=CLASSES,
    )  # fmt: skip
    _add_regions_part(bench)
    session = fake_session({(_p3(50, 50), 1): 5.0}, num_classes=2)
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(session))
    out = run.evaluate("r1", bench, drive_root=drive, data_root=data)
    group = out["parts"]["oamtcd_group"]
    assert [u["unit_id"] for u in group["units"]] == ["t1"] and group["window_m"] == 120.0
    assert "f1_ci95" in group and "forgetting_pass" not in group
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(fake_session()))
    assert "oamtcd_group" not in run.evaluate("r1", bench, drive_root=drive, data_root=data)["parts"]


def test_evaluate_skips_an_unstaged_regions_part(monkeypatch, setup, fake_session):
    data, drive, _, bench = setup
    _add_regions_part(bench)
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(fake_session()))
    assert set(run.evaluate("r1", bench, drive_root=drive, data_root=data)["parts"]) == {"val", "neon"}


def test_regions_part_units_must_match_freeze(monkeypatch, setup, write_region, fake_session):
    data, drive, _, bench = setup
    write_region("t1", tiles=((8, 0),), root=data / "oamtcd", source="oamtcd", split="test")
    _add_regions_part(bench, unit_ids=["t9"])
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(fake_session()))
    with pytest.raises(ValueError, match="frozen"):
        run.evaluate("r1", bench, drive_root=drive, data_root=data)


def test_evaluate_refuses_changed_benchmark_units(monkeypatch, setup, fake_session):
    data, drive, _, bench = setup
    _plot(data / "NeonTreeEvaluation", "SITE_003", [])
    monkeypatch.setattr(run, "_run_fn", lambda weights, device: ort_run_fn(fake_session()))
    with pytest.raises(ValueError, match="frozen"):
        run.evaluate("r1", bench, drive_root=drive, data_root=data)


def test_export_writes_onnx_and_passes_parity(setup):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from fair_trees.train import build_model

    data, drive, _, bench = setup
    torch.manual_seed(0)
    torch.save(build_model(None, trainable_layers=0).state_dict(), drive / "runs" / "r1" / "best.pt")
    result = run.export("r1", bench, drive_root=drive, data_root=data)
    assert result["pass"] is True and result["max_abs_diff"] <= 2e-3
    assert (drive / "runs" / "r1" / "model.onnx").stat().st_size > 1_000_000
    assert json.loads((drive / "runs" / "r1" / "export.json").read_text()) == result


def _fake_run(drive, run_id, digests, unit_counts, window_m=None):
    d = drive / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    with (d / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, ["epoch", "sampler/plan_sha256", "sampler/sample_ids_sha256"])
        w.writeheader()
        for e, (p, s) in enumerate(digests):
            w.writerow({"epoch": e, "sampler/plan_sha256": p, "sampler/sample_ids_sha256": s})
    units = [
        {"unit_id": f"u{i}", "meta": {}, "tp": tp, "fp": fp, "fn": fn}
        for i, (tp, fp, fn) in enumerate(unit_counts)
    ]
    tp, fp, fn = (sum(c[k] for c in unit_counts) for k in range(3))
    f1 = 2 * tp / (2 * tp + fp + fn)
    val = {"f1": f1, "f1_ci95": [f1 - 0.05, f1 + 0.05], "units": units}
    if window_m is not None:  # older eval.json files carry none
        val["window_m"] = window_m
    (d / "eval.json").write_text(json.dumps({"parts": {"val": val}}))


def test_compare_runs_pairs_two_named_parts(tmp_path):
    # a two-class run's val_group against a tree-only run's crown-merge baseline: same units, other windows
    _fake_run(tmp_path, "groups", [("p0", "s0")], [(1, 0, 0)], window_m=120.0)
    _fake_run(tmp_path, "merged", [("p0", "s0")], [(0, 1, 1)], window_m=60.0)
    for run_id, part in (("groups", "val_group"), ("merged", "val_group_merged")):
        path = tmp_path / "runs" / run_id / "eval.json"
        path.write_text(json.dumps({"parts": {part: json.loads(path.read_text())["parts"]["val"]}}))
    r = run.compare_runs(
        "merged", "groups", drive_root=tmp_path, part_a="val_group_merged", part_b="val_group"
    )
    assert (r["f1_a"], r["f1_b"], r["paired_diff"][0]) == (0.0, 1.0, 1.0)
    assert (r["part_a"], r["part_b"]) == ("val_group_merged", "val_group")


def test_compare_runs_at_each_runs_picked_threshold(tmp_path):
    for run_id, picked, counts in (("k1", 0.15, [[3, 1, 0]]), ("k2", 0.2, [[2, 0, 1]])):
        _fake_run(tmp_path, run_id, [("p0", "s0")], [(0, 9, 9)])
        path = tmp_path / "runs" / run_id / "eval.json"
        doc = json.loads(path.read_text())
        doc["parts"]["val"] |= {
            "picked_threshold": picked,
            "pick_sweep": [
                {"threshold": t, "units": counts if t == picked else [[0, 0, 0]]} for t in (0.15, 0.2)
            ],
        }
        path.write_text(json.dumps(doc))
    r = run.compare_runs("k1", "k2", drive_root=tmp_path, picked=True)
    assert (r["f1_a"], r["f1_b"]) == (pytest.approx(6 / 7), pytest.approx(4 / 5))
    assert r["paired_diff"][0] == pytest.approx(4 / 5 - 6 / 7) and r["picked"] == (0.15, 0.2)


def test_compare_runs(tmp_path):
    counts = [(8, 2, 1), (5, 1, 2), (9, 0, 1), (4, 3, 2)]
    _fake_run(tmp_path, "a", [("p0", "s0"), ("p1", "s1")], counts)
    _fake_run(tmp_path, "b", [("p0", "s0"), ("p1", "s1")], counts)
    r = run.compare_runs("a", "b", drive_root=tmp_path)
    assert r["plan_equal"] and r["sample_ids_equal"] and r["b_f1_in_a_ci"]
    assert r["paired_diff"][0] == 0.0
    _fake_run(tmp_path, "c", [("p0", "s0"), ("p1", "sX")], counts)
    assert run.compare_runs("a", "c", drive_root=tmp_path)["sample_ids_equal"] is False


def test_compare_runs_refuses_different_val_windows(tmp_path):
    counts = [(8, 2, 1), (5, 1, 2)]
    _fake_run(tmp_path, "a", [("p0", "s0")], counts, window_m=40.0)
    _fake_run(tmp_path, "b", [("p0", "s0")], counts, window_m=60.0)
    with pytest.raises(ValueError, match=r"40\.0.*60\.0"):
        run.compare_runs("a", "b", drive_root=tmp_path)


def test_compare_runs_treats_a_missing_val_window_as_40(tmp_path):
    counts = [(8, 2, 1), (5, 1, 2)]
    _fake_run(tmp_path, "a", [("p0", "s0")], counts)
    _fake_run(tmp_path, "b", [("p0", "s0")], counts, window_m=40.0)
    _fake_run(tmp_path, "c", [("p0", "s0")], counts, window_m=60.0)
    assert (
        run.compare_runs("a", "b", drive_root=tmp_path)["f1_a"]
        == run.compare_runs("b", "a", drive_root=tmp_path)["f1_a"]
    )
    with pytest.raises(ValueError, match="window"):
        run.compare_runs("a", "c", drive_root=tmp_path)


def test_compare_runs_without_common_epochs_compares_nothing(tmp_path):
    counts = [(8, 2, 1), (5, 1, 2)]
    _fake_run(tmp_path, "a", [("p0", "s0")], counts)
    _fake_run(tmp_path, "b", [], counts)
    r = run.compare_runs("a", "b", drive_root=tmp_path)
    assert r["plan_equal"] is None and r["sample_ids_equal"] is None
