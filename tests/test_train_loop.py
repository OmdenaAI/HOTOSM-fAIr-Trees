import csv
import dataclasses
import json
import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

import fair_trees.train as T  # noqa: E402
from fair_trees.dataset import load_regions  # noqa: E402
from fair_trees.params import AugmentParams, InferenceParams, TrainParams  # noqa: E402

X0 = Y0 = 2**19
CFG = TrainParams(
    epochs=3,
    batch_size=2,
    samples_per_epoch=6,
    patience=2,
    ckpt_every_min=1e3,
    amp=False,
    augment=AugmentParams(downscale=False),
)
KEYS = ("epoch", "sampler/plan_sha256", "sampler/sample_ids_sha256", "val/f1_max", "improved")


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("weights") / "start.pt"
    torch.save(T.build_model(None, trainable_layers=0).state_dict(), path)
    return path


@pytest.fixture
def regions(write_region):
    d = write_region("t", tiles=((0, 0), (1, 0)), boxes_px=[(30, 30, 60, 70)])
    write_region("v", tiles=((4, 0),), boxes_px=[(30, 30, 60, 70)], split="val")
    rs = load_regions(d.parent)
    return [r for r in rs if r.split == "train"], [r for r in rs if r.split == "val"]


class Script:
    """A cheap loss and scripted val metrics per epoch; optionally 'kills' the runtime at a forward call."""

    def __init__(self, f1s, kill_at=None):
        self.f1s, self.kill_at, self.calls = list(f1s), kill_at, 0

    def forward_loss(self, model, images, image_sizes, targets):
        self.calls += 1
        if self.kill_at is not None and self.calls == self.kill_at:
            raise KeyboardInterrupt("runtime killed")
        w = model.head.classification_head.cls_logits.weight
        return {"classification": (w * 0).sum() + 1.0, "bbox_regression": (w * 0).sum()}

    def validate(self, model, device, units, params, epoch):
        f1 = self.f1s[min(epoch, len(self.f1s) - 1)]
        return {"f1_max": f1, "conf_at_max": 0.3, "ap40": f1, "aoi_f1": f1, "seconds": 0.01}


def _run(monkeypatch, script, regions, weights, out, cfg=CFG, **kw):
    monkeypatch.setattr(T, "forward_loss", script.forward_loss)
    monkeypatch.setattr(T, "_validate", script.validate)
    return T.train(
        cfg, *regions, weights=weights, out_dir=out, num_workers=0, tmp_dir=out.parent / "tmp", **kw
    )


def _view(history):
    return [{k: h[k] for k in KEYS} for h in history]


def test_validation_uses_the_given_params_and_defaults_to_40_m(monkeypatch, regions, weights, tmp_path):
    seen = []
    script = Script([0.5])
    validate = script.validate
    script.validate = lambda model, device, units, params, epoch: (
        seen.append(params) or validate(model, device, units, params, epoch)
    )
    cfg = dataclasses.replace(CFG, epochs=1)
    _run(monkeypatch, script, regions, weights, tmp_path / "default", cfg=cfg)
    given = InferenceParams(window_m=60.0)
    _run(monkeypatch, script, regions, weights, tmp_path / "given", cfg=cfg, val_params=given)
    assert seen == [InferenceParams(), given] and seen[0].window_m == 40.0


def test_early_stopping_best_and_logs(monkeypatch, regions, weights, tmp_path):
    seen = []
    cfg = dataclasses.replace(CFG, epochs=5, patience=1)
    script = Script([0.50, 0.60, 0.603, 0.603, 0.70])
    res = _run(
        monkeypatch,
        script,
        regions,
        weights,
        tmp_path / "run",
        cfg=cfg,
        on_epoch_end=lambda m, e: seen.append((e, m)),
    )
    # epoch 2 is within min_delta of 0.60 but has a higher AP, so it improves; epoch 3 does not, and stops
    assert [h["improved"] for h in res.history] == [True, True, True, False]
    assert res.best_metrics["val/f1_max"] == 0.603 and res.best_checkpoint.name == "best.pt"
    assert len(torch.load(res.best_checkpoint, weights_only=True)) == 301  # weights only
    assert [e for e, _ in seen] == [0, 1, 2, 3]
    assert all(isinstance(v, float) for _, m in seen for v in m.values())
    expected = {
        "train/loss_cls",
        "train/loss_reg",
        "train/l2sp",
        "train/lr",
        "train/samples_per_s",
        "val/f1_max",
        "val/ap40",
        "val/seconds",
        "sampler/region_repeat_max",
    }
    assert expected <= set(seen[0][1])
    lrs = [m["train/lr"] for _, m in seen]
    assert all(a > b for a, b in zip(lrs, lrs[1:], strict=False))  # cosine over the planned total steps
    with (tmp_path / "run" / "metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and rows[2]["improved"] == "True"


@pytest.mark.parametrize("ckpt_every_min", [1e-9, 1e3])  # mid-epoch checkpoints vs epoch-end only
def test_resume_replays_the_same_samples(monkeypatch, regions, weights, tmp_path, ckpt_every_min):
    cfg = dataclasses.replace(CFG, ckpt_every_min=ckpt_every_min)
    ref = _run(monkeypatch, Script([0.1, 0.2, 0.3]), regions, weights, tmp_path / "ref", cfg=cfg)
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, Script([0.1, 0.2, 0.3], kill_at=5), regions, weights, tmp_path / "res", cfg=cfg)
    res = _run(monkeypatch, Script([0.1, 0.2, 0.3]), regions, weights, tmp_path / "res", cfg=cfg)
    assert _view(res.history) == _view(ref.history)


def test_changed_config_is_refused(monkeypatch, regions, weights, tmp_path):
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        _run(monkeypatch, Script([0.1] * 3, kill_at=4), regions, weights, out)
    other = dataclasses.replace(CFG, learning_rate=5e-3)
    with pytest.raises(ValueError, match="config"):
        _run(monkeypatch, Script([0.1] * 3), regions, weights, out, cfg=other)
    assert (
        len(_run(monkeypatch, Script([0.1] * 3), regions, weights, out, cfg=other, force=True).history) == 3
    )


def test_fresh_start_refuses_a_used_run_dir(monkeypatch, regions, weights, tmp_path):
    out = tmp_path / "run"
    _run(monkeypatch, Script([0.1, 0.2, 0.3]), regions, weights, out)
    with pytest.raises(ValueError, match="resume"):
        _run(monkeypatch, Script([0.1] * 3), regions, weights, out, resume=False)


def test_finished_run_returns_without_training(monkeypatch, regions, weights, tmp_path):
    out = tmp_path / "run"
    first = _run(monkeypatch, Script([0.1, 0.2, 0.3]), regions, weights, out)
    best_bytes = first.best_checkpoint.read_bytes()
    again = Script([0.9] * 3)
    second = _run(monkeypatch, again, regions, weights, out)
    assert again.calls == 0 and _view(second.history) == _view(first.history)
    assert second.best_checkpoint.read_bytes() == best_bytes


def test_no_val_regions_keeps_last(monkeypatch, regions, weights, tmp_path, caplog):
    script = Script([0.0])
    monkeypatch.setattr(T, "forward_loss", script.forward_loss)
    with caplog.at_level(logging.WARNING, logger="fair_trees.train"):
        res = T.train(
            CFG,
            regions[0],
            [],
            weights=weights,
            out_dir=tmp_path / "run",
            num_workers=0,
            tmp_dir=tmp_path / "tmp",
        )
    assert len(res.history) == 3 and all(h["improved"] for h in res.history)
    assert "val" in caplog.text


def test_no_train_regions_is_an_error(weights, tmp_path):
    with pytest.raises(ValueError, match="train"):
        T.train(CFG, [], [], weights=weights, out_dir=tmp_path / "run")


def test_fine_tune_on_fair_chips(monkeypatch, write_oam_chip, geo, weights, tmp_path):
    chips = [
        write_oam_chip(4 * X0 + i, 4 * Y0, 22) for i in range(4)
    ]  # z22: outside the default zoom weights
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps(geo.feature_collection([geo.box_polygon((5, 5, 15, 15))])))  # no "label"
    script = Script([0.2, 0.3])
    monkeypatch.setattr(T, "forward_loss", script.forward_loss)
    monkeypatch.setattr(T, "_validate", script.validate)
    res = T.fine_tune(
        chips[:3],
        chips[3:],
        labels,
        {"epochs": 2, "batch_size": 2, "amp": False},
        weights=weights,
        out_dir=tmp_path / "ft",
        num_workers=0,
        tmp_dir=tmp_path / "tmp",
    )
    assert len(res.history) == 2 and script.calls == 4  # 3 train chips per epoch at batch 2: 2 steps
    assert res.history[0]["sampler/draws/zoom/22"] == 3.0


def test_fine_tune_keeps_k1_unless_asked(monkeypatch, write_oam_chip, geo, weights, tmp_path):
    chips = [write_oam_chip(4 * X0 + i, 4 * Y0, 22) for i in range(3)]
    labels = tmp_path / "labels.geojson"
    fc = geo.feature_collection([geo.box_polygon((5, 5, 15, 15))])
    fc["features"] += geo.feature_collection([geo.box_polygon((40, 40, 90, 90))], label=2)["features"]
    labels.write_text(json.dumps(fc))
    monkeypatch.setattr(T, "_validate", Script([0.2]).validate)
    hp = {"epochs": 1, "batch_size": 2, "samples_per_epoch": 2, "amp": False}
    for name, extra, k in (("k1", {}, 1), ("k2", {"num_classes": 2}, 2)):
        res = T.fine_tune(
            chips[:2],
            chips[2:],
            labels,
            hp | extra,
            weights=weights,
            out_dir=tmp_path / name,
            num_workers=0,
            tmp_dir=tmp_path / "tmp",
        )
        state = torch.load(res.best_checkpoint, weights_only=True)
        assert T.checkpoint_num_classes(state) == k


def test_two_class_fine_tune_crops_wide_enough_for_groups(
    monkeypatch, write_oam_chip, geo, weights, tmp_path
):
    chips = [write_oam_chip(4 * X0 + i, 4 * Y0, 22) for i in range(2)]
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps(geo.feature_collection([geo.box_polygon((5, 5, 15, 15))])))
    seen = []
    monkeypatch.setattr(T, "train", lambda cfg, *a, **k: seen.append(cfg.augment.crop_m))
    k2 = tmp_path / "k2.pt"
    torch.save(T.build_model(None, trainable_layers=0, num_classes=2).state_dict(), k2)
    for hp, start in (
        ({}, weights),
        ({"num_classes": 2}, weights),
        ({}, k2),
        ({"num_classes": 2, "augment": {"crop_m": [30, 60]}}, weights),
    ):
        T.fine_tune(chips[:1], chips[1:], labels, hp, weights=start, out_dir=tmp_path / "ft")
    assert seen == [(30.0, 80.0), (30.0, 120.0), (30.0, 120.0), (30.0, 60.0)]


def test_two_classes_without_group_labels_are_refused(monkeypatch, regions, weights, tmp_path):
    script = Script([0.1])
    cfg = dataclasses.replace(CFG, num_classes=2)
    with pytest.raises(ValueError, match="tree_group"):
        _run(monkeypatch, script, regions, weights, tmp_path / "run", cfg=cfg)
    assert script.calls == 0 and not (tmp_path / "run").exists()


def test_fine_tune_refuses_to_resume_on_changed_data(monkeypatch, write_oam_chip, geo, weights, tmp_path):
    chips = [write_oam_chip(4 * X0 + i, 4 * Y0, 22) for i in range(4)]
    labels = tmp_path / "labels.geojson"
    labels.write_text(json.dumps(geo.feature_collection([geo.box_polygon((5, 5, 15, 15))])))
    script = Script([0.2, 0.3])
    monkeypatch.setattr(T, "forward_loss", script.forward_loss)
    monkeypatch.setattr(T, "_validate", script.validate)

    def run(train_chips, val_chips):
        return T.fine_tune(
            train_chips,
            val_chips,
            labels,
            {"epochs": 2, "batch_size": 2, "amp": False},
            weights=weights,
            out_dir=tmp_path / "ft",
            num_workers=0,
            tmp_dir=tmp_path / "tmp",
        )

    first = run(chips[:3], chips[3:])
    assert run(chips[:3], chips[3:]).history == first.history and script.calls == 4  # same data: finished run
    with pytest.raises(ValueError, match="manifest hash changed"):
        run(chips[:2], chips[2:])  # a chip moved from train to val
    labels.write_text(json.dumps(geo.feature_collection([geo.box_polygon((5, 5, 25, 25))])))
    with pytest.raises(ValueError, match="manifest hash changed"):
        run(chips[:3], chips[3:])


@pytest.mark.parametrize("saved_enabled", [False, True])
def test_scaler_state_crosses_amp_modes(saved_enabled):
    # a run saved on CPU or with bf16 (disabled scaler) resumes under fp16 (enabled), and the reverse
    saved = torch.amp.GradScaler("cpu", enabled=saved_enabled).state_dict()
    target = torch.amp.GradScaler("cpu", enabled=not saved_enabled)
    T._load_scaler_state(target, saved)
    assert target.is_enabled() is not saved_enabled


def test_forced_fresh_start_supersedes_the_old_run(monkeypatch, regions, weights, tmp_path, caplog):
    out = tmp_path / "run"
    first = _run(monkeypatch, Script([0.1, 0.2, 0.3]), regions, weights, out)
    old_names = sorted(p.name for p in out.iterdir() if p.is_file())
    with caplog.at_level(logging.WARNING, logger="fair_trees.train"):
        second = _run(monkeypatch, Script([0.4, 0.5, 0.6]), regions, weights, out, resume=False, force=True)
    assert sorted(p.name for p in (out / "superseded-0").iterdir()) == old_names
    assert "superseded-0" in caplog.text
    assert first.last_checkpoint.name in {p.name for p in out.glob("ckpt-*.pt")}  # the new run's own
    assert [h["val/f1_max"] for h in second.history] == [0.4, 0.5, 0.6]


def test_resume_without_a_readable_checkpoint_is_refused(monkeypatch, regions, weights, tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    for name in (T.checkpoint_name(0, 2), T.checkpoint_name(1, 0)):
        (out / name).write_bytes(b"not a checkpoint")
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    script = Script([0.1] * 3)
    with pytest.raises(RuntimeError, match="no readable checkpoint"):
        _run(monkeypatch, script, regions, weights, out)
    assert script.calls == 0 and {p.name: p.read_bytes() for p in out.iterdir()} == before


def _stub_reports(monkeypatch, by_source):
    """``by_source`` maps a source, or ``(source, class_index)``, to the (f1_max, ap40) it reports."""
    calls = []

    def fake_evaluate_units(run_fn, units, params, *, class_index=0):
        calls.append([u.meta["source"] for u in units])
        source = units[0].meta["source"]
        f1, ap = by_source.get((source, class_index)) or by_source[source]
        pick_sweep = [{"threshold": 0.25, "f1": f1}] if source == "big" else []
        rows = [{"tp": 300, "fn": 0}] if source == "big" else []
        return SimpleNamespace(
            f1_max=f1, conf_at_max=0.3, ap40=ap, aoi_f1=f1, units=rows, pick_sweep=pick_sweep
        )

    monkeypatch.setattr(T, "evaluate_units", fake_evaluate_units)
    return calls


def _units(*sources):
    return [SimpleNamespace(meta={"source": s}) for s in sources]


def test_validate_balances_sources(monkeypatch):
    calls = _stub_reports(monkeypatch, {"a": (0.8, 0.6), "b": (0.2, 0.4)})

    units = {"tree": _units("a", "b", "a", "b")}
    v = T._validate(torch.nn.Identity(), torch.device("cpu"), units, InferenceParams(), 0)

    assert v["f1_max"] == pytest.approx(0.5) and v["ap40"] == pytest.approx(0.5)
    assert v["f1_max_a"] == 0.8 and v["f1_max_b"] == 0.2 and v["ap40_a"] == 0.6
    # MLflow's file store keeps a metric as a file, so no key may also be the folder of another
    assert not any(k2.startswith(f"{k1}/") for k1 in v for k2 in v)
    assert sorted(calls) == [["a", "a"], ["b", "b"]]


def test_validate_single_source_unchanged(monkeypatch):
    calls = _stub_reports(monkeypatch, {"a": (0.7, 0.5)})

    v = T._validate(
        torch.nn.Identity(), torch.device("cpu"), {"tree": _units("a", "a")}, InferenceParams(), 0
    )

    assert calls == [["a", "a"]]
    assert set(v) == {"f1_max", "conf_at_max", "ap40", "aoi_f1", "seconds", "threshold_tree"}
    assert v["f1_max"] == 0.7
    assert v["threshold_tree"] == 0.15  # no ground truth to pick from: the one-class prior


def test_validate_means_over_source_and_class(monkeypatch):
    calls = _stub_reports(monkeypatch, {("hot", 0): (0.6, 0.5), ("hot", 1): (0.2, 0.1)})
    units = {"tree": _units("hot", "hot"), "tree_group": _units("hot")}

    v = T._validate(torch.nn.Identity(), torch.device("cpu"), units, InferenceParams(), 0)

    assert v["f1_max"] == pytest.approx(0.4) and v["ap40"] == pytest.approx(0.3)
    assert v["f1_max_hot"] == 0.6 and v["f1_max_hot_tree_group"] == 0.2 and v["ap40_hot_tree_group"] == 0.1
    assert sorted(calls) == [["hot"], ["hot", "hot"]]


def test_validate_picks_a_threshold_per_class(monkeypatch):
    _stub_reports(monkeypatch, {"big": (0.6, 0.5), "small": (0.4, 0.3)})
    units = {"tree": _units("big", "small"), "tree_group": _units("small")}

    v = T._validate(torch.nn.Identity(), torch.device("cpu"), units, InferenceParams(), 0)

    # trees: a pick (0.25) from the large source and the two-class prior (0.2) from the small one, averaged
    assert v["threshold_tree"] == pytest.approx(0.225) and v["threshold_tree_group"] == 0.2


def test_best_epoch_thresholds_are_saved(monkeypatch, regions, weights, tmp_path):
    script = Script([0.5, 0.6, 0.55])
    validate = script.validate
    script.validate = lambda model, device, units, params, epoch: (
        validate(model, device, units, params, epoch) | {"threshold_tree": 0.1 + 0.05 * epoch}
    )
    res = _run(monkeypatch, script, regions, weights, tmp_path / "run")
    assert res.thresholds == {"tree": pytest.approx(0.15)}
    assert json.loads((tmp_path / "run" / "thresholds.json").read_text()) == {"tree": pytest.approx(0.15)}
