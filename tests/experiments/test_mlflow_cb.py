import json
import os
import sys
from types import SimpleNamespace

from experiments.mlflow_cb import make_mlflow_callback


class FakeMlflow:
    def __init__(self):
        self.log = []

    def set_tracking_uri(self, uri):
        self.log.append(("uri", uri, os.environ.get("MLFLOW_ALLOW_FILE_STORE")))

    def set_experiment(self, name):
        self.log.append(("experiment", name))

    def start_run(self, run_id=None, run_name=None):
        self.log.append(("start", run_id, run_name))
        fake = self

        class Run:
            info = SimpleNamespace(run_id=run_id or "new-id")

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                fake.log.append(("end",))
                return False

        return Run()

    def log_params(self, params):
        self.log.append(("params", params))

    def log_metrics(self, metrics, step):
        self.log.append(("metrics", step, metrics))


def test_mlflow_callback_resumes_the_same_run(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    env = tmp_path / "env.json"
    env.write_text("{}")
    cb = make_mlflow_callback(tmp_path / "mlruns", "fair_trees", env, {"run_id": "r1", "hp": {"lr": 0.001}})
    cb({"val/f1_max": 0.5}, 0)
    cb({"val/f1_max": 0.6}, 1)
    assert json.loads(env.read_text())["mlflow_run_id"] == "new-id"
    assert [e for e in fake.log if e[0] == "start"] == [("start", None, "r1"), ("start", "new-id", None)]
    assert [e for e in fake.log if e[0] == "params"] == [("params", {"run_id": "r1", "hp.lr": "0.001"})]
    assert ("metrics", 1, {"val/f1_max": 0.6}) in fake.log
    assert fake.log.count(("end",)) == 2


def test_mlflow_callback_allows_the_file_store(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)
    env = tmp_path / "env.json"
    env.write_text("{}")
    make_mlflow_callback(tmp_path / "mlruns", "fair_trees", env, {"run_id": "r1"})({"val/f1_max": 0.5}, 0)
    # set before the tracking store is opened
    assert fake.log[0] == ("uri", (tmp_path / "mlruns").resolve().as_uri(), "true")


def test_mlflow_callback_replaces_characters_mlflow_rejects(monkeypatch, tmp_path):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    env = tmp_path / "env.json"
    env.write_text("{}")
    cb = make_mlflow_callback(tmp_path / "mlruns", "fair_trees", env, {"biome": {"A, B & C": 1}})
    cb({"sampler/draws/biome/Mediterranean Forests, Woodlands & Scrub": 3, "val/f1_max": 0.5}, 0)
    names = {"sampler/draws/biome/Mediterranean Forests_ Woodlands _ Scrub": 3, "val/f1_max": 0.5}
    assert ("metrics", 0, names) in fake.log
    assert ("params", {"biome.A_ B _ C": "1"}) in fake.log
