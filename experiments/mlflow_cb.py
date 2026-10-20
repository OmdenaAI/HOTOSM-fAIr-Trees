"""Per-epoch MLflow logging that survives a killed Colab runtime: params and metrics only, never artefacts.

The MLflow run id is kept in the run's ``env.json``, so a resumed training run logs into the same MLflow
run. Each call opens and closes the run, so a runtime killed between epochs leaves no run dangling.
"""

import json
import logging
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_PARAM_CHARS = 500
_BAD_NAME_CHARS = re.compile(r"[^A-Za-z0-9_\-. :/]")  # MLflow accepts only these in metric and param names


def _name(key: str) -> str:
    """``key`` with each character MLflow rejects (e.g. the ``,`` and ``&`` in biome names) as ``_``."""
    return _BAD_NAME_CHARS.sub("_", key)


def _flatten(params: Mapping[str, object], prefix: str = "") -> dict[str, str]:
    """Nested dicts joined with ``.``; every value as a string cut to MLflow's param length."""
    flat = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[_name(name)] = str(value)[:MAX_PARAM_CHARS]
    return flat


def make_mlflow_callback(
    tracking_dir: Path, experiment: str, env_path: Path, params: Mapping[str, object]
) -> Callable[[dict, int], None]:
    """An ``on_epoch_end(metrics, epoch)`` that logs to a file-store MLflow under ``tracking_dir``."""

    def callback(metrics: dict, epoch: int) -> None:
        import mlflow

        # newer MLflow (3.17.1 on Colab) refuses a file store unless this opts in; a folder on Drive
        # needs no database, and SQLite on the Drive mount risks locking errors
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        metrics = {_name(k): v for k, v in metrics.items()}
        mlflow.set_tracking_uri(Path(tracking_dir).resolve().as_uri())
        mlflow.set_experiment(experiment)
        env = json.loads(env_path.read_text())
        run_id = env.get("mlflow_run_id")
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(metrics, step=epoch)
            return
        with mlflow.start_run(run_name=params.get("run_id")) as run:
            env["mlflow_run_id"] = run.info.run_id
            env_path.write_text(json.dumps(env, indent=2) + "\n")
            logger.info(f"new MLflow run {run.info.run_id}")
            mlflow.log_params(_flatten(params))
            mlflow.log_metrics(metrics, step=epoch)

    return callback
