"""Small MLflow helpers shared by the QLoRA training notebooks and scripts."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import mlflow
import yaml
from transformers import TrainerCallback


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(_flatten(item, name))
        elif isinstance(item, (str, int, float, bool)):
            flattened[name] = item
    return flattened


def load_experiment_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    for field in ("experiment_name", "run_name", "dataset_version", "base_model", "seed"):
        if field not in config:
            raise ValueError(f"Experiment config is missing {field!r}")
    return config


def start_run(project_root: Path, config_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Start an MLflow run and log immutable setup evidence."""
    config = load_experiment_config(config_path)
    tracking_uri = f"sqlite:///{(project_root / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["experiment_name"])
    if mlflow.active_run():
        raise RuntimeError("An MLflow run is already active; end it before starting a new training run.")
    mlflow.start_run(run_name=config["run_name"])
    mlflow.set_tags(
        {
            "run_kind": "qlora-training",
            "base_model": config["base_model"],
            "base_model_revision": str(config.get("base_model_revision", "unspecified")),
            "dataset_version": config["dataset_version"],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
    )
    mlflow.log_params(_flatten(config))
    mlflow.log_artifact(str(config_path), artifact_path="config")
    if manifest_path.is_file():
        mlflow.log_artifact(str(manifest_path), artifact_path="dataset")
    else:
        mlflow.set_tag("dataset_manifest", "missing")
    return config


class MLflowTrainerCallback(TrainerCallback):
    """Copy Trainer log events into the currently active MLflow run."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero and mlflow.active_run():
            metrics = {
                key: float(value)
                for key, value in logs.items()
                if isinstance(value, (int, float))
            }
            if metrics:
                mlflow.log_metrics(metrics, step=state.global_step)
        return control


def finish_run(run_dir: Path, metrics: dict[str, Any], *, log_adapter: bool = False) -> None:
    """Log durable training artifacts and close the active MLflow run."""
    if not mlflow.active_run():
        return
    numeric_metrics = {
        key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))
    }
    if numeric_metrics:
        mlflow.log_metrics(numeric_metrics)
    for name in ("run_config.json", "train_metrics.json"):
        path = run_dir / name
        if path.is_file():
            mlflow.log_artifact(str(path), artifact_path="reports")
    checkpoint_state = run_dir / "checkpoints" / "trainer_state.json"
    if checkpoint_state.is_file():
        mlflow.log_artifact(str(checkpoint_state), artifact_path="reports")
    adapter_dir = run_dir / "adapter"
    mlflow.set_tag("adapter_path", str(adapter_dir))
    if log_adapter and adapter_dir.is_dir():
        mlflow.log_artifacts(str(adapter_dir), artifact_path="adapter")
    mlflow.end_run(status="FINISHED")


def fail_run() -> None:
    """Mark an open run as failed when notebook training raises."""
    if mlflow.active_run():
        mlflow.end_run(status="FAILED")
