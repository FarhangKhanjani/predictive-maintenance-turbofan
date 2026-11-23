"""MLflow experiment tracking and model registry helpers.

Every experiment run records enough to be reproduced and audited:
- params: model hyperparameters (defaults merged in), feature config, RUL cap,
  seed, number of folds, dataset subset
- tags: git commit (and whether the tree was dirty), data checksum, run kind
- metrics: all CV and test metrics from ``rul.train``
- artifacts: the fitted pipeline with signature and input example, the feature
  config as JSON, and diagnostic plots

Promotion is a separate step: ``promote`` reads the latest benchmark from the
tracking server and moves the ``champion`` and ``challenger`` aliases. Serving
code loads ``models:/<name>@champion`` and never needs a version number.

The tracking URI comes from ``MLFLOW_TRACKING_URI`` and defaults to a local
SQLite file (``mlflow.db``) with artifacts in ``mlartifacts/``.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from rul import config
from rul.data import data_checksum
from rul.features import FeatureConfig
from rul.models import DEFAULT_PARAMS

DEFAULT_TRACKING_URI = f"sqlite:///{(config.PROJECT_ROOT / 'mlflow.db').as_posix()}"
SELECTION_RULE = "lowest cv_rmse among the runs of the latest benchmark"


def experiment_name(subset: str) -> str:
    """One MLflow experiment per dataset subset."""
    return f"turbofan-rul-{subset.lower()}"


def registered_model_name(subset: str) -> str:
    """One registered model per dataset subset."""
    return f"turbofan-rul-{subset.lower()}"


def artifact_location(tracking_uri: str) -> str | None:
    """Artifact folder for a new experiment.

    A SQLite store keeps artifacts in ``mlartifacts/`` next to the database
    file. A tracking server decides on its own, so this returns ``None``.
    """
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    return (Path(tracking_uri.removeprefix(prefix)).parent / "mlartifacts").as_uri()


def configure(subset: str, tracking_uri: str | None = None) -> str:
    """Point MLflow at the tracking store and select (or create) the experiment.

    Returns the experiment id.
    """
    mlflow.set_tracking_uri(
        tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    )
    name = experiment_name(subset)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name, artifact_location=artifact_location(mlflow.get_tracking_uri())
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def git_info() -> dict[str, str]:
    """Current commit hash and whether uncommitted changes exist ("unknown" outside git).

    Generated results under ``reports/`` do not count as changes: one command
    rewriting its JSON output must not mark the next command's runs as dirty.
    """

    def run(*args: str) -> str:
        out = subprocess.run(
            ["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    try:
        return {
            "git_commit": run("rev-parse", "HEAD"),
            "git_dirty": str(bool(run("status", "--porcelain", "--", ".", ":!reports"))),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}


def run_tags(subset: str, kind: str) -> dict[str, str]:
    """Tags shared by every run of one command invocation."""
    return {"kind": kind, "subset": subset, "data_sha256": data_checksum(subset), **git_info()}


@contextmanager
def parent_run(subset: str, kind: str, enabled: bool = True) -> Iterator[Any]:
    """Group the runs of one command (benchmark, ablation) under a parent run."""
    if not enabled:
        with nullcontext() as nothing:
            yield nothing
        return
    configure(subset)
    tags = run_tags(subset, kind) | {"role": "parent"}
    with mlflow.start_run(run_name=f"{kind} {subset}", tags=tags) as run:
        yield run


def flat_params(
    model_name: str,
    params: dict[str, Any] | None,
    feature_config: FeatureConfig | None,
    cap: int | None,
    subset: str,
) -> dict[str, Any]:
    """Everything that defines an experiment, flattened for ``mlflow.log_params``."""
    model_params = {**DEFAULT_PARAMS[model_name], **(params or {})}
    features = (feature_config or FeatureConfig()).to_dict()
    return {
        "model": model_name,
        **{f"model.{k}": v for k, v in model_params.items()},
        **{f"features.{k}": v for k, v in features.items()},
        "rul_cap": "none" if cap is None else cap,
        "seed": config.SEED,
        "n_folds": config.N_FOLDS,
        "subset": subset,
    }


def prediction_plot(truth: np.ndarray, pred: np.ndarray, threshold: float = 30.0) -> Figure:
    """Predicted vs true RUL on the test engines, and the signed error per engine."""
    fig = Figure(figsize=(11, 4.2))
    left, right = fig.subplots(1, 2)
    left.scatter(truth, pred, s=14, alpha=0.7)
    top = float(max(truth.max(), pred.max())) + 5
    left.plot([0, top], [0, top], color="grey", lw=1, label="perfect")
    left.axhline(threshold, color="tab:red", lw=0.8, ls="--", label=f"alert at {threshold:.0f}")
    left.set(xlabel="true RUL (cycles)", ylabel="predicted RUL (cycles)", title="Test engines")
    left.legend(loc="upper left")

    order = np.argsort(truth)
    error = pred[order] - truth[order]
    colors = np.where(error > 0, "tab:red", "tab:blue")
    right.bar(np.arange(len(error)), error, color=colors, width=0.9)
    right.axhline(0, color="grey", lw=0.8)
    right.set(
        xlabel="test engines sorted by true RUL",
        ylabel="predicted - true (cycles)",
        title="Error per engine (red = late, the risky side)",
    )
    fig.tight_layout()
    return fig


def importance_plot(pipeline: Pipeline, top: int = 20) -> Figure | None:
    """Top feature importances for tree models; ``None`` for models without them."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    names = pipeline.named_steps["features"].get_feature_names_out()
    series = pd.Series(model.feature_importances_, index=names).nlargest(top)[::-1]
    fig = Figure(figsize=(7, 0.28 * len(series) + 1))
    ax = fig.subplots()
    ax.barh(series.index, series.to_numpy())
    ax.set(xlabel="importance", title=f"Top {len(series)} features")
    fig.tight_layout()
    return fig


def log_experiment(
    label: str,
    model_name: str,
    params: dict[str, Any] | None,
    feature_config: FeatureConfig | None,
    cap: int | None,
    subset: str,
    result: dict[str, Any],
    pipeline: Pipeline,
    example: pd.DataFrame,
    test_truth: np.ndarray,
    test_pred: np.ndarray,
    kind: str = "experiment",
) -> str:
    """Log one experiment as a (nested) MLflow run and return its run id."""
    with mlflow.start_run(run_name=label, nested=mlflow.active_run() is not None) as run:
        mlflow.set_tags({**run_tags(subset, kind), "label": label})
        mlflow.log_params(flat_params(model_name, params, feature_config, cap, subset))
        metrics = {
            k: float(v)
            for k, v in result.items()
            if isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict((feature_config or FeatureConfig()).to_dict(), "feature_config.json")

        fig = prediction_plot(test_truth, test_pred)
        mlflow.log_figure(fig, "plots/test_predictions.png")
        importance = importance_plot(pipeline)
        if importance is not None:
            mlflow.log_figure(importance, "plots/feature_importance.png")

        # Raw C-MAPSS rows in, one RUL per row out: the same contract as the API.
        signature = infer_signature(example, pipeline.predict(example))
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=example,
            serialization_format="cloudpickle",
        )
        return run.info.run_id


def log_tuning(subset: str, best_params_file: Path, trials_file: Path) -> str:
    """Record a finished tuning study (best params, best CV RMSE, all trials) as one run."""
    configure(subset)
    stored = json.loads(best_params_file.read_text())
    best = {k: v for k, v in stored.items() if not k.startswith("_")}
    with mlflow.start_run(run_name=f"tuning {subset}", tags=run_tags(subset, "tuning")) as run:
        mlflow.log_params(
            {"model": "xgboost", "sampler": "optuna TPE", "n_trials": stored.get("_n_trials")}
            | {f"model.{k}": v for k, v in best.items()}
        )
        mlflow.log_metric("best_cv_rmse", float(stored["_cv_rmse"]))
        mlflow.log_artifact(str(best_params_file))
        mlflow.log_artifact(str(trials_file))
        return run.info.run_id


def latest_benchmark_runs(subset: str) -> pd.DataFrame:
    """Child runs of the most recent benchmark, best ``cv_rmse`` first."""
    experiment_id = configure(subset)
    parents = mlflow.search_runs(
        [experiment_id],
        filter_string="tags.kind = 'benchmark' AND tags.role = 'parent'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if parents.empty:
        raise LookupError(f"no benchmark runs found for {subset}; run 'rul-train benchmark' first")
    parent_id = parents.loc[0, "run_id"]
    children = mlflow.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
        order_by=["metrics.cv_rmse ASC"],
    )
    if children.empty:
        raise LookupError(f"benchmark run {parent_id} has no child runs")
    return children


def logged_model_uri(run_id: str, experiment_id: str) -> str:
    """``models:/m-...`` URI of the model a run logged (MLflow 3 logged models)."""
    models = mlflow.search_logged_models(
        experiment_ids=[experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
        output_format="list",
    )
    if not models:
        raise LookupError(f"run {run_id} did not log a model")
    return models[0].model_uri


def promote(subset: str) -> dict[str, str]:
    """Register the best two benchmark runs and point the aliases at them.

    The best run by ``cv_rmse`` becomes ``champion`` and the runner-up
    ``challenger``. The test set plays no part in the choice. Returns a mapping
    of alias to registered model version.
    """
    runs = latest_benchmark_runs(subset)
    name = registered_model_name(subset)
    client = MlflowClient()
    if not client.search_registered_models(filter_string=f"name = '{name}'"):
        client.create_registered_model(
            name,
            description=f"RUL regressor for C-MAPSS {subset}. Selection rule: {SELECTION_RULE}.",
        )
    promoted: dict[str, str] = {}
    for alias, (_, run) in zip(("champion", "challenger"), runs.head(2).iterrows(), strict=False):
        version = mlflow.register_model(
            logged_model_uri(run["run_id"], run["experiment_id"]),
            name,
            tags={
                "label": run["tags.label"],
                "cv_rmse": f"{run['metrics.cv_rmse']:.3f}",
                "git_commit": run["tags.git_commit"],
                "selection_rule": SELECTION_RULE,
            },
        )
        client.set_registered_model_alias(name, alias, version.version)
        promoted[alias] = version.version
    return promoted
