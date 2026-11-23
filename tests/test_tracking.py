"""Tests for rul.tracking against a throwaway SQLite tracking store."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from synthetic import FAST, FAST_PARAMS, degrading_fleet, truncated_fleet

from rul import tracking
from rul.train import Tracking, fit_final, make_folds, run_experiment


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One SQLite store per module: creating its schema takes a few seconds."""
    return tmp_path_factory.mktemp("mlflow") / "mlflow.db"


@pytest.fixture(autouse=True)
def isolated_tracking(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point MLflow at the temporary store; never touch the project's own mlflow.db."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{store.as_posix()}")
    # The synthetic data has no files on disk to checksum.
    monkeypatch.setattr(tracking, "data_checksum", lambda subset: "synthetic")


@pytest.fixture(scope="module")
def fleet() -> dict[str, pd.DataFrame | pd.Series]:
    test_raw, true_rul = truncated_fleet(5)
    return {"train": degrading_fleet(12), "test": test_raw, "true_rul": true_rul}


def run_benchmark(fleet: dict, models: list[str]) -> list[dict]:
    """A miniature benchmark: one parent run with one child run per model."""
    folds = make_folds(fleet["train"], FAST, cap=40, n_splits=3)
    with tracking.parent_run("FD001", "benchmark"):
        return [
            run_experiment(
                name,
                name,
                FAST_PARAMS.get(name),
                folds,
                fleet["train"],
                fleet["test"],
                fleet["true_rul"],
                FAST,
                40,
                track=Tracking("FD001", "benchmark"),
            )  # fmt: skip
            for name in models
        ]


def test_artifact_location_follows_the_sqlite_file(tmp_path: Path) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    assert tracking.artifact_location(uri) == (tmp_path / "mlartifacts").as_uri()
    assert tracking.artifact_location("http://mlflow:5000") is None


def test_flat_params_merge_defaults_and_describe_features() -> None:
    params = tracking.flat_params("xgboost", {"max_depth": 3}, FAST, None, "FD001")
    assert params["model.max_depth"] == 3
    assert params["model.n_estimators"] == 500  # default kept
    assert params["features.lookback"] == 3
    assert params["rul_cap"] == "none"


def test_run_records_params_metrics_tags_and_a_loadable_model(fleet: dict) -> None:
    (row,) = run_benchmark(fleet, ["ridge"])
    run = MlflowClient().get_run(row["run_id"])

    assert run.data.params["model"] == "ridge"
    assert run.data.params["features.lookback"] == "3"
    assert run.data.params["rul_cap"] == "40"
    assert run.data.metrics["cv_rmse"] == pytest.approx(row["cv_rmse"])
    assert "test_nasa" in run.data.metrics
    assert len(run.data.tags["git_commit"]) == 40 or run.data.tags["git_commit"] == "unknown"
    assert run.data.tags["data_sha256"] == "synthetic"
    assert run.data.tags["kind"] == "benchmark"
    assert "mlflow.parentRunId" in run.data.tags

    artifacts = {a.path for a in MlflowClient().list_artifacts(row["run_id"])}
    assert {"feature_config.json", "plots"} <= artifacts

    # The logged model reproduces a freshly fitted pipeline exactly.
    loaded = mlflow.pyfunc.load_model(f"runs:/{row['run_id']}/model")
    expected = fit_final("ridge", None, fleet["train"], FAST, cap=40).predict(fleet["test"])
    np.testing.assert_allclose(np.asarray(loaded.predict(fleet["test"])).ravel(), expected)


def test_promote_points_champion_at_lowest_cv_rmse(fleet: dict) -> None:
    rows = run_benchmark(fleet, ["mean", "xgboost"])
    best, runner_up = sorted(rows, key=lambda r: r["cv_rmse"])

    promoted = tracking.promote("FD001")

    client = MlflowClient()
    name = tracking.registered_model_name("FD001")
    champion = client.get_model_version_by_alias(name, "champion")
    challenger = client.get_model_version_by_alias(name, "challenger")
    assert champion.version == promoted["champion"]
    assert champion.run_id == best["run_id"]
    assert challenger.run_id == runner_up["run_id"]
    assert champion.tags["selection_rule"] == tracking.SELECTION_RULE

    # Serving code only needs the alias, never a version number.
    model = mlflow.pyfunc.load_model(f"models:/{name}@champion")
    assert len(model.predict(fleet["test"])) == len(fleet["test"])


def test_promote_without_a_benchmark_fails_clearly() -> None:
    with pytest.raises(LookupError, match="no benchmark runs"):
        tracking.promote("FD002")


def test_log_tuning_records_best_score_and_trials(tmp_path: Path) -> None:
    best = tmp_path / "best_params.json"
    best.write_text(json.dumps({"max_depth": 4, "_cv_rmse": 12.5, "_n_trials": 3}))
    trials = tmp_path / "tuning_trials.csv"
    trials.write_text("number,value\n0,13.0\n1,12.5\n")

    run = MlflowClient().get_run(tracking.log_tuning("FD001", best, trials))

    assert run.data.metrics["best_cv_rmse"] == 12.5
    assert run.data.params["model.max_depth"] == "4"
    assert run.data.tags["kind"] == "tuning"
    artifacts = {a.path for a in MlflowClient().list_artifacts(run.info.run_id)}
    assert artifacts == {"best_params.json", "tuning_trials.csv"}
