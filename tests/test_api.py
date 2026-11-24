"""API tests with FastAPI's TestClient and a small model trained on synthetic data.

The app is built with an injected loader, so these tests need neither the real
dataset nor an MLflow store. One test at the end covers the real registry path
against a temporary SQLite store.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.tracking import MlflowClient
from synthetic import FAST, degrading_fleet, truncated_fleet

from rul import config
from rul.api import create_app
from rul.schemas import ModelInfo
from rul.serving import LoadedModel, load_from_registry
from rul.train import fit_final

THRESHOLD = 30.0


def info(version: str = "7") -> ModelInfo:
    return ModelInfo(
        name="test-model",
        alias="champion",
        version=version,
        run_id="run-123",
        git_commit="abc",
        cv_rmse=12.3,
        selection_rule="test",
        lookback=FAST.lookback,
        n_features=10,
        alert_threshold=THRESHOLD,
    )


@pytest.fixture(scope="module")
def pipeline() -> Any:
    return fit_final("ridge", None, degrading_fleet(12), FAST, cap=40)


@pytest.fixture(scope="module")
def test_raw() -> pd.DataFrame:
    return truncated_fleet(4)[0]


@pytest.fixture
def client(pipeline: Any) -> Iterator[TestClient]:
    app = create_app(loader=lambda: LoadedModel(pipeline=pipeline, info=info()))
    with TestClient(app) as test_client:  # "with" runs the startup (model loading)
        yield test_client


def engine_payload(raw: pd.DataFrame, unit: int, engine_id: str | None = None) -> dict:
    rows = raw[raw["unit"] == unit].drop(columns="unit")
    return {
        "engine_id": engine_id or f"engine-{unit}",
        "cycles": rows.to_dict(orient="records"),
    }


def expected_rul(pipeline: Any, raw: pd.DataFrame, unit: int) -> float:
    """What the training pipeline itself predicts at the engine's last cycle."""
    rows = raw[raw["unit"] == unit]
    return max(float(pipeline.predict(rows)[-1]), 0.0)


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert body["version"] == "7"
    assert body["alias"] == "champion"
    assert body["lookback"] == FAST.lookback


def test_predict_matches_the_training_pipeline(
    client: TestClient, pipeline: Any, test_raw: pd.DataFrame
) -> None:
    response = client.post("/predict", json=engine_payload(test_raw, unit=2))
    assert response.status_code == 200
    body = response.json()
    assert body["engine_id"] == "engine-2"
    assert body["last_cycle"] == int(test_raw.loc[test_raw["unit"] == 2, "cycle"].max())
    assert body["predicted_rul"] == pytest.approx(expected_rul(pipeline, test_raw, 2), abs=0.01)
    assert body["alert"] == (body["predicted_rul"] <= THRESHOLD)
    assert body["model_version"] == "7"


def test_batch_matches_single_predictions_in_request_order(
    client: TestClient, test_raw: pd.DataFrame
) -> None:
    units = [3, 1, 4]
    batch = {"engines": [engine_payload(test_raw, u) for u in units]}
    body = client.post("/predict-batch", json=batch).json()

    assert [p["engine_id"] for p in body["predictions"]] == [f"engine-{u}" for u in units]
    for unit, prediction in zip(units, body["predictions"], strict=True):
        single = client.post("/predict", json=engine_payload(test_raw, unit)).json()
        assert prediction == single  # batching must not mix engines


def test_negative_predictions_are_clipped_and_alert(test_raw: pd.DataFrame) -> None:
    class Pessimist:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.full(len(frame), -5.0)

    app = create_app(loader=lambda: LoadedModel(pipeline=Pessimist(), info=info()))
    with TestClient(app) as test_client:
        body = test_client.post("/predict", json=engine_payload(test_raw, 1)).json()
    assert body["predicted_rul"] == 0.0
    assert body["alert"] is True


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["cycles"].pop(1), "consecutive"),
        (lambda p: p["cycles"].reverse(), "consecutive"),
        (lambda p: p["cycles"][0].pop("s7"), "s7"),
        (lambda p: p["cycles"][0].update(s99=1.0), "s99"),
        (lambda p: p["cycles"][0].update(s2="hot"), "s2"),
        (lambda p: p["cycles"][0].update(cycle=0), "cycle"),
        (lambda p: p.update(cycles=[]), "cycles"),
        (lambda p: p.update(engine_id=""), "engine_id"),
    ],
    ids=["gap", "reversed", "missing-sensor", "unknown-field", "not-a-number",
         "cycle-zero", "no-cycles", "empty-id"],
)  # fmt: skip
def test_invalid_requests_are_rejected(
    client: TestClient, test_raw: pd.DataFrame, mutate: Any, message: str
) -> None:
    payload = engine_payload(test_raw, unit=1)
    mutate(payload)
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert message in response.text


def test_batch_rejects_duplicate_engine_ids(client: TestClient, test_raw: pd.DataFrame) -> None:
    batch = {"engines": [engine_payload(test_raw, 1, "x"), engine_payload(test_raw, 2, "x")]}
    response = client.post("/predict-batch", json=batch)
    assert response.status_code == 422
    assert "unique" in response.text


def test_app_does_not_start_without_a_model() -> None:
    def broken_loader() -> LoadedModel:
        raise RuntimeError("registry unreachable")

    with (
        pytest.raises(RuntimeError, match="registry unreachable"),
        TestClient(create_app(broken_loader)),
    ):
        pass


def test_load_from_registry_resolves_the_alias(
    pipeline: Any, test_raw: pd.DataFrame, tmp_path: Path
) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    experiment_id = mlflow.create_experiment("api-test", (tmp_path / "artifacts").as_uri())
    with mlflow.start_run(experiment_id=experiment_id):
        logged = mlflow.sklearn.log_model(
            pipeline, name="model", serialization_format="cloudpickle"
        )
    version = mlflow.register_model(logged.model_uri, "rul-test", tags={"cv_rmse": "11.5"})
    MlflowClient().set_registered_model_alias("rul-test", "champion", version.version)

    model = load_from_registry("rul-test", "champion", THRESHOLD, tracking_uri=uri)

    assert model.info.version == str(version.version)
    assert model.info.cv_rmse == 11.5
    assert model.info.lookback == FAST.lookback
    sample = test_raw[list(config.COLUMNS)]
    np.testing.assert_allclose(model.pipeline.predict(sample), pipeline.predict(sample))
