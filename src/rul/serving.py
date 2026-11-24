"""Model loading and prediction logic behind the API, kept free of web concerns.

The served object is the full training pipeline (``RULFeatures`` followed by
the regressor), loaded once from the MLflow registry by alias. Requests are
turned back into the raw C-MAPSS row format, so serving runs exactly the
feature code that training ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rul import config
from rul.features import FeatureConfig
from rul.schemas import EngineHistory, ModelInfo, Prediction


@dataclass(frozen=True)
class LoadedModel:
    """A fitted pipeline plus the metadata reported by ``/model-info``."""

    pipeline: Any  # sklearn Pipeline; anything with ``predict(DataFrame)`` works
    info: ModelInfo


def load_from_registry(
    name: str, alias: str, alert_threshold: float, tracking_uri: str | None = None
) -> LoadedModel:
    """Resolve ``models:/<name>@<alias>`` and load the pipeline it points to."""
    import mlflow  # imported here so the API module stays importable without a store
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    version = MlflowClient().get_model_version_by_alias(name, alias)
    pipeline = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    features = pipeline.named_steps["features"]
    cv_rmse = version.tags.get("cv_rmse")
    info = ModelInfo(
        name=name,
        alias=alias,
        version=str(version.version),
        run_id=version.run_id or "unknown",
        git_commit=version.tags.get("git_commit", "unknown"),
        cv_rmse=float(cv_rmse) if cv_rmse is not None else None,
        selection_rule=version.tags.get("selection_rule", "unknown"),
        lookback=(features.feature_config or FeatureConfig()).lookback,
        n_features=len(features.get_feature_names_out()),
        alert_threshold=alert_threshold,
    )
    return LoadedModel(pipeline=pipeline, info=info)


def to_frame(engines: list[EngineHistory]) -> pd.DataFrame:
    """Raw C-MAPSS rows for all engines; engine i gets the internal unit number i + 1."""
    frames = []
    for unit, engine in enumerate(engines, start=1):
        rows = pd.DataFrame([reading.model_dump() for reading in engine.cycles])
        rows.insert(0, "unit", unit)
        frames.append(rows)
    return pd.concat(frames, ignore_index=True)[list(config.COLUMNS)]


def predict_engines(model: LoadedModel, engines: list[EngineHistory]) -> list[Prediction]:
    """Predict RUL at the last received cycle of every engine, in request order.

    One ``predict`` call covers the whole batch: features are computed per
    engine inside the pipeline, so engines never leak into each other.
    Predictions below zero are clipped, since "less than no life left" carries
    no extra meaning.
    """
    frame = to_frame(engines)
    pred = pd.Series(np.asarray(model.pipeline.predict(frame)).ravel(), index=frame.index)
    last_rows = frame.groupby("unit")["cycle"].idxmax().sort_index()
    threshold = model.info.alert_threshold
    predictions = []
    for engine, row in zip(engines, last_rows.to_numpy(), strict=True):
        rul = max(float(pred.loc[row]), 0.0)
        predictions.append(
            Prediction(
                engine_id=engine.engine_id,
                last_cycle=int(frame.loc[row, "cycle"]),
                predicted_rul=round(rul, 2),
                alert=rul <= threshold,
                model_version=model.info.version,
            )
        )
    return predictions
