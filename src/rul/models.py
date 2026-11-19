"""Model factory: baselines and XGBoost, all with fixed seeds.

Every model here consumes the output of ``RULFeatures``; ``make_pipeline``
chains the two into the single object that is trained, logged, and served.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from rul import config
from rul.features import FeatureConfig, RULFeatures

MODEL_NAMES = ("mean", "ridge", "random_forest", "xgboost")

# Sensible untuned defaults. Tuned XGBoost parameters override these.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "mean": {},
    "ridge": {"alpha": 1.0},
    "random_forest": {
        "n_estimators": 200,
        "max_features": 0.3,
        "min_samples_leaf": 5,
    },
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "reg_lambda": 1.0,
    },
}


def make_model(
    name: str, params: dict[str, Any] | None = None, seed: int = config.SEED
) -> RegressorMixin:
    """Build an unfitted regressor that works on ``RULFeatures`` output.

    Args:
        name: One of ``MODEL_NAMES``.
        params: Overrides for ``DEFAULT_PARAMS[name]``.
        seed: Random seed for models that use randomness.
    """
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}, expected one of {MODEL_NAMES}")
    kwargs = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "ridge":
        # Scaling matters for the ridge penalty (cycle count is on a different scale).
        return make_pipeline(StandardScaler(), Ridge(**kwargs))
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **kwargs)
    return XGBRegressor(
        objective="reg:squarederror", tree_method="hist", random_state=seed, n_jobs=-1, **kwargs
    )


def make_pipeline_for(
    name: str,
    params: dict[str, Any] | None = None,
    feature_config: FeatureConfig | None = None,
    seed: int = config.SEED,
) -> Pipeline:
    """Raw C-MAPSS rows in, RUL out: ``RULFeatures`` followed by the model."""
    return Pipeline(
        [("features", RULFeatures(feature_config)), ("model", make_model(name, params, seed))]
    )
