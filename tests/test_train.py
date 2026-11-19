"""Tests for rul.models and rul.train on a small synthetic fleet (no real data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from rul import config
from rul.features import FeatureConfig
from rul.models import MODEL_NAMES, make_model, make_pipeline_for
from rul.train import cross_validate, evaluate_on_test, fit_final, make_folds

FAST = FeatureConfig(windows=(3,), ewm_spans=(), slope_windows=(3,), lookback=3)
FAST_PARAMS = {
    "random_forest": {"n_estimators": 10},
    "xgboost": {"n_estimators": 20},
}


def degrading_fleet(n_engines: int, seed: int = 0, truncate: bool = False) -> pd.DataFrame:
    """Engines whose sensor s2 drifts upwards towards failure, plus noise sensors."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(40, 80))
        if truncate:
            life = int(rng.integers(15, life))
        cycle = np.arange(1, life + 1)
        df = pd.DataFrame(rng.normal(size=(life, config.N_SENSORS)), columns=config.SENSOR_COLUMNS)
        df["s2"] = np.exp(cycle / 20) + rng.normal(scale=0.1, size=life)
        df.insert(0, "cycle", cycle)
        df.insert(0, "unit", unit)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def train_raw() -> pd.DataFrame:
    return degrading_fleet(12)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_model_fits_and_predicts(name: str, train_raw: pd.DataFrame) -> None:
    folds = make_folds(train_raw, FAST, cap=40, n_splits=3)
    result = cross_validate(make_model(name, FAST_PARAMS.get(name)), folds)
    assert np.isfinite(result["cv_rmse"]) and result["cv_rmse"] >= 0


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_model("svm")


def test_models_are_reproducible_with_fixed_seed(train_raw: pd.DataFrame) -> None:
    folds = make_folds(train_raw, FAST, cap=40, n_splits=3)
    model = make_model("xgboost", FAST_PARAMS["xgboost"])
    assert cross_validate(model, folds)["cv_rmse"] == cross_validate(model, folds)["cv_rmse"]


def test_folds_fit_features_on_training_engines_only(train_raw: pd.DataFrame) -> None:
    folds = make_folds(train_raw, FAST, cap=40, n_splits=3)
    for fold in folds:
        assert set(fold.X_train.index).isdisjoint(fold.X_val.index)
        assert len(fold.X_train) + len(fold.X_val) == len(train_raw)
    # Validation rows differ per fold, so the learned scaling must differ too.
    first_values = [fold.X_train["s2"].iloc[0] for fold in folds]
    assert len(set(np.round(first_values, 10))) > 1


def test_cap_applies_to_training_target_but_not_true_rul(train_raw: pd.DataFrame) -> None:
    fold = make_folds(train_raw, FAST, cap=40, n_splits=3)[0]
    assert fold.y_train.max() == 40
    assert fold.y_val_true.max() > 40
    late = fold.y_val_true <= 40
    assert (fold.y_val[late] == fold.y_val_true[late]).all()


def test_signal_beats_the_mean_baseline(train_raw: pd.DataFrame) -> None:
    folds = make_folds(train_raw, FAST, cap=40, n_splits=3)
    baseline = cross_validate(make_model("mean"), folds)["cv_rmse"]
    model = cross_validate(make_model("xgboost", FAST_PARAMS["xgboost"]), folds)["cv_rmse"]
    assert model < baseline


def test_final_pipeline_scores_last_cycle_of_each_test_engine(train_raw: pd.DataFrame) -> None:
    pipeline = fit_final("ridge", None, train_raw, FAST, cap=40)
    test_raw = degrading_fleet(5, seed=1, truncate=True)
    true_rul = pd.Series(np.arange(10, 15), index=pd.RangeIndex(1, 6, name="unit"), name="rul")
    result = evaluate_on_test(pipeline, test_raw, true_rul)
    assert {"test_rmse", "test_mae", "test_nasa", "test_alert_recall"} <= set(result)
    assert result["test_rmse"] >= 0


def test_pipeline_is_clonable_for_cross_validation() -> None:
    pipeline = make_pipeline_for("xgboost", FAST_PARAMS["xgboost"], FAST)
    cloned = clone(pipeline)
    assert cloned.named_steps["features"].feature_config == FAST
