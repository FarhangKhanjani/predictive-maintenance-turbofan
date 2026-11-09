"""Time-series features per engine, computed without looking into the future.

One rule makes training and serving identical: every feature at cycle t is
computed only from the engine's rows in the last ``lookback`` cycles up to and
including t. So the API needs just the most recent ``lookback`` readings of an
engine to reproduce exactly the features the model was trained on, and the
same ``RULFeatures`` object runs in both places (no training-serving skew).

Pipeline inside ``RULFeatures``:
1. ``fit`` learns, from training engines only, which sensors carry information
   (enough distinct values) and each sensor's mean and standard deviation.
2. ``transform`` standardizes those sensors, then builds per-engine features:
   current value, rolling mean/std/min/max over several windows, an
   exponentially weighted mean, a least-squares slope, and the cycle number.

All window statistics are vectorized over a (rows x lags x sensors) array, so
the transformer is cheap enough to refit inside every cross-validation fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from rul import config


@dataclass(frozen=True)
class FeatureConfig:
    """Which features to build. Logged with every experiment run.

    Attributes:
        windows: Window sizes (cycles) for rolling mean, std, min, and max.
        ewm_spans: Spans for exponentially weighted means, truncated at ``lookback``.
        slope_windows: Window sizes for the least-squares slope (trend per cycle).
        lookback: Maximum history used by any feature; also the number of
            recent cycles the API needs per engine for exact parity.
        min_unique: Sensors with fewer distinct training values are dropped.
        include_cycle: Add the cycle number (engine age) as a feature.
    """

    windows: tuple[int, ...] = (5, 10, 20, 30)
    ewm_spans: tuple[int, ...] = (10,)
    slope_windows: tuple[int, ...] = (10, 30)
    lookback: int = 30
    min_unique: int = 3
    include_cycle: bool = True

    def __post_init__(self) -> None:
        sizes = [*self.windows, *self.slope_windows]
        if any(w < 2 for w in sizes):
            raise ValueError("window sizes must be at least 2")
        if sizes and max(sizes) > self.lookback:
            raise ValueError(f"window sizes must not exceed lookback={self.lookback}")
        if any(s < 1 for s in self.ewm_spans):
            raise ValueError("ewm spans must be at least 1")

    def to_dict(self) -> dict[str, object]:
        """Plain dict for experiment logging."""
        return asdict(self)


def _lag_stack(values: np.ndarray, position: np.ndarray, lookback: int) -> np.ndarray:
    """Build a (rows, lookback, sensors) array of past values, NaN where unavailable.

    ``stack[i, k]`` is the reading k cycles before row i of the same engine.
    Rows must be sorted by engine and cycle; ``position`` is each row's index
    within its engine (0 for the first cycle), which stops lags crossing engines.
    """
    lags = np.arange(lookback)
    source = np.arange(len(values))[:, None] - lags[None, :]
    available = position[:, None] >= lags[None, :]
    stack = values[np.where(available, source, 0)]
    stack[~available] = np.nan
    return stack


def _window_stats(window: np.ndarray) -> dict[str, np.ndarray]:
    """Mean, sample std, min, and max over axis 1, ignoring NaN (lag 0 is never NaN)."""
    count = np.sum(~np.isnan(window), axis=1)
    mean = np.nanmean(window, axis=1)
    squared = np.nansum((window - mean[:, None, :]) ** 2, axis=1)
    std = np.sqrt(np.divide(squared, count - 1, out=np.zeros_like(squared), where=count > 1))
    return {
        "mean": mean,
        "std": std,
        "min": np.nanmin(window, axis=1),
        "max": np.nanmax(window, axis=1),
    }


def _ewm(stack: np.ndarray, span: int) -> np.ndarray:
    """Exponentially weighted mean over the lag axis, like pandas ``ewm(adjust=True)``."""
    alpha = 2.0 / (span + 1.0)
    weights = (1.0 - alpha) ** np.arange(stack.shape[1])
    weights = np.where(np.isnan(stack), 0.0, weights[None, :, None])
    return np.nansum(stack * weights, axis=1) / weights.sum(axis=1)


def _slope(window: np.ndarray) -> np.ndarray:
    """Least-squares slope per cycle over axis 1 (0 when fewer than 2 points)."""
    time = -np.arange(window.shape[1], dtype=float)[None, :, None]
    valid = ~np.isnan(window)
    count = valid.sum(axis=1, keepdims=True)
    t = np.where(valid, time, 0.0)
    t_mean = t.sum(axis=1, keepdims=True) / count
    y_mean = np.nanmean(window, axis=1, keepdims=True)
    dt = np.where(valid, time - t_mean, 0.0)
    dy = np.where(valid, window - y_mean, 0.0)
    num = (dt * dy).sum(axis=1)
    den = (dt**2).sum(axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


class RULFeatures(TransformerMixin, BaseEstimator):
    """scikit-learn transformer: raw C-MAPSS rows in, one feature row per input row out.

    Input must contain ``unit``, ``cycle`` and the sensor columns, with
    consecutive cycles per engine. Output keeps the input's row order and index.
    """

    def __init__(self, feature_config: FeatureConfig | None = None) -> None:
        self.feature_config = feature_config

    def _config(self) -> FeatureConfig:
        return self.feature_config or FeatureConfig()

    def fit(self, X: pd.DataFrame, y: object = None) -> RULFeatures:  # noqa: N803
        """Learn informative sensors and their scaling from training rows only."""
        _require_columns(X, config.SENSOR_COLUMNS)
        n_unique = X[config.SENSOR_COLUMNS].nunique()
        self.sensors_ = [
            s for s in config.SENSOR_COLUMNS if n_unique[s] >= self._config().min_unique
        ]
        if not self.sensors_:
            raise ValueError("no informative sensors found in training data")
        self.mean_ = X[self.sensors_].mean().to_numpy()
        self.scale_ = X[self.sensors_].std().replace(0.0, 1.0).to_numpy()
        self.feature_names_ = self._feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Build features for every row, using only that engine's past ``lookback`` cycles."""
        check_is_fitted(self, "sensors_")
        _require_columns(X, [*config.ID_COLUMNS, *self.sensors_])
        cfg = self._config()
        if not X.index.is_unique:
            raise ValueError("input index must be unique")
        ordered = X.sort_values(["unit", "cycle"], kind="stable")
        steps = ordered.groupby("unit")["cycle"].diff().dropna()
        if (steps != 1).any():
            raise ValueError("cycles must be consecutive within each engine (no gaps or repeats)")
        values = (ordered[self.sensors_].to_numpy(dtype=float) - self.mean_) / self.scale_
        position = ordered.groupby("unit").cumcount().to_numpy()
        stack = _lag_stack(values, position, cfg.lookback)

        blocks: list[np.ndarray] = [values]
        for w in cfg.windows:
            stats = _window_stats(stack[:, :w])
            blocks += [stats["mean"], stats["std"], stats["min"], stats["max"]]
        blocks += [_ewm(stack, span) for span in cfg.ewm_spans]
        blocks += [_slope(stack[:, :w]) for w in cfg.slope_windows]
        if cfg.include_cycle:
            blocks.append(ordered[["cycle"]].to_numpy(dtype=float))

        features = pd.DataFrame(np.hstack(blocks), index=ordered.index, columns=self.feature_names_)
        return features.loc[X.index]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Feature names in output column order (scikit-learn API)."""
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)

    def _feature_names(self) -> list[str]:
        cfg = self._config()
        names = list(self.sensors_)
        for w in cfg.windows:
            for stat in ("mean", "std", "min", "max"):
                names += [f"{s}_{stat}_{w}" for s in self.sensors_]
        for span in cfg.ewm_spans:
            names += [f"{s}_ewm_{span}" for s in self.sensors_]
        for w in cfg.slope_windows:
            names += [f"{s}_slope_{w}" for s in self.sensors_]
        if cfg.include_cycle:
            names.append("cycle")
        return names


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
