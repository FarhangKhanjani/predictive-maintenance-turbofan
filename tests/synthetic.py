"""Small synthetic C-MAPSS-like fleets and fast settings shared by the tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rul import config
from rul.features import FeatureConfig

FAST = FeatureConfig(windows=(3,), ewm_spans=(), slope_windows=(3,), lookback=3)
FAST_PARAMS = {
    "random_forest": {"n_estimators": 10},
    "xgboost": {"n_estimators": 20},
}


def degrading_fleet(n_engines: int, seed: int = 0, truncate: bool = False) -> pd.DataFrame:
    """Engines whose sensor s2 drifts upwards towards failure; other columns are noise.

    With ``truncate`` the series stop before failure, like the official test set.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_engines + 1):
        life = int(rng.integers(40, 80))
        if truncate:
            life = int(rng.integers(15, life))
        cycle = np.arange(1, life + 1)
        df = pd.DataFrame(rng.normal(size=(life, config.N_SENSORS)), columns=config.SENSOR_COLUMNS)
        for column in config.SETTING_COLUMNS:
            df.insert(0, column, rng.normal(scale=0.01, size=life))
        df["s2"] = np.exp(cycle / 20) + rng.normal(scale=0.1, size=life)
        df.insert(0, "cycle", cycle)
        df.insert(0, "unit", unit)
        frames.append(df[list(config.COLUMNS)])  # same column order as the real files
    return pd.concat(frames, ignore_index=True)


def truncated_fleet(n_engines: int = 5, seed: int = 1) -> tuple[pd.DataFrame, pd.Series]:
    """Truncated engines plus a matching true-RUL series indexed by unit."""
    raw = degrading_fleet(n_engines, seed=seed, truncate=True)
    true_rul = pd.Series(
        np.arange(10, 10 + n_engines),
        index=pd.RangeIndex(1, n_engines + 1, name="unit"),
        name="rul",
    )
    return raw, true_rul
