"""Engine-level splits and the official test set.

Rows of one engine are strongly correlated: consecutive cycles have nearly
identical sensor values and RUL values that differ by 1. A random row split
puts neighbouring cycles of the same engine in both train and validation, so
the model can "look up" an engine it has already seen and validation scores
become optimistic. Splitting by engine measures what we care about:
performance on engines the model has never seen.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from rul import config


class LeakageError(RuntimeError):
    """Raised when the same engine appears on both sides of a split."""


def engine_folds(
    df: pd.DataFrame, n_splits: int = config.N_FOLDS, seed: int = config.SEED
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_index, validation_index) positions, grouped by engine.

    Every engine lands in exactly one validation fold. Engines are shuffled
    with a fixed seed, so folds are reproducible but not tied to unit order.

    Raises:
        ValueError: if there are fewer engines than folds.
    """
    groups = df["unit"].to_numpy()
    n_engines = len(np.unique(groups))
    if n_engines < n_splits:
        raise ValueError(f"need at least {n_splits} engines for {n_splits} folds, got {n_engines}")
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(df, groups=groups):
        assert_no_engine_overlap(df.iloc[train_idx], df.iloc[val_idx])
        yield train_idx, val_idx


def assert_no_engine_overlap(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Fail loudly if any engine appears in both frames.

    Called inside ``engine_folds`` as a runtime guard, so a future refactor
    cannot silently reintroduce leakage.
    """
    shared = set(train["unit"]) & set(validation["unit"])
    if shared:
        raise LeakageError(f"engines in both train and validation: {sorted(shared)}")


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """Return the last observed row of each engine, indexed by unit.

    For the official test set this is the only row with a known label. Apply it
    after feature engineering, so each row's features still summarize the
    engine's full observed history.
    """
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].set_index("unit").sort_index()


def official_test_set(test_df: pd.DataFrame, true_rul: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Align the last cycle of each test engine with its true RUL.

    The official test set is used only for the final evaluation, never for
    tuning. The true RUL is returned uncapped: the cap is a modeling choice,
    and evaluating against the raw values keeps its cost visible.

    Raises:
        ValueError: if the test engines and the RUL file do not match.
    """
    last = last_cycle(test_df)
    if not last.index.equals(true_rul.index):
        raise ValueError("test engines and true RUL units do not match")
    return last, true_rul
