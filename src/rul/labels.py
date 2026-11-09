"""Remaining useful life (RUL) targets for run-to-failure training data."""

from __future__ import annotations

import pandas as pd

from rul import config


def add_rul(
    df: pd.DataFrame, cap: int | None = config.RUL_CAP, column: str = config.TARGET_COLUMN
) -> pd.DataFrame:
    """Return a copy of ``df`` with a RUL target column.

    In the training files every engine runs until failure, so the RUL at a row
    is the engine's last cycle minus the current cycle (0 at the failure cycle).

    With ``cap`` set, the target is piecewise linear: constant at ``cap`` early
    in life, then decreasing linearly to 0. Early in life the sensors show no
    measurable degradation, so asking a model to tell 250 from 180 remaining
    cycles mostly teaches it noise; the cap encodes "healthy" as one value.

    Args:
        df: Training data with ``unit`` and ``cycle`` columns.
        cap: Maximum RUL value, or ``None`` for the raw linear target.
        column: Name of the target column to add.

    Raises:
        ValueError: if ``cap`` is not a positive integer.
    """
    if cap is not None and cap <= 0:
        raise ValueError(f"cap must be positive or None, got {cap}")
    out = df.copy()
    out[column] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    if cap is not None:
        out[column] = out[column].clip(upper=cap)
    return out
