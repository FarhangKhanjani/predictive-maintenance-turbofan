"""Tests for rul.labels: RUL target construction and the piecewise-linear cap."""

from __future__ import annotations

import pandas as pd
import pytest

from rul.labels import add_rul


@pytest.fixture
def engines() -> pd.DataFrame:
    """Two engines with lifetimes 4 and 6 cycles."""
    return pd.DataFrame({"unit": [1] * 4 + [2] * 6, "cycle": [*range(1, 5), *range(1, 7)]})


def test_uncapped_rul_counts_down_to_zero(engines: pd.DataFrame) -> None:
    out = add_rul(engines, cap=None)
    assert out.loc[out.unit == 1, "rul"].tolist() == [3, 2, 1, 0]
    assert out.loc[out.unit == 2, "rul"].tolist() == [5, 4, 3, 2, 1, 0]


def test_cap_clips_early_life_only(engines: pd.DataFrame) -> None:
    out = add_rul(engines, cap=3)
    assert out.loc[out.unit == 1, "rul"].tolist() == [3, 2, 1, 0]
    assert out.loc[out.unit == 2, "rul"].tolist() == [3, 3, 3, 2, 1, 0]


def test_add_rul_does_not_modify_input(engines: pd.DataFrame) -> None:
    add_rul(engines)
    assert "rul" not in engines.columns


def test_rul_works_with_unsorted_rows(engines: pd.DataFrame) -> None:
    shuffled = engines.sample(frac=1, random_state=0)
    out = add_rul(shuffled, cap=None)
    assert (out["rul"] == out.groupby("unit")["cycle"].transform("max") - out["cycle"]).all()


@pytest.mark.parametrize("cap", [0, -5])
def test_invalid_cap_is_rejected(engines: pd.DataFrame, cap: int) -> None:
    with pytest.raises(ValueError, match="cap"):
        add_rul(engines, cap=cap)
