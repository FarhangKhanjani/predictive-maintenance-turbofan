"""Tests for rul.splits: engine-level folds, leakage guard, and test-set alignment."""

from __future__ import annotations

import pandas as pd
import pytest

from rul.splits import (
    LeakageError,
    assert_no_engine_overlap,
    engine_folds,
    last_cycle,
    official_test_set,
)


@pytest.fixture
def fleet() -> pd.DataFrame:
    """23 engines with different lifetimes (23 is deliberately not divisible by 5)."""
    rows = [(unit, cycle) for unit in range(1, 24) for cycle in range(1, 10 + unit)]
    return pd.DataFrame(rows, columns=["unit", "cycle"])


def test_no_engine_in_both_train_and_validation(fleet: pd.DataFrame) -> None:
    """The core leakage test: every fold keeps engines on one side only."""
    for train_idx, val_idx in engine_folds(fleet, n_splits=5, seed=0):
        train_units = set(fleet.iloc[train_idx]["unit"])
        val_units = set(fleet.iloc[val_idx]["unit"])
        assert train_units.isdisjoint(val_units)
        assert train_units | val_units == set(fleet["unit"])


def test_every_engine_is_validated_exactly_once(fleet: pd.DataFrame) -> None:
    val_units = [u for _, v in engine_folds(fleet, 5, 0) for u in fleet.iloc[v]["unit"].unique()]
    assert sorted(val_units) == sorted(fleet["unit"].unique())


def test_folds_cover_every_row_exactly_once(fleet: pd.DataFrame) -> None:
    val_rows = sorted(i for _, v in engine_folds(fleet, 5, 0) for i in v)
    assert val_rows == list(range(len(fleet)))


def test_folds_are_reproducible_and_depend_on_seed(fleet: pd.DataFrame) -> None:
    def val_sets(seed: int) -> list[set[int]]:
        return [set(fleet.iloc[v]["unit"]) for _, v in engine_folds(fleet, 5, seed)]

    assert val_sets(0) == val_sets(0)
    assert val_sets(0) != val_sets(1)


def test_too_few_engines_for_folds_is_rejected(fleet: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="engines"):
        list(engine_folds(fleet[fleet.unit <= 3], n_splits=5))


def test_leakage_guard_detects_shared_engine(fleet: pd.DataFrame) -> None:
    with pytest.raises(LeakageError, match=r"\[3\]"):
        assert_no_engine_overlap(fleet[fleet.unit <= 3], fleet[fleet.unit >= 3])


def test_last_cycle_returns_final_row_per_engine(fleet: pd.DataFrame) -> None:
    last = last_cycle(fleet.sample(frac=1, random_state=0))
    assert last.index.tolist() == list(range(1, 24))
    assert (last["cycle"] == last.index + 9).all()


def test_official_test_set_aligns_with_true_rul(fleet: pd.DataFrame) -> None:
    true_rul = pd.Series(range(100, 123), index=pd.RangeIndex(1, 24, name="unit"))
    features, target = official_test_set(fleet, true_rul)
    assert len(features) == len(target) == 23
    assert features.index.equals(target.index)


def test_official_test_set_rejects_mismatched_units(fleet: pd.DataFrame) -> None:
    true_rul = pd.Series(range(22), index=pd.RangeIndex(1, 23, name="unit"))
    with pytest.raises(ValueError, match="do not match"):
        official_test_set(fleet, true_rul)
