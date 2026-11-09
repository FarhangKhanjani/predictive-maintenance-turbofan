"""Tests for rul.features: correctness, no future peeking, and training-serving parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from rul import config
from rul.features import FeatureConfig, RULFeatures

SMALL = FeatureConfig(windows=(3,), ewm_spans=(4,), slope_windows=(3,), lookback=5)


def make_fleet(lifetimes: dict[int, int], seed: int = 0) -> pd.DataFrame:
    """Synthetic engines: s2 rises linearly, s3 is noise, s1 is constant, the rest noise."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit, life in lifetimes.items():
        cycle = np.arange(1, life + 1)
        df = pd.DataFrame(rng.normal(size=(life, config.N_SENSORS)), columns=config.SENSOR_COLUMNS)
        df.insert(0, "cycle", cycle)
        df.insert(0, "unit", unit)
        df["s1"] = 518.67
        df["s2"] = 2.0 * cycle + 10.0 * unit
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def fleet() -> pd.DataFrame:
    return make_fleet({1: 12, 2: 8, 3: 4})


def fit_transform(train: pd.DataFrame, cfg: FeatureConfig = SMALL) -> pd.DataFrame:
    return RULFeatures(cfg).fit(train).transform(train)


# --- correctness on known values ------------------------------------------------


def test_window_stats_match_pandas_rolling(fleet: pd.DataFrame) -> None:
    tf = RULFeatures(SMALL).fit(fleet)
    out = tf.transform(fleet)
    scaled = (fleet["s3"] - fleet["s3"].mean()) / fleet["s3"].std()
    rolling = scaled.groupby(fleet["unit"]).rolling(3, min_periods=1)
    for stat in ("mean", "min", "max"):
        expected = getattr(rolling, stat)().reset_index(level=0, drop=True)
        np.testing.assert_allclose(out[f"s3_{stat}_3"], expected.sort_index())
    expected_std = rolling.std().reset_index(level=0, drop=True).fillna(0.0).sort_index()
    np.testing.assert_allclose(out["s3_std_3"], expected_std)


def test_ewm_matches_pandas_when_history_fits_in_lookback() -> None:
    engine = make_fleet({1: 5})  # 5 cycles == lookback, so no truncation
    out = fit_transform(engine)
    scaled = (engine["s3"] - engine["s3"].mean()) / engine["s3"].std()
    np.testing.assert_allclose(out["s3_ewm_4"], scaled.ewm(span=4, adjust=True).mean())


def test_slope_recovers_linear_trend(fleet: pd.DataFrame) -> None:
    tf = RULFeatures(SMALL).fit(fleet)
    out = tf.transform(fleet)
    # s2 rises by 2 per cycle; after scaling the slope is 2 / std
    expected = 2.0 / fleet["s2"].std()
    later_rows = fleet.groupby("unit").cumcount() >= 1
    np.testing.assert_allclose(out.loc[later_rows, "s2_slope_3"], expected)
    assert (out.loc[~later_rows, "s2_slope_3"] == 0).all()


def test_first_cycle_has_finite_features(fleet: pd.DataFrame) -> None:
    assert np.isfinite(fit_transform(fleet).to_numpy()).all()


# --- leakage and parity ---------------------------------------------------------


def test_features_do_not_use_future_cycles(fleet: pd.DataFrame) -> None:
    """Features at cycle t are identical whether or not later cycles exist."""
    tf = RULFeatures(SMALL).fit(fleet)
    full = tf.transform(fleet)
    truncated = fleet[~((fleet.unit == 1) & (fleet.cycle > 6))]
    pd.testing.assert_frame_equal(tf.transform(truncated), full.loc[truncated.index])


def test_engines_do_not_influence_each_other(fleet: pd.DataFrame) -> None:
    tf = RULFeatures(SMALL).fit(fleet)
    alone = tf.transform(fleet[fleet.unit == 2])
    together = tf.transform(fleet).loc[alone.index]
    pd.testing.assert_frame_equal(alone, together)


def test_last_lookback_cycles_reproduce_training_features(fleet: pd.DataFrame) -> None:
    """Training-serving parity: the API sends only the last `lookback` cycles."""
    tf = RULFeatures(SMALL).fit(fleet)
    engine = fleet[fleet.unit == 1]
    full_history = tf.transform(engine).iloc[-1]
    recent_only = tf.transform(engine.tail(SMALL.lookback)).iloc[-1]
    pd.testing.assert_series_equal(full_history, recent_only)


def test_scaling_and_sensor_choice_come_from_fit_data_only(fleet: pd.DataFrame) -> None:
    tf = RULFeatures(SMALL).fit(fleet)
    mean_before, sensors_before = tf.mean_.copy(), list(tf.sensors_)
    other = make_fleet({7: 10}, seed=1)
    other["s1"] = np.arange(10.0)  # varies in new data, but was constant in training
    out = tf.transform(other)
    assert tf.sensors_ == sensors_before and "s1" not in tf.sensors_
    np.testing.assert_array_equal(tf.mean_, mean_before)
    assert not any(name.startswith("s1_") for name in out.columns)


# --- interface ---------------------------------------------------------------------


def test_output_keeps_input_order_and_index(fleet: pd.DataFrame) -> None:
    tf = RULFeatures(SMALL).fit(fleet)
    shuffled = fleet.sample(frac=1, random_state=0)
    out = tf.transform(shuffled)
    assert out.index.equals(shuffled.index)
    pd.testing.assert_frame_equal(out.sort_index(), tf.transform(fleet))


def test_feature_names_and_count(fleet: pd.DataFrame) -> None:
    tf = RULFeatures(SMALL).fit(fleet)
    n = len(tf.sensors_)  # 20: every sensor except constant s1
    assert n == config.N_SENSORS - 1
    assert len(tf.get_feature_names_out()) == n * (1 + 4 + 1 + 1) + 1
    assert list(tf.transform(fleet).columns) == list(tf.get_feature_names_out())


def test_rolling_features_can_be_switched_off(fleet: pd.DataFrame) -> None:
    raw_only = FeatureConfig(windows=(), ewm_spans=(), slope_windows=())
    out = fit_transform(fleet, raw_only)
    assert list(out.columns) == [*RULFeatures(raw_only).fit(fleet).sensors_, "cycle"]


def test_transform_before_fit_raises(fleet: pd.DataFrame) -> None:
    with pytest.raises(NotFittedError):
        RULFeatures().transform(fleet)


def test_gaps_in_cycles_are_rejected(fleet: pd.DataFrame) -> None:
    tf = RULFeatures(SMALL).fit(fleet)
    with pytest.raises(ValueError, match="consecutive"):
        tf.transform(fleet[fleet.cycle != 3])


def test_missing_columns_are_rejected(fleet: pd.DataFrame) -> None:
    tf = RULFeatures(SMALL).fit(fleet)
    with pytest.raises(ValueError, match="missing columns"):
        tf.transform(fleet.drop(columns=["cycle"]))


@pytest.mark.parametrize(
    "kwargs",
    [{"windows": (40,)}, {"slope_windows": (1,)}, {"ewm_spans": (0,)}],
)
def test_invalid_feature_config_is_rejected(kwargs: dict[str, tuple[int, ...]]) -> None:
    with pytest.raises(ValueError):
        FeatureConfig(**kwargs)
