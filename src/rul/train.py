"""Controlled experiments: baselines, XGBoost tuning, and ablations.

Rules that keep comparisons fair:
- Every model sees the same engine-level folds (same seed) and the same metrics.
- Features are fit inside each fold on training engines only. They are built
  once per fold and shared by all models and tuning trials, which is
  equivalent to cross-validating the full pipeline but much cheaper.
- Model selection and tuning use cross-validation only. The official test set
  is scored once per final model and never feeds back into a choice.

Command line:
    uv run rul-train tune --trials 30   # Optuna search for XGBoost (CV only)
    uv run rul-train benchmark          # baselines vs XGBoost, CV and test
    uv run rul-train ablate             # feature, cap, and window ablations
    uv run rul-train promote            # register best benchmark run as @champion

Runs are logged to MLflow unless `--no-track` is given (see rul.tracking).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from rul import config
from rul.data import load_split, load_test_rul
from rul.evaluate import alert_metrics, mae, nasa_score, rmse
from rul.features import FeatureConfig, RULFeatures
from rul.labels import add_rul
from rul.models import make_model, make_pipeline_for
from rul.splits import engine_folds

REPORTS_DIR = config.PROJECT_ROOT / "reports"
BEST_PARAMS_FILE = REPORTS_DIR / "best_params.json"
TRIALS_FILE = REPORTS_DIR / "tuning_trials.csv"
LATE_LIFE = 125  # "late life" evaluation region: true RUL <= this value


@dataclass
class Fold:
    """Precomputed features and targets for one cross-validation fold."""

    X_train: pd.DataFrame  # noqa: N815
    y_train: pd.Series
    X_val: pd.DataFrame  # noqa: N815
    y_val: pd.Series  # training target (capped if a cap is used)
    y_val_true: pd.Series  # uncapped true RUL, same for every experiment


def make_folds(
    train_raw: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    cap: int | None = config.RUL_CAP,
    n_splits: int = config.N_FOLDS,
    seed: int = config.SEED,
) -> list[Fold]:
    """Split by engine and build features inside each fold."""
    labeled = add_rul(train_raw, cap=cap)
    true_rul = add_rul(train_raw, cap=None)[config.TARGET_COLUMN]
    folds = []
    for tr, va in engine_folds(labeled, n_splits, seed):
        train_part, val_part = labeled.iloc[tr], labeled.iloc[va]
        features = RULFeatures(feature_config).fit(train_part)
        folds.append(
            Fold(
                X_train=features.transform(train_part),
                y_train=train_part[config.TARGET_COLUMN],
                X_val=features.transform(val_part),
                y_val=val_part[config.TARGET_COLUMN],
                y_val_true=true_rul.iloc[va],
            )
        )
    return folds


def cross_validate(model: Any, folds: list[Fold]) -> dict[str, float]:
    """Fit a fresh clone of ``model`` per fold; return mean and std of fold metrics.

    ``rmse`` is measured against the training target. ``rmse_late`` uses the
    uncapped true RUL on rows with true RUL <= 125, so it is comparable across
    experiments with different caps and focuses on the part of life where
    maintenance decisions happen.
    """
    per_fold: dict[str, list[float]] = {"rmse": [], "mae": [], "nasa_mean": [], "rmse_late": []}
    start = time.perf_counter()
    for fold in folds:
        fitted = clone(model).fit(fold.X_train, fold.y_train)
        pred = fitted.predict(fold.X_val)
        late = fold.y_val_true.to_numpy() <= LATE_LIFE
        per_fold["rmse"].append(rmse(fold.y_val, pred))
        per_fold["mae"].append(mae(fold.y_val, pred))
        per_fold["nasa_mean"].append(nasa_score(fold.y_val, pred, reduce="mean"))
        per_fold["rmse_late"].append(rmse(fold.y_val_true[late], pred[late]))
    result = {f"cv_{k}": float(np.mean(v)) for k, v in per_fold.items()}
    result["cv_rmse_std"] = float(np.std(per_fold["rmse"]))
    result["cv_seconds"] = round(time.perf_counter() - start, 1)
    return result


def last_cycle_predictions(
    pipeline: Pipeline, test_raw: pd.DataFrame, true_rul: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """True and predicted RUL at the last observed cycle of each test engine (sorted by unit)."""
    pred = pd.Series(pipeline.predict(test_raw), index=test_raw.index)
    last_rows = test_raw.groupby("unit")["cycle"].idxmax().sort_index()
    return true_rul.loc[last_rows.index].to_numpy(), pred.loc[last_rows.to_numpy()].to_numpy()


def evaluate_on_test(
    pipeline: Pipeline, test_raw: pd.DataFrame, true_rul: pd.Series
) -> dict[str, float]:
    """Score a fitted pipeline on the last cycle of every official test engine."""
    truth, pred_last = last_cycle_predictions(pipeline, test_raw, true_rul)
    return {
        "test_rmse": rmse(truth, pred_last),
        "test_mae": mae(truth, pred_last),
        "test_nasa": nasa_score(truth, pred_last, reduce="sum"),
        **{f"test_{k}": v for k, v in alert_metrics(truth, pred_last).items()},
    }


def fit_final(
    name: str,
    params: dict[str, Any] | None,
    train_raw: pd.DataFrame,
    feature_config: FeatureConfig | None = None,
    cap: int | None = config.RUL_CAP,
) -> Pipeline:
    """Fit the full pipeline (features + model) on all training engines."""
    labeled = add_rul(train_raw, cap=cap)
    pipeline = make_pipeline_for(name, params, feature_config)
    return pipeline.fit(labeled, labeled[config.TARGET_COLUMN])


def tune_xgboost(folds: list[Fold], n_trials: int, seed: int = config.SEED) -> dict[str, Any]:
    """Optuna TPE search over XGBoost parameters, minimizing mean CV RMSE.

    Returns the best parameters and writes every trial to reports/tuning_trials.csv.
    """
    import optuna  # training-only dependency, imported lazily

    def objective(trial: optuna.Trial) -> float:
        params = {
            # Ranges kept modest so 30 trials x 5 folds fit a laptop budget.
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        return cross_validate(make_model("xgboost", params), folds)["cv_rmse"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

    def report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(
            f"  trial {trial.number + 1}/{n_trials}: cv_rmse={trial.value:.3f} "
            f"(best {study.best_value:.3f})",
            flush=True,
        )

    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, callbacks=[report], show_progress_bar=False)
    print(f"Tuning took {time.perf_counter() - start:.0f}s", flush=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    study.trials_dataframe().to_csv(TRIALS_FILE, index=False)
    return {**study.best_params, "_cv_rmse": study.best_value, "_n_trials": n_trials}


def load_best_params() -> dict[str, Any] | None:
    """Tuned XGBoost parameters from the last ``tune`` run, without metadata keys."""
    if not BEST_PARAMS_FILE.exists():
        return None
    stored = json.loads(BEST_PARAMS_FILE.read_text())
    return {k: v for k, v in stored.items() if not k.startswith("_")}


@dataclass(frozen=True)
class Tracking:
    """Where to log experiment runs (``None`` in place of this means no tracking)."""

    subset: str
    kind: str


def run_experiment(
    label: str,
    name: str,
    params: dict[str, Any] | None,
    folds: list[Fold],
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    true_rul: pd.Series,
    feature_config: FeatureConfig | None = None,
    cap: int | None = config.RUL_CAP,
    track: Tracking | None = None,
) -> dict[str, Any]:
    """Cross-validate, then fit on all training engines and score the test set once.

    With ``track`` set, the run is also logged to MLflow (nested under the
    current parent run) together with the fitted pipeline.
    """
    print(f"  {label} ...", flush=True)
    row: dict[str, Any] = {"experiment": label, "model": name}
    row |= cross_validate(make_model(name, params), folds)
    pipeline = fit_final(name, params, train_raw, feature_config, cap)
    row |= evaluate_on_test(pipeline, test_raw, true_rul)
    row["n_features"] = len(pipeline.named_steps["features"].get_feature_names_out())
    if track is not None:
        from rul import tracking  # training-only dependency, imported lazily

        truth, pred = last_cycle_predictions(pipeline, test_raw, true_rul)
        example = test_raw[test_raw["unit"] == test_raw["unit"].min()][list(config.COLUMNS)].head(5)
        row["run_id"] = tracking.log_experiment(
            label, name, params, feature_config, cap, track.subset, row,
            pipeline, example, truth, pred, kind=track.kind,
        )  # fmt: skip
    return row


def _parent_run(subset: str, kind: str, track: bool) -> AbstractContextManager[Any]:
    """MLflow parent run grouping one command's runs, or a no-op without tracking."""
    if not track:
        return nullcontext()
    from rul import tracking  # training-only dependency, imported lazily

    return tracking.parent_run(subset, kind)


def benchmark(subset: str = "FD001", track: bool = True) -> pd.DataFrame:
    """Baselines vs XGBoost (default and tuned) on identical folds and metrics."""
    train_raw, test_raw = load_split(subset, "train"), load_split(subset, "test")
    true_rul = load_test_rul(subset)
    folds = make_folds(train_raw)
    tuned = load_best_params()
    plan: list[tuple[str, str, dict[str, Any] | None]] = [
        ("Mean RUL", "mean", None),
        ("Ridge", "ridge", None),
        ("Random forest", "random_forest", None),
        ("XGBoost (default)", "xgboost", None),
    ]
    if tuned:
        plan.append(("XGBoost (tuned)", "xgboost", tuned))
    tracking = Tracking(subset, "benchmark") if track else None
    with _parent_run(subset, "benchmark", track):
        rows = [
            run_experiment(lbl, n, p, folds, train_raw, test_raw, true_rul, track=tracking)
            for lbl, n, p in plan
        ]
    return pd.DataFrame(rows)


def ablate(subset: str = "FD001", track: bool = True) -> pd.DataFrame:
    """Change one thing at a time with the tuned (or default) XGBoost parameters."""
    train_raw, test_raw = load_split(subset, "train"), load_split(subset, "test")
    true_rul = load_test_rul(subset)
    params = load_best_params()
    base = FeatureConfig()
    variants: list[tuple[str, FeatureConfig, int | None]] = [
        ("Reference (all features, cap 125)", base, config.RUL_CAP),
        (
            "No rolling features",
            replace(base, windows=(), ewm_spans=(), slope_windows=()),
            config.RUL_CAP,
        ),
        ("No RUL cap", base, None),
        (
            "Short windows (5, 10)",
            replace(base, windows=(5, 10), slope_windows=(10,), lookback=10),
            config.RUL_CAP,
        ),
        (
            "Long windows (10, 30, 50)",
            replace(base, windows=(10, 30, 50), slope_windows=(10, 50), lookback=50),
            config.RUL_CAP,
        ),
    ]
    rows = []
    tracking = Tracking(subset, "ablation") if track else None
    with _parent_run(subset, "ablation", track):
        for label, feature_config, cap in variants:
            folds = make_folds(train_raw, feature_config, cap)
            row = run_experiment(
                label, "xgboost", params, folds, train_raw, test_raw, true_rul,
                feature_config, cap, track=tracking,
            )  # fmt: skip
            rows.append(row | {"lookback": feature_config.lookback, "cap": cap})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    """Render selected result columns as a GitHub markdown table."""
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for _, row in df[columns].iterrows():
        cells = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save(df: pd.DataFrame, stem: str) -> Path:
    """Write results as JSON (machine-readable) and return the path."""
    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"{stem}.json"
    path.write_text(df.to_json(orient="records", indent=2))
    return path


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``uv run rul-train``."""
    parser = argparse.ArgumentParser(description="RUL experiments")
    parser.add_argument("command", choices=["tune", "log-tuning", "benchmark", "ablate", "promote"])
    parser.add_argument("--subset", default="FD001", choices=config.SUBSETS)
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials for 'tune'")
    parser.add_argument("--no-track", action="store_true", help="skip MLflow logging")
    args = parser.parse_args(argv)
    track = not args.no_track

    if args.command == "tune":
        train_raw = load_split(args.subset, "train")
        best = tune_xgboost(make_folds(train_raw), args.trials)
        REPORTS_DIR.mkdir(exist_ok=True)
        BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
        print(json.dumps(best, indent=2))
        if not track:
            return 0
    if args.command in ("tune", "log-tuning"):
        from rul import tracking

        run_id = tracking.log_tuning(args.subset, BEST_PARAMS_FILE, TRIALS_FILE)
        print(f"Logged tuning run {run_id}")
        return 0
    if args.command == "promote":
        from rul import tracking

        for alias, version in tracking.promote(args.subset).items():
            print(f"{tracking.registered_model_name(args.subset)}@{alias} -> version {version}")
        return 0

    run = benchmark if args.command == "benchmark" else ablate
    df = run(args.subset, track=track)
    path = save(df, args.command)
    cols = ["experiment", "cv_rmse", "cv_rmse_std", "cv_rmse_late", "test_rmse", "test_nasa"]
    print(to_markdown(df, cols))
    print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
