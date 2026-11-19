# Experiments (FD001)

All numbers come from `uv run rul-train tune|benchmark|ablate` and are stored in
`reports/*.json` (`reports/tuning_trials.csv` holds every tuning trial). MLflow
tracking replaces these files as the system of record in Milestone 5.

## Protocol

- **Data:** FD001 only (100 training engines run to failure, 100 truncated test engines).
- **Target:** RUL clipped at 125 cycles for training (piecewise linear label).
- **Validation:** 5-fold `GroupKFold` by engine (seed 42). Feature scaling and the
  sensor selection are fit inside each fold on the training engines only.
- **CV metrics:** computed on every row of the held-out engines.
  - `cv_rmse` is measured against the training target (capped where a cap is used).
  - `cv_rmse_late` is measured against the *uncapped* true RUL on rows with true
    RUL <= 125. It is the only CV column that is comparable between "cap" and
    "no cap" runs, and it is the part of life where maintenance decisions happen.
- **Test metrics:** one prediction per test engine (its last observed cycle)
  against the official uncapped RUL file. NASA score is the sum over 100 engines.
  The test set was never used to choose models, parameters, or features.
- **Alert:** "predicted RUL <= 30" counts as a maintenance alert. 25 of the 100
  test engines truly have 30 or fewer cycles left.

## Benchmark: same folds, same features, same metrics

| Model | CV RMSE | CV RMSE std | CV RMSE late | CV MAE | Test RMSE | Test MAE | Test NASA | Alert precision | Alert recall |
|---|---|---|---|---|---|---|---|---|---|
| Mean RUL | 41.69 | 0.12 | 43.76 | 36.98 | 43.07 | 35.90 | 33629 | n/a | 0.00 |
| Ridge | 16.23 | 0.68 | 16.84 | 13.08 | 18.12 | 15.11 | 464 | 1.00 | 0.56 |
| Random forest | 13.44 | 1.21 | 14.64 | 9.56 | 14.01 | 10.35 | 282 | 0.95 | 0.76 |
| XGBoost (default) | 12.50 | 0.67 | 13.67 | 8.88 | 13.08 | 9.29 | 248 | 0.96 | 0.88 |
| XGBoost (tuned) | **12.14** | 0.87 | 13.47 | **8.47** | 13.33 | 9.53 | 269 | 0.92 | 0.88 |

Reading the table:

- Every learned model beats the mean baseline by a wide margin, so the features carry real signal.
- The jump from ridge to trees (16.2 to 12.5) is larger than anything tuning buys.
  Degradation is non-linear, and trees handle the plateau before the cap well.
- Tuning improved CV RMSE by 0.36 cycles, which is less than one fold standard
  deviation. On the test set the tuned model is 0.25 cycles *worse*. Both gaps are
  within noise: the test set is only 100 engines.
- Cost matters too. 5-fold CV took about 46 s for default XGBoost, 115 s for tuned
  XGBoost (800 trees), and 198 s for the random forest on a laptop CPU.

## Tuning

Optuna TPE with 30 seeded trials, minimizing 5-fold CV RMSE. Best parameters
(`reports/best_params.json`): 800 trees, learning rate 0.010, max depth 5,
min child weight 7, subsample 0.60, colsample 0.75, lambda 1.54.

The best trial sits on the edge of the search space (the most trees and the lowest
learning rate allowed). A larger budget would probably find slightly more, but
given the noise level above, that is not where the remaining error lives.

## Ablations: change one thing at a time (tuned XGBoost)

| Variant | Features | CV RMSE | CV RMSE late | CV MAE | Test RMSE | Test NASA | Alert recall |
|---|---|---|---|---|---|---|---|
| Reference (windows 5/10/20/30, lookback 30, cap 125) | 281 | 12.14 | 13.47 | 8.47 | 13.33 | 269 | 0.88 |
| No rolling features (current values + cycle) | 15 | 16.70 | 18.69 | 11.48 | 18.18 | 597 | 0.72 |
| No RUL cap | 281 | 35.37 * | 23.29 | 23.73 * | 23.49 | 6944 | 0.88 |
| Short windows (5/10, lookback 10) | 155 | 16.29 | 18.16 | 11.29 | 18.17 | 670 | 0.72 |
| Long windows (10/30/50, lookback 50) | 225 | **9.96** | **11.20** | **6.90** | 13.44 | 266 | 0.92 |

\* Measured against the uncapped target, so it is not comparable with the other
rows. Compare "CV RMSE late" instead.

What the ablations show:

1. **The RUL cap is the single most important choice.** Without it, late-life
   error grows from 13.5 to 23.3 cycles and the NASA score grows 26 times.
   Early in life the sensors are flat, so a model asked to predict "250 cycles
   left" from a healthy engine can only guess, and that guessing spills into the
   late-life predictions.
2. **Temporal features matter.** Removing rolling, EWM and slope features costs
   about 4.5 cycles of RMSE. Single readings are too noisy. This settles the
   open question from the EDA, where smoothing barely changed the per-sensor
   correlation: the gain comes from combining trends, not from any one smoothed sensor.
3. **Longer history helps in CV but did not show up on the test set.** A lookback
   of 50 lowers CV RMSE by 2.2 cycles, while test RMSE stays within noise
   (13.44 vs 13.33). Possible reasons:
   - CV scores about 4,100 rows per fold, while the test set is 100 single points.
   - 7 test engines have fewer than 50 cycles of history, so their long windows
     are only partly filled.

   This is a decision point, not a result to quietly adopt.

## Limitations (honest list)

- **FD001 is the easiest subset**: one operating condition, one fault mode,
  simulated data. These numbers say nothing about FD002 to FD004 or real engines.
- **Ablations were run after tuning** with the tuned parameters fixed. The
  long-window variant might prefer different hyperparameters.
- **The test set is small.** With 100 engines, test RMSE differences below about
  1 cycle should not be used to rank models.
- **CV and test measure different things.** CV covers every point in life, test
  covers one truncation point per engine. Both are reported for that reason.
- **No uncertainty estimate.** The model gives a point prediction. A real
  maintenance decision would want an interval (for example quantile regression),
  which is out of scope for this project.
- **The tuning budget was small** (30 trials, narrowed space) because of laptop CPU time.
