# Prediction API

A FastAPI service that returns the remaining useful life (RUL) of an engine
from its raw sensor history. It serves the model registered as
`turbofan-rul-fd001@champion` (see [experiments.md](experiments.md#promotion)).

## Run it

```bash
uv run rul-serve                     # http://127.0.0.1:8000, docs at /docs
```

| Variable | Default | Meaning |
|---|---|---|
| `MLFLOW_TRACKING_URI` | MLflow default | Tracking store or server that holds the registry, for example `sqlite:///mlflow.db` |
| `MODEL_NAME` | `turbofan-rul-fd001` | Registered model to serve |
| `MODEL_ALIAS` | `champion` | Alias to serve (`challenger` for a side-by-side check) |
| `ALERT_THRESHOLD` | `30` | `alert` is true when predicted RUL is at most this many cycles |
| `HOST`, `PORT` | `127.0.0.1`, `8000` | Bind address for `rul-serve` |

The model is loaded once at startup. If the registry cannot be reached or the
alias does not exist, the service does not start. It never runs half-ready,
answering requests with errors.

## Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model_loaded": true}` |
| GET | `/model-info` | registered name, alias, version, run id, git commit, CV RMSE, selection rule, lookback, feature count, alert threshold |
| POST | `/predict` | RUL at the last cycle of one engine |
| POST | `/predict-batch` | RUL for up to 100 engines, in request order |

### Example

`examples/predict_request.json` holds the 31 recorded cycles of FD001 test
engine 1 (true RUL: 112).

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/predict_request.json
```

```json
{"engine_id": "FD001-test-1", "last_cycle": 31, "predicted_rul": 123.52,
 "alert": false, "model_version": "1"}
```

The model overestimates by about 12 cycles here. With only 31 cycles of a
healthy engine, there is little trend to read, and the prediction sits near
the 125 cap. This is the expected behavior of a capped model on young engines.

The request body is one engine with its cycles, oldest first. Each cycle has
the same fields as a row of the C-MAPSS files:

```json
{
  "engine_id": "engine-42",
  "cycles": [
    {"cycle": 1, "op1": -0.0007, "op2": -0.0004, "op3": 100.0,
     "s1": 518.67, "s2": 641.82, "...": "...", "s21": 23.419},
    {"cycle": 2, "...": "..."}
  ]
}
```

A batch wraps several of these: `{"engines": [ {...}, {...} ]}`.

## Validation

Requests are rejected with HTTP 422 and a message naming the field when:

- a sensor or setting is missing, is not a number, is NaN or infinite, or an unknown field is sent
- cycles are not consecutive and ascending (gaps or reordering would silently
  corrupt the rolling features)
- `cycle` is below 1, the cycle list is empty or longer than 1000, or `engine_id` is empty
- a batch is empty, has more than 100 engines, or repeats an `engine_id`

## Training-serving parity

The API does not compute features itself. It rebuilds the request as raw
C-MAPSS rows and calls the logged pipeline (`RULFeatures` followed by the
regressor), which is the same object that was evaluated in training. Checked
against the real champion on all 100 FD001 test engines: the API and offline
scoring differ by at most 0.005 cycles (the API rounds to 0.01), and the API's
test RMSE is 13.33, matching [experiments.md](experiments.md). The tests in
`tests/test_api.py` check the same property on synthetic data, including that
batching gives exactly the single-engine answers.

Only the last 30 cycles affect a prediction (the feature lookback). Sending
fewer works, as for young engines in training, but gives the model less trend
information.

## Limitations

- **No authentication or rate limiting.** Add them in front of the service
  (gateway or reverse proxy) before exposing it.
- **Latency is modest, not tuned.** On the development laptop, one engine took
  about 0.7 s (first call after startup) and 100 engines about 3.4 s. Most of
  that is JSON parsing and validation of every cycle. Trimming requests to the
  last 30 cycles on the client side would cut it a lot.
- **Predictions are clipped at 0 and are point estimates.** There is no
  uncertainty interval, so the alert flag is a simple threshold on a point estimate.
- **The model is a pickle.** Only serve models from a registry you control.
- **Promotion needs a restart.** The alias is resolved at startup, so moving
  `@champion` takes effect on the next restart. That is a deliberate,
  predictable rollout step, not hot reloading.
