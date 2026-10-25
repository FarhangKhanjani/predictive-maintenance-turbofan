# Turbofan RUL: predictive maintenance on NASA C-MAPSS

> Status: in progress. The full README (architecture, results, API usage) arrives in a later milestone.

Predicting the remaining useful life (RUL) of turbofan engines from simulated sensor data, with
XGBoost and time-series features, MLflow experiment tracking and model registry, a FastAPI
service in Docker, and CI on GitHub Actions.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                    # create .venv from uv.lock
uv run rul-data download   # fetch the dataset into data/raw and verify checksums
uv run pytest              # run the tests
```

With `make` available: `make install`, `make data`, `make check`.

## Data

NASA C-MAPSS Turbofan Engine Degradation Simulation (Saxena and Goebel, 2008). The data is not
redistributed here; it is downloaded from the source and verified against `data/SHA256SUMS`.
See [docs/data.md](docs/data.md) for source, terms, and structure, and
[notebooks/01_eda_fd001.ipynb](notebooks/01_eda_fd001.ipynb) for the exploratory analysis.

## License

Code: MIT, see [LICENSE](LICENSE). Dataset: NASA terms, see [docs/data.md](docs/data.md).
