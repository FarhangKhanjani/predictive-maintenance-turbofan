"""FastAPI service: remaining useful life for engines from their raw sensor history.

Endpoints:
    GET  /health         the app is up and the model is loaded
    GET  /model-info     which registered model version is serving, and why
    POST /predict        one engine's cycle history -> RUL at its last cycle
    POST /predict-batch  several engines in one call

The model is loaded once at startup from ``models:/<MODEL_NAME>@<MODEL_ALIAS>``.
If loading fails the app does not start: a service that cannot predict should
fail its health check loudly instead of answering with errors.

Configuration (environment variables):
    MLFLOW_TRACKING_URI  tracking store or server holding the registry
    MODEL_NAME           registered model (default turbofan-rul-fd001)
    MODEL_ALIAS          alias to serve (default champion)
    ALERT_THRESHOLD      alert when predicted RUL <= this many cycles (default 30)

Run locally with ``uv run rul-serve`` and open http://localhost:8000/docs.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from rul.schemas import BatchRequest, BatchResponse, EngineHistory, Health, ModelInfo, Prediction
from rul.serving import LoadedModel, load_from_registry, predict_engines

logger = logging.getLogger("rul.api")

DEFAULT_MODEL_NAME = "turbofan-rul-fd001"
DEFAULT_ALIAS = "champion"
DEFAULT_ALERT_THRESHOLD = 30.0


def load_configured_model() -> LoadedModel:
    """Load the model named by the environment (see module docstring)."""
    return load_from_registry(
        name=os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME),
        alias=os.environ.get("MODEL_ALIAS", DEFAULT_ALIAS),
        alert_threshold=float(os.environ.get("ALERT_THRESHOLD", DEFAULT_ALERT_THRESHOLD)),
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
    )


def create_app(loader: Callable[[], LoadedModel] = load_configured_model) -> FastAPI:
    """Build the app. Tests pass their own ``loader`` instead of using a registry."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model = loader()
        logger.info(
            "Serving %s version %s (@%s)", model.info.name, model.info.version, model.info.alias
        )
        app.state.model = model
        yield

    app = FastAPI(
        title="Turbofan RUL API",
        description="Remaining useful life of turbofan engines (NASA C-MAPSS FD001).",
        version="1.0.0",
        lifespan=lifespan,
    )

    def model_of(request: Request) -> LoadedModel:
        return request.app.state.model

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", model_loaded=hasattr(request.app.state, "model"))

    @app.get("/model-info")
    def model_info(request: Request) -> ModelInfo:
        return model_of(request).info

    @app.post("/predict")
    def predict(engine: EngineHistory, request: Request) -> Prediction:
        return predict_engines(model_of(request), [engine])[0]

    @app.post("/predict-batch")
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        return BatchResponse(predictions=predict_engines(model_of(request), batch.engines))

    return app


app = create_app()


def main() -> None:
    """Entry point for ``uv run rul-serve`` (development server)."""
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "rul.api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )
