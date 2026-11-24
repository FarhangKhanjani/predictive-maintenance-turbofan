"""Request and response models for the prediction API.

A request carries the raw sensor history of an engine, exactly as the C-MAPSS
files record it: one entry per cycle with the three operating settings and all
21 sensors. Features are computed server side by the same ``RULFeatures`` code
used in training, so clients never compute features themselves.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_CYCLES = 1000  # longest FD001 to FD004 engine is 543 cycles
MAX_ENGINES = 100


class CycleReading(BaseModel):
    """Operating settings and sensor readings for one engine cycle."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    cycle: int = Field(ge=1, description="Cycle number, starting at 1")
    op1: float
    op2: float
    op3: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    s7: float
    s8: float
    s9: float
    s10: float
    s11: float
    s12: float
    s13: float
    s14: float
    s15: float
    s16: float
    s17: float
    s18: float
    s19: float
    s20: float
    s21: float


class EngineHistory(BaseModel):
    """Consecutive cycles of one engine, oldest first.

    The model uses the most recent 30 cycles; sending fewer works but gives the
    model less trend information, exactly as for young engines in training.
    """

    model_config = ConfigDict(extra="forbid")

    engine_id: str = Field(min_length=1, max_length=64, examples=["engine-42"])
    cycles: list[CycleReading] = Field(min_length=1, max_length=MAX_CYCLES)

    @field_validator("cycles")
    @classmethod
    def cycles_are_consecutive(cls, cycles: list[CycleReading]) -> list[CycleReading]:
        """Rolling features assume one row per cycle with no gaps or reordering."""
        numbers = [reading.cycle for reading in cycles]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError("cycles must be consecutive and in ascending order")
        return cycles


class BatchRequest(BaseModel):
    """Several engines scored in one call."""

    model_config = ConfigDict(extra="forbid")

    engines: list[EngineHistory] = Field(min_length=1, max_length=MAX_ENGINES)

    @field_validator("engines")
    @classmethod
    def engine_ids_are_unique(cls, engines: list[EngineHistory]) -> list[EngineHistory]:
        """Each prediction must map back to exactly one engine."""
        ids = [engine.engine_id for engine in engines]
        if len(set(ids)) != len(ids):
            raise ValueError("engine_id values must be unique within a batch")
        return engines


class Prediction(BaseModel):
    """Remaining useful life at the last cycle received for one engine."""

    engine_id: str
    last_cycle: int
    predicted_rul: float = Field(description="Predicted remaining cycles, never negative")
    alert: bool = Field(description="True when predicted_rul <= the alert threshold")
    model_version: str


class BatchResponse(BaseModel):
    """Predictions in the same order as the requested engines."""

    predictions: list[Prediction]


class Health(BaseModel):
    """Liveness and readiness in one: the app only starts with a loaded model."""

    status: str
    model_loaded: bool


class ModelInfo(BaseModel):
    """Which model is serving and how it was selected."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    alias: str
    version: str
    run_id: str
    git_commit: str
    cv_rmse: float | None
    selection_rule: str
    lookback: int
    n_features: int
    alert_threshold: float
