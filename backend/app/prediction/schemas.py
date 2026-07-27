"""Schemas for what-if prediction requests and responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class PredictRequest(Frozen):
    inputs: dict[str, Any]


class ClassProbability(Frozen):
    label: str
    probability: float


class PredictResponse(Frozen):
    prediction: float | str
    probabilities: tuple[ClassProbability, ...] | None = None
    elapsed_ms: float
