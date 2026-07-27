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


class ExplanationItem(Frozen):
    feature: str
    contribution: float


class Explanation(Frozen):
    """Per-prediction TreeSHAP contributions, sorted by absolute impact.

    For classification, positive contributions push toward `toward_label`;
    for regression the contributions are in target units and the label is None.
    """

    items: tuple[ExplanationItem, ...]
    baseline: float
    toward_label: str | None = None


class PredictionInterval(Frozen):
    """80% band: the true value lands inside it about 8 times in 10."""

    low: float
    high: float


class PredictResponse(Frozen):
    prediction: float | str
    probabilities: tuple[ClassProbability, ...] | None = None
    explanation: Explanation | None = None
    interval: PredictionInterval | None = None
    elapsed_ms: float


class SensitivityRequest(Frozen):
    feature: str
    inputs: dict = {}


class SensitivityPoint(Frozen):
    value: float | str
    output: float


class SensitivityResponse(Frozen):
    feature: str
    kind: str
    output_label: str
    current_value: float | str | None
    points: tuple[SensitivityPoint, ...]
