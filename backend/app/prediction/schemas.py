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


class ForecastPoint(Frozen):
    date: str
    predicted: float
    low: float | None = None
    high: float | None = None


class ForecastResponse(Frozen):
    points: tuple[ForecastPoint, ...]
    last_actual_date: str


class ImpactPoint(Frozen):
    """One validation row's SHAP contribution for one feature."""

    contribution: float
    value_norm: float | None = None  # 0..1 position within the feature's range
    value_label: str


class FeatureImpact(Frozen):
    feature: str
    kind: str  # "numeric" | "categorical"
    mean_abs_contribution: float
    points: tuple[ImpactPoint, ...]


class CorrelationCell(Frozen):
    """Association between two columns.

    `signed` cells hold a Pearson correlation (-1..1); unsigned cells hold a
    relationship strength (0..1) because at least one column is categorical.
    """

    value: float | None = None
    signed: bool = False


class InsightsResponse(Frozen):
    """Global explainability computed from held-out validation rows."""

    columns: tuple[str, ...]  # matrix axes; the prediction is the extra last axis
    prediction_label: str
    matrix: tuple[tuple[CorrelationCell, ...], ...]
    impacts: tuple[FeatureImpact, ...]
    axis_low_label: str
    axis_high_label: str
    sample_size: int  # rows behind the SHAP impact points
    association_rows: int  # rows behind the association matrix


class GroupFlag(Frozen):
    """A category value whose rows are served notably worse than average."""

    column: str
    value: str
    rows: int
    gap_label: str  # e.g. '64% correct here vs 78% overall'


class GroupCheckResponse(Frozen):
    flagged: tuple[GroupFlag, ...]
    groups_checked: int
    overall_label: str


class BlueprintNode(Frozen):
    """One node in the surrogate flowchart: a question or a leaf."""

    samples: int
    question: str | None = None  # None = leaf
    label: str | None = None  # leaf answer, plain English
    yes: "BlueprintNode | None" = None
    no: "BlueprintNode | None" = None


class BlueprintResponse(Frozen):
    """Shallow decision-tree approximation of the model, with its fidelity."""

    root: BlueprintNode
    fidelity: float  # 0..1: how closely the flowchart tracks the real model
    output_label: str
    sample_size: int


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
