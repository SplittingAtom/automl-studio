"""Schemas for dataset suitability analysis."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.training.task_detection import Task


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class PredictorHint(Frozen):
    name: str
    share: float


class TargetCandidate(Frozen):
    """One column's suitability as a prediction target."""

    column: str
    task: Task
    score: int  # 0-100 blended suitability
    recommended: bool
    derived_like: bool = False  # probably calculated from other columns
    signal: float  # 0-1: skill over the naive baseline
    probe_score: float | None  # accuracy (classification) or R² (regression)
    baseline_score: float | None  # majority-class accuracy; 0 for regression
    reasons: tuple[str, ...]
    top_predictors: tuple[PredictorHint, ...]


class AnalysisPoint(Frozen):
    tone: Literal["good", "warn", "bad"]
    message: str


class DatasetAnalysis(Frozen):
    dataset_id: str
    rating: Literal["great", "good", "fair", "poor"]
    summary: str
    points: tuple[AnalysisPoint, ...]
    candidates: tuple[TargetCandidate, ...]


class FeatureIdea(Frozen):
    """A candidate calculated column the probe model found useful."""

    name: str
    formula: str
    share: float  # importance share in the probe fit (0..1)
    based_on: tuple[str, str]


class FeatureIdeasResponse(Frozen):
    ideas: tuple[FeatureIdea, ...]
    checked: int  # how many candidates were evaluated
