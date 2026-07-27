"""Pydantic schemas for model training requests and metadata."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

from app.datasets.schemas import ProfileWarning
from app.training.preprocessing import ExcludedColumn
from app.training.task_detection import Task
from app.training.trainer import ImportanceItem

ModelStatus = Literal["queued", "training", "complete", "failed"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TrainRequest(Frozen):
    dataset_id: str
    target_column: str
    task: Task | None = None
    excluded_columns: tuple[str, ...] = ()

    @field_validator("dataset_id", "target_column")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class InputSpecItem(Frozen):
    """Describes one what-if control: a slider (numeric) or dropdown (categorical)."""

    name: str
    kind: Literal["numeric", "categorical"]
    min_value: float | None = None
    max_value: float | None = None
    options: tuple[str, ...] | None = None
    default: float | str | None = None


class ModelMeta(Frozen):
    id: str
    dataset_id: str
    dataset_name: str
    target_column: str
    task: Task
    status: ModelStatus
    created_at: str
    error: str | None = None
    metrics: dict[str, Any] | None = None
    importance: tuple[ImportanceItem, ...] | None = None
    input_spec: tuple[InputSpecItem, ...] | None = None
    excluded_columns: tuple[ExcludedColumn, ...] = ()
    user_excluded_columns: tuple[str, ...] = ()
    warnings: tuple[ProfileWarning, ...] = ()
    n_rows_used: int | None = None
