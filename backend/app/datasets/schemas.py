"""Pydantic schemas for datasets and column profiles."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

ColumnKind = Literal["numeric", "categorical", "datetime", "id_like", "unsupported"]


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class NumericStats(Frozen):
    min: float
    max: float
    mean: float
    median: float


class TopValue(Frozen):
    value: str
    count: int


class ColumnProfile(Frozen):
    name: str
    kind: ColumnKind
    dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    stats: NumericStats | None = None
    top_values: tuple[TopValue, ...] | None = None


class ProfileWarning(Frozen):
    code: str
    message: str
    column: str | None = None


class ProfileResult(Frozen):
    columns: tuple[ColumnProfile, ...]
    warnings: tuple[ProfileWarning, ...]


class CalculatedColumnRequest(Frozen):
    name: str
    formula: str


class DatasetMeta(Frozen):
    id: str
    name: str
    source: Literal["upload", "sample", "derived"]
    created_at: str
    row_count: int
    column_count: int
    columns: tuple[ColumnProfile, ...]
    warnings: tuple[ProfileWarning, ...]


class DatasetPreview(Frozen):
    columns: tuple[str, ...]
    rows: tuple[dict, ...]
