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


class DistributionBin(Frozen):
    """One histogram bar: a numeric range, a category, or a time bucket."""

    label: str
    count: int
    low: float | None = None  # numeric bin edges, for precise tooltips
    high: float | None = None


class ExplorationStats(Frozen):
    min: float
    max: float
    mean: float
    median: float
    std: float
    outlier_count: int  # values beyond 1.5×IQR from the quartiles


class ColumnExploration(Frozen):
    name: str
    kind: ColumnKind
    missing_pct: float
    unique_count: int
    bins: tuple[DistributionBin, ...] = ()
    other_count: int = 0  # categorical rows folded beyond the top bins
    stats: ExplorationStats | None = None
    note: str | None = None  # plain-English caveat, e.g. "looks like an ID"


class ExplorationHighlight(Frozen):
    """One statistically-selected callout about the dataset."""

    tone: Literal["info", "warn"]
    message: str
    column: str | None = None


class DatasetExploration(Frozen):
    """Profiling view payload: dataset overview + per-column distributions."""

    dataset_id: str
    row_count: int
    column_count: int
    missing_cells_pct: float
    duplicate_rows: int
    columns: tuple[ColumnExploration, ...]
    highlights: tuple[ExplorationHighlight, ...] = ()
    # Bumped when the computation changes so stale disk caches are rebuilt.
    version: int = 1
