"""FeatureSpec: the immutable contract between training and prediction.

Built once at train time from the dataset, persisted with the model, and
applied identically to training data and single-row what-if inputs.
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from app.datasets.profiling import HIGH_MISSING_PCT, profile_dataframe
from app.datasets.schemas import ColumnProfile

OTHER_CATEGORY = "Other"
DEFAULT_MAX_CATEGORIES = 50


class TrainingError(Exception):
    """Raised when a dataset can't be trained on; message is user-facing."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class FeatureColumn(Frozen):
    name: str
    kind: Literal["numeric", "categorical"]
    categories: tuple[str, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    default: float | str | None = None


class TargetSpec(Frozen):
    name: str
    task: Literal["classification", "regression"]
    classes: tuple[str, ...] | None = None


class ExcludedColumn(Frozen):
    name: str
    reason: Literal["id_like", "datetime", "high_missing", "unsupported"]


class FeatureSpec(Frozen):
    features: tuple[FeatureColumn, ...]
    target: TargetSpec
    excluded: tuple[ExcludedColumn, ...]


def build_feature_spec(
    df: pd.DataFrame,
    target_column: str,
    task: Literal["classification", "regression"],
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> FeatureSpec:
    if target_column not in df.columns:
        raise TrainingError("UNKNOWN_COLUMN", f'Column "{target_column}" is not in this dataset.')
    target_series = df[target_column]
    if target_series.dropna().empty:
        raise TrainingError(
            "EMPTY_TARGET", f'"{target_column}" has no values to learn from.'
        )

    profile = {c.name: c for c in profile_dataframe(df).columns}
    features: list[FeatureColumn] = []
    excluded: list[ExcludedColumn] = []
    for name in df.columns:
        if name == target_column:
            continue
        col = profile[str(name)]
        reason = _exclusion_reason(col)
        if reason is not None:
            excluded.append(ExcludedColumn(name=col.name, reason=reason))
        elif col.kind == "numeric":
            features.append(_numeric_feature(col))
        else:
            features.append(_categorical_feature(df[name], max_categories))

    classes = None
    if task == "classification":
        classes = tuple(sorted(target_series.dropna().astype(str).unique()))
        if len(classes) < 2:
            raise TrainingError(
                "SINGLE_CLASS_TARGET",
                f'"{target_column}" only ever has one value, so there is nothing to predict.',
            )
    return FeatureSpec(
        features=tuple(features),
        target=TargetSpec(name=target_column, task=task, classes=classes),
        excluded=tuple(excluded),
    )


def apply_feature_spec(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Return a new frame with exactly the spec's columns, typed for XGBoost."""
    data = {}
    for feature in spec.features:
        source = (
            df[feature.name]
            if feature.name in df.columns
            else pd.Series([None] * len(df), index=df.index)
        )
        if feature.kind == "numeric":
            data[feature.name] = pd.to_numeric(source, errors="coerce").astype("float64")
        else:
            data[feature.name] = _to_fixed_categorical(source, feature.categories)
    return pd.DataFrame(data, index=df.index)


def _exclusion_reason(col: ColumnProfile) -> str | None:
    if col.kind in ("id_like", "datetime", "unsupported"):
        return col.kind
    if col.missing_pct > HIGH_MISSING_PCT:
        return "high_missing"
    return None


def _numeric_feature(col: ColumnProfile) -> FeatureColumn:
    stats = col.stats
    return FeatureColumn(
        name=col.name,
        kind="numeric",
        min_value=stats.min if stats else None,
        max_value=stats.max if stats else None,
        default=stats.median if stats else None,
    )


def _categorical_feature(series: pd.Series, max_categories: int) -> FeatureColumn:
    counts = series.dropna().astype(str).value_counts()
    categories = list(counts.index[:max_categories])
    if len(counts) > max_categories:
        categories.append(OTHER_CATEGORY)
    return FeatureColumn(
        name=str(series.name),
        kind="categorical",
        categories=tuple(categories),
        default=categories[0] if categories else None,
    )


def _to_fixed_categorical(series: pd.Series, categories: tuple[str, ...]) -> pd.Series:
    known = set(categories)
    has_other = OTHER_CATEGORY in known

    def normalize(value):
        if pd.isna(value):
            return None
        text = str(value)
        if text in known:
            return text
        return OTHER_CATEGORY if has_other else None

    values = [normalize(v) for v in series.tolist()]
    return pd.Series(
        pd.Categorical(values, categories=list(categories)), index=series.index
    )
