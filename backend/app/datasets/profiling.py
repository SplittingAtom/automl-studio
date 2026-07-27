"""Column profiling: pure functions that classify columns and compute stats.

The kinds drive everything downstream: which columns become features, which
get sliders vs dropdowns in the what-if UI, and which are excluded.
"""

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)

from app.datasets.schemas import (
    ColumnKind,
    ColumnProfile,
    NumericStats,
    ProfileResult,
    ProfileWarning,
    TopValue,
)

LOW_CARDINALITY_MAX = 10
ID_LIKE_INT_RATIO = 0.95
ID_LIKE_STRING_RATIO = 0.5
DATETIME_PARSE_RATIO = 0.9
HIGH_MISSING_PCT = 50.0
TOP_VALUES_COUNT = 20
NUMERIC_ONLY_PATTERN = r"[-+]?\d+(\.\d+)?"


def profile_dataframe(df: pd.DataFrame) -> ProfileResult:
    columns = tuple(_profile_column(df[name]) for name in df.columns)
    return ProfileResult(columns=columns, warnings=tuple(_build_warnings(columns)))


def _profile_column(series: pd.Series) -> ColumnProfile:
    nonnull = series.dropna()
    n = len(series)
    missing = n - len(nonnull)
    unique = int(nonnull.nunique())
    kind = _classify(series, nonnull, unique)
    return ColumnProfile(
        name=str(series.name),
        kind=kind,
        dtype=str(series.dtype),
        missing_count=missing,
        missing_pct=round(100.0 * missing / n, 1) if n else 0.0,
        unique_count=unique,
        stats=_numeric_stats(nonnull) if kind == "numeric" else None,
        top_values=_top_values(nonnull) if kind == "categorical" else None,
    )


def _classify(series: pd.Series, nonnull: pd.Series, unique: int) -> ColumnKind:
    if len(nonnull) == 0:
        return "unsupported"
    if is_bool_dtype(series):
        return "categorical"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_numeric_dtype(series):
        return _classify_numeric(series, nonnull, unique)
    return _classify_text(nonnull, unique)


def _classify_numeric(series: pd.Series, nonnull: pd.Series, unique: int) -> ColumnKind:
    if unique <= LOW_CARDINALITY_MAX:
        return "categorical"
    if is_integer_dtype(series) and unique / len(nonnull) > ID_LIKE_INT_RATIO:
        return "id_like"
    return "numeric"


def _classify_text(nonnull: pd.Series, unique: int) -> ColumnKind:
    if _looks_like_dates(nonnull):
        return "datetime"
    if unique > LOW_CARDINALITY_MAX and unique / len(nonnull) > ID_LIKE_STRING_RATIO:
        return "id_like"
    return "categorical"


def _looks_like_dates(nonnull: pd.Series) -> bool:
    sample = nonnull.head(50).astype(str)
    if sample.str.fullmatch(NUMERIC_ONLY_PATTERN).all():
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= DATETIME_PARSE_RATIO


def _numeric_stats(nonnull: pd.Series) -> NumericStats | None:
    if nonnull.empty:
        return None
    return NumericStats(
        min=float(nonnull.min()),
        max=float(nonnull.max()),
        mean=round(float(nonnull.mean()), 4),
        median=round(float(nonnull.median()), 4),
    )


def _top_values(nonnull: pd.Series) -> tuple[TopValue, ...]:
    counts = nonnull.astype(str).value_counts().head(TOP_VALUES_COUNT)
    return tuple(TopValue(value=str(v), count=int(c)) for v, c in counts.items())


def _build_warnings(columns: tuple[ColumnProfile, ...]) -> list[ProfileWarning]:
    warnings: list[ProfileWarning] = []
    for col in columns:
        if col.kind == "id_like":
            warnings.append(
                ProfileWarning(
                    code="ID_LIKE_COLUMN",
                    column=col.name,
                    message=f'"{col.name}" looks like an ID, so it will be left out of the model.',
                )
            )
        elif col.kind == "unsupported":
            warnings.append(
                ProfileWarning(
                    code="ALL_NULL_COLUMN",
                    column=col.name,
                    message=f'"{col.name}" is completely empty and will be ignored.',
                )
            )
        elif col.kind == "datetime":
            warnings.append(
                ProfileWarning(
                    code="DATETIME_COLUMN",
                    column=col.name,
                    message=f'"{col.name}" contains dates, which are not supported yet — it will be left out.',
                )
            )
        elif col.missing_pct > HIGH_MISSING_PCT:
            warnings.append(
                ProfileWarning(
                    code="HIGH_MISSING",
                    column=col.name,
                    message=(
                        f'"{col.name}" is missing {col.missing_pct:.0f}% of its values, '
                        "so it will be left out of the model."
                    ),
                )
            )
    return warnings
