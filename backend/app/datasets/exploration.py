"""Distributions for the data profiling view: one histogram per column.

Pure functions over a dataframe plus its column profiles (the profiles carry
the column kinds, so classification happens in exactly one place —
profiling.py). Numeric columns get equal-width histogram bins, categoricals
get their top categories with the tail folded into "Other", and datetimes get
time buckets sized to the span of the data.
"""

import numpy as np
import pandas as pd

from app.datasets.profiling import HIGH_MISSING_PCT
from app.datasets.schemas import (
    ColumnExploration,
    ColumnProfile,
    DatasetExploration,
    DistributionBin,
    ExplorationHighlight,
    ExplorationStats,
)

NUMERIC_BINS = 20
CATEGORY_BINS = 8
OUTLIER_IQR_FACTOR = 1.5
DAY_SPAN_MAX_DAYS = 60
MONTH_SPAN_MAX_DAYS = 4 * 365
MAX_TIME_BUCKETS = 60

CURRENT_EXPLORATION_VERSION = 2  # bump when the computation changes
SKEW_THRESHOLD = 2.0
DOMINANT_SHARE = 0.95
OUTLIER_SHARE = 0.05
CORRELATION_THRESHOLD = 0.85
CORRELATION_SAMPLE_ROWS = 4000
DUPLICATE_SHARE = 0.01
MAX_HIGHLIGHTS = 6


def explore_dataframe(
    dataset_id: str, df: pd.DataFrame, columns: tuple[ColumnProfile, ...]
) -> DatasetExploration:
    explored = tuple(_explore_column(df[profile.name], profile) for profile in columns)
    duplicate_rows = int(df.duplicated().sum())
    return DatasetExploration(
        dataset_id=dataset_id,
        row_count=len(df),
        column_count=len(df.columns),
        missing_cells_pct=_missing_cells_pct(df),
        duplicate_rows=duplicate_rows,
        columns=explored,
        highlights=_highlights(df, explored, duplicate_rows),
        version=CURRENT_EXPLORATION_VERSION,
    )


def _highlights(
    df: pd.DataFrame,
    columns: tuple[ColumnExploration, ...],
    duplicate_rows: int,
) -> tuple[ExplorationHighlight, ...]:
    """Statistically-selected callouts: only say something when it's notable."""
    found: list[ExplorationHighlight] = []
    if len(df) and duplicate_rows / len(df) > DUPLICATE_SHARE:
        found.append(
            ExplorationHighlight(
                tone="warn",
                message=f"{duplicate_rows:,} rows are exact duplicates of another row.",
            )
        )
    found.extend(_column_highlights(df, columns))
    found.extend(_correlation_highlights(df, columns))
    return tuple(found[:MAX_HIGHLIGHTS])


def _column_highlights(df, columns) -> list[ExplorationHighlight]:
    found: list[ExplorationHighlight] = []
    total = len(df)
    for column in columns:
        if column.kind == "numeric" and column.stats is not None:
            values = pd.to_numeric(df[column.name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if len(values) > 20 and abs(float(values.skew())) > SKEW_THRESHOLD:
                found.append(
                    ExplorationHighlight(
                        tone="info",
                        column=column.name,
                        message=(
                            f'"{column.name}" is stretched by a few large values — '
                            "most rows sit far below the average."
                        ),
                    )
                )
            elif total and column.stats.outlier_count / total > OUTLIER_SHARE:
                found.append(
                    ExplorationHighlight(
                        tone="info",
                        column=column.name,
                        message=(
                            f'"{column.name}" has {column.stats.outlier_count:,} values '
                            "outside its typical range."
                        ),
                    )
                )
        if column.kind == "categorical" and column.bins:
            top = column.bins[0]
            nonmissing = sum(b.count for b in column.bins) + column.other_count
            if nonmissing and top.count / nonmissing >= DOMINANT_SHARE:
                share = top.count / nonmissing
                found.append(
                    ExplorationHighlight(
                        tone="warn",
                        column=column.name,
                        message=(
                            f'"{column.name}" is almost always "{top.label}" '
                            f"({share:.0%}) — it may not add much signal."
                        ),
                    )
                )
    return found


def _correlation_highlights(df, columns) -> list[ExplorationHighlight]:
    numeric_names = [c.name for c in columns if c.kind == "numeric"]
    if len(numeric_names) < 2:
        return []
    frame = df[numeric_names].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if len(frame) > CORRELATION_SAMPLE_ROWS:
        frame = frame.sample(CORRELATION_SAMPLE_ROWS, random_state=0)
    matrix = frame.corr()
    found: list[ExplorationHighlight] = []
    for i, a in enumerate(numeric_names):
        for b in numeric_names[i + 1 :]:
            r = matrix.loc[a, b]
            if pd.notna(r) and abs(float(r)) >= CORRELATION_THRESHOLD:
                direction = "together" if r > 0 else "in opposite directions"
                found.append(
                    ExplorationHighlight(
                        tone="info",
                        message=(
                            f'"{a}" and "{b}" move {direction} closely '
                            f"(correlation {float(r):.2f}) — they may carry the same "
                            "information."
                        ),
                    )
                )
    return found


def _missing_cells_pct(df: pd.DataFrame) -> float:
    if df.size == 0:
        return 0.0
    return round(100.0 * float(df.isna().to_numpy().mean()), 1)


def _explore_column(series: pd.Series, profile: ColumnProfile) -> ColumnExploration:
    base = {
        "name": profile.name,
        "kind": profile.kind,
        "missing_pct": profile.missing_pct,
        "unique_count": profile.unique_count,
        "note": _note(profile),
    }
    nonnull = series.dropna()
    if profile.kind == "numeric":
        # inf breaks np.histogram and isn't valid JSON — treat it as missing,
        # the same way calculated columns already sanitize it.
        values = (
            pd.to_numeric(nonnull, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )
        return ColumnExploration(
            **base, bins=_numeric_bins(values), stats=_stats(values)
        )
    if profile.kind == "categorical":
        bins, other = _category_bins(nonnull)
        return ColumnExploration(**base, bins=bins, other_count=other)
    if profile.kind == "datetime":
        return ColumnExploration(**base, bins=_time_bins(nonnull))
    return ColumnExploration(**base)  # id_like / unsupported: note only


def _note(profile: ColumnProfile) -> str | None:
    if profile.kind == "id_like":
        return "Looks like an ID — models leave it out."
    if profile.kind == "unsupported":
        return "Completely empty."
    if profile.kind == "datetime":
        return "Models use its year, month, and day of week."
    if profile.missing_pct > HIGH_MISSING_PCT:
        return f"Missing {profile.missing_pct:.0f}% of its values — models leave it out."
    return None


def _numeric_bins(values: pd.Series) -> tuple[DistributionBin, ...]:
    if values.empty:
        return ()
    n_bins = min(NUMERIC_BINS, max(int(values.nunique()), 1))
    counts, edges = np.histogram(values, bins=n_bins)
    return tuple(
        DistributionBin(
            label=f"{_compact(low)}–{_compact(high)}",
            count=int(count),
            low=round(float(low), 4),
            high=round(float(high), 4),
        )
        for count, low, high in zip(counts, edges[:-1], edges[1:])
    )


def _compact(value: float) -> str:
    return f"{value:g}" if abs(value) < 10_000 else f"{value:.3g}"


def _stats(values: pd.Series) -> ExplorationStats | None:
    if values.empty:
        return None
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    reach = OUTLIER_IQR_FACTOR * (q3 - q1)
    outliers = int(((values < q1 - reach) | (values > q3 + reach)).sum())
    return ExplorationStats(
        min=round(float(values.min()), 4),
        max=round(float(values.max()), 4),
        mean=round(float(values.mean()), 4),
        median=round(float(values.median()), 4),
        std=round(float(values.std()), 4) if len(values) > 1 else 0.0,
        outlier_count=outliers,
    )


def _category_bins(nonnull: pd.Series) -> tuple[tuple[DistributionBin, ...], int]:
    counts = nonnull.astype(str).value_counts()
    top = counts.head(CATEGORY_BINS)
    other = int(counts.iloc[CATEGORY_BINS:].sum())
    bins = tuple(
        DistributionBin(label=str(value), count=int(count))
        for value, count in top.items()
    )
    return bins, other


def _time_bins(nonnull: pd.Series) -> tuple[DistributionBin, ...]:
    parsed = pd.to_datetime(nonnull, errors="coerce", format="mixed").dropna()
    if parsed.empty:
        return ()
    span_days = (parsed.max() - parsed.min()).days
    if span_days <= DAY_SPAN_MAX_DAYS:
        frequency = "D"
    elif span_days <= MONTH_SPAN_MAX_DAYS:
        frequency = "M"
    else:
        frequency = "Y"
    periods = parsed.dt.to_period(frequency)
    counts = periods.value_counts().sort_index()
    # Fill gaps so the shape is honest — unless the range is degenerate/huge.
    full_range = pd.period_range(counts.index.min(), counts.index.max(), freq=frequency)
    if len(full_range) <= MAX_TIME_BUCKETS:
        counts = counts.reindex(full_range, fill_value=0)
    return tuple(
        DistributionBin(label=str(period), count=int(count))
        for period, count in counts.items()
    )
