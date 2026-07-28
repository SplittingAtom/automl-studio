"""Time-series preparation: chronological ordering and recent-history features.

Lag features carry "what happened recently" into each row — the standard way
to make gradient boosting work on forecasting problems. All shifts look
strictly backwards (shift(1)+) so no feature ever peeks at the current row.
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype

LAG_STEPS = (1, 2, 3)
ROLLING_WINDOW = 5


def prepare_time_frame(
    df: pd.DataFrame, time_column: str, target_column: str
) -> tuple[pd.DataFrame, list[str], int]:
    """Sort by time, add lag/rolling features of the target.

    Returns (new frame, generated column names, count of rows dropped for
    unparseable dates).
    """
    parsed = pd.to_datetime(df[time_column], errors="coerce", format="mixed")
    keep = parsed.notna()
    dropped = int((~keep).sum())
    ordered = df[keep].iloc[parsed[keep].argsort(kind="stable")].reset_index(drop=True)

    target = ordered[target_column]
    generated: dict[str, pd.Series] = {}
    for lag in LAG_STEPS:
        generated[f"{target_column}_lag{lag}"] = target.shift(lag)
    if is_numeric_dtype(target):
        generated[f"{target_column}_avg{ROLLING_WINDOW}"] = (
            target.shift(1).rolling(ROLLING_WINDOW).mean()
        )
    return ordered.assign(**generated), list(generated.keys()), dropped
