"""Recursive future forecasting for time-aware regression models.

Each future step feeds the previous predictions back in as the lag features,
date-derived features come from the future dates, and every other column is
held at its most recent observed value.
"""

import numpy as np
import pandas as pd

from app.prediction.schemas import ForecastPoint, ForecastResponse
from app.training.preprocessing import FeatureSpec, apply_feature_spec
from app.training.time_series import LAG_STEPS, ROLLING_WINDOW, prepare_time_frame

MAX_HISTORY = ROLLING_WINDOW + max(LAG_STEPS)


def compute_forecast(
    model,
    interval_model,
    spec: FeatureSpec,
    df: pd.DataFrame,
    time_column: str,
    steps: int,
) -> ForecastResponse:
    frame, _, _ = prepare_time_frame(df, time_column, spec.target.name)
    dates = pd.to_datetime(frame[time_column], errors="coerce", format="mixed")
    last_date = dates.iloc[-1]
    step_size = dates.diff().dropna().median()
    if pd.isna(step_size) or step_size <= pd.Timedelta(0):
        step_size = pd.Timedelta(days=1)

    history = list(pd.to_numeric(frame[spec.target.name], errors="coerce").tail(MAX_HISTORY))
    held_row = frame.iloc[-1].to_dict()
    date_format = "%Y-%m-%d" if step_size >= pd.Timedelta(days=1) else "%Y-%m-%d %H:%M"

    points = []
    for offset in range(1, steps + 1):
        future_date = last_date + step_size * offset
        row = dict(held_row)
        row[time_column] = future_date.strftime(date_format)
        for lag in LAG_STEPS:
            row[f"{spec.target.name}_lag{lag}"] = (
                history[-lag] if len(history) >= lag else np.nan
            )
        row[f"{spec.target.name}_avg{ROLLING_WINDOW}"] = (
            float(np.nanmean(history[-ROLLING_WINDOW:])) if history else np.nan
        )
        features = apply_feature_spec(pd.DataFrame([row]), spec)
        predicted = round(float(model.predict(features)[0]), 4)
        low = high = None
        if interval_model is not None:
            quantiles = interval_model.predict(features)[0]
            low, high = sorted(round(float(q), 4) for q in quantiles)
        history.append(predicted)
        points.append(
            ForecastPoint(
                date=future_date.strftime(date_format),
                predicted=predicted,
                low=low,
                high=high,
            )
        )
    return ForecastResponse(
        points=tuple(points),
        last_actual_date=last_date.strftime(date_format),
    )
