"""Tests for time-series mode: ordering, lag features, chronological splits."""

import io

import numpy as np
import pandas as pd
import pytest

from app.training.preprocessing import build_feature_spec
from app.training.time_series import prepare_time_frame
from app.training.trainer import train_model


def _series_df(rows=500, shuffled=False):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    trend = np.arange(rows) * 0.5
    df = pd.DataFrame(
        {
            "day": dates.strftime("%Y-%m-%d"),
            "temperature": rng.uniform(0, 30, rows),
            "sales": trend + rng.normal(0, 5, rows) + 50,
        }
    )
    if shuffled:
        df = df.sample(frac=1, random_state=7).reset_index(drop=True)
    return df


class TestPrepareTimeFrame:
    def test_sorts_rows_chronologically(self):
        frame, _, _ = prepare_time_frame(_series_df(shuffled=True), "day", "sales")
        parsed = pd.to_datetime(frame["day"])
        assert parsed.is_monotonic_increasing

    def test_adds_lag_and_rolling_features_for_numeric_target(self):
        frame, generated, _ = prepare_time_frame(_series_df(), "day", "sales")
        assert "sales_lag1" in generated and "sales_lag1" in frame.columns
        assert "sales_lag2" in generated
        assert "sales_avg5" in generated
        # lag1 of the second row equals the first row's target
        assert frame["sales_lag1"].iloc[1] == frame["sales"].iloc[0]

    def test_rolling_average_excludes_current_row(self):
        frame, _, _ = prepare_time_frame(_series_df(), "day", "sales")
        expected = frame["sales"].iloc[0:5].mean()
        assert frame["sales_avg5"].iloc[5] == pytest.approx(expected)

    def test_categorical_target_gets_lags_but_no_average(self):
        df = _series_df()
        df["state"] = np.where(df["sales"] > df["sales"].median(), "busy", "quiet")
        frame, generated, _ = prepare_time_frame(df, "day", "state")
        assert "state_lag1" in generated
        assert not any(name.endswith("_avg5") for name in generated)

    def test_unparseable_dates_dropped_and_counted(self):
        df = _series_df(100)
        df.loc[3, "day"] = "not a date"
        df.loc[7, "day"] = "???"
        frame, _, dropped = prepare_time_frame(df, "day", "sales")
        assert dropped == 2
        assert len(frame) == 98


class TestChronologicalTraining:
    @pytest.fixture(scope="class")
    def result(self):
        frame, generated, _ = prepare_time_frame(_series_df(), "day", "sales")
        spec = build_feature_spec(frame, "sales", "regression")
        return train_model(frame, spec, time_mode=True, generated_columns=tuple(generated))

    def test_validation_rows_are_the_most_recent(self, result):
        validation_dates = pd.to_datetime(result.validation["day"])
        assert validation_dates.min() > pd.Timestamp("2023-02-01")  # last ~20% of 2022-2023
        assert len(result.validation) == 100

    def test_time_split_warning_present(self, result):
        assert any(w.code == "TIME_SPLIT" for w in result.warnings)

    def test_generated_columns_warning_present(self, result):
        warning = next(w for w in result.warnings if w.code == "LAG_FEATURES")
        assert "sales_lag1" in warning.message

    def test_walk_forward_cv_runs(self, result):
        assert result.metrics["cv_folds"] == 5

    def test_lag_features_are_model_inputs(self, result):
        features = {i.feature for i in result.importance}
        assert "sales_lag1" in features


class TestTimeSeriesApi:
    def _upload(self, client, df):
        csv = df.to_csv(index=False).encode()
        return client.post(
            "/api/datasets", files={"file": ("series.csv", io.BytesIO(csv), "text/csv")}
        ).json()["data"]

    def test_time_column_flows_end_to_end(self, client):
        dataset = self._upload(client, _series_df())
        model = client.post(
            "/api/models",
            json={
                "dataset_id": dataset["id"],
                "target_column": "sales",
                "time_column": "day",
            },
        ).json()["data"]
        meta = client.get(f"/api/models/{model['id']}").json()["data"]
        assert meta["status"] == "complete", meta.get("error")
        assert meta["time_column"] == "day"
        input_names = {i["name"] for i in meta["input_spec"]}
        assert "sales_lag1" in input_names
        assert any(w["code"] == "TIME_SPLIT" for w in meta["warnings"])

    def test_non_date_time_column_rejected(self, client):
        dataset = self._upload(client, _series_df())
        resp = client.post(
            "/api/models",
            json={
                "dataset_id": dataset["id"],
                "target_column": "sales",
                "time_column": "temperature",
            },
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_TIME_COLUMN"

    def test_lag_columns_never_suggested_for_exclusion(self, client):
        dataset = self._upload(client, _series_df())
        model = client.post(
            "/api/models",
            json={
                "dataset_id": dataset["id"],
                "target_column": "sales",
                "time_column": "day",
            },
        ).json()["data"]
        meta = client.get(f"/api/models/{model['id']}").json()["data"]
        dataset_columns = {c["name"] for c in dataset["columns"]}
        assert set(meta["suggested_exclusions"]) <= dataset_columns
        assert meta["leak_suspect"] is None or meta["leak_suspect"] in dataset_columns
