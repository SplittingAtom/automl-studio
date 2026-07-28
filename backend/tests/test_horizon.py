"""Tests for the embargo gap (prediction horizon) in time-series mode."""

import io

import numpy as np
import pandas as pd
import pytest

from app.training.preprocessing import build_feature_spec
from app.training.time_series import prepare_time_frame
from app.training.trainer import TrainingError, _time_split, train_model


def _series_df(rows=500):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "day": dates.strftime("%Y-%m-%d"),
            "x": rng.uniform(0, 10, rows),
            "y": np.arange(rows) * 0.5 + rng.normal(0, 3, rows),
        }
    )


class TestTimeSplit:
    def test_no_gap_splits_at_the_cut(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))
        X_train, X_test, _, _ = _time_split(X, y, test_fraction=0.2, gap=0)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert X_test["a"].iloc[0] == 80

    def test_gap_purges_rows_before_the_test_block(self):
        X = pd.DataFrame({"a": range(100)})
        y = pd.Series(range(100))
        X_train, X_test, _, _ = _time_split(X, y, test_fraction=0.2, gap=10)
        assert len(X_train) == 70  # rows 70-79 purged
        assert X_train["a"].iloc[-1] == 69
        assert X_test["a"].iloc[0] == 80  # test block unchanged
        assert len(X_test) == 20


class TestHorizonTraining:
    def test_training_with_horizon_completes(self):
        frame, generated, _ = prepare_time_frame(_series_df(), "day", "y")
        spec = build_feature_spec(frame, "y", "regression")
        result = train_model(
            frame, spec, time_mode=True, generated_columns=tuple(generated), horizon=10
        )
        assert result.metrics["r2"] is not None
        warning = next(w for w in result.warnings if w.code == "TIME_SPLIT")
        assert "10-row gap" in warning.message

    def test_absurd_horizon_refused(self):
        frame, _, _ = prepare_time_frame(_series_df(100), "day", "y")
        spec = build_feature_spec(frame, "y", "regression")
        with pytest.raises(TrainingError) as exc:
            train_model(frame, spec, time_mode=True, horizon=79)
        assert exc.value.code == "HORIZON_TOO_LARGE"


class TestHorizonApi:
    def _upload(self, client):
        csv = _series_df().to_csv(index=False).encode()
        return client.post(
            "/api/datasets", files={"file": ("series.csv", io.BytesIO(csv), "text/csv")}
        ).json()["data"]

    def test_horizon_flows_end_to_end(self, client):
        dataset = self._upload(client)
        model = client.post(
            "/api/models",
            json={
                "dataset_id": dataset["id"],
                "target_column": "y",
                "time_column": "day",
                "horizon": 10,
            },
        ).json()["data"]
        meta = client.get(f"/api/models/{model['id']}").json()["data"]
        assert meta["status"] == "complete", meta.get("error")
        assert meta["horizon"] == 10

    def test_horizon_without_time_column_rejected(self, client):
        dataset = self._upload(client)
        resp = client.post(
            "/api/models",
            json={"dataset_id": dataset["id"], "target_column": "y", "horizon": 5},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_HORIZON"

    def test_negative_horizon_rejected(self, client):
        dataset = self._upload(client)
        resp = client.post(
            "/api/models",
            json={
                "dataset_id": dataset["id"],
                "target_column": "y",
                "time_column": "day",
                "horizon": -3,
            },
        )
        assert resp.status_code == 422
