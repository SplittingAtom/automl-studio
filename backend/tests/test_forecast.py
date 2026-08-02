"""Tests for recursive future forecasting on time-aware models."""

import io

import numpy as np
import pandas as pd
import pytest


def _series_df(rows=400):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2022-01-01", periods=rows, freq="D")
    seasonal = 10 * np.sin(np.arange(rows) * 2 * np.pi / 7)  # weekly rhythm
    return pd.DataFrame(
        {
            "day": dates.strftime("%Y-%m-%d"),
            "temperature": rng.uniform(0, 30, rows),
            "sales": 100 + np.arange(rows) * 0.3 + seasonal + rng.normal(0, 2, rows),
        }
    )


@pytest.fixture
def time_model(client):
    csv = _series_df().to_csv(index=False).encode()
    dataset = client.post(
        "/api/datasets", files={"file": ("series.csv", io.BytesIO(csv), "text/csv")}
    ).json()["data"]
    model = client.post(
        "/api/models",
        json={"dataset_id": dataset["id"], "target_column": "sales", "time_column": "day"},
    ).json()["data"]
    meta = client.get(f"/api/models/{model['id']}").json()["data"]
    assert meta["status"] == "complete", meta.get("error")
    return meta


class TestForecastEndpoint:
    def test_forecast_returns_future_points(self, client, time_model):
        resp = client.get(f"/api/models/{time_model['id']}/forecast?steps=30")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert len(data["points"]) == 30
        first = data["points"][0]
        # Dataset ends 2023-02-04; the forecast must continue daily beyond it
        assert first["date"] == "2023-02-05"
        assert data["points"][-1]["date"] == "2023-03-06"
        assert all(np.isfinite(p["predicted"]) for p in data["points"])

    def test_forecast_carries_prediction_band(self, client, time_model):
        data = client.get(f"/api/models/{time_model['id']}/forecast?steps=10").json()["data"]
        for point in data["points"]:
            assert point["low"] is not None and point["high"] is not None
            assert point["low"] <= point["predicted"] <= point["high"] or (
                point["low"] <= point["high"]
            )

    def test_forecast_is_not_flat(self, client, time_model):
        # Weekly seasonality via date features + lags should produce variation
        data = client.get(f"/api/models/{time_model['id']}/forecast?steps=21").json()["data"]
        values = [p["predicted"] for p in data["points"]]
        assert np.std(values) > 0.5

    def test_steps_bounds_enforced(self, client, time_model):
        assert (
            client.get(f"/api/models/{time_model['id']}/forecast?steps=0").status_code == 422
        )
        assert (
            client.get(f"/api/models/{time_model['id']}/forecast?steps=999").status_code
            == 422
        )

    def test_non_time_model_rejected(self, client):
        model = client.post(
            "/api/models",
            json={"dataset_id": "ds_sample_titanic", "target_column": "fare",
                  "task": "regression"},
        ).json()["data"]
        resp = client.get(f"/api/models/{model['id']}/forecast?steps=10")
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "NOT_A_TIME_MODEL"
