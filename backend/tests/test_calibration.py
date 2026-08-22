"""Calibration curve: predicted probability vs actual outcome rate."""

import numpy as np
import pytest

from app.training.trainer import _calibration

TITANIC_ID = "ds_sample_titanic"


class TestCalibrationCurve:
    def test_perfectly_calibrated_probabilities_have_low_error(self):
        rng = np.random.default_rng(41)
        probabilities = rng.uniform(0, 1, 5000)
        actual = (rng.uniform(0, 1, 5000) < probabilities).astype(int)
        result = _calibration(actual, probabilities)
        assert result["error"] < 0.05
        assert abs(result["bias"]) < 0.05

    def test_overconfident_model_has_positive_bias(self):
        rng = np.random.default_rng(42)
        # Model says 90% but reality is 50%
        probabilities = np.full(1000, 0.9)
        actual = (rng.uniform(0, 1, 1000) < 0.5).astype(int)
        result = _calibration(actual, probabilities)
        assert result["error"] > 0.3
        assert result["bias"] > 0.3

    def test_points_cover_every_row(self):
        rng = np.random.default_rng(43)
        probabilities = rng.uniform(0, 1, 500)
        actual = rng.integers(0, 2, 500)
        result = _calibration(actual, probabilities)
        assert sum(p["count"] for p in result["points"]) == 500
        for point in result["points"]:
            assert 0.0 <= point["predicted"] <= 1.0
            assert 0.0 <= point["actual"] <= 1.0


class TestCalibrationInTraining:
    def test_binary_model_gets_calibration_metrics(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
        )
        assert resp.status_code == 200, resp.text
        model_id = resp.json()["data"]["id"]
        metrics = client.get(f"/api/models/{model_id}").json()["data"]["metrics"]
        assert "calibration" in metrics
        assert len(metrics["calibration"]["points"]) > 2
        assert metrics["calibration"]["error"] < 0.5

    def test_regression_model_has_no_calibration(self, client):
        resp = client.post(
            "/api/models",
            json={"dataset_id": TITANIC_ID, "target_column": "fare", "task": "regression"},
        )
        assert resp.status_code == 200, resp.text
        model_id = resp.json()["data"]["id"]
        metrics = client.get(f"/api/models/{model_id}").json()["data"]["metrics"]
        assert "calibration" not in metrics
