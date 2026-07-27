"""Tests for model-rigor features: CV, threshold curve, intervals, tuning, validation."""

import numpy as np
import pandas as pd
import pytest

from app.prediction.predictor import predict
from app.training.preprocessing import build_feature_spec
from app.training.trainer import train_model

TITANIC_ID = "ds_sample_titanic"


def _classification_df(rows=600):
    rng = np.random.default_rng(42)
    signal = rng.uniform(0, 10, rows)
    color = rng.choice(["red", "blue"], rows)
    noise = rng.normal(0, 2, rows)
    label = np.where(signal + (color == "red") * 3 + noise > 7, "yes", "no")
    return pd.DataFrame({"signal": signal, "color": color, "label": label})


def _regression_df(rows=600):
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, rows)
    return pd.DataFrame({"x": x, "z": rng.uniform(0, 5, rows), "y": 3 * x + rng.normal(0, 1, rows)})


@pytest.fixture(scope="module")
def classification_result():
    df = _classification_df()
    return train_model(df, build_feature_spec(df, "label", "classification"))


@pytest.fixture(scope="module")
def regression_result():
    df = _regression_df()
    return train_model(df, build_feature_spec(df, "y", "regression"))


class TestCrossValidation:
    def test_classification_cv_stats(self, classification_result):
        m = classification_result.metrics
        assert 0 <= m["cv_mean"] <= 1
        assert m["cv_std"] >= 0
        assert m["cv_folds"] == 5

    def test_regression_cv_stats(self, regression_result):
        m = regression_result.metrics
        assert m["cv_mean"] > 0.5  # strongly linear data
        assert m["cv_folds"] == 5


class TestThresholdCurve:
    def test_binary_gets_a_curve(self, classification_result):
        curve = classification_result.metrics["threshold_curve"]
        assert len(curve) == 21
        assert curve[0]["threshold"] == 0.0
        assert curve[-1]["threshold"] == 1.0
        # At threshold 0 everything is flagged -> recall is 1
        assert curve[0]["recall"] == 1.0
        # Recall never increases as the threshold rises
        recalls = [p["recall"] for p in curve]
        assert all(a >= b for a, b in zip(recalls, recalls[1:]))

    def test_regression_has_no_curve(self, regression_result):
        assert "threshold_curve" not in regression_result.metrics


class TestPredictionIntervals:
    def test_regression_prediction_has_interval(self, regression_result):
        df = _regression_df()
        spec = build_feature_spec(df, "y", "regression")
        resp = predict(
            regression_result.model,
            spec,
            {"x": 5.0, "z": 2.0},
            interval_model=regression_result.interval_model,
        )
        assert resp.interval is not None
        assert resp.interval.low <= resp.interval.high
        # 3*5 = 15; an 80% band around it should be roughly a few units wide
        assert resp.interval.low < 15 < resp.interval.high

    def test_classification_has_no_interval(self, classification_result):
        df = _classification_df()
        spec = build_feature_spec(df, "label", "classification")
        resp = predict(classification_result.model, spec, {"signal": 5, "color": "red"})
        assert resp.interval is None


class TestValidationFrame:
    def test_classification_frame_shape(self, classification_result):
        frame = classification_result.validation
        assert len(frame) == 120  # 20% of 600
        assert "predicted" in frame.columns
        assert "confidence" in frame.columns
        assert frame["correct"].dtype == bool
        accuracy = frame["correct"].mean()
        assert abs(accuracy - classification_result.metrics["accuracy"]) < 0.01

    def test_regression_frame_has_error(self, regression_result):
        frame = regression_result.validation
        assert "predicted" in frame.columns
        assert "error" in frame.columns
        assert abs(frame["error"].abs().mean() - regression_result.metrics["mae"]) < 0.5


class TestThoroughEffort:
    def test_thorough_training_completes(self):
        df = _classification_df(400)
        spec = build_feature_spec(df, "label", "classification")
        result = train_model(df, spec, effort="thorough")
        assert result.metrics["accuracy"] > 0.6
        assert result.metrics.get("tuning_trials", 0) > 0


class TestValidationApi:
    @pytest.fixture
    def model_id(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
        )
        return resp.json()["data"]["id"]

    def test_validation_rows_endpoint(self, client, model_id):
        data = client.get(f"/api/models/{model_id}/validation?rows=10").json()["data"]
        assert len(data["rows"]) == 10
        assert data["total_rows"] == 179
        assert "predicted" in data["columns"]
        assert "correct" in data["columns"]

    def test_validation_download_is_csv(self, client, model_id):
        resp = client.get(f"/api/models/{model_id}/validation/download")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        header = resp.text.splitlines()[0]
        assert "predicted" in header

    def test_validation_missing_model_404(self, client):
        assert client.get("/api/models/mdl_nope/validation").status_code == 404
