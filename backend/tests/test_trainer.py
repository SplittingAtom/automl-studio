"""Tests for model training on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from app.training.preprocessing import TrainingError, build_feature_spec
from app.training.trainer import train_model


def _classification_df(rows=600):
    rng = np.random.default_rng(42)
    signal = rng.uniform(0, 10, rows)
    noise = rng.uniform(0, 10, rows)
    color = rng.choice(["red", "blue", "green"], rows)
    label = np.where(signal + (color == "red") * 3 > 6, "yes", "no")
    return pd.DataFrame({"signal": signal, "noise": noise, "color": color, "label": label})


def _regression_df(rows=600):
    rng = np.random.default_rng(42)
    x1 = rng.uniform(0, 10, rows)
    x2 = rng.uniform(0, 10, rows)
    y = 3.0 * x1 + rng.normal(0, 1, rows)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


class TestClassification:
    @pytest.fixture(scope="class")
    def result(self):
        df = _classification_df()
        spec = build_feature_spec(df, "label", "classification")
        return train_model(df, spec)

    def test_metrics_present(self, result):
        m = result.metrics
        assert 0.7 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1_weighted"] <= 1.0
        assert m["roc_auc"] is not None
        assert len(m["confusion_matrix"]) == 2
        assert m["classes"] == ["no", "yes"]

    def test_importance_normalized_and_sorted(self, result):
        scores = [i.score for i in result.importance]
        assert abs(sum(scores) - 1.0) < 0.01
        assert scores == sorted(scores, reverse=True)
        assert result.importance[0].feature in {"signal", "color"}

    def test_rows_used_reported(self, result):
        assert result.n_rows_used == 600


class TestRegression:
    @pytest.fixture(scope="class")
    def result(self):
        df = _regression_df()
        spec = build_feature_spec(df, "y", "regression")
        return train_model(df, spec)

    def test_metrics_present(self, result):
        m = result.metrics
        assert m["r2"] > 0.8
        assert m["mae"] >= 0
        assert m["rmse"] >= m["mae"]
        assert "target_mean" in m

    def test_informative_feature_dominates(self, result):
        top = result.importance[0]
        assert top.feature == "x1"
        assert top.score > 0.5


class TestGuardrails:
    def test_tiny_dataset_refused(self):
        df = _classification_df(30)
        spec = build_feature_spec(df, "label", "classification")
        with pytest.raises(TrainingError):
            train_model(df, spec)

    def test_rows_with_missing_target_dropped(self):
        df = _classification_df(300)
        df.loc[:19, "label"] = None
        spec = build_feature_spec(df, "label", "classification")
        result = train_model(df, spec)
        assert result.n_rows_used == 280
        assert any(w.code == "TARGET_MISSING_DROPPED" for w in result.warnings)

    def test_mostly_missing_target_refused(self):
        df = _classification_df(300)
        df.loc[: int(300 * 0.4), "label"] = None
        spec = build_feature_spec(df, "label", "classification")
        with pytest.raises(TrainingError):
            train_model(df, spec)

    def test_leakage_warning_on_perfect_score(self):
        rng = np.random.default_rng(1)
        y = rng.choice(["a", "b"], 400)
        df = pd.DataFrame({"leak": y, "noise": rng.uniform(0, 1, 400), "target": y})
        spec = build_feature_spec(df, "target", "classification")
        result = train_model(df, spec)
        warning = next(w for w in result.warnings if w.code == "POSSIBLE_LEAKAGE")
        assert warning.column == "leak"

    def test_no_leakage_warning_when_dataset_is_genuinely_easy(self):
        # Perfectly separable from several features together — not leakage
        rng = np.random.default_rng(2)
        rows = 400
        a = rng.uniform(0, 10, rows)
        b = rng.uniform(0, 10, rows)
        df = pd.DataFrame({"a": a, "b": b, "target": np.where(a + b > 10, "hi", "lo")})
        spec = build_feature_spec(df, "target", "classification")
        result = train_model(df, spec)
        top_share = result.importance[0].score
        if result.metrics["accuracy"] > 0.999:
            assert top_share < 0.9  # importance is shared, so no warning fires
        assert not any(w.code == "POSSIBLE_LEAKAGE" for w in result.warnings)

    def test_row_cap_sampling(self):
        df = _classification_df(2000)
        spec = build_feature_spec(df, "label", "classification")
        result = train_model(df, spec, max_rows=1000)
        assert result.n_rows_used == 1000
        assert any(w.code == "ROW_SAMPLE" for w in result.warnings)
