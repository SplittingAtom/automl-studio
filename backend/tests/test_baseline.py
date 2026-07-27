"""Tests for the naive + linear baselines trained alongside XGBoost."""

import numpy as np
import pandas as pd

from app.training.preprocessing import build_feature_spec
from app.training.trainer import train_model


def _linear_regression_df(rows=500):
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, rows)
    return pd.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 1, rows)})


def _xor_classification_df(rows=800):
    """Pure interaction — invisible to a linear model, easy for trees."""
    rng = np.random.default_rng(42)
    a = rng.uniform(0, 10, rows)
    b = rng.uniform(0, 10, rows)
    label = np.where((a > 5) != (b > 5), "odd", "even")
    return pd.DataFrame({"a": a, "b": b, "label": label})


class TestBaselineMetrics:
    def test_classification_reports_naive_and_linear(self):
        df = _xor_classification_df()
        spec = build_feature_spec(df, "label", "classification")
        metrics = train_model(df, spec).metrics
        assert 0 <= metrics["baseline_accuracy"] <= 1
        assert 0 <= metrics["linear_accuracy"] <= 1

    def test_regression_reports_naive_and_linear(self):
        df = _linear_regression_df()
        spec = build_feature_spec(df, "y", "regression")
        metrics = train_model(df, spec).metrics
        assert metrics["baseline_mae"] > 0
        assert metrics["linear_mae"] > 0
        # Predicting the average must be worse than the actual model
        assert metrics["baseline_mae"] > metrics["mae"]

    def test_linear_baseline_wins_on_linear_data(self):
        df = _linear_regression_df()
        spec = build_feature_spec(df, "y", "regression")
        metrics = train_model(df, spec).metrics
        assert metrics["linear_r2"] > 0.85


class TestSimpleRelationshipsWarning:
    def test_fires_when_linear_matches_xgboost(self):
        df = _linear_regression_df()
        spec = build_feature_spec(df, "y", "regression")
        result = train_model(df, spec)
        assert any(w.code == "SIMPLE_RELATIONSHIPS" for w in result.warnings)

    def test_silent_when_relationships_are_nonlinear(self):
        df = _xor_classification_df()
        spec = build_feature_spec(df, "label", "classification")
        result = train_model(df, spec)
        # Logistic regression can't see XOR; XGBoost can — big gap, no warning
        assert result.metrics["accuracy"] - result.metrics["linear_accuracy"] > 0.1
        assert not any(w.code == "SIMPLE_RELATIONSHIPS" for w in result.warnings)
