"""Tests for per-prediction explanations and sensitivity curves."""

import numpy as np
import pandas as pd
import pytest

from app.api.envelope import AppError
from app.prediction.predictor import predict
from app.prediction.sensitivity import compute_sensitivity
from app.training.preprocessing import build_feature_spec
from app.training.trainer import train_model


@pytest.fixture(scope="module")
def classifier():
    rng = np.random.default_rng(42)
    rows = 500
    signal = rng.uniform(0, 10, rows)
    color = rng.choice(["red", "blue"], rows)
    label = np.where(signal + (color == "red") * 4 > 8, "yes", "no")
    df = pd.DataFrame(
        {"signal": signal, "noise": rng.uniform(0, 10, rows), "color": color, "label": label}
    )
    spec = build_feature_spec(df, "label", "classification")
    return train_model(df, spec).model, spec


@pytest.fixture(scope="module")
def regressor():
    rng = np.random.default_rng(42)
    rows = 500
    x1 = rng.uniform(0, 10, rows)
    df = pd.DataFrame(
        {"x1": x1, "x2": rng.uniform(0, 10, rows), "y": 3 * x1 + rng.normal(0, 0.5, rows)}
    )
    spec = build_feature_spec(df, "y", "regression")
    return train_model(df, spec).model, spec


class TestExplanation:
    def test_every_feature_gets_a_contribution(self, classifier):
        model, spec = classifier
        resp = predict(model, spec, {"signal": 9, "color": "red"})
        assert resp.explanation is not None
        assert {i.feature for i in resp.explanation.items} == {"signal", "noise", "color"}

    def test_sorted_by_absolute_contribution(self, classifier):
        model, spec = classifier
        items = predict(model, spec, {"signal": 9, "color": "red"}).explanation.items
        magnitudes = [abs(i.contribution) for i in items]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_informative_feature_dominates(self, regressor):
        model, spec = regressor
        items = predict(model, spec, {"x1": 9.5, "x2": 5}).explanation.items
        assert items[0].feature == "x1"
        assert items[0].contribution > 0  # x1 far above average pushes prediction up

    def test_classification_labels_direction(self, classifier):
        model, spec = classifier
        explanation = predict(model, spec, {"signal": 9, "color": "red"}).explanation
        assert explanation.toward_label == "yes"  # positive contribution -> class[1]

    def test_regression_has_no_direction_label(self, regressor):
        model, spec = regressor
        explanation = predict(model, spec, {"x1": 5, "x2": 5}).explanation
        assert explanation.toward_label is None


class TestSensitivity:
    def test_numeric_grid_spans_range(self, regressor):
        model, spec = regressor
        result = compute_sensitivity(model, spec, {"x1": 5, "x2": 5}, "x1")
        assert len(result.points) == 21
        values = [p.value for p in result.points]
        assert values[0] == pytest.approx(0, abs=0.5)
        assert values[-1] == pytest.approx(10, abs=0.5)

    def test_regression_outputs_track_signal(self, regressor):
        model, spec = regressor
        result = compute_sensitivity(model, spec, {"x1": 5, "x2": 5}, "x1")
        outputs = [p.output for p in result.points]
        assert outputs[-1] > outputs[0] + 10  # y ~= 3 * x1

    def test_categorical_grid_is_category_list(self, classifier):
        model, spec = classifier
        result = compute_sensitivity(model, spec, {"signal": 5, "color": "blue"}, "color")
        assert {p.value for p in result.points} == {"red", "blue"}

    def test_classification_outputs_are_probabilities(self, classifier):
        model, spec = classifier
        result = compute_sensitivity(model, spec, {"signal": 5, "color": "blue"}, "signal")
        assert all(0 <= p.output <= 1 for p in result.points)
        assert result.output_label

    def test_unknown_feature_rejected(self, regressor):
        model, spec = regressor
        with pytest.raises(AppError) as exc:
            compute_sensitivity(model, spec, {"x1": 5}, "bogus")
        assert exc.value.status_code == 422
