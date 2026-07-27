"""Tests for what-if prediction serving."""

import numpy as np
import pandas as pd
import pytest

from app.api.envelope import AppError
from app.prediction.model_cache import ModelCache
from app.prediction.predictor import predict
from app.training.preprocessing import build_feature_spec
from app.training.trainer import train_model


@pytest.fixture(scope="module")
def trained():
    rng = np.random.default_rng(42)
    rows = 400
    df = pd.DataFrame(
        {
            "age": rng.uniform(18, 90, rows),
            "sex": rng.choice(["male", "female"], rows),
            "outcome": rng.choice(["died", "survived"], rows),
        }
    )
    spec = build_feature_spec(df, "outcome", "classification")
    result = train_model(df, spec)
    return result.model, spec


class TestPredict:
    def test_valid_inputs_return_class_and_probabilities(self, trained):
        model, spec = trained
        resp = predict(model, spec, {"age": 30, "sex": "female"})
        assert resp.prediction in {"died", "survived"}
        assert len(resp.probabilities) == 2
        assert abs(sum(p.probability for p in resp.probabilities) - 1.0) < 0.01

    def test_missing_input_uses_default(self, trained):
        model, spec = trained
        resp = predict(model, spec, {"sex": "male"})
        assert resp.prediction in {"died", "survived"}

    def test_unknown_input_rejected(self, trained):
        model, spec = trained
        with pytest.raises(AppError) as exc:
            predict(model, spec, {"age": 30, "bogus": 1})
        assert exc.value.status_code == 422

    def test_non_numeric_value_for_numeric_feature_rejected(self, trained):
        model, spec = trained
        with pytest.raises(AppError) as exc:
            predict(model, spec, {"age": "not a number"})
        assert exc.value.status_code == 422

    def test_warm_prediction_is_fast(self, trained):
        model, spec = trained
        predict(model, spec, {"age": 40, "sex": "male"})  # warm up
        resp = predict(model, spec, {"age": 41, "sex": "male"})
        assert resp.elapsed_ms < 100


class TestModelCache:
    def test_loads_once_then_hits_cache(self):
        calls = []
        cache = ModelCache(capacity=2)

        def loader():
            calls.append(1)
            return "artifact"

        assert cache.get("m1", loader) == "artifact"
        assert cache.get("m1", loader) == "artifact"
        assert len(calls) == 1

    def test_evicts_least_recently_used(self):
        cache = ModelCache(capacity=2)
        loads = {"a": 0, "b": 0, "c": 0}

        def loader_for(key):
            def load():
                loads[key] += 1
                return key

            return load

        cache.get("a", loader_for("a"))
        cache.get("b", loader_for("b"))
        cache.get("c", loader_for("c"))  # evicts "a"
        cache.get("a", loader_for("a"))
        assert loads["a"] == 2
