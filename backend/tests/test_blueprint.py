"""Surrogate decision tree: a shallow, readable approximation of the model."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier, XGBRegressor

from app.prediction.blueprint import compute_blueprint
from app.training.preprocessing import apply_feature_spec, build_feature_spec

TITANIC_ID = "ds_sample_titanic"


def _fit(df, target, task):
    spec = build_feature_spec(df, target, task)
    X = apply_feature_spec(df, spec)
    y = df[target] if task == "regression" else df[target].astype("category").cat.codes
    cls = XGBClassifier if task == "classification" else XGBRegressor
    model = cls(
        n_estimators=40, max_depth=4, tree_method="hist",
        enable_categorical=True, verbosity=0,
    ).fit(X, y)
    return model, spec


def _leaves(node):
    if node.question is None:
        return [node]
    return _leaves(node.yes) + _leaves(node.no)


def _depth(node):
    if node.question is None:
        return 0
    return 1 + max(_depth(node.yes), _depth(node.no))


class TestRegressionBlueprint:
    @pytest.fixture(scope="class")
    def blueprint(self):
        rng = np.random.default_rng(21)
        n = 600
        x = rng.uniform(0, 10, n)
        other = rng.normal(size=n)
        df = pd.DataFrame({"x": x, "other": other, "y": 3 * x + rng.normal(scale=0.5, size=n)})
        model, spec = _fit(df, "y", "regression")
        return compute_blueprint(model, spec, df)

    def test_root_splits_on_the_driver(self, blueprint):
        assert blueprint.root.question is not None
        assert "x" in blueprint.root.question

    def test_tree_is_shallow_and_faithful(self, blueprint):
        assert _depth(blueprint.root) <= 3
        assert blueprint.fidelity > 0.7  # near-linear signal: easy to mimic

    def test_leaves_carry_plain_labels_and_counts(self, blueprint):
        for leaf in _leaves(blueprint.root):
            assert leaf.label
            assert leaf.samples > 0


class TestClassificationBlueprint:
    def test_binary_leaves_speak_in_chances(self):
        rng = np.random.default_rng(22)
        n = 500
        a = rng.normal(size=n)
        group = rng.choice(["yes", "no"], n)
        outcome = np.where(a + (group == "yes") + rng.normal(scale=0.4, size=n) > 0.5, "won", "lost")
        df = pd.DataFrame({"a": a, "group": group, "outcome": outcome})
        model, spec = _fit(df, "outcome", "classification")
        blueprint = compute_blueprint(model, spec, df)
        assert 0.0 <= blueprint.fidelity <= 1.0
        assert any("won" in leaf.label for leaf in _leaves(blueprint.root))

    def test_categorical_split_reads_as_a_question(self):
        rng = np.random.default_rng(23)
        n = 400
        group = rng.choice(["gold", "silver"], n)
        outcome = np.where(
            (group == "gold") ^ (rng.uniform(size=n) < 0.05), "kept", "churned"
        )
        df = pd.DataFrame({"group": group, "noise": rng.normal(size=n), "outcome": outcome})
        model, spec = _fit(df, "outcome", "classification")
        blueprint = compute_blueprint(model, spec, df)
        assert blueprint.root.question is not None
        assert "group" in blueprint.root.question
        assert "?" in blueprint.root.question


class TestBlueprintApi:
    @pytest.fixture
    def trained_model_id(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]["id"]

    def test_blueprint_endpoint_and_cache(self, client, settings, trained_model_id):
        resp = client.get(f"/api/models/{trained_model_id}/blueprint")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["root"]["question"] is not None
        assert 0.0 <= data["fidelity"] <= 1.0
        assert (settings.data_dir / "models" / trained_model_id / "blueprint.json").is_file()

    def test_unknown_model_404(self, client):
        assert client.get("/api/models/mdl_nope/blueprint").status_code == 404
