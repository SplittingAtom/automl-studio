"""Expert tuning: validated hyperparameter overrides + monotone constraints."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.training.preprocessing import apply_feature_spec, build_feature_spec
from app.training.trainer import XGB_PARAMS, TuningOverrides, train_model

TITANIC_ID = "ds_sample_titanic"


@pytest.fixture(scope="module")
def regression_frame():
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(0, 10, size=n)
    other = rng.normal(size=n)
    y = 2 * x + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"x": x, "other": other, "target": y})


class TestOverrideValidation:
    def test_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            TuningOverrides(max_depth=99)
        with pytest.raises(ValidationError):
            TuningOverrides(learning_rate=0.0)

    def test_monotone_directions_restricted(self):
        with pytest.raises(ValidationError):
            TuningOverrides(monotone_constraints={"x": 2})
        assert TuningOverrides(monotone_constraints={"x": -1}).monotone_constraints == {
            "x": -1
        }


class TestOverridesApplied:
    def test_overrides_reach_the_model(self, regression_frame):
        spec = build_feature_spec(regression_frame, "target", "regression")
        result = train_model(
            regression_frame,
            spec,
            overrides=TuningOverrides(max_depth=3, learning_rate=0.3, n_estimators=60),
        )
        params = result.model.get_params()
        assert params["max_depth"] == 3
        assert params["learning_rate"] == 0.3
        assert params["n_estimators"] == 60

    def test_unset_fields_keep_defaults(self, regression_frame):
        spec = build_feature_spec(regression_frame, "target", "regression")
        result = train_model(
            regression_frame, spec, overrides=TuningOverrides(max_depth=4)
        )
        params = result.model.get_params()
        assert params["max_depth"] == 4
        assert params["learning_rate"] == XGB_PARAMS["learning_rate"]

    def test_monotone_constraint_enforced(self, regression_frame):
        spec = build_feature_spec(regression_frame, "target", "regression")
        result = train_model(
            regression_frame,
            spec,
            overrides=TuningOverrides(monotone_constraints={"x": 1}),
        )
        grid = pd.DataFrame({"x": np.linspace(0, 10, 100), "other": 0.0})
        predictions = result.model.predict(apply_feature_spec(grid, spec))
        assert np.all(np.diff(predictions) >= -1e-6)


class TestTuningApi:
    @pytest.fixture
    def baseline_id(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]["id"]

    def test_tuned_run_stores_overrides_and_baseline(self, client, baseline_id):
        resp = client.post(
            "/api/models",
            json={
                "dataset_id": TITANIC_ID,
                "target_column": "survived",
                "overrides": {"max_depth": 3, "subsample": 0.8},
                "baseline_model_id": baseline_id,
            },
        )
        assert resp.status_code == 200, resp.text
        data = client.get(f"/api/models/{resp.json()['data']['id']}").json()["data"]
        assert data["status"] == "complete"
        assert data["overrides"]["max_depth"] == 3
        assert data["overrides"]["subsample"] == 0.8
        assert data["baseline_model_id"] == baseline_id

    def test_out_of_range_override_rejected(self, client):
        resp = client.post(
            "/api/models",
            json={
                "dataset_id": TITANIC_ID,
                "target_column": "survived",
                "overrides": {"max_depth": 99},
            },
        )
        assert resp.status_code == 422

    def test_monotone_on_non_numeric_column_rejected(self, client):
        resp = client.post(
            "/api/models",
            json={
                "dataset_id": TITANIC_ID,
                "target_column": "survived",
                "overrides": {"monotone_constraints": {"sex": 1}},
            },
        )
        assert resp.status_code == 422
        assert "sex" in resp.json()["error"]["message"]

    def test_run_label_stored(self, client):
        resp = client.post(
            "/api/models",
            json={
                "dataset_id": TITANIC_ID,
                "target_column": "survived",
                "label": "  Simpler & steadier  ",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["label"] == "Simpler & steadier"

    def test_untuned_run_has_no_overrides(self, client, baseline_id):
        data = client.get(f"/api/models/{baseline_id}").json()["data"]
        assert data["overrides"] is None
        assert data["baseline_model_id"] is None
