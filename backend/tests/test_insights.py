"""Model insights: association matrix + SHAP impact summary on validation rows."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier, XGBRegressor

from app.prediction.insights import (
    compute_insights,
    correlation_ratio,
    cramers_v,
    pearson,
)
from app.training.preprocessing import apply_feature_spec, build_feature_spec

TITANIC_ID = "ds_sample_titanic"


def _fit(df, target, task):
    spec = build_feature_spec(df, target, task)
    X = apply_feature_spec(df, spec)
    cls = XGBClassifier if task == "classification" else XGBRegressor
    y = df[target] if task == "regression" else df[target].astype("category").cat.codes
    model = cls(
        n_estimators=30,
        max_depth=3,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
    ).fit(X, y)
    return model, spec


@pytest.fixture(scope="module")
def regression_setup():
    rng = np.random.default_rng(0)
    n = 300
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    group = rng.choice(["x", "y"], size=n)
    target = 3 * a + 2 * (group == "x") + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"a": a, "b": b, "group": group, "price": target})
    model, spec = _fit(df, "price", "regression")
    return df, model, spec


class TestAssociations:
    def test_pearson_signs(self):
        x = pd.Series(np.arange(50, dtype=float))
        assert pearson(x, x * 2 + 1) == pytest.approx(1.0)
        assert pearson(x, -x) == pytest.approx(-1.0)

    def test_pearson_constant_is_none(self):
        x = pd.Series(np.arange(10, dtype=float))
        assert pearson(x, pd.Series([5.0] * 10)) is None

    def test_pearson_ignores_missing_pairs(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 100.0])
        assert pearson(x, y) == pytest.approx(1.0)

    def test_correlation_ratio_separated_groups(self):
        cat = pd.Series(["a"] * 50 + ["b"] * 50, dtype="category")
        values = pd.Series([0.0] * 50 + [10.0] * 50)
        assert correlation_ratio(cat, values) == pytest.approx(1.0)

    def test_correlation_ratio_unrelated_is_small(self):
        rng = np.random.default_rng(1)
        cat = pd.Series(rng.choice(["a", "b"], 500), dtype="category")
        values = pd.Series(rng.normal(size=500))
        assert correlation_ratio(cat, values) < 0.2

    def test_cramers_v_identical_columns(self):
        cat = pd.Series(["a", "b", "c"] * 40, dtype="category")
        assert cramers_v(cat, cat) == pytest.approx(1.0, abs=0.05)

    def test_cramers_v_single_level_is_none(self):
        cat = pd.Series(["a"] * 20, dtype="category")
        other = pd.Series(["x", "y"] * 10, dtype="category")
        assert cramers_v(cat, other) is None


class TestComputeInsightsRegression:
    def test_matrix_shape_and_bounds(self, regression_setup):
        df, model, spec = regression_setup
        insights = compute_insights(model, spec, df)
        n_axes = len(insights.columns) + 1  # + prediction row/col
        assert len(insights.matrix) == n_axes
        assert all(len(row) == n_axes for row in insights.matrix)
        for row in insights.matrix:
            for cell in row:
                if cell.value is not None:
                    assert -1.0 <= cell.value <= 1.0

    def test_prediction_tracks_strongest_driver(self, regression_setup):
        df, model, spec = regression_setup
        insights = compute_insights(model, spec, df)
        i = insights.columns.index("a")
        cell = insights.matrix[i][-1]  # a ↔ prediction
        assert cell.signed
        assert cell.value > 0.8

    def test_impacts_ranked_and_dominated_by_a(self, regression_setup):
        df, model, spec = regression_setup
        insights = compute_insights(model, spec, df)
        assert insights.impacts[0].feature == "a"
        strengths = [f.mean_abs_contribution for f in insights.impacts]
        assert strengths == sorted(strengths, reverse=True)

    def test_impact_points_have_normalized_values(self, regression_setup):
        df, model, spec = regression_setup
        insights = compute_insights(model, spec, df)
        by_name = {f.feature: f for f in insights.impacts}
        for point in by_name["a"].points:
            assert point.value_norm is not None
            assert 0.0 <= point.value_norm <= 1.0
        for point in by_name["group"].points:
            assert point.value_norm is None  # categories have no numeric position
            assert point.value_label in {"x", "y"}

    def test_regression_axis_labels_name_the_target(self, regression_setup):
        df, model, spec = regression_setup
        insights = compute_insights(model, spec, df)
        assert "lower" in insights.axis_low_label
        assert "higher" in insights.axis_high_label

    def test_sample_size_reported(self, regression_setup):
        df, model, spec = regression_setup
        insights = compute_insights(model, spec, df)
        assert insights.sample_size <= len(df)


class TestReviewRegressions:
    def test_unused_features_are_left_out_of_impacts(self):
        rng = np.random.default_rng(3)
        n = 200
        a = rng.normal(size=n)
        df = pd.DataFrame(
            {
                "a": a,
                "constant": [1.0] * n,  # never split on → contribution exactly 0
                "price": 3 * a + rng.normal(scale=0.1, size=n),
            }
        )
        model, spec = _fit(df, "price", "regression")
        insights = compute_insights(model, spec, df)
        assert "constant" not in {f.feature for f in insights.impacts}

    def test_association_rows_reported(self, regression_setup):
        df, model, spec = regression_setup
        insights = compute_insights(model, spec, df)
        assert insights.association_rows == len(df)

    def test_numeric_looking_categories_survive_csv_round_trip(self, tmp_path):
        from app.training.repository import ModelRepository

        repo = ModelRepository(tmp_path)
        frame = pd.DataFrame({"code": ["01", "02", "01"], "n": [1, 2, 3]})
        (tmp_path / "mdl_x").mkdir()
        repo.save_validation("mdl_x", frame, text_columns=("code",))
        loaded = repo.load_validation("mdl_x")  # dtypes come from the sidecar
        assert list(loaded["code"]) == ["01", "02", "01"]


class TestComputeInsightsClassification:
    def test_binary_labels_name_both_classes(self):
        rng = np.random.default_rng(2)
        n = 300
        a = rng.normal(size=n)
        outcome = np.where(a + rng.normal(scale=0.3, size=n) > 0, "yes", "no")
        df = pd.DataFrame({"a": a, "b": rng.normal(size=n), "outcome": outcome})
        model, spec = _fit(df, "outcome", "classification")
        insights = compute_insights(model, spec, df)
        assert '"no"' in insights.axis_low_label
        assert '"yes"' in insights.axis_high_label
        # The driver correlates with the chance of "yes"
        i = insights.columns.index("a")
        assert insights.matrix[i][-1].value > 0.6


class TestInsightsApi:
    @pytest.fixture
    def trained_model_id(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]["id"]

    def test_insights_endpoint_shape(self, client, trained_model_id):
        resp = client.get(f"/api/models/{trained_model_id}/insights")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert len(data["matrix"]) == len(data["columns"]) + 1
        assert data["prediction_label"]
        assert len(data["impacts"]) > 0
        assert all(len(f["points"]) > 0 for f in data["impacts"])

    def test_missing_model_404(self, client):
        resp = client.get("/api/models/mdl_nope/insights")
        assert resp.status_code == 404

    def test_insights_cached_to_disk(self, client, settings, trained_model_id):
        client.get(f"/api/models/{trained_model_id}/insights")
        cache_file = settings.data_dir / "models" / trained_model_id / "insights.json"
        assert cache_file.is_file()
        # Second call serves the cached payload identically
        again = client.get(f"/api/models/{trained_model_id}/insights")
        assert again.status_code == 200
