"""Tests for FeatureSpec construction and application."""

import numpy as np
import pandas as pd
import pytest

from app.training.preprocessing import (
    TrainingError,
    apply_feature_spec,
    build_feature_spec,
)


def _make_df(rows=200):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.uniform(1, 80, rows),
            "sex": rng.choice(["male", "female"], rows),
            "record_id": range(rows),
            "mostly_missing": [None] * (rows - 10) + list(range(10)),
            "outcome": rng.choice([0, 1], rows),
        }
    )


class TestBuildFeatureSpec:
    def test_target_not_in_features(self):
        spec = build_feature_spec(_make_df(), "outcome", "classification")
        assert "outcome" not in [f.name for f in spec.features]

    def test_id_like_columns_excluded_with_reason(self):
        spec = build_feature_spec(_make_df(), "outcome", "classification")
        excluded = {e.name: e.reason for e in spec.excluded}
        assert excluded["record_id"] == "id_like"

    def test_high_missing_columns_excluded(self):
        spec = build_feature_spec(_make_df(), "outcome", "classification")
        excluded = {e.name: e.reason for e in spec.excluded}
        assert excluded["mostly_missing"] == "high_missing"

    def test_numeric_feature_has_bounds_and_default(self):
        spec = build_feature_spec(_make_df(), "outcome", "classification")
        age = next(f for f in spec.features if f.name == "age")
        assert age.kind == "numeric"
        assert age.min_value is not None and age.max_value is not None
        assert age.min_value <= age.default <= age.max_value

    def test_categorical_feature_has_categories_and_default(self):
        spec = build_feature_spec(_make_df(), "outcome", "classification")
        sex = next(f for f in spec.features if f.name == "sex")
        assert sex.kind == "categorical"
        assert set(sex.categories) == {"male", "female"}
        assert sex.default in sex.categories

    def test_classification_target_records_classes(self):
        spec = build_feature_spec(_make_df(), "outcome", "classification")
        assert spec.target.classes == ("0", "1")

    def test_regression_target_has_no_classes(self):
        df = _make_df()
        df["price"] = np.random.default_rng(1).uniform(100, 500, len(df))
        spec = build_feature_spec(df, "price", "regression")
        assert spec.target.classes is None

    def test_high_cardinality_capped_with_other(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            {
                "city": rng.choice([f"city_{i}" for i in range(60)], 3000),
                "y": rng.choice([0, 1], 3000),
            }
        )
        spec = build_feature_spec(df, "y", "classification")
        city = next(f for f in spec.features if f.name == "city")
        assert len(city.categories) == 51  # top 50 + "Other"
        assert "Other" in city.categories

    def test_all_null_target_rejected(self):
        df = pd.DataFrame({"x": range(60), "y": [None] * 60})
        with pytest.raises(TrainingError):
            build_feature_spec(df, "y", "classification")

    def test_missing_target_column_rejected(self):
        with pytest.raises(TrainingError):
            build_feature_spec(_make_df(), "nope", "classification")


class TestApplyFeatureSpec:
    def test_columns_match_spec_order(self):
        df = _make_df()
        spec = build_feature_spec(df, "outcome", "classification")
        X = apply_feature_spec(df, spec)
        assert list(X.columns) == [f.name for f in spec.features]

    def test_unseen_category_maps_to_nan_without_other(self):
        df = _make_df()
        spec = build_feature_spec(df, "outcome", "classification")
        row = pd.DataFrame([{"age": 30.0, "sex": "unknown_value"}])
        X = apply_feature_spec(row, spec)
        assert pd.isna(X["sex"].iloc[0])

    def test_overflow_category_maps_to_other(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame(
            {
                "city": rng.choice([f"city_{i}" for i in range(60)], 3000),
                "y": rng.choice([0, 1], 3000),
            }
        )
        spec = build_feature_spec(df, "y", "classification")
        city = next(f for f in spec.features if f.name == "city")
        rare = next(c for c in (f"city_{i}" for i in range(60)) if c not in city.categories)
        X = apply_feature_spec(pd.DataFrame([{"city": rare}]), spec)
        assert X["city"].iloc[0] == "Other"

    def test_numeric_strings_coerced(self):
        df = _make_df()
        spec = build_feature_spec(df, "outcome", "classification")
        X = apply_feature_spec(pd.DataFrame([{"age": "42.5", "sex": "male"}]), spec)
        assert X["age"].iloc[0] == 42.5

    def test_missing_column_becomes_nan(self):
        df = _make_df()
        spec = build_feature_spec(df, "outcome", "classification")
        X = apply_feature_spec(pd.DataFrame([{"sex": "male"}]), spec)
        assert pd.isna(X["age"].iloc[0])

    def test_does_not_mutate_input(self):
        df = _make_df()
        spec = build_feature_spec(df, "outcome", "classification")
        before = df.copy()
        apply_feature_spec(df, spec)
        pd.testing.assert_frame_equal(df, before)
