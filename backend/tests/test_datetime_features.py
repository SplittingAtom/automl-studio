"""Tests for datetime feature derivation (year/month/dayofweek)."""

import numpy as np
import pandas as pd

from app.training.preprocessing import apply_feature_spec, build_feature_spec


def _df(rows=300):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=rows, freq="7D")
    return pd.DataFrame(
        {
            "signup_date": dates.strftime("%Y-%m-%d"),
            "amount": rng.uniform(10, 500, rows),
            "churned": rng.choice([0, 1], rows),
        }
    )


class TestBuildSpec:
    def test_datetime_column_becomes_derived_numeric_features(self):
        spec = build_feature_spec(_df(), "churned", "classification")
        names = [f.name for f in spec.features]
        assert "signup_date_year" in names
        assert "signup_date_month" in names
        assert "signup_date_dayofweek" not in names  # weekly cadence -> constant weekday

    def test_derived_features_record_their_source(self):
        spec = build_feature_spec(_df(), "churned", "classification")
        year = next(f for f in spec.features if f.name == "signup_date_year")
        assert year.derived_from == "signup_date"
        assert year.date_part == "year"
        assert year.kind == "numeric"

    def test_derived_bounds_feed_sliders(self):
        spec = build_feature_spec(_df(), "churned", "classification")
        month = next(f for f in spec.features if f.name == "signup_date_month")
        assert month.min_value == 1
        assert month.max_value == 12

    def test_raw_datetime_column_not_excluded(self):
        spec = build_feature_spec(_df(), "churned", "classification")
        assert "signup_date" not in {e.name for e in spec.excluded}


class TestApplySpec:
    def test_training_path_derives_from_raw_column(self):
        df = _df()
        spec = build_feature_spec(df, "churned", "classification")
        X = apply_feature_spec(df, spec)
        assert X["signup_date_year"].iloc[0] == 2020.0
        assert X["signup_date_month"].iloc[0] == 1.0

    def test_predict_path_uses_derived_values_directly(self):
        spec = build_feature_spec(_df(), "churned", "classification")
        row = pd.DataFrame([{"signup_date_year": 2023, "signup_date_month": 6, "amount": 100}])
        X = apply_feature_spec(row, spec)
        assert X["signup_date_year"].iloc[0] == 2023.0
        assert X["signup_date_month"].iloc[0] == 6.0

    def test_unparseable_dates_become_nan(self):
        df = _df()
        spec = build_feature_spec(df, "churned", "classification")
        bad = pd.DataFrame([{"signup_date": "not a date", "amount": 50}])
        X = apply_feature_spec(bad, spec)
        assert pd.isna(X["signup_date_year"].iloc[0])
