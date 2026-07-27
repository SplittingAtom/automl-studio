"""Tests for dataset profiling — column kind detection and stats."""

import numpy as np
import pandas as pd

from app.datasets.profiling import profile_dataframe


def _col(result, name):
    return next(c for c in result.columns if c.name == name)


def _codes(result):
    return {w.code for w in result.warnings}


class TestColumnKinds:
    def test_continuous_floats_are_numeric(self):
        df = pd.DataFrame({"x": np.linspace(0.5, 99.5, 200)})
        assert _col(profile_dataframe(df), "x").kind == "numeric"

    def test_high_unique_integers_are_id_like(self):
        df = pd.DataFrame({"passenger_id": range(1, 201)})
        result = profile_dataframe(df)
        assert _col(result, "passenger_id").kind == "id_like"
        assert "ID_LIKE_COLUMN" in _codes(result)

    def test_low_cardinality_integers_are_categorical(self):
        df = pd.DataFrame({"pclass": [1, 2, 3] * 50})
        assert _col(profile_dataframe(df), "pclass").kind == "categorical"

    def test_binary_numeric_is_categorical(self):
        df = pd.DataFrame({"survived": [0, 1] * 100})
        assert _col(profile_dataframe(df), "survived").kind == "categorical"

    def test_low_cardinality_strings_are_categorical(self):
        df = pd.DataFrame({"embarked": ["S", "C", "Q"] * 40})
        assert _col(profile_dataframe(df), "embarked").kind == "categorical"

    def test_unique_strings_are_id_like(self):
        df = pd.DataFrame({"name": [f"person_{i}" for i in range(50)]})
        assert _col(profile_dataframe(df), "name").kind == "id_like"

    def test_booleans_are_categorical(self):
        df = pd.DataFrame({"alone": [True, False] * 30})
        assert _col(profile_dataframe(df), "alone").kind == "categorical"

    def test_all_null_column_is_unsupported(self):
        df = pd.DataFrame({"empty": [None] * 20, "x": range(20)})
        result = profile_dataframe(df)
        assert _col(result, "empty").kind == "unsupported"
        assert "ALL_NULL_COLUMN" in _codes(result)

    def test_date_strings_are_datetime(self):
        dates = pd.date_range("2024-01-01", periods=60).strftime("%Y-%m-%d")
        df = pd.DataFrame({"signup_date": list(dates)})
        assert _col(profile_dataframe(df), "signup_date").kind == "datetime"

    def test_numeric_strings_are_not_datetime(self):
        df = pd.DataFrame({"code": [str(i) for i in range(100, 115)] * 10})
        assert _col(profile_dataframe(df), "code").kind == "categorical"


class TestStatsAndWarnings:
    def test_missing_percentage(self):
        df = pd.DataFrame({"age": [10.0, 20.0, None, None]})
        col = _col(profile_dataframe(df), "age")
        assert col.missing_count == 2
        assert col.missing_pct == 50.0

    def test_numeric_stats_feed_slider_bounds(self):
        df = pd.DataFrame({"fare": [10.0, 20.0, 30.0, 40.0]})
        stats = _col(profile_dataframe(df), "fare").stats
        assert stats is None  # only 4 unique values -> categorical, no stats

        df = pd.DataFrame({"fare": np.linspace(5.0, 100.0, 50)})
        stats = _col(profile_dataframe(df), "fare").stats
        assert stats.min == 5.0
        assert stats.max == 100.0
        assert stats.median == 52.5

    def test_categorical_top_values(self):
        df = pd.DataFrame({"sex": ["male"] * 60 + ["female"] * 40})
        top = _col(profile_dataframe(df), "sex").top_values
        assert top[0].value == "male"
        assert top[0].count == 60

    def test_high_missing_warning(self):
        df = pd.DataFrame({"deck": ["A"] * 2 + [None] * 8, "x": range(10)})
        assert "HIGH_MISSING" in _codes(profile_dataframe(df))

    def test_single_row_does_not_crash(self):
        df = pd.DataFrame({"a": [1], "b": ["x"]})
        result = profile_dataframe(df)
        assert len(result.columns) == 2
