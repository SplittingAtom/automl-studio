"""Tests for automatic task detection from the target column."""

import pandas as pd

from app.training.task_detection import detect_task


def test_continuous_floats_are_regression():
    s = pd.Series([1.5, 2.7, 3.14, 4.2, 5.9] * 10 + list(range(20)))
    assert detect_task(s) == "regression"


def test_binary_integers_are_classification():
    assert detect_task(pd.Series([0, 1] * 50)) == "classification"


def test_strings_are_classification():
    assert detect_task(pd.Series(["yes", "no"] * 30)) == "classification"


def test_booleans_are_classification():
    assert detect_task(pd.Series([True, False] * 30)) == "classification"


def test_many_unique_integers_are_regression():
    assert detect_task(pd.Series(range(100))) == "regression"


def test_few_unique_numerics_are_classification():
    assert detect_task(pd.Series([1, 2, 3, 4, 5] * 20)) == "classification"


def test_ignores_missing_values():
    assert detect_task(pd.Series([0, 1, None] * 30)) == "classification"
