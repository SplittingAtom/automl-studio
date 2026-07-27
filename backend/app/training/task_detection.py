"""Detect whether a target column implies classification or regression."""

from typing import Literal

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

Task = Literal["classification", "regression"]

CLASSIFICATION_MAX_UNIQUE = 10


def detect_task(target: pd.Series) -> Task:
    nonnull = target.dropna()
    if is_bool_dtype(target) or not is_numeric_dtype(target):
        return "classification"
    if nonnull.nunique() <= CLASSIFICATION_MAX_UNIQUE:
        return "classification"
    return "regression"
