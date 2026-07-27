"""Naive and linear baselines trained alongside XGBoost.

These exist for honesty, not accuracy: they let the UI answer "is this model
actually better than something simple?" — and for regression, the linear
model covers the tree family's extrapolation blind spot.
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.training.preprocessing import FeatureSpec

logger = logging.getLogger(__name__)


def baseline_metrics(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    spec: FeatureSpec,
) -> dict:
    """Return baseline_* and linear_* metric entries; never raises."""
    is_classification = spec.target.task == "classification"
    metrics = _naive_metrics(y_train, y_test, is_classification)
    try:
        metrics.update(
            _linear_metrics(X_train, X_test, y_train, y_test, spec, is_classification)
        )
    except Exception:
        logger.exception("Linear baseline failed; reporting naive baseline only")
    return metrics


def _naive_metrics(y_train, y_test, is_classification) -> dict:
    if is_classification:
        majority = y_train.value_counts().idxmax()
        return {"baseline_accuracy": round(float((y_test == majority).mean()), 4)}
    mean_prediction = float(y_train.mean())
    return {"baseline_mae": round(float((y_test - mean_prediction).abs().mean()), 4)}


def _linear_metrics(X_train, X_test, y_train, y_test, spec, is_classification) -> dict:
    numeric = [f.name for f in spec.features if f.kind == "numeric"]
    categorical = [f.name for f in spec.features if f.kind == "categorical"]
    model = Pipeline(
        [
            ("prep", _preprocessor(numeric, categorical)),
            (
                "model",
                LogisticRegression(max_iter=1000) if is_classification else Ridge(),
            ),
        ]
    )
    model.fit(_plain(X_train, categorical), y_train)
    predicted = model.predict(_plain(X_test, categorical))
    if is_classification:
        return {"linear_accuracy": round(float(accuracy_score(y_test, predicted)), 4)}
    return {
        "linear_mae": round(float(mean_absolute_error(y_test, predicted)), 4),
        "linear_r2": round(float(r2_score(y_test, predicted)), 4),
    }


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def _plain(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """sklearn imputers want plain object columns, not pandas Categorical."""
    converted = X.copy()
    for name in categorical:
        converted[name] = converted[name].astype(object)
    return converted
