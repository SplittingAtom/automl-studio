"""Train an XGBoost model with sensible defaults — no hyperparameter search."""

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from app.datasets.schemas import ProfileWarning
from app.training.preprocessing import FeatureSpec, TrainingError, apply_feature_spec

SEED = 42
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.1
EARLY_STOPPING_MIN_ROWS = 200
MIN_TRAINING_ROWS = 50
MAX_TARGET_MISSING_PCT = 30.0
IMBALANCE_WARNING_FRACTION = 0.10
LEAKAGE_SCORE_THRESHOLD = 0.999

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "tree_method": "hist",
    "enable_categorical": True,
    "importance_type": "gain",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


class ImportanceItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    feature: str
    score: float


class TrainingResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    metrics: dict[str, Any]
    importance: tuple[ImportanceItem, ...]
    warnings: tuple[ProfileWarning, ...]
    n_rows_used: int
    best_iteration: int | None
    model: Any  # fitted XGBoost estimator; persisted via joblib, never JSON


def train_model(
    df: pd.DataFrame, spec: FeatureSpec, max_rows: int = 100_000
) -> TrainingResult:
    warnings: list[ProfileWarning] = []
    data, n_dropped = _drop_missing_target(df, spec, warnings)
    if len(data) < MIN_TRAINING_ROWS:
        raise TrainingError(
            "TOO_FEW_ROWS",
            f"Only {len(data)} usable rows — at least {MIN_TRAINING_ROWS} are needed to train a model.",
        )
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=SEED)
        warnings.append(
            ProfileWarning(
                code="ROW_SAMPLE",
                message=f"Trained on a random sample of {max_rows:,} rows for speed.",
            )
        )

    X = apply_feature_spec(data, spec)
    if not spec.features:
        raise TrainingError(
            "NO_USABLE_FEATURES", "No usable input columns were found in this dataset."
        )
    is_classification = spec.target.task == "classification"
    y = _encode_target(data, spec, warnings)

    stratify = y if is_classification and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=stratify
    )
    balance = (
        is_classification
        and y.value_counts(normalize=True).min() < IMBALANCE_WARNING_FRACTION
    )
    model, best_iteration = _fit(X_train, y_train, is_classification, balance)
    metrics = (
        _classification_metrics(model, X_test, y_test, spec)
        if is_classification
        else _regression_metrics(model, X_test, y_test)
    )
    importance = _importance(model, X.columns)
    _check_leakage(metrics, importance, warnings)
    return TrainingResult(
        metrics=metrics,
        importance=importance,
        warnings=tuple(warnings),
        n_rows_used=len(data),
        best_iteration=best_iteration,
        model=model,
    )


def _drop_missing_target(df, spec, warnings) -> tuple[pd.DataFrame, int]:
    mask = df[spec.target.name].notna()
    n_dropped = int((~mask).sum())
    if len(df) and 100.0 * n_dropped / len(df) > MAX_TARGET_MISSING_PCT:
        raise TrainingError(
            "TARGET_TOO_SPARSE",
            f'More than {MAX_TARGET_MISSING_PCT:.0f}% of "{spec.target.name}" values are missing — '
            "pick a column with more complete data.",
        )
    if n_dropped:
        warnings.append(
            ProfileWarning(
                code="TARGET_MISSING_DROPPED",
                message=f'{n_dropped} rows were skipped because "{spec.target.name}" was empty.',
            )
        )
    return df[mask], n_dropped


def _encode_target(data, spec, warnings) -> pd.Series:
    target = data[spec.target.name]
    if spec.target.task == "regression":
        return pd.to_numeric(target, errors="coerce")
    class_index = {c: i for i, c in enumerate(spec.target.classes)}
    y = target.astype(str).map(class_index)
    counts = y.value_counts(normalize=True)
    if counts.min() < IMBALANCE_WARNING_FRACTION:
        rare = spec.target.classes[counts.idxmin()]
        warnings.append(
            ProfileWarning(
                code="CLASS_IMBALANCE",
                message=(
                    f'"{rare}" appears in under {IMBALANCE_WARNING_FRACTION:.0%} of rows — '
                    "those rows were given extra weight during training, but predictions "
                    "for it may still be less reliable."
                ),
            )
        )
    return y


def _fit(X_train, y_train, is_classification, balance=False):
    estimator_cls = XGBClassifier if is_classification else XGBRegressor
    use_early_stopping = len(X_train) >= EARLY_STOPPING_MIN_ROWS
    if not use_early_stopping:
        model = estimator_cls(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=_weights(y_train, balance), verbose=False)
        return model, None

    stratify = y_train if is_classification and y_train.value_counts().min() >= 2 else None
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_FRACTION, random_state=SEED, stratify=stratify
    )
    model = estimator_cls(**XGB_PARAMS, early_stopping_rounds=25)
    model.fit(
        X_fit,
        y_fit,
        sample_weight=_weights(y_fit, balance),
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model, int(model.best_iteration)


def _weights(y, balance: bool):
    """Give minority-class rows extra weight when the target is imbalanced."""
    return compute_sample_weight("balanced", y) if balance else None


def _classification_metrics(model, X_test, y_test, spec) -> dict[str, Any]:
    predicted = model.predict(X_test)
    classes = list(spec.target.classes)
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
        "f1_weighted": round(float(f1_score(y_test, predicted, average="weighted")), 4),
        "roc_auc": None,
        "confusion_matrix": confusion_matrix(
            y_test, predicted, labels=range(len(classes))
        ).tolist(),
        "classes": classes,
        "test_rows": len(X_test),
    }
    if len(classes) == 2:
        probabilities = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, probabilities)), 4)
    return metrics


def _regression_metrics(model, X_test, y_test) -> dict[str, Any]:
    predicted = model.predict(X_test)
    return {
        "r2": round(float(r2_score(y_test, predicted)), 4),
        "mae": round(float(mean_absolute_error(y_test, predicted)), 4),
        "rmse": round(float(root_mean_squared_error(y_test, predicted)), 4),
        "target_mean": round(float(y_test.mean()), 4),
        "target_std": round(float(y_test.std()), 4),
        "test_rows": len(X_test),
    }


LEAKAGE_IMPORTANCE_SHARE = 0.9


def _check_leakage(
    metrics: dict[str, Any],
    importance: tuple[ImportanceItem, ...],
    warnings: list[ProfileWarning],
) -> None:
    """A perfect score alone isn't leakage — some datasets are genuinely easy.

    The leakage signature is a perfect score AND one column doing nearly all
    the work (a look-alike of the target).
    """
    score = metrics.get("accuracy") or metrics.get("r2")
    if score is None or score <= LEAKAGE_SCORE_THRESHOLD or not importance:
        return
    top = importance[0]
    if top.score < LEAKAGE_IMPORTANCE_SHARE:
        return
    warnings.append(
        ProfileWarning(
            code="POSSIBLE_LEAKAGE",
            column=top.feature,
            message=(
                f'The model is suspiciously perfect, and "{top.feature}" is doing almost '
                "all the work — it may already contain the answer. Consider leaving it "
                "out and retraining."
            ),
        )
    )


def _importance(model, feature_names) -> tuple[ImportanceItem, ...]:
    raw = model.feature_importances_
    total = float(raw.sum())
    if total <= 0:
        return tuple(
            ImportanceItem(feature=str(f), score=0.0) for f in feature_names
        )
    items = [
        ImportanceItem(feature=str(name), score=round(float(score) / total, 4))
        for name, score in zip(feature_names, raw)
    ]
    return tuple(sorted(items, key=lambda i: i.score, reverse=True))
