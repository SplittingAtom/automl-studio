"""Train an XGBoost model with sensible defaults.

Standard effort uses fixed parameters; "thorough" adds a small random search.
Every run also produces 5-fold CV stats, a held-out validation frame, a
threshold curve (binary), and a quantile interval model (regression).
"""

import logging
from typing import Any, Literal

import numpy as np
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
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from app.datasets.schemas import ProfileWarning
from app.training.baseline import baseline_metrics
from app.training.preprocessing import FeatureSpec, TrainingError, apply_feature_spec

logger = logging.getLogger(__name__)

SEED = 42
TEST_FRACTION = 0.2
VALIDATION_FRACTION = 0.1
EARLY_STOPPING_MIN_ROWS = 200
MIN_TRAINING_ROWS = 50
MAX_TARGET_MISSING_PCT = 30.0
IMBALANCE_WARNING_FRACTION = 0.10
LEAKAGE_SCORE_THRESHOLD = 0.999
CV_FOLDS = 5
CV_MAX_ROWS = 8000
THRESHOLD_STEPS = 21
INTERVAL_QUANTILES = [0.1, 0.9]  # 80% prediction band
TUNING_TRIALS = 12
TUNING_MAX_ROWS = 10_000

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
    interval_model: Any = None  # quantile regressor for prediction bands
    validation: Any = None  # held-out rows with actual vs predicted (DataFrame)


def train_model(
    df: pd.DataFrame,
    spec: FeatureSpec,
    max_rows: int = 100_000,
    effort: Literal["standard", "thorough"] = "standard",
    time_mode: bool = False,
    generated_columns: tuple[str, ...] = (),
    horizon: int = 0,
) -> TrainingResult:
    warnings: list[ProfileWarning] = []
    data, n_dropped = _drop_missing_target(df, spec, warnings)
    if len(data) < MIN_TRAINING_ROWS:
        raise TrainingError(
            "TOO_FEW_ROWS",
            f"Only {len(data)} usable rows — at least {MIN_TRAINING_ROWS} are needed to train a model.",
        )
    if len(data) > max_rows:
        if time_mode:
            data = data.tail(max_rows)
            message = f"Trained on the most recent {max_rows:,} rows for speed."
        else:
            data = data.sample(max_rows, random_state=SEED)
            message = f"Trained on a random sample of {max_rows:,} rows for speed."
        warnings.append(ProfileWarning(code="ROW_SAMPLE", message=message))
    if time_mode:
        gap_note = (
            f" A {horizon}-row gap was left between training and test rows so a "
            f"{horizon}-step-ahead target can't overlap the test period."
            if horizon
            else ""
        )
        warnings.append(
            ProfileWarning(
                code="TIME_SPLIT",
                message=(
                    "Time-aware training: the model was tested on the most recent "
                    f"{TEST_FRACTION:.0%} of rows — it never saw the future while learning."
                    + gap_note
                ),
            )
        )
        if generated_columns:
            warnings.append(
                ProfileWarning(
                    code="LAG_FEATURES",
                    message=(
                        "Added recent-history columns based on past values of the target: "
                        f"{', '.join(generated_columns)}."
                    ),
                )
            )

    X = apply_feature_spec(data, spec)
    if not spec.features:
        raise TrainingError(
            "NO_USABLE_FEATURES", "No usable input columns were found in this dataset."
        )
    is_classification = spec.target.task == "classification"
    y = _encode_target(data, spec, warnings)

    if time_mode:
        X_train, X_test, y_train, y_test = _time_split(X, y, TEST_FRACTION, horizon)
        if len(X_train) < MIN_TRAINING_ROWS // 2:
            raise TrainingError(
                "HORIZON_TOO_LARGE",
                f"A prediction horizon of {horizon} rows leaves too little training "
                "data. Use a smaller horizon or more rows.",
            )
    else:
        stratify = y if is_classification and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=stratify
        )
    balance = (
        is_classification
        and y.value_counts(normalize=True).min() < IMBALANCE_WARNING_FRACTION
    )
    params = dict(XGB_PARAMS)
    tuning_trials = 0
    if effort == "thorough" and len(X_train) >= EARLY_STOPPING_MIN_ROWS:
        params, tuning_trials = _tuned_params(
            X_train, y_train, is_classification, balance, time_mode, horizon
        )
    model, best_iteration = _fit(
        X_train, y_train, is_classification, balance, params, time_mode, horizon
    )
    metrics = (
        _classification_metrics(model, X_test, y_test, spec)
        if is_classification
        else _regression_metrics(model, X_test, y_test)
    )
    metrics.update(baseline_metrics(X_train, X_test, y_train, y_test, spec))
    metrics.update(
        _cross_validate(X, y, is_classification, best_iteration, params, time_mode, horizon)
    )
    if tuning_trials:
        metrics["tuning_trials"] = tuning_trials
    importance = _importance(model, X.columns)
    _check_leakage(metrics, importance, warnings)
    _check_simple_relationships(metrics, is_classification, warnings)
    return TrainingResult(
        metrics=metrics,
        importance=importance,
        warnings=tuple(warnings),
        n_rows_used=len(data),
        best_iteration=best_iteration,
        model=model,
        interval_model=None
        if is_classification
        else _fit_interval_model(X_train, y_train, best_iteration),
        validation=_validation_frame(data, X_test, y_test, spec, model),
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


def _time_split(X, y, test_fraction, gap):
    """Chronological split with an embargo: purge `gap` rows before the test
    block so multi-horizon targets in late training rows can't overlap it."""
    cut = int(len(X) * (1 - test_fraction))
    train_end = max(0, cut - gap)
    return X.iloc[:train_end], X.iloc[cut:], y.iloc[:train_end], y.iloc[cut:]


def _train_val_split(X_train, y_train, is_classification, time_mode, gap=0):
    """Validation slice for early stopping — the most recent rows in time mode."""
    if time_mode:
        return _time_split(X_train, y_train, VALIDATION_FRACTION, gap)
    stratify = y_train if is_classification and y_train.value_counts().min() >= 2 else None
    return train_test_split(
        X_train, y_train, test_size=VALIDATION_FRACTION, random_state=SEED, stratify=stratify
    )


def _fit(
    X_train, y_train, is_classification, balance=False, params=None, time_mode=False, gap=0
):
    params = params or XGB_PARAMS
    estimator_cls = XGBClassifier if is_classification else XGBRegressor
    use_early_stopping = len(X_train) >= EARLY_STOPPING_MIN_ROWS
    if not use_early_stopping:
        model = estimator_cls(**params)
        model.fit(X_train, y_train, sample_weight=_weights(y_train, balance), verbose=False)
        return model, None

    X_fit, X_val, y_fit, y_val = _train_val_split(
        X_train, y_train, is_classification, time_mode, gap
    )
    model = estimator_cls(**params, early_stopping_rounds=25)
    model.fit(
        X_fit,
        y_fit,
        sample_weight=_weights(y_fit, balance),
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model, int(model.best_iteration)


def _tuned_params(X_train, y_train, is_classification, balance, time_mode=False, gap=0):
    """Small random search; scores each trial on an early-stopping val split."""
    rng = np.random.default_rng(SEED)
    X_fit, X_val, y_fit, y_val = _train_val_split(
        X_train, y_train, is_classification, time_mode, gap
    )
    if len(X_fit) > TUNING_MAX_ROWS:
        if time_mode:
            X_fit, y_fit = X_fit.tail(TUNING_MAX_ROWS), y_fit.tail(TUNING_MAX_ROWS)
        else:
            sampled = X_fit.sample(TUNING_MAX_ROWS, random_state=SEED)
            X_fit, y_fit = sampled, y_fit.loc[sampled.index]
    estimator_cls = XGBClassifier if is_classification else XGBRegressor
    best_params, best_score = dict(XGB_PARAMS), None
    for _ in range(TUNING_TRIALS):
        trial = {
            **XGB_PARAMS,
            "max_depth": int(rng.integers(3, 9)),
            "learning_rate": round(float(10 ** rng.uniform(-2, -0.7)), 4),
            "subsample": round(float(rng.uniform(0.6, 1.0)), 3),
            "colsample_bytree": round(float(rng.uniform(0.6, 1.0)), 3),
            "min_child_weight": int(rng.choice([1, 2, 4, 8])),
        }
        model = estimator_cls(**trial, early_stopping_rounds=25)
        model.fit(
            X_fit,
            y_fit,
            sample_weight=_weights(y_fit, balance),
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        score = float(model.best_score)  # logloss / rmse: lower is better
        if best_score is None or score < best_score:
            best_score, best_params = score, trial
    return best_params, TUNING_TRIALS


def _cross_validate(
    X, y, is_classification, best_iteration, params, time_mode=False, gap=0
) -> dict[str, Any]:
    """K-fold spread of the headline metric — how stable is this model?

    Time mode uses walk-forward (expanding-window) folds: every fold trains
    on the past and scores on the block that follows it.
    """
    try:
        if len(X) > CV_MAX_ROWS:
            if time_mode:
                X, y = X.tail(CV_MAX_ROWS), y.tail(CV_MAX_ROWS)
            else:
                X = X.sample(CV_MAX_ROWS, random_state=SEED)
                y = y.loc[X.index]
        n_estimators = min(int(best_iteration) + 1 if best_iteration else 300, 300)
        cv_params = {**params, "n_estimators": n_estimators}
        use_stratified = (
            not time_mode and is_classification and y.value_counts().min() >= CV_FOLDS
        )
        if time_mode:
            splitter = TimeSeriesSplit(n_splits=CV_FOLDS, gap=gap)
        elif use_stratified:
            splitter = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
        else:
            splitter = KFold(CV_FOLDS, shuffle=True, random_state=SEED)
        estimator_cls = XGBClassifier if is_classification else XGBRegressor
        scores = []
        for train_idx, test_idx in splitter.split(X, y if use_stratified else None):
            model = estimator_cls(**cv_params)
            model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
            predicted = model.predict(X.iloc[test_idx])
            actual = y.iloc[test_idx]
            score = (
                accuracy_score(actual, predicted)
                if is_classification
                else r2_score(actual, predicted)
            )
            scores.append(float(score))
        return {
            "cv_mean": round(float(np.mean(scores)), 4),
            "cv_std": round(float(np.std(scores)), 4),
            "cv_folds": CV_FOLDS,
        }
    except Exception:
        logger.exception("Cross-validation failed; continuing without it")
        return {}


def _fit_interval_model(X_train, y_train, best_iteration):
    """Quantile regressor giving an 80% prediction band."""
    try:
        params = {
            **XGB_PARAMS,
            "objective": "reg:quantileerror",
            "quantile_alpha": INTERVAL_QUANTILES,
            "n_estimators": min(int(best_iteration) + 1 if best_iteration else 200, 200),
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        return model
    except Exception:
        logger.exception("Interval model failed; predictions will have no band")
        return None


def _validation_frame(data, X_test, y_test, spec, model) -> pd.DataFrame:
    """Held-out rows with actual vs predicted, for the validation view."""
    frame = data.loc[X_test.index].copy()
    predicted_col = _free_name(frame, "predicted")
    if spec.target.task == "classification":
        probabilities = model.predict_proba(X_test)
        winners = probabilities.argmax(axis=1)
        classes = spec.target.classes
        frame[predicted_col] = [classes[i] for i in winners]
        frame[_free_name(frame, "confidence")] = probabilities.max(axis=1).round(4)
        actual = frame[spec.target.name].astype(str).to_numpy()
        frame[_free_name(frame, "correct")] = actual == frame[predicted_col].to_numpy()
    else:
        predictions = model.predict(X_test).astype(float).round(4)
        frame[predicted_col] = predictions
        actual_values = pd.to_numeric(frame[spec.target.name], errors="coerce")
        frame[_free_name(frame, "error")] = (predictions - actual_values).round(4)
    return frame


def _free_name(frame, name: str) -> str:
    return name if name not in frame.columns else f"{name}_model"


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
        metrics["threshold_curve"] = _threshold_curve(y_test.to_numpy(), probabilities)
    return metrics


def _threshold_curve(actual: np.ndarray, probability_positive: np.ndarray) -> list[dict]:
    """Precision/recall/accuracy at each decision threshold, for the UI slider."""
    points = []
    total_positive = int((actual == 1).sum())
    for threshold in np.linspace(0.0, 1.0, THRESHOLD_STEPS):
        flagged = probability_positive >= threshold
        true_positive = int(((actual == 1) & flagged).sum())
        false_positive = int(((actual == 0) & flagged).sum())
        n_flagged = true_positive + false_positive
        points.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(true_positive / n_flagged, 4) if n_flagged else None,
                "recall": round(true_positive / total_positive, 4) if total_positive else 0.0,
                "accuracy": round(float((flagged == (actual == 1)).mean()), 4),
                "flagged_pct": round(100.0 * n_flagged / len(actual), 1),
            }
        )
    return points


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


SIMPLE_CLASSIFICATION_GAP = 0.02
SIMPLE_REGRESSION_GAP = 0.03


def _check_simple_relationships(
    metrics: dict[str, Any], is_classification: bool, warnings: list[ProfileWarning]
) -> None:
    """If a linear model nearly matches XGBoost, the patterns are mostly simple."""
    if is_classification:
        linear, main = metrics.get("linear_accuracy"), metrics.get("accuracy")
        close = linear is not None and linear >= main - SIMPLE_CLASSIFICATION_GAP
        extra = ""
    else:
        linear, main = metrics.get("linear_r2"), metrics.get("r2")
        close = linear is not None and linear >= main - SIMPLE_REGRESSION_GAP
        extra = (
            " For what-if values outside the range of your data, simple trends like "
            "this are often more trustworthy than the model's flat extrapolation."
        )
    if close:
        warnings.append(
            ProfileWarning(
                code="SIMPLE_RELATIONSHIPS",
                message=(
                    "A basic statistical model does nearly as well as the advanced one — "
                    "the patterns in your data are mostly simple. That's fine; it just "
                    "means the extra complexity isn't adding much." + extra
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
