"""Dataset suitability analysis: recommend targets and grade the dataset.

For each eligible column, a small, fast XGBoost probe measures how well the
OTHER columns predict it, compared to a naive baseline (majority class /
mean). That predictability signal — not just structure — drives the ranking,
and the probe's feature importances become "best predictor" hints.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from app.analysis.schemas import (
    AnalysisPoint,
    DatasetAnalysis,
    PredictorHint,
    TargetCandidate,
)
from app.datasets.schemas import ColumnProfile
from app.training.preprocessing import TrainingError, apply_feature_spec, build_feature_spec
from app.training.task_detection import Task, detect_task

logger = logging.getLogger(__name__)

SEED = 42
PROBE_ROWS = 3000
PROBE_MIN_ROWS = 40
MAX_PROBED_CANDIDATES = 30
MAX_TARGET_CLASSES = 20
MAX_TARGET_MISSING_PCT = 30.0
TOP_PREDICTORS = 5
RECOMMEND_MIN_SCORE = 55
RECOMMEND_MIN_SIGNAL = 0.15
NEAR_DUPLICATE_SHARE = 0.9
NEAR_DUPLICATE_SIGNAL = 0.75
NEAR_DUPLICATE_PENALTY = 25
# Near-perfect predictability usually means the column is CALCULATED from the
# others (MonthlyCharges = sum of services) — true, but worthless to predict.
DERIVED_CLASSIFICATION_ACCURACY = 0.99
DERIVED_REGRESSION_R2 = 0.99
DERIVED_PENALTY = 40

PROBE_PARAMS = {
    "n_estimators": 40,
    "max_depth": 4,
    "learning_rate": 0.2,
    "tree_method": "hist",
    "enable_categorical": True,
    "importance_type": "gain",
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": 0,
}


def analyze_dataset(
    dataset_id: str, df: pd.DataFrame, columns: tuple[ColumnProfile, ...]
) -> DatasetAnalysis:
    sample = df.sample(PROBE_ROWS, random_state=SEED) if len(df) > PROBE_ROWS else df
    eligible = _eligible_columns(df, columns)
    shortlisted = sorted(eligible, key=lambda c: _structural_score(c), reverse=True)
    truncated = len(shortlisted) > MAX_PROBED_CANDIDATES
    shortlisted = shortlisted[:MAX_PROBED_CANDIDATES]

    candidates = []
    for column in shortlisted:
        candidate = _evaluate_candidate(sample, df, column)
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = tuple(
        c.model_copy(
            update={
                "recommended": not c.derived_like
                and c.score >= RECOMMEND_MIN_SCORE
                and c.signal >= RECOMMEND_MIN_SIGNAL
            }
        )
        for c in candidates
    )
    rating, summary, points = _overall(df, columns, candidates, truncated)
    return DatasetAnalysis(
        dataset_id=dataset_id,
        rating=rating,
        summary=summary,
        points=tuple(points),
        candidates=candidates,
    )


def _eligible_columns(df, columns) -> list[ColumnProfile]:
    eligible = []
    for column in columns:
        if column.kind not in ("numeric", "categorical"):
            continue
        if column.missing_pct > MAX_TARGET_MISSING_PCT or column.unique_count < 2:
            continue
        if column.kind == "categorical" and column.unique_count > MAX_TARGET_CLASSES:
            continue
        eligible.append(column)
    return eligible


def _structural_score(column: ColumnProfile) -> float:
    completeness = (100.0 - column.missing_pct) / 100.0
    balance = _balance(column)
    return completeness + balance


def _evaluate_candidate(
    sample: pd.DataFrame, full_df: pd.DataFrame, column: ColumnProfile
) -> TargetCandidate | None:
    task = detect_task(full_df[column.name])
    try:
        probe = _probe(sample, column.name, task)
    except TrainingError:
        return None
    except Exception:
        logger.exception("Probe failed for column %s", column.name)
        return None
    if probe is None:
        return None
    probe_score, baseline, signal, top_predictors = probe
    score = _blended_score(column, task, signal)
    reasons = _reasons(column, task, probe_score, baseline, signal)
    caution = _caution(task, probe_score, signal, top_predictors)
    derived_like = caution is not None
    if derived_like:
        message, penalty = caution
        score = max(0, score - penalty)
        reasons.insert(0, message)
    return TargetCandidate(
        column=column.name,
        task=task,
        score=score,
        recommended=False,  # finalized after ranking
        derived_like=derived_like,
        signal=round(signal, 4),
        probe_score=round(probe_score, 4),
        baseline_score=round(baseline, 4),
        reasons=tuple(reasons),
        top_predictors=top_predictors,
    )


def _caution(task, probe_score, signal, top_predictors) -> tuple[str, int] | None:
    """Detect targets that are technically predictable but pointless to predict."""
    dominant = top_predictors and top_predictors[0].share >= NEAR_DUPLICATE_SHARE
    derived_threshold = (
        DERIVED_CLASSIFICATION_ACCURACY if task == "classification" else DERIVED_REGRESSION_R2
    )
    if probe_score >= derived_threshold and dominant:
        return (
            f'Almost perfectly explained by "{top_predictors[0].name}" alone — they '
            "may be two versions of the same thing.",
            DERIVED_PENALTY,
        )
    if probe_score >= derived_threshold:
        return (
            "So predictable it's probably calculated from the other columns — "
            "predicting it wouldn't tell you anything new.",
            DERIVED_PENALTY,
        )
    if dominant and signal >= NEAR_DUPLICATE_SIGNAL:
        return (
            f'Almost entirely explained by "{top_predictors[0].name}" alone — check '
            "they aren't two versions of the same thing.",
            NEAR_DUPLICATE_PENALTY,
        )
    return None


def _probe(sample: pd.DataFrame, target: str, task: Task):
    data = sample[sample[target].notna()]
    if len(data) < PROBE_MIN_ROWS:
        return None
    spec = build_feature_spec(data, target, task)
    if not spec.features:
        return None
    X = apply_feature_spec(data, spec)
    if task == "classification":
        classes = {c: i for i, c in enumerate(spec.target.classes)}
        y = data[target].astype(str).map(classes)
    else:
        y = pd.to_numeric(data[target], errors="coerce")

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=stratify
    )
    estimator_cls = XGBClassifier if task == "classification" else XGBRegressor
    model = estimator_cls(**PROBE_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    if task == "classification":
        probe_score = float((model.predict(X_test) == y_test).mean())
        baseline = float(y_test.value_counts(normalize=True).max())
        signal = max(0.0, (probe_score - baseline) / max(1e-9, 1.0 - baseline))
    else:
        probe_score = float(r2_score(y_test, model.predict(X_test)))
        baseline = 0.0
        signal = min(1.0, max(0.0, probe_score))
    return probe_score, baseline, signal, _top_predictors(model, X.columns)


def _top_predictors(model, feature_names) -> tuple[PredictorHint, ...]:
    raw = model.feature_importances_
    total = float(raw.sum())
    if total <= 0:
        return ()
    ranked = sorted(zip(feature_names, raw), key=lambda pair: pair[1], reverse=True)
    return tuple(
        PredictorHint(name=str(name), share=round(float(value) / total, 4))
        for name, value in ranked[:TOP_PREDICTORS]
        if value / total >= 0.02
    )


def _balance(column: ColumnProfile) -> float:
    """0-1: how evenly spread a categorical column's values are (1 for numeric)."""
    if column.kind != "categorical" or not column.top_values:
        return 1.0
    counts = np.array([t.count for t in column.top_values], dtype=float)
    fractions = counts / counts.sum()
    entropy = float(-(fractions * np.log(fractions)).sum())
    max_entropy = np.log(len(fractions)) if len(fractions) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _simplicity(column: ColumnProfile, task: Task) -> float:
    if task == "regression" or column.unique_count == 2:
        return 1.0
    return 0.7 if column.unique_count <= 10 else 0.3


def _blended_score(column: ColumnProfile, task: Task, signal: float) -> int:
    completeness = (100.0 - column.missing_pct) / 100.0
    score = (
        55 * signal
        + 20 * completeness
        + 15 * _balance(column)
        + 10 * _simplicity(column, task)
    )
    return int(round(score))


def _reasons(column, task, probe_score, baseline, signal) -> list[str]:
    reasons = []
    if task == "classification":
        if signal >= 0.15:
            reasons.append(
                f"The other columns predict it well — {probe_score:.0%} correct vs "
                f"{baseline:.0%} by always guessing the most common value."
            )
        else:
            reasons.append("The other columns barely predict it — expect a weak model.")
        if column.unique_count == 2:
            reasons.append("Simple yes/no-style outcome.")
        balance = _balance(column)
        if balance < 0.5 and column.top_values:
            reasons.append(f'Imbalanced — "{column.top_values[0].value}" dominates.')
    else:
        if signal >= 0.15:
            reasons.append(
                f"The other columns explain {max(0.0, probe_score):.0%} of its variation."
            )
        else:
            reasons.append("The other columns explain almost none of its variation.")
    if column.missing_pct == 0:
        reasons.append("No missing values.")
    elif column.missing_pct > 5:
        reasons.append(f"{column.missing_pct:.0f}% of its values are missing.")
    return reasons


def _overall(df, columns, candidates, truncated):
    points: list[AnalysisPoint] = []
    rows = len(df)

    if rows < 100:
        points.append(
            AnalysisPoint(
                tone="bad",
                message=f"Very small dataset ({rows} rows) — results will be unstable.",
            )
        )
    elif rows < 500:
        points.append(
            AnalysisPoint(
                tone="warn",
                message=f"Smallish dataset ({rows} rows) — results may vary between runs.",
            )
        )
    else:
        points.append(
            AnalysisPoint(tone="good", message=f"{rows:,} rows — plenty to learn from.")
        )

    unusable = [c for c in columns if c.kind in ("id_like", "unsupported")]
    heavy_missing = [
        c for c in columns if c.kind in ("numeric", "categorical") and c.missing_pct > 50
    ]
    usable = len(columns) - len(unusable)
    points.append(
        AnalysisPoint(tone="good", message=f"{usable} of {len(columns)} columns are usable.")
    )
    if unusable:
        names = ", ".join(c.name for c in unusable[:4])
        points.append(
            AnalysisPoint(
                tone="warn",
                message=f"Ignoring {len(unusable)} ID-like or empty column(s): {names}.",
            )
        )
    if heavy_missing:
        names = ", ".join(c.name for c in heavy_missing[:4])
        points.append(
            AnalysisPoint(
                tone="warn", message=f"Heavy missing values in: {names}."
            )
        )
    if truncated:
        points.append(
            AnalysisPoint(
                tone="warn",
                message=f"Only the {MAX_PROBED_CANDIDATES} most promising columns were "
                "analyzed as prediction targets.",
            )
        )

    # Derived look-alikes don't make good headline recommendations
    best = next((c for c in candidates if not c.derived_like), None) or (
        candidates[0] if candidates else None
    )
    if best is None:
        points.append(
            AnalysisPoint(tone="bad", message="No column looks predictable from the others.")
        )
        return "poor", "This dataset isn't a good fit for training a model.", points

    if best.signal >= 0.2:
        points.append(
            AnalysisPoint(
                tone="good",
                message=f'"{best.column}" looks clearly predictable from the other columns.',
            )
        )
    elif best.signal >= 0.05:
        points.append(
            AnalysisPoint(
                tone="warn",
                message=f'"{best.column}" is only weakly predictable — models will be rough.',
            )
        )
    else:
        points.append(
            AnalysisPoint(
                tone="bad", message="Nothing here is predictable from the other columns."
            )
        )

    if best.signal < 0.05 or rows < 50:
        rating = "poor"
        summary = "This dataset will struggle to produce a useful model."
    elif best.signal >= 0.5 and rows >= 500 and best.score >= 70:
        rating = "great"
        summary = f'Looks great for modeling — try predicting "{best.column}".'
    elif best.signal < 0.2 or rows < 300:
        rating = "fair"
        summary = f'Workable with caveats — "{best.column}" is your best bet.'
    else:
        rating = "good"
        summary = f'Good fit for modeling — "{best.column}" is a strong target.'
    return rating, summary, points
