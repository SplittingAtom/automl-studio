"""Suggest calculated columns a model might benefit from.

A bounded, explainable take on automatic feature engineering: build simple
ratio/product/difference candidates from the model's strongest numeric
columns, add them to a quick probe fit, and keep only the candidates the
probe actually leans on. Every suggestion is a plain formula the user can
read, add as a calculated column, and retrain with.
"""

import re

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from app.analysis.schemas import FeatureIdea, FeatureIdeasResponse
from app.training.preprocessing import FeatureSpec, apply_feature_spec
from app.training.trainer import ImportanceItem

SEED = 42
MAX_BASE_FEATURES = 4
PROBE_ROWS = 3000
MIN_SHARE = 0.05
MAX_IDEAS = 5
MAX_MISSING_SHARE = 0.3
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

PROBE_PARAMS = dict(
    n_estimators=60,
    max_depth=4,
    learning_rate=0.1,
    tree_method="hist",
    enable_categorical=True,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)


def compute_feature_ideas(
    df: pd.DataFrame, spec: FeatureSpec, importance: tuple[ImportanceItem, ...]
) -> FeatureIdeasResponse:
    base = _base_features(df, spec, importance)
    if len(base) < 2:
        return FeatureIdeasResponse(ideas=(), checked=0)

    data = _probe_rows(df, spec)
    candidates = _candidates(data, base)
    if not candidates:
        return FeatureIdeasResponse(ideas=(), checked=0)

    shares = _probe_shares(data, spec, candidates)
    ideas = tuple(
        FeatureIdea(
            name=name,
            formula=formula,
            share=round(share, 4),
            based_on=(a, b),
        )
        for (name, formula, a, b), share in zip(candidates, shares)
        if share >= MIN_SHARE
    )
    ranked = tuple(sorted(ideas, key=lambda i: i.share, reverse=True)[:MAX_IDEAS])
    return FeatureIdeasResponse(ideas=ranked, checked=len(candidates))


def _base_features(df, spec, importance) -> list[str]:
    """The strongest numeric dataset columns, in importance order."""
    scores = {item.feature: item.score for item in importance}
    usable = [
        f.name
        for f in spec.features
        if f.kind == "numeric"
        and f.derived_from is None  # date parts aren't dataset columns
        and f.name in df.columns
        and IDENTIFIER.match(f.name)
    ]
    ranked = sorted(usable, key=lambda name: scores.get(name, 0.0), reverse=True)
    return ranked[:MAX_BASE_FEATURES]


def _probe_rows(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    data = df[df[spec.target.name].notna()]
    if len(data) > PROBE_ROWS:
        data = data.sample(PROBE_ROWS, random_state=SEED)
    return data


def _candidates(data: pd.DataFrame, base: list[str]) -> list[tuple[str, str, str, str]]:
    """(name, formula, a, b) tuples whose computed values look usable."""
    existing = {str(c) for c in data.columns}
    found: list[tuple[str, str, str, str]] = []
    for i, a in enumerate(base):
        for b in base[i + 1 :]:
            for name, formula in (
                (f"{a}_per_{b}", f"{a} / {b}"),
                (f"{a}_times_{b}", f"{a} * {b}"),
                (f"{a}_minus_{b}", f"{a} - {b}"),
            ):
                if name in existing:
                    continue
                values = _evaluate(data, formula)
                if values is not None:
                    found.append((name, formula, a, b))
    return found


def _evaluate(data: pd.DataFrame, formula: str) -> pd.Series | None:
    try:
        values = pd.to_numeric(data.eval(formula), errors="coerce")
    except Exception:
        return None
    values = values.replace([np.inf, -np.inf], np.nan)
    if values.isna().mean() > MAX_MISSING_SHARE or values.nunique() < 2:
        return None
    return values


def _probe_shares(
    data: pd.DataFrame, spec: FeatureSpec, candidates: list[tuple[str, str, str, str]]
) -> list[float]:
    """Importance share of each candidate in a quick fit alongside the originals."""
    X = apply_feature_spec(data, spec)
    for name, formula, _, _ in candidates:
        X[name] = _evaluate(data, formula)
    y = _encode_target(data, spec)
    estimator_cls = (
        XGBClassifier if spec.target.task == "classification" else XGBRegressor
    )
    model = estimator_cls(**PROBE_PARAMS)
    model.fit(X, y, verbose=False)
    raw = model.feature_importances_
    total = float(raw.sum()) or 1.0
    by_name = {str(col): float(score) / total for col, score in zip(X.columns, raw)}
    return [by_name.get(name, 0.0) for name, _, _, _ in candidates]


def _encode_target(data: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    target = data[spec.target.name]
    if spec.target.task == "regression":
        return pd.to_numeric(target, errors="coerce")
    class_index = {c: i for i, c in enumerate(spec.target.classes or ())}
    return target.astype(str).map(class_index)
