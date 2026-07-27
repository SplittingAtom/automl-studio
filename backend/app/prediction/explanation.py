"""Per-prediction explanations using XGBoost's built-in TreeSHAP.

`pred_contribs=True` gives exact SHAP contributions straight from the
booster — no extra dependency. Binary/multiclass contributions are in
log-odds space; regression contributions are in target units.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from app.prediction.schemas import Explanation, ExplanationItem
from app.training.preprocessing import FeatureSpec


def explain_prediction(model, spec: FeatureSpec, frame: pd.DataFrame) -> Explanation:
    contributions, bias = _contributions(model, spec, frame)
    items = sorted(
        (
            ExplanationItem(feature=name, contribution=round(float(value), 4))
            for name, value in zip(frame.columns, contributions)
        ),
        key=lambda item: abs(item.contribution),
        reverse=True,
    )
    return Explanation(
        items=tuple(items),
        baseline=round(float(bias), 4),
        toward_label=_toward_label(model, spec, frame),
    )


def _contributions(model, spec: FeatureSpec, frame: pd.DataFrame):
    booster = model.get_booster()
    matrix = xgb.DMatrix(frame, enable_categorical=True)
    iteration_range = _best_iteration_range(model)
    raw = booster.predict(matrix, pred_contribs=True, iteration_range=iteration_range)
    if raw.ndim == 3:  # multiclass: (rows, classes, features + bias)
        class_index = int(np.argmax(model.predict_proba(frame)[0]))
        row = raw[0, class_index]
    else:
        row = raw[0]
    return row[:-1], row[-1]  # last entry is the bias/baseline


def _best_iteration_range(model) -> tuple[int, int] | None:
    best_iteration = getattr(model, "best_iteration", None)
    return (0, best_iteration + 1) if best_iteration is not None else None


def _toward_label(model, spec: FeatureSpec, frame: pd.DataFrame) -> str | None:
    classes = spec.target.classes
    if classes is None:
        return None
    if len(classes) == 2:
        return classes[1]  # binary contributions are log-odds of the second class
    class_index = int(np.argmax(model.predict_proba(frame)[0]))
    return classes[class_index]
