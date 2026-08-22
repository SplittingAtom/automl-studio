"""Sensitivity curves: vary one feature across its range, hold the rest fixed.

This is an ICE curve for the user's current what-if scenario — the intuitive
"how does age affect this prediction?" view.
"""

import numpy as np
import pandas as pd

from app.api.envelope import AppError
from app.prediction.labels import prediction_output_label
from app.prediction.predictor import validated_row
from app.prediction.schemas import SensitivityPoint, SensitivityResponse
from app.training.preprocessing import FeatureColumn, FeatureSpec, apply_feature_spec

GRID_POINTS = 21


def compute_sensitivity(
    model, spec: FeatureSpec, inputs: dict, feature_name: str
) -> SensitivityResponse:
    feature = next((f for f in spec.features if f.name == feature_name), None)
    if feature is None:
        raise AppError(
            "UNKNOWN_FEATURE",
            f'"{feature_name}" is not one of this model\'s inputs.',
            status_code=422,
        )
    row = validated_row(spec, inputs)
    grid = _grid(feature)
    frame = pd.DataFrame([{**row, feature_name: value} for value in grid])
    X = apply_feature_spec(frame, spec)
    outputs, output_label = _outputs(model, spec, X)
    points = tuple(
        SensitivityPoint(
            value=value if isinstance(value, str) else round(float(value), 4),
            output=round(float(output), 4),
        )
        for value, output in zip(grid, outputs)
    )
    return SensitivityResponse(
        feature=feature_name,
        kind=feature.kind,
        output_label=output_label,
        current_value=row.get(feature_name),
        points=points,
    )


def _grid(feature: FeatureColumn) -> list:
    if feature.kind == "categorical":
        return list(feature.categories or ())
    low = feature.min_value if feature.min_value is not None else 0.0
    high = feature.max_value if feature.max_value is not None else 1.0
    if low == high:
        return [low]
    return list(np.linspace(low, high, GRID_POINTS))


def _outputs(model, spec: FeatureSpec, X: pd.DataFrame):
    if spec.target.task == "regression":
        return model.predict(X), prediction_output_label(spec)
    probabilities = model.predict_proba(X)
    classes = spec.target.classes or ()
    if len(classes) == 2:
        class_index = 1
    else:
        # Track the class the model currently favors (averaged over the grid)
        class_index = int(np.argmax(probabilities.mean(axis=0)))
    label = prediction_output_label(spec, class_index) if classes else 'Chance of "?"'
    return probabilities[:, class_index], label
