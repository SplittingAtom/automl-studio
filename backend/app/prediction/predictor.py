"""Single-row what-if prediction: validate inputs, fill defaults, predict."""

import time

import pandas as pd

from app.api.envelope import AppError
from app.prediction.schemas import ClassProbability, PredictResponse
from app.training.preprocessing import FeatureSpec, apply_feature_spec


def predict(model, spec: FeatureSpec, inputs: dict) -> PredictResponse:
    row = _validated_row(spec, inputs)
    frame = apply_feature_spec(pd.DataFrame([row]), spec)
    started = time.perf_counter()
    if spec.target.task == "classification":
        response = _predict_class(model, spec, frame)
    else:
        response = {"prediction": round(float(model.predict(frame)[0]), 4), "probabilities": None}
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return PredictResponse(**response, elapsed_ms=elapsed_ms)


def _validated_row(spec: FeatureSpec, inputs: dict) -> dict:
    known = {f.name for f in spec.features}
    unknown = set(inputs) - known
    if unknown:
        raise AppError(
            "UNKNOWN_INPUT",
            f"These inputs aren't part of the model: {', '.join(sorted(unknown))}.",
            status_code=422,
        )
    row = {}
    for feature in spec.features:
        value = inputs.get(feature.name, feature.default)
        if feature.kind == "numeric" and value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise AppError(
                    "INVALID_INPUT",
                    f'"{feature.name}" must be a number.',
                    status_code=422,
                ) from None
        row[feature.name] = value
    return row


def _predict_class(model, spec: FeatureSpec, frame: pd.DataFrame) -> dict:
    probabilities = model.predict_proba(frame)[0]
    classes = spec.target.classes
    best = int(probabilities.argmax())
    return {
        "prediction": classes[best],
        "probabilities": tuple(
            ClassProbability(label=label, probability=round(float(p), 4))
            for label, p in zip(classes, probabilities)
        ),
    }
