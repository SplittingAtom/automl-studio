"""Model export: a self-contained scoring kit.

The zip holds the model in XGBoost's portable JSON format (no pickle), the
FeatureSpec contract, and a standalone predict.py that reapplies the exact
preprocessing — anyone with `pip install pandas xgboost` can score a CSV.
"""

import io
import zipfile

from app.training.preprocessing import FeatureSpec
from app.training.schemas import ModelMeta

PREDICT_SCRIPT = '''#!/usr/bin/env python3
"""Score a CSV with this exported AutoML Studio model.

Usage:
    python predict.py your_data.csv [output.csv]

Requires: pip install pandas xgboost   (macOS also: brew install libomp)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "feature_spec.json").read_text())


def apply_spec(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of the app's apply_feature_spec — keep behavior identical."""
    data = {}
    parsed_dates = {}
    for feature in SPEC["features"]:
        name = feature["name"]
        if name in df.columns:
            source = df[name]
        elif feature.get("derived_from") and feature["derived_from"] in df.columns:
            raw = feature["derived_from"]
            if raw not in parsed_dates:
                parsed_dates[raw] = pd.to_datetime(df[raw], errors="coerce", format="mixed")
            source = getattr(parsed_dates[raw].dt, feature["date_part"])
        else:
            source = pd.Series([None] * len(df), index=df.index)
        if feature["kind"] == "numeric":
            data[name] = pd.to_numeric(source, errors="coerce").astype("float64")
        else:
            categories = list(feature["categories"] or [])
            known = set(categories)
            has_other = "Other" in known
            values = []
            for value in source.tolist():
                if pd.isna(value):
                    values.append(None)
                    continue
                text = str(value)
                if text in known:
                    values.append(text)
                else:
                    values.append("Other" if has_other else None)
            data[name] = pd.Series(
                pd.Categorical(values, categories=categories), index=df.index
            )
    return pd.DataFrame(data, index=df.index)


def load_booster(filename: str) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(HERE / filename))
    return booster


def iteration_range(booster: xgb.Booster):
    best = booster.attr("best_iteration")
    return (0, int(best) + 1) if best is not None else None


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    source = Path(sys.argv[1])
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix(".predictions.csv")

    df = pd.read_csv(source)
    matrix = xgb.DMatrix(apply_spec(df), enable_categorical=True)
    booster = load_booster("model.json")
    limit = iteration_range(booster)
    raw = booster.predict(matrix, iteration_range=limit) if limit else booster.predict(matrix)

    result = df.copy()
    target = SPEC["target"]
    if target["task"] == "classification":
        classes = target["classes"]
        proba = np.column_stack([1 - raw, raw]) if raw.ndim == 1 else raw
        winners = proba.argmax(axis=1)
        result["predicted"] = [classes[i] for i in winners]
        for index, label in enumerate(classes):
            result[f"probability_{label}"] = proba[:, index].round(4)
    else:
        result["predicted"] = raw.round(4)
        interval_path = HERE / "interval_model.json"
        if interval_path.is_file():
            quantiles = load_booster("interval_model.json").predict(matrix)
            result["predicted_low"] = np.minimum(quantiles[:, 0], quantiles[:, 1]).round(4)
            result["predicted_high"] = np.maximum(quantiles[:, 0], quantiles[:, 1]).round(4)

    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} predictions to {output}")


if __name__ == "__main__":
    main()
'''


def build_export_zip(meta: ModelMeta, artifact: dict) -> bytes:
    model = artifact["model"]
    spec: FeatureSpec = artifact["spec"]
    interval_model = artifact.get("interval_model")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("model.json", bytes(model.get_booster().save_raw(raw_format="json")))
        bundle.writestr("feature_spec.json", spec.model_dump_json(indent=2))
        if interval_model is not None:
            bundle.writestr(
                "interval_model.json",
                bytes(interval_model.get_booster().save_raw(raw_format="json")),
            )
        bundle.writestr("predict.py", PREDICT_SCRIPT)
        bundle.writestr("README.md", _readme(meta))
    return buffer.getvalue()


def _readme(meta: ModelMeta) -> str:
    metrics = meta.metrics or {}
    if meta.task == "classification":
        quality = f"{metrics.get('accuracy', '?')} accuracy on held-out rows"
    else:
        quality = f"R² {metrics.get('r2', '?')} on held-out rows"
    return f"""# AutoML Studio model export

- **Predicts:** {meta.target_column}
- **Trained on:** {meta.dataset_name} ({meta.n_rows_used} rows)
- **Type:** {meta.task}
- **Quality:** {quality}
- **Exported model id:** {meta.id}

## Score new data

```
pip install pandas xgboost        # macOS also: brew install libomp
python predict.py your_data.csv
```

Your CSV needs the same input columns the model was trained on (see
`feature_spec.json`); extra columns are ignored and missing ones are treated
as unknown. Predictions are written next to your file as
`your_data.predictions.csv` with a `predicted` column appended{
        " (plus an 80% low/high band)" if meta.task == "regression" else
        " (plus per-class probabilities)"}.

## Files

- `model.json` — the trained XGBoost model (portable JSON, no pickle)
- `feature_spec.json` — the preprocessing contract (columns, categories, date parts)
- `predict.py` — standalone scorer, no AutoML Studio required
{"- `interval_model.json` — quantile model for the prediction band" if meta.task == "regression" else ""}"""
