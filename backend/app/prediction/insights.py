"""Global explainability computed from the held-out validation rows.

Two views of "how do columns relate to the prediction":

- An association matrix between every model input and the prediction.
  Numeric pairs get a signed Pearson correlation; pairs involving a
  categorical column get an unsigned strength (correlation ratio between
  categorical and numeric, Cramér's V between two categoricals) so every
  column type lands on the same 0..1 scale.
- A SHAP impact summary: TreeSHAP contributions per validation row for the
  most influential features, with each row's feature value attached so the
  UI can color points by value (the classic beeswarm view).

Everything is computed on validation rows the model never trained on.
"""

import numpy as np
import pandas as pd

from app.prediction.explanation import batch_contributions
from app.prediction.labels import prediction_output_label
from app.prediction.schemas import (
    CorrelationCell,
    FeatureImpact,
    ImpactPoint,
    InsightsResponse,
)
from app.training.preprocessing import FeatureSpec, apply_feature_spec

SEED = 42
IMPACT_SAMPLE_ROWS = 200
MAX_IMPACT_FEATURES = 8
MAX_MATRIX_FEATURES = 11  # + the prediction axis = a readable 12x12 grid
ASSOCIATION_SAMPLE_ROWS = 4000
MIN_PAIR_ROWS = 3


def compute_insights(model, spec: FeatureSpec, validation: pd.DataFrame) -> InsightsResponse:
    X = apply_feature_spec(validation, spec)
    impact_frame = _sample(X, IMPACT_SAMPLE_ROWS)
    class_index, toward_label = _class_focus(model, spec, impact_frame)
    contributions = batch_contributions(model, impact_frame, class_index)[:, :-1]
    mean_abs = np.abs(contributions).mean(axis=0)
    ranking = list(np.argsort(mean_abs)[::-1])

    # A feature the model never uses contributes exactly 0 to every row —
    # a beeswarm row of dots stacked on zero is noise, so leave it out.
    used = [index for index in ranking if mean_abs[index] > 0]
    impacts = tuple(
        _feature_impact(spec, impact_frame, contributions, mean_abs, index)
        for index in used[:MAX_IMPACT_FEATURES]
    )

    matrix_indices = ranking[:MAX_MATRIX_FEATURES]
    association_frame = _sample(X, ASSOCIATION_SAMPLE_ROWS)
    prediction, prediction_label = _prediction_series(model, spec, association_frame)
    axes = [association_frame.iloc[:, i] for i in matrix_indices] + [prediction]
    low_label, high_label = _axis_labels(spec, toward_label)
    return InsightsResponse(
        columns=tuple(str(X.columns[i]) for i in matrix_indices),
        prediction_label=prediction_label,
        matrix=_association_matrix(axes),
        impacts=impacts,
        axis_low_label=low_label,
        axis_high_label=high_label,
        sample_size=len(impact_frame),
        association_rows=len(association_frame),
    )


def _sample(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    return frame.sample(cap, random_state=SEED)


def _class_focus(model, spec: FeatureSpec, frame: pd.DataFrame) -> tuple[int, str | None]:
    """Which class the contributions point toward.

    Binary boosters already return log-odds of the second class; for
    multiclass we summarize toward the class the model predicts most often.
    """
    classes = spec.target.classes
    if classes is None:
        return 0, None
    if len(classes) == 2:
        return 0, classes[1]
    predicted = np.asarray(model.predict(frame), dtype=int)
    class_index = int(np.bincount(predicted, minlength=len(classes)).argmax())
    return class_index, classes[class_index]


def _feature_impact(
    spec: FeatureSpec,
    frame: pd.DataFrame,
    contributions: np.ndarray,
    mean_abs: np.ndarray,
    index: int,
) -> FeatureImpact:
    feature = spec.features[index]
    values = frame.iloc[:, index]
    norms = _normalized_values(values) if feature.kind == "numeric" else None
    points = tuple(
        ImpactPoint(
            contribution=round(float(contributions[row, index]), 4),
            value_norm=None if norms is None else norms[row],
            value_label=_value_label(values.iloc[row]),
        )
        for row in range(len(frame))
    )
    return FeatureImpact(
        feature=feature.name,
        kind=feature.kind,
        mean_abs_contribution=round(float(mean_abs[index]), 4),
        points=points,
    )


def _normalized_values(values: pd.Series) -> list[float | None]:
    """Each value's 0..1 position within the feature's observed range."""
    numbers = pd.to_numeric(values, errors="coerce")
    low, high = numbers.min(), numbers.max()
    span = float(high - low) if pd.notna(high) and pd.notna(low) else 0.0

    def norm(value) -> float | None:
        if pd.isna(value):
            return None
        if span == 0.0:
            return 0.5
        return round(float((value - low) / span), 4)

    return [norm(v) for v in numbers]


def _value_label(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _prediction_series(model, spec: FeatureSpec, frame: pd.DataFrame) -> tuple[pd.Series, str]:
    classes = spec.target.classes
    if classes is None:
        values = pd.Series(model.predict(frame).astype(float), index=frame.index)
        return values, prediction_output_label(spec)
    if len(classes) == 2:
        probabilities = model.predict_proba(frame)[:, 1]
        values = pd.Series(probabilities.astype(float), index=frame.index)
        return values, prediction_output_label(spec, class_index=1)
    predicted = np.asarray(model.predict(frame), dtype=int)
    labels = pd.Series(
        pd.Categorical([classes[i] for i in predicted], categories=list(classes)),
        index=frame.index,
    )
    return labels, prediction_output_label(spec)


def _axis_labels(spec: FeatureSpec, toward_label: str | None) -> tuple[str, str]:
    classes = spec.target.classes
    if classes is None:
        target = spec.target.name
        return f"pushes {target} lower", f"pushes {target} higher"
    if len(classes) == 2:
        return f'toward "{classes[0]}"', f'toward "{classes[1]}"'
    return f'away from "{toward_label}"', f'toward "{toward_label}"'


def _association_matrix(axes: list[pd.Series]) -> tuple[tuple[CorrelationCell, ...], ...]:
    size = len(axes)
    cells: list[list[CorrelationCell]] = [
        [CorrelationCell() for _ in range(size)] for _ in range(size)
    ]
    for i in range(size):
        for j in range(i, size):
            cell = _association(axes[i], axes[j])
            cells[i][j] = cell
            cells[j][i] = cell
    return tuple(tuple(row) for row in cells)


def _association(a: pd.Series, b: pd.Series) -> CorrelationCell:
    a_categorical = isinstance(a.dtype, pd.CategoricalDtype)
    b_categorical = isinstance(b.dtype, pd.CategoricalDtype)
    if not a_categorical and not b_categorical:
        return _cell(pearson(a, b), signed=True)
    if a_categorical and b_categorical:
        return _cell(cramers_v(a, b), signed=False)
    cat, num = (a, b) if a_categorical else (b, a)
    return _cell(correlation_ratio(cat, num), signed=False)


def _cell(value: float | None, signed: bool) -> CorrelationCell:
    if value is None:
        return CorrelationCell()
    return CorrelationCell(value=round(value, 3), signed=signed)


def pearson(a: pd.Series, b: pd.Series) -> float | None:
    """Signed correlation between two numeric columns, on complete pairs."""
    mask = a.notna() & b.notna()
    x, y = a[mask], b[mask]
    if len(x) < MIN_PAIR_ROWS or x.nunique() < 2 or y.nunique() < 2:
        return None
    value = float(np.corrcoef(x.astype(float), y.astype(float))[0, 1])
    if np.isnan(value):
        return None
    return float(np.clip(value, -1.0, 1.0))


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    """Strength (0..1) of a categorical column's relationship with a numeric one."""
    mask = categories.notna() & values.notna()
    cats, nums = categories[mask], values[mask].astype(float)
    if len(nums) < MIN_PAIR_ROWS or cats.nunique() < 2:
        return None
    overall_mean = nums.mean()
    ss_total = float(((nums - overall_mean) ** 2).sum())
    if ss_total == 0.0:
        return None
    groups = nums.groupby(cats, observed=True)
    ss_between = float((groups.size() * (groups.mean() - overall_mean) ** 2).sum())
    return float(np.clip(np.sqrt(ss_between / ss_total), 0.0, 1.0))


def cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Strength (0..1) of association between two categorical columns."""
    mask = a.notna() & b.notna()
    if int(mask.sum()) < MIN_PAIR_ROWS:
        return None
    table = pd.crosstab(a[mask], b[mask]).to_numpy(dtype=float)
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    n_rows, n_cols = table.shape
    if min(n_rows, n_cols) < 2:
        return None
    total = table.sum()
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / total
    chi2 = float(((table - expected) ** 2 / expected).sum())
    v = np.sqrt(chi2 / total / (min(n_rows, n_cols) - 1))
    return float(np.clip(v, 0.0, 1.0))
