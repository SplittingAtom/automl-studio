"""Surrogate decision tree: "how the model thinks", as a readable flowchart.

A depth-3 sklearn tree is fit to the XGBoost model's own outputs on the
held-out validation rows, then serialized as plain-English questions. The
fidelity score says how closely the flowchart tracks the real model — it's
an approximation by design, and honest about being one.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.prediction.schemas import BlueprintNode, BlueprintResponse
from app.training.preprocessing import FeatureSpec, apply_feature_spec

SEED = 42
MAX_DEPTH = 3
MAX_ROWS = 4000
MIN_LEAF_FRACTION = 0.05


def compute_blueprint(model, spec: FeatureSpec, validation: pd.DataFrame) -> BlueprintResponse:
    X = apply_feature_spec(validation, spec)
    if len(X) > MAX_ROWS:
        X = X.sample(MAX_ROWS, random_state=SEED)

    target, is_label_tree, output_label = _model_outputs(model, spec, X)
    encoded, dummy_origins = _encode(X)
    min_leaf = max(20, int(len(encoded) * MIN_LEAF_FRACTION))
    tree_cls = DecisionTreeClassifier if is_label_tree else DecisionTreeRegressor
    tree = tree_cls(max_depth=MAX_DEPTH, min_samples_leaf=min_leaf, random_state=SEED)
    tree.fit(encoded, target)

    mimic = tree.predict(encoded)
    fidelity = (
        float(accuracy_score(target, mimic))
        if is_label_tree
        else max(0.0, float(r2_score(target, mimic)))
    )
    root = _serialize(tree, 0, list(encoded.columns), dummy_origins, spec, is_label_tree)
    return BlueprintResponse(
        root=root,
        fidelity=round(fidelity, 3),
        output_label=output_label,
        sample_size=len(encoded),
    )


def _model_outputs(model, spec: FeatureSpec, X: pd.DataFrame):
    """What the surrogate should mimic, plus how to phrase it."""
    classes = spec.target.classes
    if classes is None:
        return model.predict(X), False, f"Predicted {spec.target.name}"
    if len(classes) == 2:
        # Mimic the probability, not the hard label — richer leaves.
        return model.predict_proba(X)[:, 1], False, f'Chance of "{classes[1]}"'
    predicted = np.asarray(model.predict(X), dtype=int)
    return predicted, True, f"Predicted {spec.target.name}"


def _encode(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    """One-hot categoricals so tree splits become yes/no category questions.

    Returns the encoded frame plus a map of dummy-column name -> (column,
    category) so splits can be phrased without parsing names.
    """
    parts: dict[str, pd.Series] = {}
    dummy_origins: dict[str, tuple[str, str]] = {}
    for column in X.columns:
        series = X[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            for category in series.cat.categories:
                dummy_name = f"{column}={category}"
                parts[dummy_name] = (series == category).astype(float)
                dummy_origins[dummy_name] = (str(column), str(category))
        else:
            parts[str(column)] = series.astype(float)
    return pd.DataFrame(parts, index=X.index), dummy_origins


def _serialize(tree, node_id, feature_names, dummy_origins, spec, is_label_tree) -> BlueprintNode:
    inner = tree.tree_
    if inner.children_left[node_id] == -1:  # leaf
        return BlueprintNode(
            label=_leaf_label(tree, inner, node_id, spec, is_label_tree),
            samples=int(inner.n_node_samples[node_id]),
        )
    name = feature_names[inner.feature[node_id]]
    threshold = float(inner.threshold[node_id])
    left = _serialize(tree, inner.children_left[node_id], feature_names, dummy_origins, spec, is_label_tree)
    right = _serialize(tree, inner.children_right[node_id], feature_names, dummy_origins, spec, is_label_tree)
    if name in dummy_origins:
        column, category = dummy_origins[name]
        # Left branch is dummy ≤ 0.5, i.e. NOT this category.
        question, yes, no = f'Is {column} "{category}"?', right, left
    else:
        question, yes, no = f"Is {name} ≤ {threshold:g}?", left, right
    return BlueprintNode(
        question=question,
        samples=int(inner.n_node_samples[node_id]),
        yes=yes,
        no=no,
    )


def _leaf_label(tree, inner, node_id, spec: FeatureSpec, is_label_tree: bool) -> str:
    classes = spec.target.classes
    if is_label_tree:  # multiclass: value holds per-class counts
        counts = inner.value[node_id][0]
        winner = int(np.argmax(counts))
        share = counts[winner] / counts.sum() if counts.sum() else 0.0
        label = classes[int(tree.classes_[winner])] if classes else str(winner)
        return f'usually "{label}" ({share:.0%})'
    value = float(inner.value[node_id][0][0])
    if classes is not None and len(classes) == 2:  # binary: value is a probability
        return f'about {value:.0%} chance of "{classes[1]}"'
    return f"≈ {value:g}"
