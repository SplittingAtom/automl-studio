"""Where does the model struggle? Per-group performance on validation rows.

For every categorical input, compare each group's score against the overall
score on the held-out rows and flag groups that are served notably worse.
Doubles as a plain-English fairness check: "predictions are less reliable
for embarked = Q" is exactly a disparate-impact statement, minus the jargon.
"""

import pandas as pd

from app.prediction.schemas import GroupCheckResponse, GroupFlag
from app.training.schemas import ModelMeta

MIN_GROUP_ROWS = 20
ACCURACY_GAP = 0.07  # flag groups this many points below overall accuracy
ERROR_FACTOR = 1.3  # flag groups whose typical error is 30%+ worse
MAX_GROUPS_PER_COLUMN = 12


def compute_group_check(meta: ModelMeta, validation: pd.DataFrame) -> GroupCheckResponse:
    predicted_col = "predicted" if "predicted" in validation.columns else "predicted_model"
    if predicted_col not in validation.columns:
        return GroupCheckResponse(flagged=(), groups_checked=0, overall_label="")

    if meta.task == "classification":
        scores = (
            validation[meta.target_column].astype(str)
            == validation[predicted_col].astype(str)
        ).astype(float)
        overall = float(scores.mean())
        overall_label = f"{overall:.0%} of held-out predictions are correct overall"
    else:
        actual = pd.to_numeric(validation[meta.target_column], errors="coerce")
        predicted = pd.to_numeric(validation[predicted_col], errors="coerce")
        scores = (predicted - actual).abs()
        overall = float(scores.mean())
        overall_label = f"typical error is {overall:g} overall"

    flagged: list[GroupFlag] = []
    checked = 0
    for column in _categorical_columns(meta, validation):
        groups = scores.groupby(validation[column].astype(str))
        sizes = groups.size().sort_values(ascending=False)
        for value in sizes.index[:MAX_GROUPS_PER_COLUMN]:
            size = int(sizes[value])
            if size < MIN_GROUP_ROWS:
                continue
            checked += 1
            group_score = float(groups.mean()[value])
            flag = _flag(meta.task, column, str(value), size, group_score, overall)
            if flag is not None:
                flagged.append(flag)
    flagged.sort(key=lambda f: f.rows, reverse=True)
    return GroupCheckResponse(
        flagged=tuple(flagged),
        groups_checked=checked,
        overall_label=overall_label,
    )


def _categorical_columns(meta: ModelMeta, validation: pd.DataFrame) -> list[str]:
    return [
        item.name
        for item in meta.input_spec or ()
        if item.kind == "categorical" and item.name in validation.columns
    ]


def _flag(
    task: str, column: str, value: str, rows: int, score: float, overall: float
) -> GroupFlag | None:
    if task == "classification":
        if score >= overall - ACCURACY_GAP:
            return None
        gap_label = f"{score:.0%} correct here vs {overall:.0%} overall"
    else:
        if overall == 0 or score <= overall * ERROR_FACTOR:
            return None
        gap_label = f"typical error {score:g} here vs {overall:g} overall"
    return GroupFlag(column=column, value=value, rows=rows, gap_label=gap_label)
