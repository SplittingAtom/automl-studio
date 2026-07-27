"""Calculated columns: PowerBI-style formulas evaluated over the dataset.

Formulas run through pandas' guarded expression evaluator (df.eval), with
extra blocklists so only column arithmetic and comparisons are possible —
no attribute access, no dunders, no statements.
"""

import re

import numpy as np
import pandas as pd

from app.api.envelope import AppError

NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
# A dot after an identifier or closing bracket is attribute access; a dot in
# "0.5" is not.
ATTRIBUTE_ACCESS = re.compile(r"[A-Za-z_\)\]]\s*\.")
FORBIDDEN_TOKENS = ("__", "@", ";", "\n", "\r", "=", "[", "]")
ALLOWED_ASSIGN_EXCEPTIONS = ("==", "!=", "<=", ">=")
MAX_FORMULA_LENGTH = 500


def add_calculated_column(df: pd.DataFrame, name: str, formula: str) -> pd.DataFrame:
    """Return a new dataframe with the calculated column appended."""
    clean_name = _validated_name(df, name)
    clean_formula = _validated_formula(formula)
    try:
        result = df.eval(clean_formula)
    except Exception as exc:
        raise AppError(
            "FORMULA_ERROR",
            f"That formula couldn't be calculated: {_short(exc)}. "
            "Use column names with + - * / ( ) and comparisons like fare > 10.",
            status_code=422,
        ) from exc
    if not isinstance(result, pd.Series):
        raise AppError(
            "FORMULA_ERROR",
            "The formula must use at least one column — a constant value "
            "wouldn't help the model.",
            status_code=422,
        )
    if pd.api.types.is_numeric_dtype(result) and not pd.api.types.is_bool_dtype(result):
        result = result.replace([np.inf, -np.inf], np.nan)
    return df.assign(**{clean_name: result})


def _validated_name(df: pd.DataFrame, name: str) -> str:
    clean = name.strip()
    if not NAME_PATTERN.match(clean):
        raise AppError(
            "INVALID_COLUMN_NAME",
            "Column names must start with a letter and use only letters, "
            "numbers, and underscores.",
            status_code=422,
        )
    if clean in {str(c) for c in df.columns}:
        raise AppError(
            "DUPLICATE_COLUMN",
            f'A column called "{clean}" already exists.',
            status_code=422,
        )
    return clean


def _validated_formula(formula: str) -> str:
    clean = formula.strip()
    if not clean or len(clean) > MAX_FORMULA_LENGTH:
        raise AppError(
            "INVALID_FORMULA", "Please enter a formula.", status_code=422
        )
    stripped = clean
    for allowed in ALLOWED_ASSIGN_EXCEPTIONS:
        stripped = stripped.replace(allowed, "")
    for token in FORBIDDEN_TOKENS:
        if token in stripped:
            raise AppError(
                "INVALID_FORMULA",
                "Formulas can only use column names, numbers, math operators "
                "(+ - * /), and comparisons (== != < > <= >=).",
                status_code=422,
            )
    if ATTRIBUTE_ACCESS.search(clean):
        raise AppError(
            "INVALID_FORMULA",
            "Formulas can't use dots after names — write plain column math "
            "like fare / (sibsp + parch + 1).",
            status_code=422,
        )
    return clean


def _short(exc: Exception) -> str:
    text = str(exc).split("\n")[0]
    return text[:120] if text else "it isn't valid column math"
