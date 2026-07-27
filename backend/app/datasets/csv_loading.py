"""Robust CSV parsing for user uploads: encoding fallback + delimiter sniffing."""

import csv
import io

import pandas as pd

from app.api.envelope import AppError

SNIFF_SAMPLE_BYTES = 64 * 1024


def parse_csv_upload(content: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes, raising a friendly AppError when unusable."""
    text = _decode(content)
    delimiter = _sniff_delimiter(text)
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except Exception as exc:
        raise AppError(
            "UNPARSEABLE_CSV",
            "We couldn't read that file as a CSV. Please check it opens in a spreadsheet tool.",
            status_code=422,
        ) from exc
    if len(df) == 0 or len(df.columns) == 0:
        raise AppError(
            "EMPTY_DATASET",
            "That file has no data rows. Please upload a CSV with at least a few rows.",
            status_code=422,
        )
    return df


def _decode(content: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise AppError(
        "UNPARSEABLE_CSV",
        "We couldn't read that file's text encoding. Please save it as UTF-8 CSV.",
        status_code=422,
    )


def _sniff_delimiter(text: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(text[:SNIFF_SAMPLE_BYTES], delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","
