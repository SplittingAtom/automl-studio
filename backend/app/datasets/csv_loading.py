"""Robust upload parsing: CSV (encoding fallback + delimiter sniffing) and Excel."""

import csv
import io

import pandas as pd

from app.api.envelope import AppError

SNIFF_SAMPLE_BYTES = 64 * 1024
SUPPORTED_EXTENSIONS = (".csv", ".xlsx")


def parse_upload(content: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV or Excel file, raising a friendly AppError when unusable."""
    lowered = filename.lower()
    if lowered.endswith(".xlsx"):
        df = _parse_excel(content)
    elif lowered.endswith(".csv"):
        df = _parse_csv(content)
    else:
        raise AppError(
            "INVALID_FILE_TYPE",
            "Only CSV (.csv) and Excel (.xlsx) files are supported. "
            "Please export your data in one of those formats.",
            status_code=422,
        )
    if len(df) == 0 or len(df.columns) == 0:
        raise AppError(
            "EMPTY_DATASET",
            "That file has no data rows. Please upload a file with at least a few rows.",
            status_code=422,
        )
    return df


def _parse_excel(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise AppError(
            "UNPARSEABLE_FILE",
            "We couldn't read that Excel file. Please check it opens in a spreadsheet tool.",
            status_code=422,
        ) from exc


def _parse_csv(content: bytes) -> pd.DataFrame:
    text = _decode(content)
    delimiter = _sniff_delimiter(text)
    try:
        return pd.read_csv(io.StringIO(text), sep=delimiter)
    except Exception as exc:
        raise AppError(
            "UNPARSEABLE_FILE",
            "We couldn't read that file as a CSV. Please check it opens in a spreadsheet tool.",
            status_code=422,
        ) from exc


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
