"""Dataset routes: list, upload, profile, preview."""

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, Query, UploadFile

from app.api.deps import get_dataset_repo, get_settings
from app.api.envelope import AppError, ok
from app.config import Settings
from app.datasets.csv_loading import parse_csv_upload
from app.datasets.repository import DatasetRepository
from app.datasets.schemas import DatasetMeta, DatasetPreview

router = APIRouter(prefix="/datasets", tags=["datasets"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
RepoDep = Annotated[DatasetRepository, Depends(get_dataset_repo)]


@router.get("")
def list_datasets(repo: RepoDep) -> dict:
    return ok([m.model_dump() for m in repo.list()])


@router.post("")
async def upload_dataset(file: UploadFile, repo: RepoDep, settings: SettingsDep) -> dict:
    name = file.filename or "upload.csv"
    if not name.lower().endswith(".csv"):
        raise AppError(
            "INVALID_FILE_TYPE",
            "Only CSV files are supported. Please export your data as .csv and try again.",
            status_code=422,
        )
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise AppError(
            "FILE_TOO_LARGE",
            f"That file is too large — the limit is {limit_mb} MB.",
            status_code=413,
        )
    df = parse_csv_upload(content)
    meta = repo.save(df, name=name, source="upload")
    return ok(meta.model_dump())


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str, repo: RepoDep) -> dict:
    return ok(_require(repo, dataset_id).model_dump())


@router.get("/{dataset_id}/preview")
def preview_dataset(
    dataset_id: str,
    repo: RepoDep,
    settings: SettingsDep,
    rows: Annotated[int, Query(ge=1)] = 20,
) -> dict:
    _require(repo, dataset_id)
    df = repo.load_dataframe(dataset_id).head(min(rows, settings.preview_rows_max))
    cleaned = df.astype(object).where(pd.notna(df), None)
    preview = DatasetPreview(
        columns=tuple(str(c) for c in df.columns),
        rows=tuple(cleaned.to_dict(orient="records")),
    )
    return ok(preview.model_dump())


def _require(repo: DatasetRepository, dataset_id: str) -> DatasetMeta:
    meta = repo.get(dataset_id)
    if meta is None:
        raise AppError(
            "DATASET_NOT_FOUND", "That dataset doesn't exist anymore.", status_code=404
        )
    return meta
