"""Model routes: train, status/results, what-if predict."""

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse, Response

from app.analysis.feature_ideas import compute_feature_ideas
from app.api.deps import get_dataset_repo, get_model_cache, get_model_repo, get_settings
from app.api.envelope import AppError, ok
from app.config import Settings
from app.datasets.repository import DatasetRepository
from app.prediction.blueprint import compute_blueprint
from app.prediction.forecast import compute_forecast
from app.prediction.group_check import compute_group_check
from app.prediction.insights import compute_insights
from app.prediction.model_cache import ModelCache
from app.prediction.predictor import predict
from app.prediction.schemas import PredictRequest, SensitivityRequest
from app.prediction.sensitivity import compute_sensitivity
from app.training import service
from app.training.export import build_export_zip
from app.training.report import build_report_html
from app.training.repository import ModelRepository
from app.training.schemas import ModelMeta, TrainRequest

router = APIRouter(prefix="/models", tags=["models"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
DatasetRepoDep = Annotated[DatasetRepository, Depends(get_dataset_repo)]
ModelRepoDep = Annotated[ModelRepository, Depends(get_model_repo)]
CacheDep = Annotated[ModelCache, Depends(get_model_cache)]


@router.post("")
def create_model(
    request: TrainRequest,
    background: BackgroundTasks,
    dataset_repo: DatasetRepoDep,
    model_repo: ModelRepoDep,
    settings: SettingsDep,
) -> dict:
    meta = service.create_training_job(request, dataset_repo, model_repo)
    background.add_task(
        service.run_training_job, meta.id, dataset_repo, model_repo, settings
    )
    return ok(meta.model_dump())


@router.get("")
def list_models(model_repo: ModelRepoDep, dataset_id: str | None = None) -> dict:
    return ok([m.model_dump() for m in model_repo.list(dataset_id)])


@router.get("/{model_id}")
def get_model(model_id: str, model_repo: ModelRepoDep) -> dict:
    return ok(_require(model_repo, model_id).model_dump())


@router.post("/{model_id}/predict")
def predict_model(
    model_id: str,
    request: PredictRequest,
    model_repo: ModelRepoDep,
    cache: CacheDep,
) -> dict:
    meta = _require_complete(model_repo, model_id)
    artifact = cache.get(model_id, lambda: model_repo.load_artifact(model_id))
    response = predict(
        artifact["model"],
        artifact["spec"],
        request.inputs,
        interval_model=artifact["interval_model"],
    )
    return ok(response.model_dump())


@router.post("/{model_id}/sensitivity")
def sensitivity_model(
    model_id: str,
    request: SensitivityRequest,
    model_repo: ModelRepoDep,
    cache: CacheDep,
) -> dict:
    meta = _require_complete(model_repo, model_id)
    artifact = cache.get(model_id, lambda: model_repo.load_artifact(model_id))
    response = compute_sensitivity(
        artifact["model"], artifact["spec"], request.inputs, request.feature
    )
    return ok(response.model_dump())


@router.get("/{model_id}/insights")
def model_insights(
    model_id: str,
    model_repo: ModelRepoDep,
    cache: CacheDep,
) -> dict:
    """Global explainability: computed once per model, then served from disk."""
    _require_complete(model_repo, model_id)
    cached = model_repo.get_insights(model_id)
    if cached is not None:
        return ok(cached.model_dump())
    artifact = cache.get(model_id, lambda: model_repo.load_artifact(model_id))
    frame = _require_validation(model_repo, model_id)
    response = compute_insights(artifact["model"], artifact["spec"], frame)
    model_repo.save_insights(model_id, response)
    return ok(response.model_dump())


@router.get("/{model_id}/validation")
def get_validation(
    model_id: str,
    model_repo: ModelRepoDep,
    rows: Annotated[int, Query(ge=1, le=200)] = 25,
) -> dict:
    _require_complete(model_repo, model_id)
    frame = _require_validation(model_repo, model_id)
    head = frame.head(rows)
    cleaned = head.astype(object).where(pd.notna(head), None)
    return ok(
        {
            "columns": [str(c) for c in frame.columns],
            "rows": cleaned.to_dict(orient="records"),
            "total_rows": len(frame),
        }
    )


@router.get("/{model_id}/validation/download")
def download_validation(model_id: str, model_repo: ModelRepoDep) -> FileResponse:
    _require_complete(model_repo, model_id)
    path = model_repo.validation_path(model_id)
    if not path.is_file():
        raise AppError("VALIDATION_NOT_FOUND", VALIDATION_MISSING_MESSAGE, 404)
    return FileResponse(
        path, media_type="text/csv", filename=f"validation_{model_id}.csv"
    )


@router.get("/{model_id}/forecast")
def forecast_model(
    model_id: str,
    model_repo: ModelRepoDep,
    dataset_repo: DatasetRepoDep,
    cache: CacheDep,
    steps: Annotated[int, Query(ge=1, le=365)] = 30,
) -> dict:
    meta = _require_complete(model_repo, model_id)
    if meta.time_column is None or meta.task != "regression":
        raise AppError(
            "NOT_A_TIME_MODEL",
            "Future forecasting needs a time-aware model predicting a number — "
            "train with a time column selected.",
            status_code=422,
        )
    artifact = cache.get(model_id, lambda: model_repo.load_artifact(model_id))
    df = dataset_repo.load_dataframe(meta.dataset_id)
    response = compute_forecast(
        artifact["model"],
        artifact["interval_model"],
        artifact["spec"],
        df,
        meta.time_column,
        steps,
    )
    return ok(response.model_dump())


@router.get("/{model_id}/group-check")
def model_group_check(model_id: str, model_repo: ModelRepoDep) -> dict:
    """Per-group reliability on validation rows; computed once, then cached."""
    meta = _require_complete(model_repo, model_id)
    cached = model_repo.get_group_check(model_id)
    if cached is not None:
        return ok(cached.model_dump())
    frame = _require_validation(model_repo, model_id)
    response = compute_group_check(meta, frame)
    model_repo.save_group_check(model_id, response)
    return ok(response.model_dump())


@router.get("/{model_id}/blueprint")
def model_blueprint(
    model_id: str,
    model_repo: ModelRepoDep,
    cache: CacheDep,
) -> dict:
    """Surrogate flowchart; computed once per model, then served from disk."""
    _require_complete(model_repo, model_id)
    cached = model_repo.get_blueprint(model_id)
    if cached is not None:
        return ok(cached.model_dump())
    artifact = cache.get(model_id, lambda: model_repo.load_artifact(model_id))
    frame = _require_validation(model_repo, model_id)
    response = compute_blueprint(artifact["model"], artifact["spec"], frame)
    model_repo.save_blueprint(model_id, response)
    return ok(response.model_dump())


@router.get("/{model_id}/feature-ideas")
def model_feature_ideas(
    model_id: str,
    model_repo: ModelRepoDep,
    dataset_repo: DatasetRepoDep,
    cache: CacheDep,
) -> dict:
    """Suggested calculated columns; computed once per model, then cached."""
    meta = _require_complete(model_repo, model_id)
    cached = model_repo.get_feature_ideas(model_id)
    if cached is not None:
        return ok(cached.model_dump())
    artifact = cache.get(model_id, lambda: model_repo.load_artifact(model_id))
    df = dataset_repo.load_dataframe(meta.dataset_id)
    response = compute_feature_ideas(df, artifact["spec"], meta.importance or ())
    model_repo.save_feature_ideas(model_id, response)
    return ok(response.model_dump())


@router.get("/{model_id}/report")
def model_report(model_id: str, model_repo: ModelRepoDep) -> Response:
    """Shareable HTML report built entirely from stored results."""
    meta = _require_complete(model_repo, model_id)
    html = build_report_html(meta, model_repo.get_insights(model_id))
    return Response(
        content=html,
        media_type="text/html",
        headers={
            "Content-Disposition": f'attachment; filename="model_report_{model_id}.html"'
        },
    )


@router.get("/{model_id}/export")
def export_model(model_id: str, model_repo: ModelRepoDep, cache: CacheDep) -> Response:
    meta = _require_complete(model_repo, model_id)
    artifact = cache.get(model_id, lambda: model_repo.load_artifact(model_id))
    payload = build_export_zip(meta, artifact)
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="automl_model_{model_id}.zip"'
        },
    )


VALIDATION_MISSING_MESSAGE = (
    "No validation rows were saved for this model — it may predate this feature. "
    "Retrain to get them."
)


def _require_validation(repo: ModelRepository, model_id: str):
    frame = repo.load_validation(model_id)
    if frame is None:
        raise AppError("VALIDATION_NOT_FOUND", VALIDATION_MISSING_MESSAGE, 404)
    return frame


def _require_complete(repo: ModelRepository, model_id: str) -> ModelMeta:
    meta = _require(repo, model_id)
    if meta.status != "complete":
        raise AppError(
            "MODEL_NOT_READY",
            "This model hasn't finished training yet.",
            status_code=409,
        )
    return meta


def _require(repo: ModelRepository, model_id: str) -> ModelMeta:
    meta = repo.get(model_id)
    if meta is None:
        raise AppError("MODEL_NOT_FOUND", "That model doesn't exist.", status_code=404)
    return meta
