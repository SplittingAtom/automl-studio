"""Model routes: train, status/results, what-if predict."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends

from app.api.deps import get_dataset_repo, get_model_cache, get_model_repo, get_settings
from app.api.envelope import AppError, ok
from app.config import Settings
from app.datasets.repository import DatasetRepository
from app.prediction.model_cache import ModelCache
from app.prediction.predictor import predict
from app.prediction.schemas import PredictRequest
from app.training import service
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
    meta = _require(model_repo, model_id)
    if meta.status != "complete":
        raise AppError(
            "MODEL_NOT_READY",
            "This model hasn't finished training yet.",
            status_code=409,
        )
    model, spec = cache.get(model_id, lambda: model_repo.load_artifact(model_id))
    response = predict(model, spec, request.inputs)
    return ok(response.model_dump())


def _require(repo: ModelRepository, model_id: str) -> ModelMeta:
    meta = repo.get(model_id)
    if meta is None:
        raise AppError("MODEL_NOT_FOUND", "That model doesn't exist.", status_code=404)
    return meta
