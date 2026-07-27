"""Training orchestration: create the job record, run it, record the outcome."""

import logging
import uuid
from datetime import UTC, datetime

from app.api.envelope import AppError
from app.config import Settings
from app.datasets.repository import DatasetRepository
from app.training.preprocessing import FeatureSpec, TrainingError, build_feature_spec
from app.training.repository import ModelRepository
from app.training.schemas import InputSpecItem, ModelMeta, TrainRequest
from app.training.task_detection import detect_task
from app.training.trainer import train_model

logger = logging.getLogger(__name__)


def create_training_job(
    request: TrainRequest,
    dataset_repo: DatasetRepository,
    model_repo: ModelRepository,
) -> ModelMeta:
    dataset = dataset_repo.get(request.dataset_id)
    if dataset is None:
        raise AppError(
            "DATASET_NOT_FOUND", "That dataset doesn't exist anymore.", status_code=404
        )
    column = next((c for c in dataset.columns if c.name == request.target_column), None)
    if column is None:
        raise AppError(
            "UNKNOWN_COLUMN",
            f'"{request.target_column}" is not a column in {dataset.name}.',
            status_code=422,
        )
    if column.kind == "unsupported":
        raise AppError(
            "UNSUPPORTED_TARGET",
            f'"{column.name}" is completely empty, so it can\'t be predicted.',
            status_code=422,
        )
    task = request.task or detect_task(dataset_repo.load_dataframe(dataset.id)[column.name])
    meta = ModelMeta(
        id=f"mdl_{uuid.uuid4().hex[:10]}",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        target_column=request.target_column,
        task=task,
        status="queued",
        created_at=datetime.now(UTC).isoformat(),
    )
    return model_repo.save_meta(meta)


def run_training_job(
    model_id: str,
    dataset_repo: DatasetRepository,
    model_repo: ModelRepository,
    settings: Settings,
) -> None:
    meta = model_repo.get(model_id)
    if meta is None:
        logger.error("Training job %s vanished before it could run", model_id)
        return
    model_repo.save_meta(meta.model_copy(update={"status": "training"}))
    try:
        df = dataset_repo.load_dataframe(meta.dataset_id)
        spec = build_feature_spec(
            df, meta.target_column, meta.task, max_categories=settings.max_categories
        )
        result = train_model(df, spec, max_rows=settings.max_training_rows)
        model_repo.save_artifact(model_id, result.model, spec)
        model_repo.save_meta(
            meta.model_copy(
                update={
                    "status": "complete",
                    "metrics": result.metrics,
                    "importance": result.importance,
                    "input_spec": build_input_spec(spec),
                    "excluded_columns": spec.excluded,
                    "warnings": result.warnings,
                    "n_rows_used": result.n_rows_used,
                }
            )
        )
    except TrainingError as exc:
        logger.info("Training %s failed: %s", model_id, exc.message)
        model_repo.save_meta(meta.model_copy(update={"status": "failed", "error": exc.message}))
    except Exception:
        logger.exception("Unexpected training failure for %s", model_id)
        model_repo.save_meta(
            meta.model_copy(
                update={
                    "status": "failed",
                    "error": "Training failed unexpectedly. Try a different dataset or target column.",
                }
            )
        )


def build_input_spec(spec: FeatureSpec) -> tuple[InputSpecItem, ...]:
    return tuple(
        InputSpecItem(
            name=f.name,
            kind=f.kind,
            min_value=f.min_value,
            max_value=f.max_value,
            options=f.categories,
            default=f.default,
        )
        for f in spec.features
    )
