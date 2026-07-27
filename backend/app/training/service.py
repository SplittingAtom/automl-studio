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
from app.training.trainer import ImportanceItem, train_model

LOW_IMPORTANCE_SHARE = 0.01
MIN_REMAINING_COLUMNS = 2

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
    if column.kind not in ("numeric", "categorical"):
        reasons = {
            "unsupported": "is completely empty",
            "id_like": "looks like an ID",
            "datetime": "contains dates",
        }
        raise AppError(
            "UNSUPPORTED_TARGET",
            f'"{column.name}" {reasons.get(column.kind, "is not usable")}, '
            "so it can't be predicted. Pick a number or category column.",
            status_code=422,
        )
    _validate_exclusions(request, dataset)
    task = request.task or detect_task(dataset_repo.load_dataframe(dataset.id)[column.name])
    meta = ModelMeta(
        id=f"mdl_{uuid.uuid4().hex[:10]}",
        dataset_id=dataset.id,
        dataset_name=dataset.name,
        target_column=request.target_column,
        task=task,
        status="queued",
        effort=request.effort,
        created_at=datetime.now(UTC).isoformat(),
        user_excluded_columns=request.excluded_columns,
    )
    return model_repo.save_meta(meta)


def _validate_exclusions(request: TrainRequest, dataset) -> None:
    known = {c.name for c in dataset.columns}
    unknown = set(request.excluded_columns) - known
    if unknown:
        raise AppError(
            "UNKNOWN_COLUMN",
            f"These columns aren't in the dataset: {', '.join(sorted(unknown))}.",
            status_code=422,
        )
    if request.target_column in request.excluded_columns:
        raise AppError(
            "TARGET_EXCLUDED",
            "The column you're predicting can't also be left out of the model.",
            status_code=422,
        )


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
            df,
            meta.target_column,
            meta.task,
            max_categories=settings.max_categories,
            user_excluded=meta.user_excluded_columns,
        )
        result = train_model(
            df, spec, max_rows=settings.max_training_rows, effort=meta.effort
        )
        model_repo.save_artifact(model_id, result.model, spec, result.interval_model)
        if result.validation is not None:
            model_repo.save_validation(model_id, result.validation)
        model_repo.save_meta(
            meta.model_copy(
                update={
                    "status": "complete",
                    "metrics": result.metrics,
                    "importance": result.importance,
                    "input_spec": build_input_spec(spec),
                    "excluded_columns": spec.excluded,
                    "suggested_exclusions": suggest_exclusions(spec, result.importance),
                    "leak_suspect": _leak_suspect(spec, result.warnings),
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


def suggest_exclusions(
    spec: FeatureSpec, importance: tuple[ImportanceItem, ...]
) -> tuple[str, ...]:
    """Dataset columns whose features all scored below the importance floor.

    Derived date features are grouped by their source column — the raw column
    is only suggested when every part of it was useless.
    """
    scores = {item.feature: item.score for item in importance}
    by_source: dict[str, list[float]] = {}
    for feature in spec.features:
        source = feature.derived_from or feature.name
        by_source.setdefault(source, []).append(scores.get(feature.name, 0.0))
    suggested = tuple(
        source
        for source, values in by_source.items()
        if max(values) < LOW_IMPORTANCE_SHARE
    )
    if len(by_source) - len(suggested) < MIN_REMAINING_COLUMNS:
        return ()
    return suggested


def _leak_suspect(spec: FeatureSpec, warnings) -> str | None:
    """Map a POSSIBLE_LEAKAGE warning's feature back to its dataset column."""
    warning = next((w for w in warnings if w.code == "POSSIBLE_LEAKAGE"), None)
    if warning is None or warning.column is None:
        return None
    feature = next((f for f in spec.features if f.name == warning.column), None)
    if feature is None:
        return warning.column
    return feature.derived_from or feature.name


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
