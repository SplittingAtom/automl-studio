"""FastAPI application factory."""

import logging

from fastapi import FastAPI

from app.api import datasets, models
from app.api.envelope import register_exception_handlers
from app.config import Settings
from app.datasets.repository import DatasetRepository
from app.datasets.samples import seed_samples
from app.prediction.model_cache import ModelCache
from app.training.repository import ModelRepository

logging.basicConfig(level=logging.INFO)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    dataset_repo = DatasetRepository(settings.datasets_dir)
    model_repo = ModelRepository(settings.models_dir)

    seed_samples(dataset_repo, settings.sample_data_dir)
    model_repo.fail_interrupted()

    app = FastAPI(title="AutoML Prototype", version="0.1.0")
    app.state.settings = settings
    app.state.dataset_repo = dataset_repo
    app.state.model_repo = model_repo
    app.state.model_cache = ModelCache(settings.model_cache_size)

    register_exception_handlers(app)
    app.include_router(datasets.router, prefix="/api")
    app.include_router(models.router, prefix="/api")
    return app
