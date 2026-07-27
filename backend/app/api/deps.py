"""FastAPI dependencies pulling shared services off app state."""

from fastapi import Request

from app.config import Settings
from app.datasets.repository import DatasetRepository
from app.prediction.model_cache import ModelCache
from app.training.repository import ModelRepository


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_dataset_repo(request: Request) -> DatasetRepository:
    return request.app.state.dataset_repo


def get_model_repo(request: Request) -> ModelRepository:
    return request.app.state.model_repo


def get_model_cache(request: Request) -> ModelCache:
    return request.app.state.model_cache
