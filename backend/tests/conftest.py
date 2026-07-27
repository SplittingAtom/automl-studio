"""Shared fixtures: app + client backed by a temp data directory."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=tmp_path / "data",
        sample_data_dir=BACKEND_ROOT / "sample_data",
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    return TestClient(app)
