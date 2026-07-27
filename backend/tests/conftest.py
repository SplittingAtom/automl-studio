"""Shared fixtures: app + client backed by a temp data directory.

Unit/API tests seed only the Titanic sample to stay fast; the full sample
catalog is covered by test_samples_integration.py.
"""

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parent.parent

FAST_SAMPLES = ("titanic.csv",)


@pytest.fixture(scope="session")
def fast_sample_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("samples")
    for filename in FAST_SAMPLES:
        shutil.copy(BACKEND_ROOT / "sample_data" / filename, directory / filename)
    return directory


@pytest.fixture
def settings(tmp_path, fast_sample_dir):
    return Settings(
        data_dir=tmp_path / "data",
        sample_data_dir=fast_sample_dir,
    )


@pytest.fixture
def client(settings):
    app = create_app(settings)
    return TestClient(app)
