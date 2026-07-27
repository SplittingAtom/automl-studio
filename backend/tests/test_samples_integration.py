"""Every bundled sample dataset must train successfully on its headline target.

This is the guardrail against shipping a sample that makes a bad first
impression (leakage, unusable columns, mis-detected task).
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parent.parent

# (dataset_id, headline target, expected task, minimum acceptable quality)
SAMPLE_EXPECTATIONS = [
    ("ds_sample_titanic", "survived", "classification", ("accuracy", 0.7)),
    ("ds_sample_housing", "median_house_value", "regression", ("r2", 0.7)),
    ("ds_sample_churn", "Churn", "classification", ("accuracy", 0.7)),
    ("ds_sample_penguins", "species", "classification", ("accuracy", 0.9)),
    ("ds_sample_bikes", "rentals", "regression", ("r2", 0.5)),
    ("ds_sample_diamonds", "price", "regression", ("r2", 0.9)),
    ("ds_sample_income", "income", "classification", ("accuracy", 0.8)),
    ("ds_sample_heart", "disease", "classification", ("accuracy", 0.7)),
]


@pytest.fixture(scope="module")
def full_client(tmp_path_factory):
    settings = Settings(
        data_dir=tmp_path_factory.mktemp("data"),
        sample_data_dir=BACKEND_ROOT / "sample_data",
    )
    return TestClient(create_app(settings))


@pytest.mark.parametrize(
    "dataset_id,target,task,quality", SAMPLE_EXPECTATIONS, ids=[s[0] for s in SAMPLE_EXPECTATIONS]
)
def test_sample_trains_cleanly(full_client, dataset_id, target, task, quality):
    created = full_client.post(
        "/api/models", json={"dataset_id": dataset_id, "target_column": target}
    )
    assert created.status_code == 200, created.text
    assert created.json()["data"]["task"] == task

    model = full_client.get(f"/api/models/{created.json()['data']['id']}").json()["data"]
    assert model["status"] == "complete", model.get("error")

    metric_name, minimum = quality
    assert model["metrics"][metric_name] >= minimum, (
        f"{dataset_id}: {metric_name}={model['metrics'][metric_name]} below {minimum}"
    )
    # No sample should ship with a leakage tell
    assert not any(w["code"] == "POSSIBLE_LEAKAGE" for w in model["warnings"]), model["warnings"]
    # And the what-if panel must have something to adjust
    assert len(model["input_spec"]) >= 3
