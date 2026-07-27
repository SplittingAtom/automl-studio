"""Contract tests for the models API — train on the seeded Titanic sample."""

import pytest

TITANIC_ID = "ds_sample_titanic"


@pytest.fixture
def trained_model_id(client):
    resp = client.post(
        "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
    )
    assert resp.status_code == 200, resp.text
    model_id = resp.json()["data"]["id"]
    # TestClient runs BackgroundTasks before returning, so training is done.
    return model_id


class TestTraining:
    def test_create_detects_classification(self, client):
        data = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
        ).json()["data"]
        assert data["task"] == "classification"

    def test_completed_model_has_results(self, client, trained_model_id):
        data = client.get(f"/api/models/{trained_model_id}").json()["data"]
        assert data["status"] == "complete"
        assert data["metrics"]["accuracy"] > 0.6
        assert len(data["importance"]) > 0
        assert len(data["input_spec"]) > 0
        excluded = {e["name"] for e in data["excluded_columns"]}
        assert "deck" in excluded  # 77% missing on Titanic

    def test_input_spec_drives_ui_controls(self, client, trained_model_id):
        spec = client.get(f"/api/models/{trained_model_id}").json()["data"]["input_spec"]
        by_name = {i["name"]: i for i in spec}
        assert by_name["age"]["kind"] == "numeric"
        assert by_name["age"]["min_value"] is not None
        assert by_name["sex"]["kind"] == "categorical"
        assert set(by_name["sex"]["options"]) == {"male", "female"}

    def test_unknown_dataset_404(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": "ds_nope", "target_column": "x"}
        )
        assert resp.status_code == 404

    def test_bad_target_column_422(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "nope"}
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "UNKNOWN_COLUMN"

    def test_task_override_respected(self, client):
        data = client.post(
            "/api/models",
            json={"dataset_id": TITANIC_ID, "target_column": "fare", "task": "regression"},
        ).json()["data"]
        assert data["task"] == "regression"


class TestPredict:
    def test_what_if_prediction(self, client, trained_model_id):
        resp = client.post(
            f"/api/models/{trained_model_id}/predict",
            json={"inputs": {"age": 8, "sex": "female", "pclass": "1"}},
        )
        data = resp.json()["data"]
        assert data["prediction"] in {"0", "1"}
        assert len(data["probabilities"]) == 2

    def test_unknown_input_rejected(self, client, trained_model_id):
        resp = client.post(
            f"/api/models/{trained_model_id}/predict", json={"inputs": {"bogus": 1}}
        )
        assert resp.status_code == 422

    def test_predict_on_missing_model_404(self, client):
        resp = client.post("/api/models/mdl_nope/predict", json={"inputs": {}})
        assert resp.status_code == 404
