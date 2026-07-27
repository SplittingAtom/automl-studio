"""API tests for: user exclusions, model listing, sensitivity, xlsx upload,
imbalance handling, and target-kind rejection."""

import io

import numpy as np
import pandas as pd
import pytest

TITANIC_ID = "ds_sample_titanic"


@pytest.fixture
def trained(client):
    resp = client.post(
        "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
    )
    return resp.json()["data"]["id"]


class TestUserExclusions:
    def test_excluded_columns_left_out_of_model(self, client):
        data = client.post(
            "/api/models",
            json={
                "dataset_id": TITANIC_ID,
                "target_column": "survived",
                "excluded_columns": ["fare", "embarked"],
            },
        ).json()["data"]
        result = client.get(f"/api/models/{data['id']}").json()["data"]
        assert result["status"] == "complete"
        input_names = {i["name"] for i in result["input_spec"]}
        assert "fare" not in input_names and "embarked" not in input_names
        excluded = {e["name"]: e["reason"] for e in result["excluded_columns"]}
        assert excluded["fare"] == "user_excluded"

    def test_excluding_the_target_rejected(self, client):
        resp = client.post(
            "/api/models",
            json={
                "dataset_id": TITANIC_ID,
                "target_column": "survived",
                "excluded_columns": ["survived"],
            },
        )
        assert resp.status_code == 422

    def test_excluding_unknown_column_rejected(self, client):
        resp = client.post(
            "/api/models",
            json={
                "dataset_id": TITANIC_ID,
                "target_column": "survived",
                "excluded_columns": ["bogus"],
            },
        )
        assert resp.status_code == 422


class TestModelListing:
    def test_list_filters_by_dataset(self, client, trained):
        listed = client.get(f"/api/models?dataset_id={TITANIC_ID}").json()["data"]
        assert any(m["id"] == trained for m in listed)
        assert all(m["dataset_id"] == TITANIC_ID for m in listed)
        assert client.get("/api/models?dataset_id=ds_nope").json()["data"] == []


class TestSensitivity:
    def test_numeric_sensitivity_endpoint(self, client, trained):
        resp = client.post(
            f"/api/models/{trained}/sensitivity",
            json={"feature": "age", "inputs": {"sex": "female", "pclass": "1"}},
        )
        data = resp.json()["data"]
        assert data["feature"] == "age"
        assert len(data["points"]) == 21
        assert all(0 <= p["output"] <= 1 for p in data["points"])

    def test_categorical_sensitivity_endpoint(self, client, trained):
        data = client.post(
            f"/api/models/{trained}/sensitivity",
            json={"feature": "sex", "inputs": {}},
        ).json()["data"]
        assert {p["value"] for p in data["points"]} == {"male", "female"}

    def test_unknown_feature_422(self, client, trained):
        resp = client.post(
            f"/api/models/{trained}/sensitivity", json={"feature": "bogus", "inputs": {}}
        )
        assert resp.status_code == 422


class TestPredictExplanation:
    def test_prediction_includes_explanation(self, client, trained):
        data = client.post(
            f"/api/models/{trained}/predict",
            json={"inputs": {"age": 8, "sex": "female", "pclass": "1"}},
        ).json()["data"]
        assert data["explanation"] is not None
        features = {i["feature"] for i in data["explanation"]["items"]}
        assert "sex" in features


class TestXlsxUpload:
    def test_xlsx_parses(self, client):
        df = pd.DataFrame(
            {"age": np.linspace(20, 60, 40), "outcome": ["a", "b"] * 20}
        )
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        resp = client.post(
            "/api/datasets",
            files={
                "file": (
                    "data.xlsx",
                    io.BytesIO(buffer.getvalue()),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["data"]["row_count"] == 40

    def test_xls_still_rejected(self, client):
        resp = client.post(
            "/api/datasets", files={"file": ("old.xls", io.BytesIO(b"x"), "application/x-xls")}
        )
        assert resp.status_code == 422


class TestTargetKindRejection:
    def test_id_like_target_rejected(self, client):
        csv = ("record_id,value\n" + "".join(f"{i},{i % 3}\n" for i in range(100))).encode()
        ds = client.post(
            "/api/datasets", files={"file": ("ids.csv", io.BytesIO(csv), "text/csv")}
        ).json()["data"]
        resp = client.post(
            "/api/models", json={"dataset_id": ds["id"], "target_column": "record_id"}
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "UNSUPPORTED_TARGET"


class TestImbalanceHandling:
    def test_imbalanced_training_warns_and_completes(self, client):
        rng = np.random.default_rng(7)
        labels = ["common"] * 380 + ["rare"] * 20
        csv = ("x,label\n" + "".join(f"{rng.uniform():.4f},{l}\n" for l in labels)).encode()
        ds = client.post(
            "/api/datasets", files={"file": ("imb.csv", io.BytesIO(csv), "text/csv")}
        ).json()["data"]
        model = client.post(
            "/api/models", json={"dataset_id": ds["id"], "target_column": "label"}
        ).json()["data"]
        result = client.get(f"/api/models/{model['id']}").json()["data"]
        assert result["status"] == "complete"
        assert any(w["code"] == "CLASS_IMBALANCE" for w in result["warnings"])
