"""Tests for calculated columns: formula validation, evaluation, API flow."""

import io

import numpy as np
import pandas as pd
import pytest

from app.api.envelope import AppError
from app.datasets.calculated import add_calculated_column


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "fare": [10.0, 20.0, 0.0, 40.0],
            "sibsp": [1, 0, 2, 0],
            "parch": [0, 0, 1, 0],
            "sex": ["male", "female", "male", "female"],
        }
    )


class TestFormulas:
    def test_arithmetic_ratio(self, df):
        result = add_calculated_column(df, "fare_per_person", "fare / (sibsp + parch + 1)")
        assert list(result["fare_per_person"]) == [5.0, 20.0, 0.0, 40.0]

    def test_original_dataframe_not_mutated(self, df):
        before = df.copy()
        add_calculated_column(df, "x", "fare * 2")
        pd.testing.assert_frame_equal(df, before)

    def test_boolean_from_comparison(self, df):
        result = add_calculated_column(df, "expensive", "fare > 15")
        assert list(result["expensive"]) == [False, True, False, True]

    def test_string_equality(self, df):
        result = add_calculated_column(df, "is_male", "sex == 'male'")
        assert list(result["is_male"]) == [True, False, True, False]

    def test_division_by_zero_becomes_missing(self, df):
        result = add_calculated_column(df, "ratio", "sibsp / parch")
        assert result["ratio"].isna().sum() >= 1  # inf coerced to NaN
        assert not np.isinf(result["ratio"].dropna()).any()


class TestValidation:
    def test_unknown_column_rejected(self, df):
        with pytest.raises(AppError) as exc:
            add_calculated_column(df, "x", "bogus_column * 2")
        assert exc.value.status_code == 422

    def test_duplicate_name_rejected(self, df):
        with pytest.raises(AppError) as exc:
            add_calculated_column(df, "fare", "sibsp + 1")
        assert exc.value.code == "DUPLICATE_COLUMN"

    def test_invalid_name_rejected(self, df):
        with pytest.raises(AppError):
            add_calculated_column(df, "bad name!", "fare * 2")

    def test_dunder_blocked(self, df):
        with pytest.raises(AppError) as exc:
            add_calculated_column(df, "x", "__import__('os')")
        assert exc.value.code == "INVALID_FORMULA"

    def test_attribute_access_blocked(self, df):
        with pytest.raises(AppError) as exc:
            add_calculated_column(df, "x", "fare.abs()")
        assert exc.value.code == "INVALID_FORMULA"

    def test_constant_formula_rejected(self, df):
        with pytest.raises(AppError) as exc:
            add_calculated_column(df, "x", "1 + 1")
        assert exc.value.status_code == 422

    def test_decimals_are_fine(self, df):
        result = add_calculated_column(df, "half_fare", "fare * 0.5")
        assert result["half_fare"].iloc[1] == 10.0


class TestCalculatedApi:
    def _upload(self, client):
        csv = b"fare,sibsp,parch,outcome\n" + b"".join(
            f"{10 + i}.0,{i % 3},{i % 2},{i % 2}\n".encode() for i in range(80)
        )
        return client.post(
            "/api/datasets", files={"file": ("t.csv", io.BytesIO(csv), "text/csv")}
        ).json()["data"]

    def test_creates_new_derived_dataset(self, client):
        original = self._upload(client)
        resp = client.post(
            f"/api/datasets/{original['id']}/calculated",
            json={"name": "fare_per_person", "formula": "fare / (sibsp + parch + 1)"},
        )
        assert resp.status_code == 200, resp.text
        derived = resp.json()["data"]
        assert derived["id"] != original["id"]
        assert derived["source"] == "derived"
        assert derived["row_count"] == original["row_count"]
        assert "fare_per_person" in [c["name"] for c in derived["columns"]]
        assert "fare_per_person" in derived["name"]

    def test_original_dataset_unchanged(self, client):
        original = self._upload(client)
        client.post(
            f"/api/datasets/{original['id']}/calculated",
            json={"name": "x", "formula": "fare * 2"},
        )
        fresh = client.get(f"/api/datasets/{original['id']}").json()["data"]
        assert "x" not in [c["name"] for c in fresh["columns"]]

    def test_formula_error_is_friendly_422(self, client):
        original = self._upload(client)
        resp = client.post(
            f"/api/datasets/{original['id']}/calculated",
            json={"name": "x", "formula": "fare +* 2"},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["message"]

    def test_derived_dataset_is_trainable(self, client):
        original = self._upload(client)
        derived = client.post(
            f"/api/datasets/{original['id']}/calculated",
            json={"name": "fare_per_person", "formula": "fare / (sibsp + parch + 1)"},
        ).json()["data"]
        model = client.post(
            "/api/models",
            json={"dataset_id": derived["id"], "target_column": "outcome"},
        ).json()["data"]
        meta = client.get(f"/api/models/{model['id']}").json()["data"]
        assert meta["status"] == "complete"
        assert "fare_per_person" in {i["name"] for i in meta["input_spec"]}
