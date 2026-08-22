"""Per-group performance: find where the model is notably less reliable."""

import numpy as np
import pandas as pd
import pytest

from app.prediction.group_check import compute_group_check
from app.training.schemas import InputSpecItem, ModelMeta

TITANIC_ID = "ds_sample_titanic"


def _meta(task, target, categorical_columns):
    return ModelMeta.model_validate(
        dict(
            id="mdl_x",
            dataset_id="ds_x",
            dataset_name="Data",
            target_column=target,
            task=task,
            status="complete",
            created_at="2026-08-12T10:00:00+00:00",
            input_spec=tuple(
                InputSpecItem(name=c, kind="categorical", options=()) for c in categorical_columns
            ),
        )
    )


class TestClassificationGroups:
    def test_flags_the_badly_served_group(self):
        rng = np.random.default_rng(31)
        n = 400
        region = np.array(["north"] * 200 + ["south"] * 200)
        actual = rng.choice(["yes", "no"], n)
        predicted = actual.copy()
        south = np.arange(200, 400)
        flip = rng.choice(south, size=80, replace=False)  # 60% accuracy in south
        predicted[flip] = np.where(predicted[flip] == "yes", "no", "yes")
        frame = pd.DataFrame({"region": region, "outcome": actual, "predicted": predicted})
        meta = _meta("classification", "outcome", ["region"])
        result = compute_group_check(meta, frame)
        assert any(f.column == "region" and f.value == "south" for f in result.flagged)
        assert not any(f.value == "north" for f in result.flagged)
        south_flag = next(f for f in result.flagged if f.value == "south")
        assert "%" in south_flag.gap_label

    def test_small_groups_are_ignored(self):
        frame = pd.DataFrame(
            {
                "region": ["north"] * 95 + ["tiny"] * 5,
                "outcome": ["yes"] * 100,
                "predicted": ["yes"] * 95 + ["no"] * 5,  # tiny group is 0% correct
            }
        )
        meta = _meta("classification", "outcome", ["region"])
        result = compute_group_check(meta, frame)
        assert result.flagged == ()


class TestRegressionGroups:
    def test_flags_group_with_larger_errors(self):
        rng = np.random.default_rng(32)
        n = 300
        kind = np.array(["easy"] * 200 + ["hard"] * 100)
        actual = rng.normal(100, 10, n)
        noise = np.where(kind == "hard", 25.0, 2.0)
        predicted = actual + rng.normal(0, 1, n) * noise
        frame = pd.DataFrame({"kind": kind, "price": actual, "predicted": predicted})
        meta = _meta("regression", "price", ["kind"])
        result = compute_group_check(meta, frame)
        assert any(f.value == "hard" for f in result.flagged)
        assert not any(f.value == "easy" for f in result.flagged)


class TestGroupCheckApi:
    @pytest.fixture
    def trained_model_id(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]["id"]

    def test_endpoint_shape_and_cache(self, client, settings, trained_model_id):
        resp = client.get(f"/api/models/{trained_model_id}/group-check")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["groups_checked"] > 0
        assert isinstance(data["flagged"], list)
        assert (settings.data_dir / "models" / trained_model_id / "group_check.json").is_file()

    def test_unknown_model_404(self, client):
        assert client.get("/api/models/mdl_nope/group-check").status_code == 404
