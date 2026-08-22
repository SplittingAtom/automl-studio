"""Feature ideas: candidate calculated columns scored by a quick probe fit."""

import numpy as np
import pandas as pd
import pytest

from app.analysis.feature_ideas import compute_feature_ideas
from app.training.preprocessing import build_feature_spec
from app.training.trainer import ImportanceItem

TITANIC_ID = "ds_sample_titanic"


def _importance(*names: str) -> tuple[ImportanceItem, ...]:
    n = len(names)
    return tuple(
        ImportanceItem(feature=name, score=(n - i) / n) for i, name in enumerate(names)
    )


class TestComputeFeatureIdeas:
    def test_ratio_signal_is_discovered(self):
        rng = np.random.default_rng(11)
        n = 500
        distance = rng.uniform(10, 100, size=n)
        duration = rng.uniform(1, 10, size=n)
        noise_col = rng.normal(size=n)
        target = 5 * (distance / duration) + rng.normal(scale=1.0, size=n)
        df = pd.DataFrame(
            {
                "distance": distance,
                "duration": duration,
                "noise_col": noise_col,
                "speed_target": target,
            }
        )
        spec = build_feature_spec(df, "speed_target", "regression")
        ideas = compute_feature_ideas(df, spec, _importance("distance", "duration", "noise_col"))
        formulas = [idea.formula for idea in ideas.ideas]
        assert "distance / duration" in formulas
        top = ideas.ideas[0]
        assert top.share > 0.2  # the ratio should dominate the probe

    def test_idea_names_are_valid_column_names(self):
        rng = np.random.default_rng(12)
        n = 300
        df = pd.DataFrame(
            {
                "a": rng.uniform(1, 5, n),
                "b": rng.uniform(1, 5, n),
                "y": rng.normal(size=n),
            }
        )
        spec = build_feature_spec(df, "y", "regression")
        ideas = compute_feature_ideas(df, spec, _importance("a", "b"))
        for idea in ideas.ideas:
            assert idea.name.isidentifier()
            assert idea.name not in df.columns

    def test_needs_two_numeric_features(self):
        rng = np.random.default_rng(13)
        df = pd.DataFrame(
            {
                "a": rng.uniform(1, 5, 300),
                "group": rng.choice(["x", "y"], 300),
                "y": rng.normal(size=300),
            }
        )
        spec = build_feature_spec(df, "y", "regression")
        ideas = compute_feature_ideas(df, spec, _importance("a"))
        assert ideas.ideas == ()
        assert ideas.checked == 0


class TestFeatureIdeasApi:
    @pytest.fixture
    def trained_model_id(self, client):
        resp = client.post(
            "/api/models",
            json={"dataset_id": TITANIC_ID, "target_column": "survived"},
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]["id"]

    def test_ideas_endpoint_shape_and_cache(self, client, settings, trained_model_id):
        resp = client.get(f"/api/models/{trained_model_id}/feature-ideas")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert isinstance(data["ideas"], list)
        assert data["checked"] >= 0
        cache = settings.data_dir / "models" / trained_model_id / "feature_ideas.json"
        assert cache.is_file()

    def test_unknown_model_404(self, client):
        assert client.get("/api/models/mdl_nope/feature-ideas").status_code == 404
