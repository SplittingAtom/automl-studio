"""Tests for dataset suitability analysis and target recommendation."""

import numpy as np
import pandas as pd
import pytest

from app.analysis.analyzer import analyze_dataset
from app.datasets.profiling import profile_dataframe

TITANIC_ID = "ds_sample_titanic"


def _analyze(df, dataset_id="ds_test"):
    profile = profile_dataframe(df)
    return analyze_dataset(dataset_id, df, profile.columns)


@pytest.fixture(scope="module")
def predictable_df():
    rng = np.random.default_rng(42)
    rows = 600
    driver = rng.uniform(0, 10, rows)
    group = rng.choice(["a", "b", "c"], rows)
    # Noisy boundary: strongly but not perfectly predictable, like real data
    latent = driver + (group == "a") * 4 + rng.normal(0, 1.5, rows)
    return pd.DataFrame(
        {
            "driver": driver,
            "group": group,
            "outcome": np.where(latent > 8, "yes", "no"),
            "pure_noise": rng.choice(["x", "y"], rows),
            "record_id": range(rows),
        }
    )


class TestTargetCandidates:
    def test_predictable_column_ranked_first(self, predictable_df):
        analysis = _analyze(predictable_df)
        assert analysis.candidates[0].column == "outcome"
        assert analysis.candidates[0].recommended

    def test_predictable_column_shows_signal_and_baseline(self, predictable_df):
        outcome = next(c for c in _analyze(predictable_df).candidates if c.column == "outcome")
        assert outcome.signal > 0.4
        assert outcome.probe_score is not None and outcome.baseline_score is not None
        assert outcome.probe_score > outcome.baseline_score

    def test_noise_column_scores_low(self, predictable_df):
        noise = next(c for c in _analyze(predictable_df).candidates if c.column == "pure_noise")
        assert noise.signal < 0.15
        assert not noise.recommended

    def test_id_columns_are_not_candidates(self, predictable_df):
        columns = {c.column for c in _analyze(predictable_df).candidates}
        assert "record_id" not in columns

    def test_top_predictors_identified(self, predictable_df):
        outcome = next(c for c in _analyze(predictable_df).candidates if c.column == "outcome")
        top_two = {p.name for p in outcome.top_predictors[:2]}
        assert top_two == {"driver", "group"}  # the informative pair, not noise

    def test_every_candidate_has_plain_english_reasons(self, predictable_df):
        for candidate in _analyze(predictable_df).candidates:
            assert len(candidate.reasons) >= 1

    def test_regression_candidate_supported(self):
        rng = np.random.default_rng(42)
        rows = 500
        x = rng.uniform(0, 10, rows)
        df = pd.DataFrame({"x": x, "price": 3 * x + rng.normal(0, 1, rows)})
        analysis = _analyze(df)
        price = next(c for c in analysis.candidates if c.column == "price")
        assert price.task == "regression"
        assert price.signal > 0.5

    def test_near_duplicate_column_flagged_and_penalized(self):
        rng = np.random.default_rng(42)
        rows = 500
        status = rng.choice(["active", "churned"], rows)
        df = pd.DataFrame(
            {
                "status": status,
                "status_copy": status,  # a look-alike column
                "noise": rng.uniform(0, 1, rows),
            }
        )
        analysis = _analyze(df)
        status_candidate = next(c for c in analysis.candidates if c.column == "status")
        assert any("two versions" in r for r in status_candidate.reasons)
        clean = _analyze(df.drop(columns=["status_copy"]))
        clean_status = next((c for c in clean.candidates if c.column == "status"), None)
        # Without the duplicate, status is pure noise — no duplicate flag
        assert clean_status is None or not any(
            "two versions" in r for r in clean_status.reasons
        )

    def test_calculated_column_flagged_as_derived(self):
        rng = np.random.default_rng(42)
        rows = 500
        a = rng.uniform(0, 10, rows)
        b = rng.uniform(0, 10, rows)
        df = pd.DataFrame(
            {"a": a, "b": b, "total": a + b, "label": rng.choice(["x", "y"], rows)}
        )
        analysis = _analyze(df)
        total = next(c for c in analysis.candidates if c.column == "total")
        assert total.derived_like
        assert not total.recommended
        assert any("calculated" in r or "two versions" in r for r in total.reasons)
        # And the headline summary must not recommend a derived column
        assert '"total"' not in analysis.summary

    def test_high_cardinality_categorical_not_a_candidate(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "city": rng.choice([f"c{i}" for i in range(40)], 400),
                "x": rng.uniform(0, 1, 400),
            }
        )
        assert "city" not in {c.column for c in _analyze(df).candidates}


class TestOverallAssessment:
    def test_predictable_dataset_rated_well(self, predictable_df):
        analysis = _analyze(predictable_df)
        assert analysis.rating in ("great", "good")
        assert analysis.summary
        assert len(analysis.points) >= 2

    def test_pure_noise_dataset_rated_poorly(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.uniform(0, 1, 300),
                "b": rng.uniform(0, 1, 300),
                "c": rng.choice(["x", "y"], 300),
            }
        )
        analysis = _analyze(df)
        assert analysis.rating in ("fair", "poor")

    def test_tiny_dataset_flagged(self):
        rng = np.random.default_rng(42)
        rows = 60
        x = rng.uniform(0, 10, rows)
        df = pd.DataFrame({"x": x, "y": np.where(x > 5, "hi", "lo")})
        analysis = _analyze(df)
        assert any("small" in p.message.lower() for p in analysis.points)


class TestAnalysisApi:
    def test_analysis_endpoint_and_caching(self, client):
        first = client.get(f"/api/datasets/{TITANIC_ID}/analysis")
        assert first.status_code == 200, first.text
        data = first.json()["data"]
        assert data["rating"] in ("great", "good", "fair", "poor")
        top_columns = [c["column"] for c in data["candidates"][:3]]
        assert "survived" in top_columns  # sex is also legitimately predictable
        # Second call must hit the disk cache and return identical content
        assert client.get(f"/api/datasets/{TITANIC_ID}/analysis").json()["data"] == data

    def test_analysis_missing_dataset_404(self, client):
        assert client.get("/api/datasets/ds_nope/analysis").status_code == 404
