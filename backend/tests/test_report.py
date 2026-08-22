"""Downloadable model report: self-contained HTML built from stored results."""

import pytest

from app.training.report import build_report_html
from app.training.schemas import ModelMeta

TITANIC_ID = "ds_sample_titanic"


def _meta(**updates) -> ModelMeta:
    base = dict(
        id="mdl_test",
        dataset_id="ds_x",
        dataset_name="My Data",
        target_column="price",
        task="regression",
        status="complete",
        created_at="2026-08-12T10:00:00+00:00",
        metrics={
            "r2": 0.81,
            "mae": 12.5,
            "rmse": 20.1,
            "cv_mean": 0.79,
            "cv_std": 0.03,
            "cv_folds": 5,
            "baseline_r2": 0.0,
            "linear_r2": 0.7,
            "test_rows": 200,
        },
        importance=({"feature": "sqft", "score": 0.6}, {"feature": "age", "score": 0.4}),
        warnings=({"code": "ROW_SAMPLE", "message": "Sampled rows for speed.", "column": None},),
        n_rows_used=1000,
    )
    base.update(updates)
    return ModelMeta.model_validate(base)


class TestBuildReportHtml:
    def test_contains_the_essentials(self):
        html = build_report_html(_meta(), insights=None)
        assert "price" in html
        assert "My Data" in html
        assert "0.81" in html  # headline R²
        assert "sqft" in html  # importance
        assert "Sampled rows for speed." in html  # warnings
        assert "<!doctype html>" in html.lower()

    def test_mentions_validation_approach(self):
        html = build_report_html(_meta(), insights=None)
        assert "20%" in html  # held-out share
        random_split = "random" in html.lower()
        assert random_split

    def test_time_aware_models_described_differently(self):
        html = build_report_html(
            _meta(time_column="date", horizon=7), insights=None
        )
        assert "most recent" in html.lower()
        assert "7" in html

    def test_overrides_are_listed(self):
        html = build_report_html(
            _meta(overrides={"max_depth": 3, "monotone_constraints": {"sqft": 1}}),
            insights=None,
        )
        assert "max_depth" in html
        assert "sqft" in html

    def test_untrusted_names_are_escaped(self):
        html = build_report_html(
            _meta(target_column="<script>alert(1)</script>"), insights=None
        )
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html


class TestReportApi:
    @pytest.fixture
    def trained_model_id(self, client):
        resp = client.post(
            "/api/models", json={"dataset_id": TITANIC_ID, "target_column": "survived"}
        )
        assert resp.status_code == 200, resp.text
        return resp.json()["data"]["id"]

    def test_report_downloads_as_html(self, client, trained_model_id):
        resp = client.get(f"/api/models/{trained_model_id}/report")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "attachment" in resp.headers["content-disposition"]
        assert "survived" in resp.text
        assert "Titanic" in resp.text

    def test_unknown_model_404(self, client):
        assert client.get("/api/models/mdl_nope/report").status_code == 404
