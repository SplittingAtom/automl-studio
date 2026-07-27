"""Tests for the model export scoring kit — including actually running it."""

import io
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest


def _upload_and_train(client, df, target, task=None):
    csv = df.to_csv(index=False).encode()
    dataset = client.post(
        "/api/datasets", files={"file": ("data.csv", io.BytesIO(csv), "text/csv")}
    ).json()["data"]
    body = {"dataset_id": dataset["id"], "target_column": target}
    if task:
        body["task"] = task
    model = client.post("/api/models", json=body).json()["data"]
    meta = client.get(f"/api/models/{model['id']}").json()["data"]
    assert meta["status"] == "complete", meta.get("error")
    return meta


def _classification_df(rows=400):
    rng = np.random.default_rng(42)
    signal = rng.uniform(0, 10, rows)
    color = rng.choice(["red", "blue"], rows)
    return pd.DataFrame(
        {
            "signal": signal,
            "color": color,
            "label": np.where(signal + (color == "red") * 2 + rng.normal(0, 1, rows) > 6, "hi", "lo"),
        }
    )


def _regression_df(rows=400):
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, rows)
    return pd.DataFrame({"x": x, "y": 3 * x + rng.normal(0, 1, rows)})


def _export_zip(client, model_id):
    resp = client.get(f"/api/models/{model_id}/export")
    assert resp.status_code == 200, resp.text
    assert "application/zip" in resp.headers["content-type"]
    return zipfile.ZipFile(io.BytesIO(resp.content))


class TestExportContents:
    def test_classification_kit_members(self, client):
        meta = _upload_and_train(client, _classification_df(), "label")
        names = set(_export_zip(client, meta["id"]).namelist())
        assert {"model.json", "feature_spec.json", "predict.py", "README.md"} <= names
        assert "interval_model.json" not in names

    def test_regression_kit_includes_interval_model(self, client):
        meta = _upload_and_train(client, _regression_df(), "y")
        names = set(_export_zip(client, meta["id"]).namelist())
        assert "interval_model.json" in names

    def test_export_of_missing_model_404(self, client):
        assert client.get("/api/models/mdl_nope/export").status_code == 404


class TestExportedScriptRuns:
    """The real proof: unzip the kit and score a CSV with plain python."""

    def _run_kit(self, client, meta, df, tmp_path):
        kit = _export_zip(client, meta["id"])
        kit.extractall(tmp_path)
        input_csv = tmp_path / "new_data.csv"
        df.drop(columns=[meta["target_column"]]).to_csv(input_csv, index=False)
        result = subprocess.run(
            [sys.executable, "predict.py", str(input_csv)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        return pd.read_csv(tmp_path / "new_data.predictions.csv")

    def test_classification_kit_scores_correctly(self, client, tmp_path):
        df = _classification_df()
        meta = _upload_and_train(client, df, "label")
        scored = self._run_kit(client, meta, df, tmp_path)
        assert len(scored) == len(df)
        assert set(scored["predicted"].unique()) <= {"hi", "lo"}
        assert "probability_hi" in scored.columns
        # Predictions must agree with the true pattern most of the time
        expected = df["label"].to_numpy()
        agreement = (scored["predicted"].to_numpy() == expected).mean()
        assert agreement > 0.75

    def test_regression_kit_scores_with_intervals(self, client, tmp_path):
        df = _regression_df()
        meta = _upload_and_train(client, df, "y")
        scored = self._run_kit(client, meta, df, tmp_path)
        assert "predicted" in scored.columns
        assert (scored["predicted_low"] <= scored["predicted_high"]).all()
        # y = 3x: predictions should track the signal closely
        correlation = np.corrcoef(scored["predicted"], df["x"])[0, 1]
        assert correlation > 0.95
