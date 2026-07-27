"""Tests for retrain suggestions: low-importance exclusions and leak suspects."""

import io

import numpy as np
import pandas as pd

from app.training.preprocessing import build_feature_spec
from app.training.service import suggest_exclusions
from app.training.trainer import ImportanceItem


def _importance(**scores) -> tuple[ImportanceItem, ...]:
    return tuple(ImportanceItem(feature=k, score=v) for k, v in scores.items())


def _spec(df, target="y"):
    return build_feature_spec(df, target, "classification")


class TestSuggestExclusions:
    def test_low_importance_columns_suggested(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.uniform(0, 1, 100),
                "b": rng.uniform(0, 1, 100),
                "c": rng.uniform(0, 1, 100),
                "y": rng.choice([0, 1], 100),
            }
        )
        suggested = suggest_exclusions(
            _spec(df), _importance(a=0.7, b=0.295, c=0.005)
        )
        assert suggested == ("c",)

    def test_date_column_suggested_only_if_all_parts_useless(self):
        rng = np.random.default_rng(42)
        # Spans several years so the _year part actually gets derived
        dates = pd.date_range("2020-01-01", periods=120, freq="10D").strftime("%Y-%m-%d")
        df = pd.DataFrame(
            {
                "when": list(dates),
                "amount": rng.uniform(0, 1, 120),
                "other": rng.uniform(0, 1, 120),
                "y": rng.choice([0, 1], 120),
            }
        )
        spec = _spec(df)
        all_parts_low = suggest_exclusions(
            spec,
            _importance(
                amount=0.99, other=0.02, when_year=0.003, when_month=0.002, when_dayofweek=0.001
            ),
        )
        assert all_parts_low == ("when",)

        one_part_matters = suggest_exclusions(
            spec,
            _importance(
                amount=0.9, other=0.02, when_year=0.08, when_month=0.002, when_dayofweek=0.001
            ),
        )
        assert one_part_matters == ()

    def test_never_suggests_below_two_remaining_columns(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "a": rng.uniform(0, 1, 100),
                "b": rng.uniform(0, 1, 100),
                "c": rng.uniform(0, 1, 100),
                "y": rng.choice([0, 1], 100),
            }
        )
        suggested = suggest_exclusions(_spec(df), _importance(a=0.99, b=0.005, c=0.005))
        assert suggested == ()


class TestLeakSuspectApi:
    def test_leaky_dataset_names_the_suspect_column(self, client):
        rng = np.random.default_rng(7)
        labels = rng.choice(["won", "lost"], 400)
        rows = "".join(
            f"{label},{label},{rng.uniform():.4f},{rng.uniform():.4f}\n" for label in labels
        )
        csv = ("outcome,outcome_copy,f1,f2\n" + rows).encode()
        ds = client.post(
            "/api/datasets", files={"file": ("leaky.csv", io.BytesIO(csv), "text/csv")}
        ).json()["data"]
        model = client.post(
            "/api/models", json={"dataset_id": ds["id"], "target_column": "outcome"}
        ).json()["data"]
        meta = client.get(f"/api/models/{model['id']}").json()["data"]
        assert meta["status"] == "complete"
        assert meta["leak_suspect"] == "outcome_copy"

    def test_clean_dataset_has_no_leak_suspect(self, client):
        model = client.post(
            "/api/models",
            json={"dataset_id": "ds_sample_titanic", "target_column": "survived"},
        ).json()["data"]
        meta = client.get(f"/api/models/{model['id']}").json()["data"]
        assert meta["leak_suspect"] is None
        assert isinstance(meta["suggested_exclusions"], list)
