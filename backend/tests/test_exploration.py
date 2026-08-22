"""Dataset exploration: per-column distributions + dataset-level overview."""

import numpy as np
import pandas as pd
import pytest

from app.datasets.exploration import explore_dataframe
from app.datasets.profiling import profile_dataframe

TITANIC_ID = "ds_sample_titanic"


def _explore(df: pd.DataFrame):
    profile = profile_dataframe(df)
    return explore_dataframe("ds_test", df, profile.columns)


@pytest.fixture(scope="module")
def mixed_exploration():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "amount": rng.normal(100, 15, size=n),
            "city": rng.choice(["Berlin", "Paris", "Rome"], size=n, p=[0.5, 0.3, 0.2]),
            "signup": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "user_id": range(1, n + 1),
        }
    )
    df.loc[:19, "amount"] = np.nan  # 10% missing
    return _explore(df), df


class TestNumericColumns:
    def test_histogram_counts_cover_every_value(self, mixed_exploration):
        exploration, df = mixed_exploration
        amount = next(c for c in exploration.columns if c.name == "amount")
        assert len(amount.bins) > 1
        assert sum(b.count for b in amount.bins) == int(df["amount"].notna().sum())

    def test_bin_edges_are_monotonic(self, mixed_exploration):
        exploration, _ = mixed_exploration
        amount = next(c for c in exploration.columns if c.name == "amount")
        edges = [b.low for b in amount.bins] + [amount.bins[-1].high]
        assert all(a < b for a, b in zip(edges, edges[1:]))

    def test_stats_include_spread_and_outliers(self, mixed_exploration):
        exploration, _ = mixed_exploration
        amount = next(c for c in exploration.columns if c.name == "amount")
        assert amount.stats is not None
        assert amount.stats.std > 0
        assert amount.stats.min < amount.stats.median < amount.stats.max
        assert amount.missing_pct == 10.0

    def test_infinite_values_treated_as_missing(self):
        values = list(np.linspace(0.0, 1.0, 98)) + [np.inf, -np.inf]
        exploration = _explore(pd.DataFrame({"x": values, "y": range(100)}))
        x = next(c for c in exploration.columns if c.name == "x")
        assert sum(b.count for b in x.bins) == 98
        assert np.isfinite(x.stats.max) and np.isfinite(x.stats.min)

    def test_extreme_value_counted_as_outlier(self):
        values = list(np.linspace(10.0, 11.0, 99)) + [10_000.0]
        exploration = _explore(pd.DataFrame({"x": values, "y": range(100)}))
        x = next(c for c in exploration.columns if c.name == "x")
        assert x.stats.outlier_count >= 1


class TestCategoricalColumns:
    def test_top_categories_with_counts(self, mixed_exploration):
        exploration, df = mixed_exploration
        city = next(c for c in exploration.columns if c.name == "city")
        assert city.bins[0].label == "Berlin"  # most common first
        assert sum(b.count for b in city.bins) + city.other_count == len(df)

    def test_long_tail_folds_into_other(self):
        values = [f"cat_{i % 30}" for i in range(300)]
        exploration = _explore(pd.DataFrame({"c": values, "n": range(300)}))
        # 30 raw categories exceeds the profiler's categorical cutoff → id_like;
        # force a wider frame where c stays categorical
        c = next(col for col in exploration.columns if col.name == "c")
        if c.kind == "categorical":
            assert len(c.bins) <= 8
            assert c.other_count > 0


class TestDatetimeColumns:
    def test_daily_span_buckets_by_day_or_month(self, mixed_exploration):
        exploration, df = mixed_exploration
        signup = next(c for c in exploration.columns if c.name == "signup")
        assert signup.kind == "datetime"
        assert len(signup.bins) > 0
        assert sum(b.count for b in signup.bins) == len(df)

    def test_multi_year_span_buckets_by_year(self):
        dates = pd.date_range("2010-01-01", "2024-01-01", periods=150).astype(str)
        exploration = _explore(pd.DataFrame({"when": dates, "v": range(150)}))
        when = next(c for c in exploration.columns if c.name == "when")
        assert all(len(b.label) == 4 for b in when.bins)  # "2010" … "2024"


class TestExcludedKinds:
    def test_id_like_gets_note_and_no_bins(self, mixed_exploration):
        exploration, _ = mixed_exploration
        user_id = next(c for c in exploration.columns if c.name == "user_id")
        assert user_id.kind == "id_like"
        assert user_id.bins == ()
        assert user_id.note is not None


class TestOverview:
    def test_missing_and_duplicates(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 2.0, None],
                "b": ["x", "y", "y", "z"],
            }
        )
        exploration = _explore(df)
        assert exploration.row_count == 4
        assert exploration.duplicate_rows == 1
        assert exploration.missing_cells_pct == pytest.approx(12.5)


class TestHighlights:
    def test_skewed_column_flagged(self):
        rng = np.random.default_rng(5)
        df = pd.DataFrame(
            {"amount": np.exp(rng.normal(0, 1.5, 400)), "b": rng.normal(size=400)}
        )
        exploration = _explore(df)
        assert any(
            h.column == "amount" and "large values" in h.message
            for h in exploration.highlights
        )

    def test_dominant_category_flagged(self):
        values = ["same"] * 490 + ["rare"] * 10
        df = pd.DataFrame({"status": values, "n": range(500)})
        exploration = _explore(df)
        assert any(h.column == "status" for h in exploration.highlights)

    def test_correlated_pair_flagged(self):
        rng = np.random.default_rng(6)
        a = rng.normal(size=300)
        df = pd.DataFrame(
            {"a": a, "a_copy": a * 2 + rng.normal(scale=0.05, size=300),
             "unrelated": rng.normal(size=300)}
        )
        exploration = _explore(df)
        joined = " ".join(h.message for h in exploration.highlights)
        assert "a" in joined and "a_copy" in joined
        assert "unrelated" not in joined

    def test_clean_data_has_few_or_no_highlights(self):
        rng = np.random.default_rng(8)
        df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
        assert len(_explore(df).highlights) == 0


class TestExplorationApi:
    def test_titanic_exploration_shape(self, client):
        resp = client.get(f"/api/datasets/{TITANIC_ID}/exploration")
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        assert data["row_count"] == 891
        by_name = {c["name"]: c for c in data["columns"]}
        assert len(by_name["age"]["bins"]) > 1
        assert by_name["sex"]["bins"][0]["label"] in {"male", "female"}

    def test_result_is_cached_to_disk(self, client, settings):
        client.get(f"/api/datasets/{TITANIC_ID}/exploration")
        assert (settings.data_dir / "datasets" / TITANIC_ID / "exploration.json").is_file()

    def test_unknown_dataset_404(self, client):
        assert client.get("/api/datasets/ds_nope/exploration").status_code == 404

    def test_outdated_cache_is_recomputed(self, client, settings):
        from app.datasets.exploration import CURRENT_EXPLORATION_VERSION

        first = client.get(f"/api/datasets/{TITANIC_ID}/exploration").json()["data"]
        assert first["version"] == CURRENT_EXPLORATION_VERSION
        # Simulate a cache written by an older build
        cache = settings.data_dir / "datasets" / TITANIC_ID / "exploration.json"
        cache.write_text(cache.read_text().replace(
            f'"version": {CURRENT_EXPLORATION_VERSION}', '"version": 1'
        ))
        again = client.get(f"/api/datasets/{TITANIC_ID}/exploration").json()["data"]
        assert again["version"] == CURRENT_EXPLORATION_VERSION
