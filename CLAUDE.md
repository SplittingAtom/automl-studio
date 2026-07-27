# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A prototype AutoML web app bringing PowerBI-style ease of use to XGBoost: upload a tabular CSV (or pick a bundled sample), choose a target column, the app trains a model automatically, and the user explores it through a live what-if panel and a feature-importance chart. The target user is an analyst, not an ML engineer — all UI copy is plain English, ML jargon is hidden, and training "just works" with sensible defaults. Local single-user prototype: no auth, no database; datasets and models persist to disk under `backend/app/data/` (gitignored).

## Commands

```bash
make install        # uv sync (backend) + npm install (frontend)
make dev            # run both servers: FastAPI on :8000, Vite on :5173
make test           # pytest + vitest

# Backend (from backend/)
uv run pytest                          # all tests
uv run pytest tests/test_trainer.py -k leakage   # single test
uv run uvicorn app.main:create_app --factory --reload --port 8000

# Frontend (from frontend/)
npm run dev / npm test / npm run typecheck / npm run lint
```

There is no module-level `app` in `app/main.py` — always launch uvicorn with `--factory`. macOS needs `brew install libomp` for XGBoost.

The Vite dev server proxies `/api` to `localhost:8000` (no CORS anywhere); the frontend only ever calls relative `/api/...` paths.

## Architecture

Monorepo: `backend/` (FastAPI + XGBoost, uv-managed) and `frontend/` (React 19 + Vite + TanStack Query + Recharts + zod).

### The FeatureSpec contract (most important concept)

`backend/app/training/preprocessing.py` defines `FeatureSpec` — an immutable description of the model's inputs (feature columns with kinds/categories/bounds/defaults, target with classes, excluded columns with reasons). It is built once at train time, persisted inside `model.joblib` alongside the fitted model, and `apply_feature_spec()` is the single function that converts raw data into the model's input frame — used identically for training data and single-row what-if predictions. The API's `input_spec` (which drives the frontend's sliders/dropdowns) is derived directly from it. Any preprocessing change must go through this contract or train/predict will disagree.

### Data flow

1. **Upload** → `api/datasets.py` → `csv_loading.py` (CSV with encoding fallback + delimiter sniffing, or .xlsx via openpyxl) → `datasets/profiling.py` classifies every column (`numeric`/`categorical`/`id_like`/`datetime`/`unsupported`) → profile + plain-English warnings stored in `meta.json`. Column-kind heuristics live only in `profiling.py`; they decide feature inclusion, slider vs dropdown, and target eligibility.
1b. **Calculated columns** → `POST /api/datasets/{id}/calculated` (`datasets/calculated.py`) evaluates a user formula via guarded `df.eval` (blocklist: dunders, attribute access, assignment, indexing) and materializes a NEW dataset with `source: "derived"` — datasets stay immutable, and the new column flows through analysis/training/what-if like any other (it can even be the target). inf → NaN; formula errors return friendly 422s.
2. **Analyze** → `GET /api/datasets/{id}/analysis` (`analysis/analyzer.py`) grades dataset suitability and ranks target candidates by actually probing predictability: a small fast XGBoost per eligible column measures how well the other columns predict it vs a naive baseline (rows capped at 3k, candidates at 30; cached to `analysis.json` since datasets are immutable). Near-perfect probes (accuracy ≥ 0.99 / R² ≥ 0.99) or one predictor holding ≥ 90% importance mark a candidate `derived_like` — penalized, never recommended, skipped for the headline summary (they're calculated columns like MonthlyCharges, not interesting targets). The probes' importances feed "best predictor" hints in the configure UI.
3. **Train** → `api/models.py` → `training/service.py` creates a `queued` job record, runs it via FastAPI `BackgroundTasks` (status: `queued → training → complete|failed`, frontend polls 1s). `trainer.py` does split/early-stopping/metrics/importance, with balanced sample weights when the minority class is <10%. Every run also produces: 5-fold CV stats (`cv_mean/cv_std`, capped at 8k rows), a persisted validation frame of the held-out 20% with actual-vs-predicted (`validation.csv`, served by `GET /{id}/validation` + `/download`), a threshold curve for binary classification (`metrics.threshold_curve`, drives the cut-off slider), and for regression a quantile interval model (80% band, stored in the joblib artifact, surfaced as `PredictResponse.interval`). `TrainRequest.effort: "thorough"` runs a 12-trial random search before the final fit. `training/baseline.py` also trains a naive baseline and a linear model (its own contained one-hot/impute path — it never touches the FeatureSpec contract) purely for context: metrics gain `baseline_*`/`linear_*` keys, and a `SIMPLE_RELATIONSHIPS` warning fires when the linear model nearly matches XGBoost. No preprocessing pipeline: XGBoost native categoricals (`enable_categorical=True`), NaN passthrough, top-50 category cap with "Other". Datetime columns become derived numeric features (`{col}_year/_month/_dayofweek`) via `FeatureColumn.derived_from` — at train time they're parsed from the raw column, at predict time the what-if inputs carry the derived names directly. Users can exclude columns per-run (`TrainRequest.excluded_columns` → reason `user_excluded`); `GET /api/models?dataset_id=` lists runs for the comparison page.
4. **Predict** → `prediction/predictor.py` validates what-if inputs against the FeatureSpec (unknown key → 422, missing key → default), served from `model_cache.py` (in-memory LRU) at ~1ms warm. Every prediction includes a per-prediction explanation from `prediction/explanation.py` — XGBoost's native TreeSHAP (`pred_contribs=True`), deliberately no `shap` dependency; classification contributions are log-odds toward `toward_label` (binary: always `classes[1]`). `prediction/sensitivity.py` serves ICE curves (`POST /{id}/sensitivity`): vary one feature over its range, others held at the current what-if values.

### Conventions

- **API envelope**: every response is `{success, data, error: {code, message}|null, meta}` — routes return `ok(...)` or raise `AppError(code, message, status)`; handlers in `api/envelope.py`. Error messages are user-facing: friendly, no jargon, no stack traces.
- **Frontend boundary**: `src/api/schemas.ts` (zod) mirrors the backend Pydantic schemas field-for-field; `client.ts` unwraps the envelope and throws typed `ApiError`. Keep the two schema files in sync when changing API shapes.
- **Suggest-and-retrain**: training also computes `suggested_exclusions` (dataset columns whose features all scored <1% importance; derived date parts grouped by source) and `leak_suspect` (POSSIBLE_LEAKAGE feature mapped back to its dataset column). The results page renders these as one-click "Retrain without it" actions that re-POST `/api/models` with merged `excluded_columns` — never silently drop columns; suggest and let the user compare runs.
- **Warnings pattern**: guardrails don't block, they warn — `ProfileWarning{code, message, column}` objects flow from profiling/training through meta into the UI's `WarningBanner` verbatim (e.g. `POSSIBLE_LEAKAGE` when test score > 0.999, `ROW_SAMPLE` when >100k rows sampled, `CLASS_IMBALANCE`). Refusals (`TrainingError`) are reserved for unusable data (<50 rows, >30% missing target).
- **Immutability**: all Pydantic models are frozen (tuples, `model_copy(update=...)` for changes); repositories (`datasets/repository.py`, `training/repository.py`) take an injected root path — tests construct them on `tmp_path`, so tests never share state.
- **Sample datasets**: `backend/sample_data/*.csv` are committed and seeded on startup with fixed ids (`ds_sample_titanic`, `ds_sample_housing`) by `datasets/samples.py`. Titanic is deliberately trimmed to the classic 8 columns — the seaborn original contains `alive`, a duplicate of `survived` that leaks the answer.

### Testing

Backend: contract tests per router in `tests/test_api_*.py` (note: `TestClient` runs `BackgroundTasks` synchronously, so training is complete when the POST returns) plus unit tests on the pure modules (`profiling`, `task_detection`, `preprocessing`, `trainer`, `predictor`) with synthetic seeded dataframes. Frontend: vitest for the API client (envelope/error paths), what-if reducer, and formatters. Write tests first for any new pure-function behavior.
