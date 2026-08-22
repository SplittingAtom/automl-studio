# AutoML Studio

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PowerBI-style ease of use for machine learning. Upload a tabular CSV (or pick a
bundled sample), choose the column you want to predict, and the app trains an
XGBoost model automatically — then lets you explore it through live what-if
analysis, plain-English explainability, and honest quality checks. Built for
analysts: no ML background needed, no jargon in the UI, sensible defaults
everywhere.

## What it does

- **Data profiling** — per-column distributions, stats, outlier counts, and
  statistically-selected "Worth a look" callouts (skew, duplicates, correlated
  columns), plus PowerBI-style calculated columns.
- **Automatic training** — target-suitability analysis, honest train/test
  evaluation with cross-validated consistency ranges, naive and linear
  baselines for context, leakage detection, and class-imbalance handling.
- **Explainability** — live what-if scenarios with per-prediction explanations,
  sensitivity curves, a SHAP beeswarm, an association heatmap, a simplified
  decision flowchart, per-group reliability checks, and probability
  calibration.
- **Improvement loop** — one-click retrain suggestions, probe-scored
  calculated-column ideas, auto-compare of three training approaches into a
  ranked leaderboard, and expert fine-tuning with safe-ranged knobs and
  monotonicity constraints.
- **Time series** — chronological evaluation with embargo gaps, lag features,
  and recursive future forecasting with prediction bands.
- **Sharing** — a downloadable one-page HTML model report and a self-contained
  scoring kit (`model + predict.py`) that runs anywhere with Python.

There's a full feature guide and how-to documentation inside the app at
`/help`.

## Setup

### Prerequisites

- **Python 3.12+** and [uv](https://docs.astral.sh/uv/) (backend package
  manager)
- **Node.js 20+** and npm
- **macOS only**: XGBoost needs OpenMP — `brew install libomp`

### Install & run

```bash
git clone https://github.com/SplittingAtom/automl-studio.git
cd automl-studio
make install   # uv sync (backend) + npm install (frontend)
make dev       # FastAPI on :8000 + Vite on :5173
```

Open **http://localhost:5173**, pick a sample dataset (Titanic is a good
start), and build your first model.

### Useful commands

| Command | What it does |
|---|---|
| `make dev` | Run both servers with hot reload |
| `make backend` / `make frontend` | Run one side only |
| `make test` | Backend pytest + frontend vitest |
| `cd backend && uv run pytest` | Backend tests only |
| `cd frontend && npm run typecheck && npm run lint` | Frontend checks |

The Vite dev server proxies `/api` to the backend, so the frontend only ever
calls relative paths — no CORS configuration needed.

## Notes

- This is a **local, single-user prototype**: no auth, no database. Uploaded
  datasets and trained models persist to disk under `backend/app/data/`
  (gitignored).
- The backend has no module-level app object — always launch uvicorn with
  `--factory` (the Makefile does this for you).
- Sample datasets are seeded automatically on first startup.

## Architecture (short version)

A monorepo: `backend/` (FastAPI + XGBoost, uv-managed) and `frontend/`
(React 19 + Vite + TanStack Query + Recharts + zod). The core contract is the
**FeatureSpec** — an immutable description of the model's inputs built at
train time and applied identically to training data and what-if predictions,
so train and predict can never disagree. Expensive derived views (profiling,
insights, reports) are computed once per immutable dataset/model and cached to
disk as JSON. See `CLAUDE.md` for the full architecture notes.

## License

This project is licensed under the [MIT License](LICENSE).
