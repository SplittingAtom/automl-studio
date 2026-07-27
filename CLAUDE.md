# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Status

**Greenfield — no code exists yet.** This repository currently contains only this file. There are no build, test, or lint commands to document until the initial implementation lands. Update this file as soon as the stack, tooling, and structure are established.

## What This Project Is

A prototype AutoML web application that brings PowerBI-style ease of use to traditional ML models (e.g., XGBoost).

Core user flow:

1. **Connect data** — the user points the web app at a tabular dataset.
2. **Build a model** — the user selects an outcome/target column and the app trains an ML model to predict it (XGBoost or similar gradient-boosted / traditional tabular models, not deep learning).
3. **Explore interactively** — an intuitive interface lets the user adjust input variables ("what-if" style) and immediately see how predictions change.

## Design Principles

- **Ease of use over configurability.** The target user is a PowerBI-style analyst, not an ML engineer. Sensible defaults everywhere; ML jargon hidden or explained. Model training should "just work" from a dataset + target column.
- **Prototype scope.** Favor the simplest thing that demonstrates the flow end-to-end (upload → train → interactive what-if exploration). Avoid premature investment in auth, multi-tenancy, or production infrastructure unless asked.
- **Tabular data first.** CSV/spreadsheet-shaped data is the primary input. Expect messy real-world data: missing values, mixed types, and categorical columns need automatic handling.
- **Interactivity is the point.** The differentiator is the variable-adjustment UI for exploring predictions, not model sophistication. Fast feedback (sub-second prediction on slider changes) matters more than squeezing out model accuracy.

## Decisions Not Yet Made

The following are open — confirm with the user before locking in, then record the choice here:

- Frontend framework and charting library
- Backend language/framework (Python is the natural fit for XGBoost/scikit-learn, but not yet decided)
- How datasets are provided (file upload, URL, database connection)
- Classification vs. regression support (likely both)
- Where/how trained models are persisted
