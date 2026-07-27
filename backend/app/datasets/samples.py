"""Bundled sample datasets seeded on startup so the app is demoable instantly."""

import logging
from pathlib import Path

import pandas as pd

from app.datasets.repository import DatasetRepository

logger = logging.getLogger(__name__)

SAMPLE_DATASETS = (
    {"id": "ds_sample_titanic", "filename": "titanic.csv", "name": "Titanic Passengers"},
    {"id": "ds_sample_housing", "filename": "california_housing.csv", "name": "California Housing"},
    {"id": "ds_sample_churn", "filename": "telco_churn.csv", "name": "Telco Customer Churn"},
    {"id": "ds_sample_penguins", "filename": "penguins.csv", "name": "Palmer Penguins"},
    {"id": "ds_sample_bikes", "filename": "bike_rentals.csv", "name": "Bike Rental Demand"},
    {"id": "ds_sample_diamonds", "filename": "diamonds.csv", "name": "Diamond Prices"},
    {"id": "ds_sample_income", "filename": "census_income.csv", "name": "Census Income"},
    {"id": "ds_sample_heart", "filename": "heart_disease.csv", "name": "Heart Disease"},
)


def seed_samples(repo: DatasetRepository, sample_dir: Path) -> None:
    for sample in SAMPLE_DATASETS:
        if repo.get(sample["id"]) is not None:
            continue
        path = sample_dir / sample["filename"]
        if not path.is_file():
            logger.warning("Sample dataset missing, skipping: %s", path)
            continue
        df = pd.read_csv(path)
        repo.save(df, name=sample["name"], source="sample", dataset_id=sample["id"])
        logger.info("Seeded sample dataset %s (%d rows)", sample["id"], len(df))
