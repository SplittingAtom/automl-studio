"""Filesystem-backed dataset storage: data.csv + meta.json per dataset."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from app.analysis.schemas import DatasetAnalysis
from app.datasets.profiling import profile_dataframe
from app.datasets.schemas import DatasetMeta

META_FILE = "meta.json"
DATA_FILE = "data.csv"
ANALYSIS_FILE = "analysis.json"


class DatasetRepository:
    def __init__(self, root: Path):
        self._root = root
        root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        source: str,
        dataset_id: str | None = None,
    ) -> DatasetMeta:
        profile = profile_dataframe(df)
        meta = DatasetMeta(
            id=dataset_id or f"ds_{uuid.uuid4().hex[:10]}",
            name=name,
            source=source,
            created_at=datetime.now(UTC).isoformat(),
            row_count=len(df),
            column_count=len(df.columns),
            columns=profile.columns,
            warnings=profile.warnings,
        )
        target = self._root / meta.id
        target.mkdir(parents=True, exist_ok=True)
        df.to_csv(target / DATA_FILE, index=False)
        (target / META_FILE).write_text(meta.model_dump_json(indent=2))
        return meta

    def get(self, dataset_id: str) -> DatasetMeta | None:
        meta_path = self._root / dataset_id / META_FILE
        if not meta_path.is_file():
            return None
        return DatasetMeta.model_validate(json.loads(meta_path.read_text()))

    def list(self) -> list[DatasetMeta]:
        metas = [
            self.get(entry.name)
            for entry in sorted(self._root.iterdir())
            if (entry / META_FILE).is_file()
        ]
        return sorted(
            (m for m in metas if m is not None),
            key=lambda m: m.created_at,
            reverse=True,
        )

    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        return pd.read_csv(self._root / dataset_id / DATA_FILE)

    def get_analysis(self, dataset_id: str) -> "DatasetAnalysis | None":
        path = self._root / dataset_id / ANALYSIS_FILE
        if not path.is_file():
            return None
        return DatasetAnalysis.model_validate(json.loads(path.read_text()))

    def save_analysis(self, dataset_id: str, analysis: "DatasetAnalysis") -> None:
        (self._root / dataset_id / ANALYSIS_FILE).write_text(
            analysis.model_dump_json(indent=2)
        )
