"""Filesystem-backed model storage: meta.json + model.joblib per model."""

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from app.analysis.schemas import FeatureIdeasResponse
from app.prediction.schemas import (
    BlueprintResponse,
    GroupCheckResponse,
    InsightsResponse,
)
from app.training.preprocessing import FeatureSpec
from app.training.schemas import ModelMeta

logger = logging.getLogger(__name__)

META_FILE = "meta.json"
ARTIFACT_FILE = "model.joblib"
VALIDATION_FILE = "validation.csv"
VALIDATION_DTYPES_FILE = "validation_dtypes.json"
INSIGHTS_FILE = "insights.json"
FEATURE_IDEAS_FILE = "feature_ideas.json"
BLUEPRINT_FILE = "blueprint.json"
GROUP_CHECK_FILE = "group_check.json"


class ModelRepository:
    def __init__(self, root: Path):
        self._root = root
        root.mkdir(parents=True, exist_ok=True)

    def save_meta(self, meta: ModelMeta) -> ModelMeta:
        target = self._root / meta.id
        target.mkdir(parents=True, exist_ok=True)
        (target / META_FILE).write_text(meta.model_dump_json(indent=2))
        return meta

    def get(self, model_id: str) -> ModelMeta | None:
        meta_path = self._root / model_id / META_FILE
        if not meta_path.is_file():
            return None
        return ModelMeta.model_validate(json.loads(meta_path.read_text()))

    def list(self, dataset_id: str | None = None) -> list[ModelMeta]:
        metas = [
            self.get(entry.name)
            for entry in sorted(self._root.iterdir())
            if (entry / META_FILE).is_file()
        ]
        found = [m for m in metas if m is not None]
        if dataset_id is not None:
            found = [m for m in found if m.dataset_id == dataset_id]
        return sorted(found, key=lambda m: m.created_at, reverse=True)

    def save_artifact(
        self, model_id: str, model, spec: FeatureSpec, interval_model=None
    ) -> None:
        joblib.dump(
            {"model": model, "spec": spec, "interval_model": interval_model},
            self._root / model_id / ARTIFACT_FILE,
        )

    def load_artifact(self, model_id: str) -> dict:
        artifact = joblib.load(self._root / model_id / ARTIFACT_FILE)
        return {
            "model": artifact["model"],
            "spec": artifact["spec"],
            "interval_model": artifact.get("interval_model"),
        }

    def save_validation(
        self, model_id: str, frame: pd.DataFrame, text_columns: tuple[str, ...] = ()
    ) -> None:
        """Persist the validation frame, remembering which columns are text.

        Without the sidecar, categorical values that look numeric ("01",
        "TRUE") would be re-inferred on read into dtypes that no longer match
        their FeatureSpec categories.
        """
        frame.to_csv(self._root / model_id / VALIDATION_FILE, index=False)
        (self._root / model_id / VALIDATION_DTYPES_FILE).write_text(
            json.dumps({"text_columns": list(text_columns)})
        )

    def validation_path(self, model_id: str) -> Path:
        return self._root / model_id / VALIDATION_FILE

    def load_validation(self, model_id: str) -> pd.DataFrame | None:
        """Read the validation frame back, text columns preserved as strings."""
        path = self.validation_path(model_id)
        if not path.is_file():
            return None
        dtypes_path = self._root / model_id / VALIDATION_DTYPES_FILE
        text_columns: list[str] = (
            json.loads(dtypes_path.read_text()).get("text_columns", [])
            if dtypes_path.is_file()
            else []
        )
        return pd.read_csv(path, dtype=dict.fromkeys(text_columns, str))

    def _load_cached(self, model_id: str, filename: str, model_cls):
        path = self._root / model_id / filename
        if not path.is_file():
            return None
        return model_cls.model_validate(json.loads(path.read_text()))

    def _save_cached(self, model_id: str, filename: str, payload) -> None:
        (self._root / model_id / filename).write_text(payload.model_dump_json(indent=2))

    def get_insights(self, model_id: str) -> InsightsResponse | None:
        return self._load_cached(model_id, INSIGHTS_FILE, InsightsResponse)

    def save_insights(self, model_id: str, insights: InsightsResponse) -> None:
        self._save_cached(model_id, INSIGHTS_FILE, insights)

    def get_feature_ideas(self, model_id: str) -> FeatureIdeasResponse | None:
        return self._load_cached(model_id, FEATURE_IDEAS_FILE, FeatureIdeasResponse)

    def save_feature_ideas(self, model_id: str, ideas: FeatureIdeasResponse) -> None:
        self._save_cached(model_id, FEATURE_IDEAS_FILE, ideas)

    def get_blueprint(self, model_id: str) -> BlueprintResponse | None:
        return self._load_cached(model_id, BLUEPRINT_FILE, BlueprintResponse)

    def save_blueprint(self, model_id: str, blueprint: BlueprintResponse) -> None:
        self._save_cached(model_id, BLUEPRINT_FILE, blueprint)

    def get_group_check(self, model_id: str) -> GroupCheckResponse | None:
        return self._load_cached(model_id, GROUP_CHECK_FILE, GroupCheckResponse)

    def save_group_check(self, model_id: str, check: GroupCheckResponse) -> None:
        self._save_cached(model_id, GROUP_CHECK_FILE, check)

    def fail_interrupted(self) -> None:
        """On startup, mark models stuck in queued/training as failed."""
        for entry in self._root.iterdir():
            meta = self.get(entry.name)
            if meta is not None and meta.status in ("queued", "training"):
                self.save_meta(
                    meta.model_copy(
                        update={
                            "status": "failed",
                            "error": "Training was interrupted by a server restart. Please train again.",
                        }
                    )
                )
                logger.info("Marked interrupted model %s as failed", meta.id)
