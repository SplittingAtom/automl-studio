"""Filesystem-backed model storage: meta.json + model.joblib per model."""

import json
import logging
from pathlib import Path

import joblib

from app.training.preprocessing import FeatureSpec
from app.training.schemas import ModelMeta

logger = logging.getLogger(__name__)

META_FILE = "meta.json"
ARTIFACT_FILE = "model.joblib"


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

    def save_artifact(self, model_id: str, model, spec: FeatureSpec) -> None:
        joblib.dump({"model": model, "spec": spec}, self._root / model_id / ARTIFACT_FILE)

    def load_artifact(self, model_id: str) -> tuple[object, FeatureSpec]:
        artifact = joblib.load(self._root / model_id / ARTIFACT_FILE)
        return artifact["model"], artifact["spec"]

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
