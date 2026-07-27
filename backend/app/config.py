"""Application settings. Frozen so config is never mutated at runtime."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    data_dir: Path = BACKEND_ROOT / "app" / "data"
    sample_data_dir: Path = BACKEND_ROOT / "sample_data"

    max_upload_bytes: int = 50 * 1024 * 1024
    max_training_rows: int = 100_000
    min_training_rows: int = 50
    small_dataset_rows: int = 500
    max_target_missing_pct: float = 30.0
    max_categories: int = 50
    profile_top_values: int = 20
    preview_rows_default: int = 20
    preview_rows_max: int = 200
    model_cache_size: int = 8

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"
