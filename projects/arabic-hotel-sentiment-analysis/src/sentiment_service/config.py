from dataclasses import dataclass
import os
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    model_path: Path = Path(os.getenv("MODEL_PATH", "artifacts/tfidf_svm.joblib"))
    model_version: str = os.getenv("MODEL_VERSION", "tfidf-svm-v1")
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "100"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
